"""
ModelServer - Optional HTTP server for model inference.

Provides a simple REST API for model predictions using Flask.
Can be extended to use FastAPI for async/production deployments.

Usage:
    # Start server
    python scripts/serve_model.py --bundle ./bundles/xgb_h20 --port 8080

    # Make predictions
    curl -X POST http://localhost:8080/predict \
        -H "Content-Type: application/json" \
        -d '{"features": [[0.1, 0.2, ...]]}'

Note:
    Flask is optional. Install with: pip install flask
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)


# =============================================================================
# SERVER CONFIGURATION
# =============================================================================

@dataclass
class ServerConfig:
    """Configuration for model server."""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    max_batch_size: int = 1000
    timeout_seconds: float = 30.0
    enable_metrics: bool = True


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

@dataclass
class PredictionRequest:
    """Request format for predictions."""
    features: list[list[float]]  # 2D array of features
    calibrate: bool = True
    return_probabilities: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionRequest:
        return cls(
            features=data["features"],
            calibrate=data.get("calibrate", True),
            return_probabilities=data.get("return_probabilities", True),
        )


@dataclass
class PredictionResponse:
    """Response format for predictions."""
    predictions: list[int]
    probabilities: list[list[float]] | None = None
    confidence: list[float] | None = None
    inference_time_ms: float = 0.0
    model_name: str = ""
    horizon: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = {
            "predictions": self.predictions,
            "inference_time_ms": self.inference_time_ms,
            "model_name": self.model_name,
            "horizon": self.horizon,
        }
        if self.probabilities is not None:
            result["probabilities"] = self.probabilities
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


# =============================================================================
# MODEL SERVER
# =============================================================================

class ModelServer:
    """
    Simple HTTP server for model inference.

    Provides REST endpoints for:
    - /health - Health check
    - /info - Model information
    - /predict - Single/batch predictions
    - /metrics - (optional) Server metrics

    Example:
        >>> server = ModelServer.from_bundle("./bundles/xgb_h20")
        >>> server.run(host="0.0.0.0", port=8080)
    """

    def __init__(
        self,
        pipeline: InferencePipeline,
        config: ServerConfig | None = None,
    ) -> None:
        """
        Initialize ModelServer.

        Args:
            pipeline: InferencePipeline for predictions
            config: Server configuration
        """
        self.pipeline = pipeline
        self.config = config or ServerConfig()

        # Metrics tracking
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0

        self._app = None

    @classmethod
    def from_bundle(
        cls,
        path: str | Path,
        config: ServerConfig | None = None,
    ) -> ModelServer:
        """Create server from a model bundle."""
        pipeline = InferencePipeline.from_bundle(path)
        return cls(pipeline, config)

    @classmethod
    def from_bundles(
        cls,
        paths: list[str | Path],
        config: ServerConfig | None = None,
    ) -> ModelServer:
        """Create server from multiple bundles (ensemble)."""
        pipeline = InferencePipeline.from_bundles(paths)
        return cls(pipeline, config)

    def create_app(self):
        """
        Create Flask application.

        Returns:
            Flask app instance

        Raises:
            ImportError: If Flask is not installed
        """
        try:
            from flask import Flask, jsonify, request
        except ImportError:
            raise ImportError(
                "Flask is required for model serving. "
                "Install with: pip install flask"
            )

        app = Flask(__name__)

        @app.route("/health", methods=["GET"])
        def health():
            """Health check endpoint."""
            return jsonify({"status": "healthy"})

        @app.route("/info", methods=["GET"])
        def info():
            """Model information endpoint."""
            return jsonify({
                "models": self.pipeline.get_model_info(),
                "horizon": self.pipeline.horizon,
                "n_features": len(self.pipeline.feature_columns),
                "feature_columns": self.pipeline.feature_columns,
            })

        @app.route("/predict", methods=["POST"])
        def predict():
            """Prediction endpoint."""
            start_time = time.perf_counter()

            try:
                data = request.get_json()
                req = PredictionRequest.from_dict(data)

                # Validate batch size
                if len(req.features) > self.config.max_batch_size:
                    return jsonify({
                        "error": f"Batch size exceeds maximum ({self.config.max_batch_size})"
                    }), 400

                # Make predictions
                X = np.array(req.features, dtype=np.float32)
                result = self.pipeline.predict(X, calibrate=req.calibrate)

                inference_time = (time.perf_counter() - start_time) * 1000

                # Build response
                response = PredictionResponse(
                    predictions=result.predictions.class_predictions.tolist(),
                    inference_time_ms=inference_time,
                    model_name=self.pipeline.model_names[0],
                    horizon=self.pipeline.horizon,
                )

                if req.return_probabilities:
                    response.probabilities = result.predictions.class_probabilities.tolist()
                    response.confidence = result.predictions.confidence.tolist()

                # Update metrics
                self._request_count += 1
                self._total_latency_ms += inference_time

                return jsonify(response.to_dict())

            except Exception as e:
                self._error_count += 1
                logger.error(f"Prediction error: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/predict_ensemble", methods=["POST"])
        def predict_ensemble():
            """Ensemble prediction endpoint."""
            if self.pipeline.n_models < 2:
                return jsonify({
                    "error": "Ensemble requires multiple models"
                }), 400

            start_time = time.perf_counter()

            try:
                data = request.get_json()
                req = PredictionRequest.from_dict(data)

                X = np.array(req.features, dtype=np.float32)
                method = data.get("voting_method", "soft_vote")

                result = self.pipeline.predict_ensemble(
                    X, method=method, calibrate=req.calibrate
                )

                return jsonify({
                    "predictions": result.predictions.class_predictions.tolist(),
                    "probabilities": result.predictions.class_probabilities.tolist() if req.return_probabilities else None,
                    "confidence": result.predictions.confidence.tolist() if req.return_probabilities else None,
                    "voting_method": result.voting_method,
                    "inference_time_ms": result.inference_time_ms,
                    "models_used": [r.model_name for r in result.individual_results],
                })

            except Exception as e:
                logger.error(f"Ensemble prediction error: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/metrics", methods=["GET"])
        def metrics():
            """Server metrics endpoint."""
            if not self.config.enable_metrics:
                return jsonify({"error": "Metrics disabled"}), 404

            avg_latency = (
                self._total_latency_ms / self._request_count
                if self._request_count > 0 else 0
            )

            return jsonify({
                "request_count": self._request_count,
                "error_count": self._error_count,
                "error_rate": self._error_count / max(self._request_count, 1),
                "average_latency_ms": avg_latency,
            })

        self._app = app
        return app

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
    ) -> None:
        """
        Run the server.

        Args:
            host: Host to bind to
            port: Port to listen on
            debug: Enable debug mode
        """
        if self._app is None:
            self.create_app()

        host = host or self.config.host
        port = port or self.config.port
        debug = debug if debug is not None else self.config.debug

        logger.info(f"Starting model server on {host}:{port}")
        logger.info(f"Models: {self.pipeline.model_names}")

        self._app.run(host=host, port=port, debug=debug)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def start_server(
    bundle_path: str | Path,
    host: str = "0.0.0.0",
    port: int = 8080,
    debug: bool = False,
) -> None:
    """
    Start model server with a single bundle.

    Args:
        bundle_path: Path to model bundle
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode
    """
    config = ServerConfig(host=host, port=port, debug=debug)
    server = ModelServer.from_bundle(bundle_path, config)
    server.run()


__all__ = [
    "ModelServer",
    "ServerConfig",
    "PredictionRequest",
    "PredictionResponse",
    "start_server",
]
