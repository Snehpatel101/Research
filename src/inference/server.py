"""
ModelServer - HTTP server for model inference using FastAPI.

Provides a production-ready REST API for model predictions with:
- Async support for high concurrency
- Automatic OpenAPI documentation (Swagger UI at /docs)
- Pydantic validation for request/response
- Health checks, metrics, and model info endpoints

Usage:
    # Start server
    python scripts/serve_model.py --bundle ./bundles/xgb_h20 --port 8080

    # Make predictions
    curl -X POST http://localhost:8080/predict \
        -H "Content-Type: application/json" \
        -d '{"features": [[0.1, 0.2, ...]]}'

    # View API docs
    http://localhost:8080/docs

Note:
    FastAPI is optional. Install with: pip install fastapi uvicorn
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
    workers: int = 1


# =============================================================================
# PYDANTIC REQUEST/RESPONSE MODELS
# =============================================================================

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda *args, **kwargs: None


class PredictionRequest(BaseModel if PYDANTIC_AVAILABLE else object):
    """Request format for predictions."""

    features: list[list[float]] = Field(..., description="2D array of features")
    calibrate: bool = Field(default=True, description="Apply calibration if available")
    return_probabilities: bool = Field(default=True, description="Return class probabilities")

    if not PYDANTIC_AVAILABLE:

        def __init__(self, features, calibrate=True, return_probabilities=True):
            self.features = features
            self.calibrate = calibrate
            self.return_probabilities = return_probabilities

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> PredictionRequest:
            return cls(
                features=data["features"],
                calibrate=data.get("calibrate", True),
                return_probabilities=data.get("return_probabilities", True),
            )


class PredictionResponse(BaseModel if PYDANTIC_AVAILABLE else object):
    """Response format for predictions."""

    predictions: list[int] = Field(..., description="Predicted class labels")
    probabilities: list[list[float]] | None = Field(None, description="Class probabilities")
    confidence: list[float] | None = Field(None, description="Prediction confidence scores")
    inference_time_ms: float = Field(0.0, description="Inference time in milliseconds")
    model_name: str = Field("", description="Name of the model used")
    horizon: int = Field(0, description="Prediction horizon")

    if not PYDANTIC_AVAILABLE:

        def __init__(
            self,
            predictions,
            probabilities=None,
            confidence=None,
            inference_time_ms=0.0,
            model_name="",
            horizon=0,
        ):
            self.predictions = predictions
            self.probabilities = probabilities
            self.confidence = confidence
            self.inference_time_ms = inference_time_ms
            self.model_name = model_name
            self.horizon = horizon

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


class EnsemblePredictionRequest(BaseModel if PYDANTIC_AVAILABLE else object):
    """Request format for ensemble predictions."""

    features: list[list[float]] = Field(..., description="2D array of features")
    calibrate: bool = Field(default=True, description="Apply calibration if available")
    return_probabilities: bool = Field(default=True, description="Return class probabilities")
    voting_method: str = Field(
        default="soft_vote", description="Voting method: soft_vote or hard_vote"
    )


class HealthResponse(BaseModel if PYDANTIC_AVAILABLE else object):
    """Health check response."""

    status: str = Field(..., description="Server status")


class ModelInfoResponse(BaseModel if PYDANTIC_AVAILABLE else object):
    """Model information response."""

    models: list[dict[str, Any]] = Field(..., description="Information about loaded models")
    horizon: int = Field(..., description="Prediction horizon")
    n_features: int = Field(..., description="Number of input features")
    feature_columns: list[str] = Field(..., description="Feature column names")


class MetricsResponse(BaseModel if PYDANTIC_AVAILABLE else object):
    """Server metrics response."""

    request_count: int = Field(..., description="Total number of requests")
    error_count: int = Field(..., description="Total number of errors")
    error_rate: float = Field(..., description="Error rate")
    average_latency_ms: float = Field(..., description="Average latency in milliseconds")


class ErrorResponse(BaseModel if PYDANTIC_AVAILABLE else object):
    """Error response."""

    error: str = Field(..., description="Error message")


# =============================================================================
# MODEL SERVER
# =============================================================================


class ModelServer:
    """
    Production-ready HTTP server for model inference using FastAPI.

    Provides REST endpoints for:
    - GET /health - Health check
    - GET /info - Model information
    - POST /predict - Single/batch predictions
    - POST /predict_ensemble - Ensemble predictions
    - GET /metrics - Server metrics
    - GET /docs - Swagger UI documentation

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
        Create FastAPI application.

        Returns:
            FastAPI app instance

        Raises:
            ImportError: If FastAPI is not installed
        """
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
        except ImportError:
            raise ImportError(
                "FastAPI is required for model serving. "
                "Install with: pip install fastapi uvicorn"
            )

        app = FastAPI(
            title="ML Model Server",
            description="Production-ready REST API for ML model inference",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        @app.get("/health", response_model=HealthResponse, tags=["Health"])
        async def health():
            """Health check endpoint."""
            return {"status": "healthy"}

        @app.get("/info", response_model=ModelInfoResponse, tags=["Info"])
        async def info():
            """Model information endpoint."""
            return {
                "models": self.pipeline.get_model_info(),
                "horizon": self.pipeline.horizon,
                "n_features": len(self.pipeline.feature_columns),
                "feature_columns": self.pipeline.feature_columns,
            }

        @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
        async def predict(request: PredictionRequest):
            """
            Make predictions on input features.

            Accepts a 2D array of features and returns class predictions
            with optional probabilities and confidence scores.
            """
            start_time = time.perf_counter()

            try:
                # Validate batch size
                if len(request.features) > self.config.max_batch_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Batch size exceeds maximum ({self.config.max_batch_size})",
                    )

                # Make predictions
                X = np.array(request.features, dtype=np.float32)
                result = self.pipeline.predict(X, calibrate=request.calibrate)

                inference_time = (time.perf_counter() - start_time) * 1000

                # Build response
                response = {
                    "predictions": result.predictions.class_predictions.tolist(),
                    "inference_time_ms": inference_time,
                    "model_name": self.pipeline.model_names[0],
                    "horizon": self.pipeline.horizon,
                }

                if request.return_probabilities:
                    response["probabilities"] = result.predictions.class_probabilities.tolist()
                    response["confidence"] = result.predictions.confidence.tolist()

                # Update metrics
                self._request_count += 1
                self._total_latency_ms += inference_time

                return response

            except HTTPException:
                raise
            except Exception as e:
                self._error_count += 1
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/predict_ensemble", tags=["Prediction"])
        async def predict_ensemble(request: EnsemblePredictionRequest):
            """
            Make ensemble predictions using multiple models.

            Combines predictions from all loaded models using the specified
            voting method (soft_vote or hard_vote).
            """
            if self.pipeline.n_models < 2:
                raise HTTPException(status_code=400, detail="Ensemble requires multiple models")

            time.perf_counter()

            try:
                X = np.array(request.features, dtype=np.float32)

                result = self.pipeline.predict_ensemble(
                    X, method=request.voting_method, calibrate=request.calibrate
                )

                response = {
                    "predictions": result.predictions.class_predictions.tolist(),
                    "voting_method": result.voting_method,
                    "inference_time_ms": result.inference_time_ms,
                    "models_used": [r.model_name for r in result.individual_results],
                }

                if request.return_probabilities:
                    response["probabilities"] = result.predictions.class_probabilities.tolist()
                    response["confidence"] = result.predictions.confidence.tolist()

                return response

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Ensemble prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
        async def metrics():
            """Server metrics endpoint."""
            if not self.config.enable_metrics:
                raise HTTPException(status_code=404, detail="Metrics disabled")

            avg_latency = (
                self._total_latency_ms / self._request_count if self._request_count > 0 else 0
            )

            return {
                "request_count": self._request_count,
                "error_count": self._error_count,
                "error_rate": self._error_count / max(self._request_count, 1),
                "average_latency_ms": avg_latency,
            }

        self._app = app
        return app

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
        workers: int | None = None,
    ) -> None:
        """
        Run the server using uvicorn.

        Args:
            host: Host to bind to
            port: Port to listen on
            debug: Enable debug mode (reload)
            workers: Number of worker processes
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required to run the server. " "Install with: pip install uvicorn"
            )

        if self._app is None:
            self.create_app()

        host = host or self.config.host
        port = port or self.config.port
        debug = debug if debug is not None else self.config.debug
        workers = workers or self.config.workers

        logger.info(f"Starting model server on {host}:{port}")
        logger.info(f"Models: {self.pipeline.model_names}")
        logger.info(f"API documentation available at http://{host}:{port}/docs")

        uvicorn.run(
            self._app,
            host=host,
            port=port,
            reload=debug,
            workers=workers if not debug else 1,
        )

    def get_app(self):
        """
        Get the FastAPI app instance for external ASGI servers.

        Useful for deployment with gunicorn, hypercorn, etc.

        Example:
            # In your_app.py
            server = ModelServer.from_bundle("./bundles/model")
            app = server.get_app()

            # Run with: gunicorn your_app:app -w 4 -k uvicorn.workers.UvicornWorker
        """
        if self._app is None:
            self.create_app()
        return self._app


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def start_server(
    bundle_path: str | Path,
    host: str = "0.0.0.0",
    port: int = 8080,
    debug: bool = False,
    workers: int = 1,
) -> None:
    """
    Start model server with a single bundle.

    Args:
        bundle_path: Path to model bundle
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode (hot reload)
        workers: Number of worker processes
    """
    config = ServerConfig(host=host, port=port, debug=debug, workers=workers)
    server = ModelServer.from_bundle(bundle_path, config)
    server.run()


def create_app_from_bundle(bundle_path: str | Path) -> Any:
    """
    Create FastAPI app from a model bundle.

    Useful for deployment with external ASGI servers.

    Args:
        bundle_path: Path to model bundle

    Returns:
        FastAPI application instance

    Example:
        # In app.py
        app = create_app_from_bundle("./bundles/model")

        # Run with: uvicorn app:app --workers 4
    """
    server = ModelServer.from_bundle(bundle_path)
    return server.get_app()


__all__ = [
    "ModelServer",
    "ServerConfig",
    "PredictionRequest",
    "PredictionResponse",
    "EnsemblePredictionRequest",
    "HealthResponse",
    "ModelInfoResponse",
    "MetricsResponse",
    "start_server",
    "create_app_from_bundle",
]
