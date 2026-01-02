"""
InferencePipeline - End-to-end inference orchestration.

Provides a high-level interface for making predictions with bundled models:
- Single predictions
- Batch predictions
- Ensemble predictions (multiple bundles)
- Streaming predictions

Usage:
    from src.inference import InferencePipeline

    # Single model inference
    pipeline = InferencePipeline.from_bundle("./bundles/xgb_h20")
    predictions = pipeline.predict(X_new)

    # Ensemble inference
    pipeline = InferencePipeline.from_bundles([
        "./bundles/xgb_h20",
        "./bundles/lgbm_h20",
        "./bundles/lstm_h20",
    ])
    ensemble_predictions = pipeline.predict_ensemble(X_new)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.bundle import ModelBundle
from src.models.base import PredictionOutput

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class InferenceResult:
    """
    Result from inference pipeline.

    Attributes:
        predictions: PredictionOutput with class predictions and probabilities
        inference_time_ms: Time taken for inference in milliseconds
        model_name: Name of the model used
        horizon: Prediction horizon
        n_samples: Number of samples processed
        metadata: Additional inference metadata
    """

    predictions: PredictionOutput
    inference_time_ms: float
    model_name: str
    horizon: int
    n_samples: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert predictions to DataFrame."""
        return pd.DataFrame(
            {
                "prediction": self.predictions.class_predictions,
                "prob_short": self.predictions.class_probabilities[:, 0],
                "prob_neutral": self.predictions.class_probabilities[:, 1],
                "prob_long": self.predictions.class_probabilities[:, 2],
                "confidence": self.predictions.confidence,
            }
        )


@dataclass
class EnsembleResult:
    """
    Result from ensemble inference.

    Attributes:
        predictions: Combined PredictionOutput
        individual_results: Results from each model
        voting_method: Method used for combining predictions
        inference_time_ms: Total time in milliseconds
    """

    predictions: PredictionOutput
    individual_results: list[InferenceResult]
    voting_method: str
    inference_time_ms: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert ensemble predictions to DataFrame."""
        df = pd.DataFrame(
            {
                "ensemble_prediction": self.predictions.class_predictions,
                "ensemble_prob_short": self.predictions.class_probabilities[:, 0],
                "ensemble_prob_neutral": self.predictions.class_probabilities[:, 1],
                "ensemble_prob_long": self.predictions.class_probabilities[:, 2],
                "ensemble_confidence": self.predictions.confidence,
            }
        )

        # Add individual model predictions
        for result in self.individual_results:
            prefix = result.model_name
            df[f"{prefix}_pred"] = result.predictions.class_predictions
            df[f"{prefix}_conf"] = result.predictions.confidence

        return df


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================


class InferencePipeline:
    """
    High-level inference orchestration.

    Manages one or more model bundles and provides unified prediction interface.
    Supports single-model, multi-model, and ensemble inference modes.

    Example:
        >>> # Single model
        >>> pipeline = InferencePipeline.from_bundle("./bundles/xgb_h20")
        >>> result = pipeline.predict(X_test)
        >>> print(f"Predictions: {result.predictions.class_predictions}")

        >>> # Ensemble of models
        >>> pipeline = InferencePipeline.from_bundles([
        ...     "./bundles/xgb_h20",
        ...     "./bundles/lgbm_h20",
        ... ])
        >>> result = pipeline.predict_ensemble(X_test, method="soft_vote")
    """

    def __init__(
        self,
        bundles: list[ModelBundle],
        default_voting: str = "soft_vote",
    ) -> None:
        """
        Initialize InferencePipeline.

        Args:
            bundles: List of ModelBundle instances
            default_voting: Default ensemble voting method
        """
        if not bundles:
            raise ValueError("At least one bundle is required")

        self.bundles = bundles
        self.default_voting = default_voting

        # Validate bundles have compatible horizons
        horizons = {b.metadata.horizon for b in bundles}
        if len(horizons) > 1:
            logger.warning(
                f"Bundles have different horizons: {horizons}. "
                "Ensemble predictions may not be meaningful."
            )

        self._primary_bundle = bundles[0]

    @classmethod
    def from_bundle(cls, path: str | Path) -> InferencePipeline:
        """
        Create pipeline from a single bundle.

        Args:
            path: Path to bundle directory

        Returns:
            InferencePipeline with single bundle
        """
        bundle = ModelBundle.load(path)
        return cls([bundle])

    @classmethod
    def from_bundles(
        cls,
        paths: list[str | Path],
        default_voting: str = "soft_vote",
    ) -> InferencePipeline:
        """
        Create pipeline from multiple bundles.

        Args:
            paths: List of bundle paths
            default_voting: Default voting method for ensemble

        Returns:
            InferencePipeline with multiple bundles
        """
        bundles = [ModelBundle.load(p) for p in paths]
        return cls(bundles, default_voting=default_voting)

    @property
    def n_models(self) -> int:
        """Number of models in pipeline."""
        return len(self.bundles)

    @property
    def model_names(self) -> list[str]:
        """Names of all models."""
        return [b.metadata.model_name for b in self.bundles]

    @property
    def feature_columns(self) -> list[str]:
        """Feature columns from primary bundle."""
        return self._primary_bundle.feature_columns

    @property
    def horizon(self) -> int:
        """Horizon from primary bundle."""
        return self._primary_bundle.metadata.horizon

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> InferenceResult:
        """
        Make predictions using primary bundle.

        Args:
            X: Input features
            calibrate: Whether to apply calibration

        Returns:
            InferenceResult with predictions and timing
        """
        return self._predict_single(self._primary_bundle, X, calibrate)

    def predict_all(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> list[InferenceResult]:
        """
        Get predictions from all models.

        Args:
            X: Input features
            calibrate: Whether to apply calibration

        Returns:
            List of InferenceResult, one per model
        """
        return [self._predict_single(bundle, X, calibrate) for bundle in self.bundles]

    def predict_ensemble(
        self,
        X: pd.DataFrame | np.ndarray,
        method: str | None = None,
        weights: list[float] | None = None,
        calibrate: bool = True,
    ) -> EnsembleResult:
        """
        Make ensemble predictions combining all models.

        Args:
            X: Input features
            method: Voting method ("soft_vote", "hard_vote", "weighted")
            weights: Model weights for weighted voting
            calibrate: Whether to apply calibration

        Returns:
            EnsembleResult with combined predictions
        """
        method = method or self.default_voting
        start_time = time.perf_counter()

        # Get predictions from all models
        individual_results = self.predict_all(X, calibrate)

        # Combine predictions
        combined = self._combine_predictions(individual_results, method, weights)

        total_time = (time.perf_counter() - start_time) * 1000

        return EnsembleResult(
            predictions=combined,
            individual_results=individual_results,
            voting_method=method,
            inference_time_ms=total_time,
        )

    def _predict_single(
        self,
        bundle: ModelBundle,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool,
    ) -> InferenceResult:
        """Make predictions with a single bundle."""
        start_time = time.perf_counter()

        predictions = bundle.predict(X, calibrate=calibrate)

        inference_time = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            predictions=predictions,
            inference_time_ms=inference_time,
            model_name=bundle.metadata.model_name,
            horizon=bundle.metadata.horizon,
            n_samples=predictions.n_samples,
            metadata={
                "calibrated": calibrate and bundle.calibrator is not None,
                "model_family": bundle.metadata.model_family,
            },
        )

    def _combine_predictions(
        self,
        results: list[InferenceResult],
        method: str,
        weights: list[float] | None,
    ) -> PredictionOutput:
        """Combine predictions from multiple models."""
        if method == "soft_vote":
            return self._soft_vote(results, weights)
        elif method == "hard_vote":
            return self._hard_vote(results, weights)
        elif method == "weighted":
            if weights is None:
                raise ValueError("weights required for weighted voting")
            return self._soft_vote(results, weights)
        else:
            raise ValueError(f"Unknown voting method: {method}")

    def _soft_vote(
        self,
        results: list[InferenceResult],
        weights: list[float] | None,
    ) -> PredictionOutput:
        """Average probabilities across models."""
        if weights is None:
            weights = [1.0] * len(results)

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Average probabilities
        n_samples = results[0].predictions.n_samples
        n_classes = results[0].predictions.n_classes
        avg_probs = np.zeros((n_samples, n_classes))

        for result, w in zip(results, weights, strict=False):
            avg_probs += w * result.predictions.class_probabilities

        # Get predictions from averaged probabilities
        class_predictions = np.argmax(avg_probs, axis=1) - 1  # Map to -1, 0, 1
        confidence = np.max(avg_probs, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=avg_probs,
            confidence=confidence,
            metadata={"method": "soft_vote", "n_models": len(results)},
        )

    def _hard_vote(
        self,
        results: list[InferenceResult],
        weights: list[float] | None,
    ) -> PredictionOutput:
        """Majority vote on class predictions."""
        if weights is None:
            weights = [1.0] * len(results)

        n_samples = results[0].predictions.n_samples
        n_classes = results[0].predictions.n_classes

        # Count votes per class (weighted)
        vote_counts = np.zeros((n_samples, n_classes))
        for result, w in zip(results, weights, strict=False):
            preds = result.predictions.class_predictions + 1  # Map to 0, 1, 2
            for i, pred in enumerate(preds):
                vote_counts[i, int(pred)] += w

        # Get majority class
        majority_class = np.argmax(vote_counts, axis=1)
        class_predictions = majority_class - 1  # Map back to -1, 0, 1

        # Confidence from vote proportion
        confidence = np.max(vote_counts, axis=1) / np.sum(vote_counts, axis=1)

        # Use average probabilities from winning class
        avg_probs = np.zeros((n_samples, n_classes))
        for result in results:
            avg_probs += result.predictions.class_probabilities
        avg_probs /= len(results)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=avg_probs,
            confidence=confidence,
            metadata={"method": "hard_vote", "n_models": len(results)},
        )

    def get_model_info(self) -> list[dict[str, Any]]:
        """Get information about all models in pipeline."""
        return [
            {
                "name": b.metadata.model_name,
                "family": b.metadata.model_family,
                "horizon": b.metadata.horizon,
                "features": b.metadata.n_features,
                "has_calibrator": b.metadata.has_calibrator,
                "requires_sequences": b.metadata.requires_sequences,
            }
            for b in self.bundles
        ]

    def validate(self) -> dict[str, Any]:
        """Validate all bundles in pipeline."""
        validations = {}
        all_valid = True

        for bundle in self.bundles:
            result = bundle.validate()
            validations[bundle.metadata.model_name] = result
            if not result["valid"]:
                all_valid = False

        return {
            "valid": all_valid,
            "n_models": len(self.bundles),
            "models": validations,
        }

    def __repr__(self) -> str:
        return f"InferencePipeline(models={self.model_names}, " f"horizon={self.horizon})"


__all__ = [
    "InferencePipeline",
    "InferenceResult",
    "EnsembleResult",
]
