"""
Conformal Prediction Module.

Provides prediction sets with finite-sample coverage guarantees.
Implements split conformal prediction for multi-class classification.

Key Features:
- Prediction sets with guaranteed coverage (e.g., 90%)
- Adaptive set sizes based on model uncertainty
- Coverage validation and diagnostics
- Integration with existing calibration pipeline

Usage:
    from src.models.calibration import ConformalPredictor, ConformalConfig

    # Fit on calibration set
    conformal = ConformalPredictor(ConformalConfig(confidence_level=0.90))
    metrics = conformal.fit(y_cal, probas_cal)

    # Generate prediction sets
    pred_sets, set_sizes = conformal.predict_sets(probas_test)
    # pred_sets is binary matrix: 1 if class in set, 0 otherwise

Note:
    Split conformal requires a held-out calibration set separate from
    training and test. Use validation fold or dedicated calibration split.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ConformalConfig:
    """Configuration for conformal prediction.

    Attributes:
        confidence_level: Desired coverage level (1 - error_rate)
        method: Conformal method ('lac' for least ambiguous, 'aps' for adaptive)
        allow_empty_sets: Whether to allow empty prediction sets
        epsilon: Numerical stability constant
    """
    confidence_level: float = 0.90
    method: Literal["lac", "aps", "naive"] = "lac"
    allow_empty_sets: bool = False
    epsilon: float = 1e-7

    def __post_init__(self) -> None:
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0.5, 1.0), got {self.confidence_level}"
            )
        if self.method not in ("lac", "aps", "naive"):
            raise ValueError(f"Unknown method: {self.method}")


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class ConformalMetrics:
    """Metrics for conformal prediction quality.

    Attributes:
        empirical_coverage: Actual coverage on calibration/test set
        average_set_size: Mean prediction set size
        singleton_rate: Fraction of predictions with exactly one class
        empty_set_rate: Fraction of empty prediction sets
        conditional_coverage: Coverage per true class
        method_used: Conformal method used
        n_samples: Number of samples evaluated
    """
    empirical_coverage: float
    average_set_size: float
    singleton_rate: float
    empty_set_rate: float
    conditional_coverage: Dict[int, float]
    method_used: str
    n_samples: int
    threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "empirical_coverage": self.empirical_coverage,
            "average_set_size": self.average_set_size,
            "singleton_rate": self.singleton_rate,
            "empty_set_rate": self.empty_set_rate,
            "conditional_coverage": self.conditional_coverage,
            "method_used": self.method_used,
            "n_samples": self.n_samples,
            "threshold": self.threshold,
        }


# =============================================================================
# CONFORMAL PREDICTOR
# =============================================================================

class ConformalPredictor:
    """
    Generate prediction sets with coverage guarantees.

    Implements split conformal prediction for multi-class classification.
    The prediction sets contain all classes whose non-conformity scores
    are below a data-dependent threshold, guaranteeing finite-sample
    coverage at the specified confidence level.

    Supported Methods:
    - 'lac' (Least Ambiguous Conformal): Uses 1 - P(true class) as score
    - 'aps' (Adaptive Prediction Sets): Cumulative sorted probability score
    - 'naive': Uses probability threshold directly

    Example:
        >>> config = ConformalConfig(confidence_level=0.90)
        >>> conformal = ConformalPredictor(config)

        >>> # Fit on calibration set (separate from train/test)
        >>> metrics = conformal.fit(y_cal, probas_cal)
        >>> print(f"Threshold: {metrics.threshold:.4f}")

        >>> # Generate prediction sets
        >>> sets, sizes = conformal.predict_sets(probas_test)
        >>> print(f"Average set size: {sizes.mean():.2f}")
    """

    def __init__(self, config: Optional[ConformalConfig] = None) -> None:
        """
        Initialize conformal predictor.

        Args:
            config: Configuration options
        """
        self.config = config or ConformalConfig()
        self._is_fitted: bool = False
        self._threshold: float = 0.0
        self._n_classes: int = 0
        self._calibration_scores: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        """Whether the predictor has been fitted."""
        return self._is_fitted

    @property
    def threshold(self) -> float:
        """Fitted non-conformity score threshold."""
        return self._threshold

    @property
    def n_classes(self) -> int:
        """Number of classes."""
        return self._n_classes

    def fit(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> ConformalMetrics:
        """
        Fit conformal predictor on calibration set.

        Computes the quantile threshold that achieves the desired coverage.

        IMPORTANT: Must use held-out calibration data (not training data).

        Args:
            y_true: True labels, shape (n_samples,)
            probabilities: Class probabilities, shape (n_samples, n_classes)

        Returns:
            ConformalMetrics with fitted predictor statistics

        Raises:
            ValueError: If inputs are invalid
        """
        y_true = np.asarray(y_true)
        probabilities = np.asarray(probabilities)

        # Validate inputs
        if y_true.ndim != 1:
            raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
        if probabilities.ndim != 2:
            raise ValueError(f"probabilities must be 2D, got shape {probabilities.shape}")
        if len(y_true) != len(probabilities):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true)}) vs "
                f"probabilities ({len(probabilities)})"
            )

        n_samples, n_classes = probabilities.shape
        self._n_classes = n_classes

        # Normalize labels to 0-indexed
        y_normalized = self._normalize_labels(y_true)

        # Compute non-conformity scores
        scores = self._compute_scores(y_normalized, probabilities)
        self._calibration_scores = scores

        # Compute threshold as quantile
        # For coverage 1-alpha, we use the (1-alpha)(1 + 1/n) quantile
        alpha = 1 - self.config.confidence_level
        quantile_level = (1 - alpha) * (1 + 1 / n_samples)
        quantile_level = min(quantile_level, 1.0)  # Clip to 1

        self._threshold = float(np.quantile(scores, quantile_level))
        self._is_fitted = True

        # Compute metrics on calibration set
        pred_sets, set_sizes = self.predict_sets(probabilities)
        metrics = self._compute_metrics(y_normalized, pred_sets, set_sizes)

        logger.info(
            f"Fitted conformal predictor: threshold={self._threshold:.4f}, "
            f"coverage={metrics.empirical_coverage:.3f}, "
            f"avg_set_size={metrics.average_set_size:.2f}"
        )

        return metrics

    def predict_sets(
        self,
        probabilities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction sets with coverage guarantee.

        Args:
            probabilities: Class probabilities, shape (n_samples, n_classes)

        Returns:
            Tuple of (prediction_sets, set_sizes) where:
            - prediction_sets: Binary matrix (n_samples, n_classes)
              1 if class in prediction set, 0 otherwise
            - set_sizes: Array of set sizes per sample

        Raises:
            RuntimeError: If predictor not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Conformal predictor not fitted. Call fit() first.")

        probabilities = np.asarray(probabilities)
        if probabilities.ndim != 2:
            raise ValueError(f"probabilities must be 2D, got shape {probabilities.shape}")

        n_samples, n_classes = probabilities.shape

        if n_classes != self._n_classes:
            raise ValueError(
                f"Expected {self._n_classes} classes, got {n_classes}"
            )

        # Compute scores for all classes
        if self.config.method == "lac":
            # LAC: Include class if 1 - P(class) <= threshold
            # Equivalently: P(class) >= 1 - threshold
            prediction_sets = (1 - probabilities) <= self._threshold

        elif self.config.method == "aps":
            # APS: Cumulative sorted probability
            prediction_sets = np.zeros((n_samples, n_classes), dtype=bool)

            for i in range(n_samples):
                sorted_idx = np.argsort(-probabilities[i])
                cumsum = np.cumsum(probabilities[i, sorted_idx])

                # Include classes until cumsum exceeds threshold
                for j, idx in enumerate(sorted_idx):
                    if j == 0:
                        prediction_sets[i, idx] = True
                    elif cumsum[j] <= self._threshold:
                        prediction_sets[i, idx] = True
                    else:
                        break

        else:  # naive
            # Naive: Include if probability >= 1 - threshold
            prediction_sets = probabilities >= (1 - self._threshold)

        # Handle empty sets
        set_sizes = prediction_sets.sum(axis=1)
        empty_mask = set_sizes == 0

        if not self.config.allow_empty_sets and empty_mask.any():
            # Add highest probability class for empty sets
            for i in np.where(empty_mask)[0]:
                best_class = np.argmax(probabilities[i])
                prediction_sets[i, best_class] = True

        set_sizes = prediction_sets.sum(axis=1)

        return prediction_sets.astype(np.int32), set_sizes

    def predict_with_rejection(
        self,
        probabilities: np.ndarray,
        reject_threshold: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with rejection for ambiguous cases.

        Args:
            probabilities: Class probabilities
            reject_threshold: Reject if set size > threshold

        Returns:
            Tuple of (predictions, rejected_mask)
            predictions[i] = -1 if rejected
        """
        pred_sets, set_sizes = self.predict_sets(probabilities)

        # Reject if set size exceeds threshold
        rejected = set_sizes > reject_threshold

        # For non-rejected, return argmax of probabilities within set
        predictions = np.full(len(probabilities), -1, dtype=np.int32)

        for i in np.where(~rejected)[0]:
            in_set = pred_sets[i].astype(bool)
            masked_probs = np.where(in_set, probabilities[i], -np.inf)
            predictions[i] = np.argmax(masked_probs)

        return predictions, rejected

    def evaluate(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> ConformalMetrics:
        """
        Evaluate conformal predictor on test set.

        Args:
            y_true: True labels
            probabilities: Class probabilities

        Returns:
            ConformalMetrics with evaluation results
        """
        if not self._is_fitted:
            raise RuntimeError("Conformal predictor not fitted. Call fit() first.")

        y_normalized = self._normalize_labels(y_true)
        pred_sets, set_sizes = self.predict_sets(probabilities)

        return self._compute_metrics(y_normalized, pred_sets, set_sizes)

    def _normalize_labels(self, y: np.ndarray) -> np.ndarray:
        """Normalize labels to 0-indexed."""
        y = np.asarray(y)
        unique = np.unique(y)

        # Handle -1, 0, 1 format
        if set(unique).issubset({-1, 0, 1}):
            return y + 1  # Map to 0, 1, 2

        # Already 0-indexed
        return y

    def _compute_scores(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Compute non-conformity scores for calibration."""
        n_samples = len(y_true)

        if self.config.method == "lac":
            # LAC: score = 1 - P(true class)
            true_probs = probabilities[np.arange(n_samples), y_true]
            scores = 1 - true_probs

        elif self.config.method == "aps":
            # APS: Cumulative probability up to and including true class
            scores = np.zeros(n_samples)

            for i in range(n_samples):
                sorted_idx = np.argsort(-probabilities[i])
                cumsum = np.cumsum(probabilities[i, sorted_idx])

                # Find position of true class
                true_pos = np.where(sorted_idx == y_true[i])[0][0]
                scores[i] = cumsum[true_pos]

        else:  # naive
            true_probs = probabilities[np.arange(n_samples), y_true]
            scores = 1 - true_probs

        return scores

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        prediction_sets: np.ndarray,
        set_sizes: np.ndarray,
    ) -> ConformalMetrics:
        """Compute conformal prediction metrics."""
        n_samples = len(y_true)

        # Coverage: true class in prediction set
        coverage = np.mean([
            prediction_sets[i, y_true[i]] for i in range(n_samples)
        ])

        # Set size statistics
        avg_size = set_sizes.mean()
        singleton_rate = (set_sizes == 1).mean()
        empty_rate = (set_sizes == 0).mean()

        # Conditional coverage per class
        conditional_coverage = {}
        for c in range(self._n_classes):
            mask = y_true == c
            if mask.sum() > 0:
                class_coverage = np.mean([
                    prediction_sets[i, c] for i in np.where(mask)[0]
                ])
                conditional_coverage[c] = float(class_coverage)

        return ConformalMetrics(
            empirical_coverage=float(coverage),
            average_set_size=float(avg_size),
            singleton_rate=float(singleton_rate),
            empty_set_rate=float(empty_rate),
            conditional_coverage=conditional_coverage,
            method_used=self.config.method,
            n_samples=n_samples,
            threshold=self._threshold,
        )

    def save(self, path: Union[str, Path]) -> None:
        """
        Save fitted conformal predictor.

        Args:
            path: File path for saving
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted predictor")

        path = Path(path)
        state = {
            "config": self.config,
            "threshold": self._threshold,
            "n_classes": self._n_classes,
            "calibration_scores": self._calibration_scores,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved conformal predictor to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConformalPredictor":
        """
        Load fitted conformal predictor.

        Args:
            path: File path to load from

        Returns:
            Loaded ConformalPredictor
        """
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        predictor = cls(config=state["config"])
        predictor._threshold = state["threshold"]
        predictor._n_classes = state["n_classes"]
        predictor._calibration_scores = state["calibration_scores"]
        predictor._is_fitted = True

        logger.info(f"Loaded conformal predictor from {path}")
        return predictor


# =============================================================================
# COVERAGE VALIDATION
# =============================================================================

def validate_coverage(
    y_true: np.ndarray,
    prediction_sets: np.ndarray,
    expected_coverage: float,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Validate that empirical coverage meets expected level.

    Args:
        y_true: True labels (0-indexed)
        prediction_sets: Binary prediction set matrix
        expected_coverage: Expected coverage level (e.g., 0.90)
        tolerance: Acceptable deviation from expected

    Returns:
        Dict with validation results
    """
    y_true = np.asarray(y_true)
    n_samples = len(y_true)

    # Compute empirical coverage
    covered = np.array([
        prediction_sets[i, y_true[i]] for i in range(n_samples)
    ])
    empirical = covered.mean()

    # Standard error of coverage
    se = np.sqrt(empirical * (1 - empirical) / n_samples)

    # Two-sided test
    z_score = (empirical - expected_coverage) / se if se > 0 else 0
    passed = abs(empirical - expected_coverage) <= tolerance

    return {
        "empirical_coverage": float(empirical),
        "expected_coverage": expected_coverage,
        "deviation": float(empirical - expected_coverage),
        "standard_error": float(se),
        "z_score": float(z_score),
        "passed": bool(passed),  # Convert to Python bool
        "tolerance": tolerance,
        "n_samples": n_samples,
    }


__all__ = [
    "ConformalPredictor",
    "ConformalConfig",
    "ConformalMetrics",
    "validate_coverage",
]
