"""
Probability Calibration for ML Trading Models.

Implements isotonic regression (for boosting) and Platt scaling (for linear)
to correct miscalibrated probability outputs.

Boosting models (XGBoost, LightGBM, CatBoost) are notoriously miscalibrated.
This module applies post-hoc calibration to correct probabilities for
downstream position sizing and ensemble stacking.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.calibration.metrics import (
    ReliabilityBins,
    compute_brier_score,
    compute_ece,
    compute_reliability_bins,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for probability calibration."""

    method: Literal["isotonic", "sigmoid", "auto"] = "auto"
    min_samples_per_class: int = 100  # Minimum samples for isotonic
    clip_probabilities: bool = True  # Clip to [epsilon, 1-epsilon]
    epsilon: float = 1e-7


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics before and after calibration."""

    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    reliability_bins: ReliabilityBins
    method_used: str

    @property
    def brier_improvement(self) -> float:
        """Relative Brier score improvement (positive = better)."""
        if self.brier_before == 0:
            return 0.0
        return (self.brier_before - self.brier_after) / self.brier_before

    @property
    def ece_improvement(self) -> float:
        """Relative ECE improvement (positive = better)."""
        if self.ece_before == 0:
            return 0.0
        return (self.ece_before - self.ece_after) / self.ece_before

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "brier_improvement": self.brier_improvement,
            "ece_improvement": self.ece_improvement,
            "method_used": self.method_used,
            "reliability_bins": self.reliability_bins.to_dict(),
        }


class ProbabilityCalibrator:
    """
    Calibrates probability outputs from classification models.

    Boosting models output miscalibrated probabilities. This class applies
    isotonic (non-parametric) or sigmoid (Platt scaling) calibration
    to correct probabilities for downstream use.

    Leakage-Safe Usage:
        - For training: fit on validation set predictions/labels
        - For CV: fit calibrator on held-out fold, not the fold being predicted

    Example:
        >>> calibrator = ProbabilityCalibrator(CalibrationConfig())
        >>> metrics = calibrator.fit(y_val, probas_val)
        >>> calibrated_probs = calibrator.calibrate(probas_test)
        >>> print(f"Brier improved: {metrics.brier_improvement:.1%}")
    """

    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        """
        Initialize ProbabilityCalibrator.

        Args:
            config: Calibration configuration. Uses defaults if None.
        """
        self.config = config or CalibrationConfig()
        self._calibrators: Dict[int, Any] = {}  # class -> calibrator
        self._is_fitted: bool = False
        self._n_classes: int = 0
        self._method_used: str = ""

    @property
    def is_fitted(self) -> bool:
        """Whether calibrator has been fitted."""
        return self._is_fitted

    def fit(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> CalibrationMetrics:
        """
        Fit calibrators on validation data.

        IMPORTANT: Must be called with a held-out validation set to avoid
        leakage. Do not fit on the same data used for model training.

        Args:
            y_true: True labels, shape (n_samples,)
            probabilities: Uncalibrated probabilities, shape (n_samples, n_classes)

        Returns:
            CalibrationMetrics with before/after quality scores

        Raises:
            ValueError: If input shapes are invalid
        """
        y_true = np.asarray(y_true).ravel()
        probabilities = np.asarray(probabilities)

        if len(y_true) != len(probabilities):
            raise ValueError(
                f"y_true length ({len(y_true)}) != probabilities length ({len(probabilities)})"
            )

        if probabilities.ndim != 2:
            raise ValueError(f"probabilities must be 2D, got shape {probabilities.shape}")

        n_samples, n_classes = probabilities.shape
        self._n_classes = n_classes

        # Normalize labels to 0-indexed (handle -1,0,1 format)
        y_normalized = self._normalize_labels(y_true, n_classes)

        # Compute pre-calibration metrics
        brier_before = compute_brier_score(y_true, probabilities)
        ece_before = compute_ece(y_true, probabilities)

        # Select calibration method
        method = self._select_method(y_normalized, n_classes)
        self._method_used = method

        logger.debug(f"Fitting calibration using method: {method}")

        # Fit one calibrator per class (one-vs-rest)
        for cls in range(n_classes):
            y_binary = (y_normalized == cls).astype(float)
            probs_cls = probabilities[:, cls]

            if method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(probs_cls, y_binary)
            else:
                # Sigmoid (Platt scaling) via logistic regression
                calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
                calibrator.fit(probs_cls.reshape(-1, 1), y_binary)

            self._calibrators[cls] = calibrator

        self._is_fitted = True

        # Compute post-calibration metrics
        calibrated = self.calibrate(probabilities)
        brier_after = compute_brier_score(y_true, calibrated)
        ece_after = compute_ece(y_true, calibrated)
        reliability = compute_reliability_bins(y_true, calibrated)

        logger.info(
            f"Calibration ({method}): "
            f"Brier {brier_before:.4f} -> {brier_after:.4f}, "
            f"ECE {ece_before:.4f} -> {ece_after:.4f}"
        )

        return CalibrationMetrics(
            brier_before=brier_before,
            brier_after=brier_after,
            ece_before=ece_before,
            ece_after=ece_after,
            reliability_bins=reliability,
            method_used=method,
        )

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probability outputs.

        Args:
            probabilities: Uncalibrated probabilities, shape (n_samples, n_classes)

        Returns:
            Calibrated probabilities, shape (n_samples, n_classes)

        Raises:
            RuntimeError: If calibrator not fitted
            ValueError: If number of classes doesn't match
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        probabilities = np.asarray(probabilities)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)

        n_samples, n_classes = probabilities.shape

        if n_classes != self._n_classes:
            raise ValueError(
                f"Expected {self._n_classes} classes, got {n_classes}"
            )

        calibrated = np.zeros_like(probabilities)

        for cls in range(n_classes):
            probs_cls = probabilities[:, cls]
            calibrator = self._calibrators[cls]

            if self._method_used == "isotonic":
                calibrated[:, cls] = calibrator.predict(probs_cls)
            else:
                # Logistic returns probability of class 1
                calibrated[:, cls] = calibrator.predict_proba(
                    probs_cls.reshape(-1, 1)
                )[:, 1]

        # Normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)  # Avoid division by zero
        calibrated = calibrated / row_sums

        # Clip to avoid extreme values
        if self.config.clip_probabilities:
            eps = self.config.epsilon
            calibrated = np.clip(calibrated, eps, 1.0 - eps)
            # Re-normalize after clipping
            calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

        return calibrated

    def save(self, path: Path) -> None:
        """
        Save calibrator to disk.

        Args:
            path: Path to save calibrator pickle file
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted calibrator")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "calibrators": self._calibrators,
            "n_classes": self._n_classes,
            "method_used": self._method_used,
            "is_fitted": self._is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.debug(f"Saved calibrator to {path}")

    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        """
        Load calibrator from disk.

        Args:
            path: Path to calibrator pickle file

        Returns:
            Loaded ProbabilityCalibrator instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        calibrator = cls(config=state["config"])
        calibrator._calibrators = state["calibrators"]
        calibrator._n_classes = state["n_classes"]
        calibrator._method_used = state["method_used"]
        calibrator._is_fitted = state["is_fitted"]

        logger.debug(f"Loaded calibrator from {path}")
        return calibrator

    def _normalize_labels(self, y_true: np.ndarray, n_classes: int) -> np.ndarray:
        """
        Normalize labels to 0-indexed format.

        Handles labels in {-1, 0, 1} format (trading signals) by mapping to {0, 1, 2}.

        Args:
            y_true: True class labels (may be -1, 0, 1 or 0, 1, 2)
            n_classes: Number of classes

        Returns:
            Labels normalized to {0, 1, ..., n_classes-1}
        """
        y_int = y_true.astype(int)

        # Check if labels contain negative values (trading signal format)
        if y_int.min() < 0:
            # Map -1 -> 0, 0 -> 1, 1 -> 2
            y_int = y_int + 1

        return y_int

    def _select_method(self, y_true: np.ndarray, n_classes: int) -> str:
        """Select calibration method based on config and data."""
        if self.config.method != "auto":
            return self.config.method

        # Check minimum samples per class for isotonic
        # y_true should already be normalized at this point
        unique_classes, class_counts = np.unique(y_true.astype(int), return_counts=True)
        min_class_count = class_counts.min() if len(class_counts) > 0 else 0

        if min_class_count >= self.config.min_samples_per_class:
            return "isotonic"
        else:
            logger.debug(
                f"Using sigmoid: min class count {min_class_count} "
                f"< {self.config.min_samples_per_class}"
            )
            return "sigmoid"


__all__ = [
    "CalibrationConfig",
    "CalibrationMetrics",
    "ProbabilityCalibrator",
]
