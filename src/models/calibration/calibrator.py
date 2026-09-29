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
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.calibration.metrics import (
    ReliabilityBins,
    compute_brier_score,
    compute_ece,
    compute_reliability_bins,
)
from src.models.common.label_mapping import map_labels_to_classes

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for probability calibration."""

    method: Literal["isotonic", "sigmoid", "auto"] = "auto"
    # "auto" uses isotonic only when every class has at least this many
    # samples; isotonic regression overfits (step functions) below ~1000.
    min_samples_per_class: int = 1000
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

    def to_dict(self) -> dict[str, Any]:
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

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        """
        Initialize ProbabilityCalibrator.

        Args:
            config: Calibration configuration. Uses defaults if None.
        """
        self.config = config or CalibrationConfig()
        self._calibrators: dict[int, Any] = {}  # class -> calibrator
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
            y_true: Trading labels, shape (n_samples,): {-1,0,1} (3-class)
                or {0,1} (binary). The class count is probabilities.shape[1].
            probabilities: Uncalibrated probabilities, shape (n_samples, n_classes)

        Returns:
            CalibrationMetrics with before/after quality scores. The "after"
            values are IN-SAMPLE (same rows the calibrator was fit on); use
            estimate_holdout_improvement() for an out-of-sample estimate.

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

        # Trading labels -> class indices ({-1,0,1} -> {0,1,2}; binary identity)
        y_normalized = map_labels_to_classes(y_true, n_classes)

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

            # Guard: sklearn needs at least 2 classes in y_binary.
            # When all OOF predictions map to a single class (common with
            # small data or 1-epoch neural models), skip calibration for
            # this class and use a pass-through identity calibrator.
            n_unique = len(np.unique(y_binary))
            if n_unique < 2:
                logger.warning(
                    f"Calibration: class {cls} has only {n_unique} unique "
                    f"label(s) in OOF — skipping calibrator (pass-through)."
                )
                self._calibrators[cls] = None  # None = pass-through
                continue

            if method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(probs_cls, y_binary)
            else:
                # Sigmoid (Platt scaling) via logistic regression
                calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
                calibrator.fit(probs_cls.reshape(-1, 1), y_binary)

            self._calibrators[cls] = calibrator

        self._is_fitted = True

        # Compute post-calibration metrics (IN-SAMPLE ONLY)
        # WARNING: These metrics are computed on the same data used to fit
        # the calibrator and may be overly optimistic. For unbiased evaluation,
        # use a separate held-out test set.
        calibrated = self.calibrate(probabilities)
        brier_after = compute_brier_score(y_true, calibrated)
        ece_after = compute_ece(y_true, calibrated)
        reliability = compute_reliability_bins(y_true, calibrated)

        logger.info(
            f"Calibration ({method}): "
            f"Brier {brier_before:.4f} -> {brier_after:.4f}, "
            f"ECE {ece_before:.4f} -> {ece_after:.4f} "
            f"(IN-SAMPLE - may be optimistic)"
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
            raise ValueError(f"Expected {self._n_classes} classes, got {n_classes}")

        calibrated = np.zeros_like(probabilities)

        for cls in range(n_classes):
            probs_cls = probabilities[:, cls]
            calibrator = self._calibrators[cls]

            if calibrator is None:
                # Pass-through: no calibrator fitted for this class
                calibrated[:, cls] = probs_cls
            elif self._method_used == "isotonic":
                calibrated[:, cls] = calibrator.predict(probs_cls)
            else:
                # Logistic returns probability of class 1
                calibrated[:, cls] = calibrator.predict_proba(probs_cls.reshape(-1, 1))[:, 1]

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

        result: np.ndarray = calibrated
        return result

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
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug(f"Saved calibrator to {path}")

    @classmethod
    def load(cls, path: Path) -> ProbabilityCalibrator:
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

        from src.core.utils.safe_pickle import safe_pickle_load

        state = safe_pickle_load(path)

        calibrator = cls(config=state["config"])
        calibrator._calibrators = state["calibrators"]
        calibrator._n_classes = state["n_classes"]
        calibrator._method_used = state["method_used"]
        calibrator._is_fitted = state["is_fitted"]

        logger.debug(f"Loaded calibrator from {path}")
        return calibrator

    def _select_method(self, y_true: np.ndarray, n_classes: int) -> str:
        """Select calibration method based on config and data."""
        if self.config.method != "auto":
            return self.config.method

        # Isotonic needs enough samples in EVERY class (absent classes count 0)
        class_counts = np.bincount(y_true.astype(int), minlength=n_classes)
        min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0

        if min_class_count >= self.config.min_samples_per_class:
            return "isotonic"
        else:
            logger.debug(
                f"Using sigmoid: min class count {min_class_count} "
                f"< {self.config.min_samples_per_class}"
            )
            return "sigmoid"


def estimate_holdout_improvement(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    config: CalibrationConfig,
    holdout_fraction: float = 0.3,
) -> dict[str, float] | None:
    """
    Out-of-sample Brier/ECE change from calibration.

    Fits a calibrator on the leading (1 - holdout_fraction) of the rows and
    scores the trailing holdout rows before and after calibration, so the
    reported gain is not measured on the rows the calibrator was fit on.

    Returns:
        Dict with brier_before/after, ece_before/after, and relative
        improvements on the holdout, or None when either part is too small.
    """
    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities)
    n_fit = int(len(y_true) * (1.0 - holdout_fraction))
    if n_fit < 2 or len(y_true) - n_fit < 2:
        return None
    calibrator = ProbabilityCalibrator(config)
    calibrator.fit(y_true[:n_fit], probabilities[:n_fit])
    y_hold, p_hold = y_true[n_fit:], probabilities[n_fit:]
    calibrated = calibrator.calibrate(p_hold)
    brier_before = compute_brier_score(y_hold, p_hold)
    brier_after = compute_brier_score(y_hold, calibrated)
    ece_before = compute_ece(y_hold, p_hold)
    ece_after = compute_ece(y_hold, calibrated)
    return {
        "brier_before": brier_before,
        "brier_after": brier_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "brier_improvement": ((brier_before - brier_after) / brier_before if brier_before else 0.0),
        "ece_improvement": (ece_before - ece_after) / ece_before if ece_before else 0.0,
        "n_holdout": float(len(y_hold)),
    }


__all__ = [
    "CalibrationConfig",
    "CalibrationMetrics",
    "ProbabilityCalibrator",
    "estimate_holdout_improvement",
]
