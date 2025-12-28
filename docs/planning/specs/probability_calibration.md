# Probability Calibration Specification

**Version:** 1.0.0
**Date:** 2025-12-28
**Priority:** P0 (Must Fix)

---

## Overview

### Problem Statement

Tree-based models (XGBoost, LightGBM, CatBoost) output probability-like scores, but these are **miscalibrated**. A model may output 0.70 confidence for a prediction, but the true accuracy at that confidence level may only be 55%. This creates three critical problems:

1. **Position Sizing Errors:** Trading strategies use confidence scores to size positions. If confidence is inflated, positions will be too large, increasing risk by 30-50%.

2. **Ensemble Stacking Distortion:** Meta-learners in stacking ensembles learn from these miscalibrated inputs, compounding the error through the ensemble.

3. **Invalid Threshold Selection:** Strategies that filter trades by confidence (e.g., "only trade if confidence > 0.6") will use incorrect thresholds.

### Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Brier Score | 0.25-0.30 | < 0.15 |
| ECE (Expected Calibration Error) | 0.10-0.15 | < 0.05 |
| Position Sizing Accuracy | Baseline | 30-50% improvement |
| Ensemble Performance | Baseline | 5-10% improvement |

### Dependencies

- `sklearn.isotonic.IsotonicRegression`
- `sklearn.linear_model.LogisticRegression`

---

## Technical Design

### Calibration Methods

**Isotonic Regression (Preferred for Boosting):**
- Non-parametric, monotonic transformation
- Works well for tree models
- Requires ≥100 samples per class
- Can overfit with small validation sets

**Platt Scaling (Sigmoid):**
- Parametric (logistic regression)
- Works with smaller sample sizes
- Assumes S-shaped calibration curve
- Better for linear models

**Auto Selection:**
```python
if min_class_count >= 100:
    use isotonic  # More flexible
else:
    use sigmoid   # More stable
```

### Calibration Metrics

**Brier Score (Multi-Class):**
```python
brier = mean(sum((probs - one_hot_labels)^2, axis=1))
```
- Lower is better
- Perfect calibration: Brier ≈ 0.0
- Random guessing: Brier ≈ 0.67 for 3 classes

**Expected Calibration Error (ECE):**
```python
# Bin predictions by confidence
for bin in bins:
    accuracy_in_bin = actual_accuracy
    confidence_in_bin = mean_confidence
    ece += |accuracy - confidence| * count_in_bin
```
- Measures alignment between confidence and accuracy
- Perfect calibration: ECE = 0.0

**Reliability Diagram:**
- X-axis: Predicted probability bins
- Y-axis: Actual accuracy in each bin
- Perfect calibration: diagonal line

---

## Implementation

### File: `src/models/calibration/calibrator.py`

```python
"""
Probability Calibration for ML Trading Models.

Implements isotonic regression (for boosting) and Platt scaling (for linear)
to correct miscalibrated probability outputs.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for probability calibration."""
    method: Literal["isotonic", "sigmoid", "auto"] = "auto"
    cv: int = 5  # CV folds for calibration
    min_samples: int = 100  # Minimum samples per class


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""
    brier_before: float
    brier_after: float
    ece_before: float  # Expected Calibration Error
    ece_after: float
    reliability_bins: Dict[str, np.ndarray]


class ProbabilityCalibrator:
    """
    Calibrates probability outputs from classification models.

    Boosting models (XGBoost, LightGBM, CatBoost) are notoriously
    miscalibrated. This class applies isotonic or sigmoid calibration
    to correct probabilities for downstream use.

    Example:
        >>> calibrator = ProbabilityCalibrator(CalibrationConfig())
        >>> calibrator.fit(y_val, probas_val)
        >>> calibrated_probs = calibrator.calibrate(probas_test)
    """

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        self._calibrators: Dict[int, Any] = {}  # One per class
        self._is_fitted: bool = False

    def fit(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> CalibrationMetrics:
        """
        Fit calibrators on validation data.

        Args:
            y_true: True labels (n_samples,)
            probabilities: Uncalibrated probabilities (n_samples, n_classes)

        Returns:
            CalibrationMetrics with before/after quality scores
        """
        n_classes = probabilities.shape[1]

        # Compute pre-calibration metrics
        brier_before = self._compute_brier(y_true, probabilities)
        ece_before = self._compute_ece(y_true, probabilities)

        # Fit calibrator for each class (one-vs-rest)
        method = self._select_method(y_true, n_classes)

        for cls in range(n_classes):
            y_binary = (y_true == cls).astype(int)
            probs_cls = probabilities[:, cls]

            if method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:
                calibrator = LogisticRegression()

            calibrator.fit(probs_cls.reshape(-1, 1), y_binary)
            self._calibrators[cls] = calibrator

        self._is_fitted = True

        # Compute post-calibration metrics
        calibrated = self.calibrate(probabilities)
        brier_after = self._compute_brier(y_true, calibrated)
        ece_after = self._compute_ece(y_true, calibrated)

        logger.info(
            f"Calibration complete: Brier {brier_before:.4f} -> {brier_after:.4f}, "
            f"ECE {ece_before:.4f} -> {ece_after:.4f}"
        )

        return CalibrationMetrics(
            brier_before=brier_before,
            brier_after=brier_after,
            ece_before=ece_before,
            ece_after=ece_after,
            reliability_bins=self._compute_reliability_bins(y_true, calibrated),
        )

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probability outputs.

        Args:
            probabilities: Uncalibrated (n_samples, n_classes)

        Returns:
            Calibrated probabilities (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        n_samples, n_classes = probabilities.shape
        calibrated = np.zeros_like(probabilities)

        for cls in range(n_classes):
            probs_cls = probabilities[:, cls].reshape(-1, 1)
            calibrated[:, cls] = self._calibrators[cls].predict(probs_cls)

        # Normalize to sum to 1
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

        return calibrated

    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'calibrators': self._calibrators,
                'is_fitted': self._is_fitted,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        """Load calibrator from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        calibrator = cls(data['config'])
        calibrator._calibrators = data['calibrators']
        calibrator._is_fitted = data['is_fitted']
        return calibrator

    def _select_method(self, y_true: np.ndarray, n_classes: int) -> str:
        """Select calibration method based on data."""
        if self.config.method != "auto":
            return self.config.method
        # Isotonic for enough samples, sigmoid otherwise
        min_class_count = min(np.bincount(y_true.astype(int), minlength=n_classes))
        return "isotonic" if min_class_count >= self.config.min_samples else "sigmoid"

    def _compute_brier(self, y_true: np.ndarray, probabilities: np.ndarray) -> float:
        """Compute multi-class Brier score."""
        n_classes = probabilities.shape[1]
        y_onehot = np.eye(n_classes)[y_true.astype(int)]
        return float(np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1)))

    def _compute_ece(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        confidences = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = accuracies[mask].mean()
                bin_confidence = confidences[mask].mean()
                ece += mask.sum() * abs(bin_accuracy - bin_confidence)

        return float(ece / len(y_true))

    def _compute_reliability_bins(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """Compute reliability diagram bins."""
        confidences = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_accuracies = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracies[i] = accuracies[mask].mean()
                bin_confidences[i] = confidences[mask].mean()
                bin_counts[i] = mask.sum()

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
        }


__all__ = ["ProbabilityCalibrator", "CalibrationConfig", "CalibrationMetrics"]
```

---

## Integration

### Modification to `src/models/trainer.py`

Add after line 234 (after evaluation metrics):

```python
# Calibration
from src.models.calibration import ProbabilityCalibrator, CalibrationConfig

logger.info("Calibrating probability outputs...")
calibrator = ProbabilityCalibrator(CalibrationConfig())
calibration_metrics = calibrator.fit(
    y_true=y_val,
    probabilities=val_predictions.class_probabilities,
)

eval_metrics["calibration"] = {
    "brier_before": calibration_metrics.brier_before,
    "brier_after": calibration_metrics.brier_after,
    "ece_before": calibration_metrics.ece_before,
    "ece_after": calibration_metrics.ece_after,
}

# Save calibrator alongside model
if not skip_save:
    calibrator_path = self.output_path / "checkpoints" / "calibrator.pkl"
    calibrator.save(calibrator_path)
    logger.info(f"Saved calibrator to {calibrator_path}")
```

### Modification to `src/cross_validation/oof_generator.py`

After generating OOF predictions, calibrate them:

```python
# After OOF prediction generation
logger.info("Calibrating OOF predictions...")
calibrator = ProbabilityCalibrator(CalibrationConfig())

# Split OOF data for calibration
cal_size = int(len(y_true) * 0.3)
cal_idx = np.random.choice(len(y_true), size=cal_size, replace=False)
train_idx = np.setdiff1d(np.arange(len(y_true)), cal_idx)

# Fit calibrator on subset
cal_metrics = calibrator.fit(
    y_true[cal_idx],
    oof_probabilities[cal_idx],
)

# Calibrate all OOF predictions
oof_probabilities = calibrator.calibrate(oof_probabilities)
```

---

## Testing

### Unit Tests

**File: `tests/models/test_calibration.py`**

```python
import numpy as np
import pytest

from src.models.calibration import ProbabilityCalibrator, CalibrationConfig


@pytest.fixture
def sample_data():
    """Generate sample calibration data."""
    np.random.seed(42)
    n_samples = 1000
    n_classes = 3

    y_true = np.random.randint(0, n_classes, n_samples)
    # Generate miscalibrated probabilities
    probs = np.random.dirichlet([1, 1, 1], n_samples)
    # Bias toward predictions
    for i in range(n_samples):
        probs[i, y_true[i]] += 0.3
    probs = probs / probs.sum(axis=1, keepdims=True)

    return y_true, probs


def test_calibration_improves_brier(sample_data):
    """Calibration should reduce Brier score."""
    y_true, probs = sample_data

    calibrator = ProbabilityCalibrator(CalibrationConfig())
    metrics = calibrator.fit(y_true, probs)

    assert metrics.brier_after < metrics.brier_before


def test_calibration_reduces_ece(sample_data):
    """Calibration should reduce ECE."""
    y_true, probs = sample_data

    calibrator = ProbabilityCalibrator(CalibrationConfig())
    metrics = calibrator.fit(y_true, probs)

    assert metrics.ece_after < metrics.ece_before


def test_calibrated_probs_sum_to_one(sample_data):
    """Calibrated probabilities should sum to 1."""
    y_true, probs = sample_data

    calibrator = ProbabilityCalibrator(CalibrationConfig())
    calibrator.fit(y_true, probs)
    calibrated = calibrator.calibrate(probs)

    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, rtol=1e-5)


def test_save_load(sample_data, tmp_path):
    """Test calibrator serialization."""
    y_true, probs = sample_data

    calibrator = ProbabilityCalibrator(CalibrationConfig())
    calibrator.fit(y_true, probs)

    # Save
    save_path = tmp_path / "calibrator.pkl"
    calibrator.save(save_path)

    # Load
    loaded = ProbabilityCalibrator.load(save_path)

    # Test predictions match
    original_cal = calibrator.calibrate(probs)
    loaded_cal = loaded.calibrate(probs)

    np.testing.assert_array_almost_equal(original_cal, loaded_cal)
```

### Integration Test

```bash
# Train model with calibration
python scripts/train_model.py --model xgboost --horizon 20

# Check logs for:
# "Calibration complete: Brier 0.2XXX -> 0.1XXX, ECE 0.1XXX -> 0.0XXX"

# Verify artifacts
ls experiments/runs/<run_id>/checkpoints/calibrator.pkl
```

---

## Acceptance Criteria

- [ ] Brier score drops from 0.25-0.30 to < 0.15
- [ ] ECE drops from 0.10-0.15 to < 0.05
- [ ] Reliability diagram shows near-diagonal alignment
- [ ] Calibrator saves/loads correctly
- [ ] Integration with `trainer.py` complete
- [ ] Integration with `oof_generator.py` complete
- [ ] All unit tests pass
- [ ] No performance regression (< 5% training time increase)

---

## Cross-References

- [ROADMAP.md](../ROADMAP.md#11-probability-calibration) - Phase 1 overview
- [GAPS_ANALYSIS.md](../GAPS_ANALYSIS.md#gap-1-no-probability-calibration) - Detailed gap analysis
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md#step-11-implement-probability-calibration) - Migration steps

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-28 | ML Engineering | Initial calibration spec from IMPLEMENTATION_PLAN.md |
