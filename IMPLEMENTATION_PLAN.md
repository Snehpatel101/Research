# IMPLEMENTATION PLAN: OHLCV ML Trading Pipeline

**Version:** 1.0.0
**Date:** 2025-12-28
**Status:** Ready for Development

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Assessment](#current-state-assessment)
3. [Phase 1: Critical Foundation (Week 1-2)](#phase-1-critical-foundation-week-1-2)
4. [Phase 2: Production Safety (Week 3-4)](#phase-2-production-safety-week-3-4)
5. [Phase 3: Performance Upgrades (Week 5-8)](#phase-3-performance-upgrades-week-5-8)
6. [Detailed Implementation Specs](#detailed-implementation-specs)
7. [Migration Guide](#migration-guide)
8. [Expected Outcomes](#expected-outcomes)
9. [Risk Mitigation](#risk-mitigation)
10. [Appendix: Code Templates](#appendix-code-templates)

---

## Executive Summary

This implementation plan addresses **12 critical issues** identified in the OHLCV ML trading pipeline. The current pipeline has a solid foundation with 12 implemented models across 4 families, 150+ features, and proper purged k-fold CV. However, several gaps prevent production deployment:

**P0 Issues (Must Fix):**
- No probability calibration (30-50% position sizing errors)
- CV loads pre-scaled data globally (leaking future statistics)
- Phase 5 (Inference) not implemented

**P1 Issues (Production Safety):**
- No online drift detection
- Label-aware purging not wired
- No walk-forward evaluation
- Sequence model CV uses wrong data structure

**Expected Impact After Implementation:**
| Metric | Before | After |
|--------|--------|-------|
| Calibration (Brier) | 0.25-0.30 | < 0.15 |
| CV Metric Inflation | +15-25% | < 5% |
| Drift Detection | None | Real-time |
| Deployment Ready | No | Yes |

**Timeline:** 8 weeks for full implementation

---

## Current State Assessment

### Pipeline Strengths

1. **Solid Model Factory Architecture**
   - 12 models registered: XGBoost, LightGBM, CatBoost, LSTM, GRU, TCN, Transformer, RF, Logistic, SVM, Voting, Stacking, Blending
   - Plugin-based registry with `@register()` decorator
   - Unified `BaseModel` interface (`src/models/base.py`)
   - `TrainingMetrics` and `PredictionOutput` standardized containers

2. **Comprehensive Data Pipeline**
   - 15 stages: ingest -> clean -> sessions -> regime -> mtf -> features -> labeling -> ga_optimize -> final_labels -> splits -> scaling -> scaled_validation -> datasets -> validation -> reporting
   - 150+ features (momentum, volatility, trend, wavelets, microstructure)
   - Triple-barrier labeling with Optuna optimization
   - `TimeSeriesDataContainer` for unified data access

3. **Cross-Validation Foundation**
   - `PurgedKFold` with configurable purge (60 bars) and embargo (1440 bars)
   - `ModelAwareCV` adapts splits per model family
   - `OOFGenerator` creates stacking datasets
   - Walk-forward feature selection exists

4. **Code Quality**
   - Type hints throughout
   - Dataclasses for configuration
   - Comprehensive docstrings
   - Logging infrastructure

### Critical Gaps

| Gap | Location | Impact | Priority |
|-----|----------|--------|----------|
| **No Probability Calibration** | `src/models/trainer.py`, `src/cross_validation/oof_generator.py` | 30-50% position sizing errors | P0 |
| **CV Leakage from Global Scaling** | `src/cross_validation/oof_generator.py` line 197-198 | Inflated metrics | P0 |
| **Phase 5 Not Implemented** | Missing `src/inference/` | No production path | P0 |
| **No Online Drift Detection** | Missing `src/monitoring/` | Silent model decay | P1 |
| **Label-Aware Purging Dead** | `src/phase1/stages/labeling/triple_barrier.py`, `src/cross_validation/cv_runner.py` | Suboptimal purging | P1 |
| **No Walk-Forward Evaluation** | Missing in `src/cross_validation/` | Misses temporal degradation | P1 |
| **Sequence Model CV Wrong** | `src/cross_validation/cv_runner.py` line 327 | Invalid neural CV | P1 |
| **MTF Lookahead Unverified** | `src/phase1/stages/mtf/` | Potential data leak | P2 |
| **No CPCV** | Missing in `src/cross_validation/` | Weak hyperparameter robustness | P2 |
| **No Regime-Adaptive Models** | Missing `src/regime/` | 20-30% Sharpe opportunity | P2 |
| **No Conformal Prediction** | Missing `src/models/conformal.py` | No uncertainty quantification | P3 |
| **Time Bars Only** | Missing `src/phase1/stages/ingest/alternative_bars.py` | Suboptimal sampling | P3 |

---

## Phase 1: Critical Foundation (Week 1-2)

### 1.1 Probability Calibration

**Problem:** Tree models (XGBoost, LightGBM, CatBoost) output miscalibrated probabilities. This causes:
- Position sizing based on wrong confidence
- Ensemble stacking learns from distorted inputs
- Threshold-based trading decisions are biased

**Solution:** Implement isotonic/Platt calibration with Brier/ECE metrics.

**Files to Create:**
```
src/models/calibration/
    __init__.py
    calibrator.py         # CalibratedPredictor class
    metrics.py            # Brier score, ECE, reliability curves
```

**Files to Modify:**
- `src/models/trainer.py` - Add calibration step after training
- `src/cross_validation/oof_generator.py` - Calibrate OOF predictions
- `src/models/base.py` - Add `calibrator` attribute to PredictionOutput

**Implementation Details:** See Section 6.1

**Acceptance Criteria:**
- Brier score < 0.15 (down from 0.25-0.30)
- ECE < 0.05
- Reliability curves show diagonal alignment

---

### 1.2 Fold-Aware Scaling in CV

**Problem:** `oof_generator.py` loads pre-scaled data from `TimeSeriesDataContainer.get_sklearn_arrays("train")`. The scaler was fit on the entire training set, so each fold's validation data has been transformed using statistics from "future" samples in other folds.

**Current Code (Problem):**
```python
# src/cross_validation/oof_generator.py line 197-198
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
# X is already scaled globally - WRONG
```

**Solution:** Scale within each fold using only that fold's training data.

**Files to Create:**
```
src/cross_validation/fold_scaling.py   # FoldAwareScaler class
```

**Files to Modify:**
- `src/cross_validation/oof_generator.py` - Add per-fold scaling
- `src/cross_validation/cv_runner.py` - Pass unscaled data option

**Implementation Details:** See Section 6.2

**Acceptance Criteria:**
- CV metrics drop 5-15% (expected - removes optimism)
- No scaling parameters leak between folds
- Fold validation uses only training fold statistics

---

### 1.3 Lookahead Audit for MTF Features

**Problem:** Multi-timeframe features (e.g., 15min, 1H aggregates) may inadvertently peek into future data if resampling isn't done with `closed='left', label='left'`.

**Solution:** Automated lookahead verification suite.

**Files to Create:**
```
src/validation/lookahead_audit.py      # LookaheadAuditor class
tests/validation/test_lookahead.py     # Automated tests
```

**Files to Modify:**
- `src/phase1/stages/mtf/generator.py` - Document/verify closed/label params
- Pipeline reporting - Add lookahead verification results

**Implementation Details:** See Section 6.3

**Acceptance Criteria:**
- 0 lookahead violations detected
- All MTF resampling uses `closed='left', label='left'`
- Corruption test passes (future corruption doesn't affect features)

---

### 1.4 Fast Ensemble Baseline

**Problem:** No quick-deploy ensemble option for production testing.

**Solution:** Implement low-latency boosting ensemble (XGBoost + LightGBM + CatBoost) with voting.

**Files to Modify:**
- `src/models/ensemble/voting.py` - Optimize for inference speed
- Add benchmark script

**Acceptance Criteria:**
- Inference latency < 5ms for 1000 samples
- Memory < 500MB total
- No GPU required

---

## Phase 2: Production Safety (Week 3-4)

### 2.1 Online Drift Detection

**Problem:** Models decay silently in production without alerts. Current pipeline only has offline PSI checks in `scaled_validation`.

**Solution:** Implement real-time drift monitoring using ADWIN for performance drift and PSI for feature drift.

**Files to Create:**
```
src/monitoring/
    __init__.py
    drift_detector.py     # DriftDetector with ADWIN, PSI
    performance_monitor.py # PerformanceMonitor with alerting
    alert_handlers.py     # Slack, email, logging handlers
```

**Dependencies:** `river` (for ADWIN)

**Implementation Details:** See Section 6.4

**Acceptance Criteria:**
- ADWIN detects performance drift within 100 samples of shift
- PSI alerts when feature drift > 0.2
- Configurable alert thresholds and handlers

---

### 2.2 Label-Aware Purging

**Problem:** `PurgedKFold` supports `label_end_times` but it's never passed. Triple-barrier labels have variable resolution times (stored in `bars_to_hit`), so fixed purge bars is a coarse approximation.

**Current Code (Unused):**
```python
# src/cross_validation/purged_kfold.py line 207-212
if label_end_times is not None and has_datetime_index:
    test_start_time = X.index[test_start]
    for i in range(purge_start):
        if label_end_times.iloc[i] >= test_start_time:
            train_mask[i] = False
```

**Solution:** Compute and persist `label_end_time` during labeling, wire it through to CV.

**Files to Modify:**
- `src/phase1/stages/labeling/triple_barrier.py` - Compute `label_end_time`
- `src/phase1/stages/datasets/container.py` - Expose `label_end_times`
- `src/cross_validation/cv_runner.py` - Pass to PurgedKFold
- `src/cross_validation/oof_generator.py` - Pass to PurgedKFold

**Implementation Details:** See Section 6.5

**Acceptance Criteria:**
- `label_end_time_h{horizon}` column persisted in parquet
- CV uses label-aware purging for overlapping events
- Validation set contains no samples whose labels depend on future test data

---

### 2.3 Walk-Forward Evaluation

**Problem:** Purged k-fold averages hide temporal degradation. A model may perform well in 2020 folds but fail in 2024.

**Solution:** Add rolling-origin walk-forward evaluator alongside k-fold.

**Files to Create:**
```
src/cross_validation/walk_forward.py   # WalkForwardEvaluator class
scripts/run_walk_forward.py            # CLI entrypoint
```

**Implementation Details:** See Section 6.6

**Acceptance Criteria:**
- Reports per-window metrics (Sharpe, F1, accuracy)
- Shows temporal degradation curve
- Supports expanding and rolling windows

---

### 2.4 CPCV (Combinatorial Purged CV)

**Problem:** Standard purged k-fold tests a single path through time. With hyperparameter tuning across many trials, winner selection may be overfitting to that specific path.

**Solution:** Implement CPCV that tests C(n,k) combinations to estimate probability of backtest overfitting (PBO).

**Files to Create:**
```
src/cross_validation/cpcv.py           # CombinatorialPurgedCV class
src/cross_validation/pbo.py            # ProbabilityOfBacktestOverfitting
```

**Reference:** Bailey et al. (2014) "The Probability of Backtest Overfitting"

**Implementation Details:** See Section 6.7

**Acceptance Criteria:**
- PBO estimate computed after hyperparameter tuning
- Warning when PBO > 0.5
- Block deployment when PBO > 0.8

---

### 2.5 Sequence Model CV Fix

**Problem:** `cv_runner.py` uses `container.get_sklearn_arrays("train")` for all models, but LSTM/GRU/TCN need 3D sequences from `get_pytorch_sequences()`.

**Current Code (Wrong):**
```python
# src/cross_validation/cv_runner.py line 327
X, y, weights = container.get_sklearn_arrays("train", return_df=True)
# This returns 2D arrays, but neural models need 3D sequences
```

**Solution:** Add model-type-aware data loading in CV.

**Files to Modify:**
- `src/cross_validation/cv_runner.py` - Check `model.requires_sequences`
- `src/cross_validation/oof_generator.py` - Handle sequence data

**Implementation Details:** See Section 6.8

**Acceptance Criteria:**
- LSTM/GRU/TCN CV uses correct sequence structure
- Sequences don't cross symbol boundaries
- Temporal ordering preserved

---

## Phase 3: Performance Upgrades (Week 5-8)

### 3.1 Inference Pipeline (Phase 5)

**Problem:** No production deployment capability. Phase 5 is documented in `docs/phases/PHASE_5.md` but `src/inference/` doesn't exist.

**Solution:** Implement complete inference pipeline with serialization, monitoring, and deployment profiles.

**Files to Create:**
```
src/inference/
    __init__.py
    pipeline.py           # InferencePipeline class
    serializer.py         # PipelineSerializer (scaler + features + models + calibrator)
    server.py             # FastAPI inference server (optional)
    batch.py              # BatchInference for offline scoring
scripts/
    serve_model.py        # Start inference server
    batch_inference.py    # Run batch predictions
```

**Implementation Details:** See Section 6.9

**Acceptance Criteria:**
- Single artifact bundle with everything needed for inference
- Inference latency < 100ms for ensemble
- Lookahead verification passes
- Drift monitoring integrated

---

### 3.2 Regime-Adaptive Models

**Problem:** Single model for all market regimes misses 20-30% Sharpe improvement from regime-specific strategies.

**Solution:** HMM-based regime detection with specialist models per regime.

**Files to Create:**
```
src/regime/
    __init__.py
    detector.py           # RegimeDetector with HMM (3 states: bull/bear/neutral)
    specialist.py         # RegimeSpecialistModel wrapper
    ensemble.py           # RegimeAdaptiveEnsemble
```

**Dependencies:** `hmmlearn`

**Implementation Details:** See Section 6.10

**Acceptance Criteria:**
- HMM converges to 3 interpretable regimes
- Specialist models train on regime-specific data
- Ensemble routes predictions through active regime

---

### 3.3 Conformal Prediction

**Problem:** No uncertainty quantification for position sizing. Model may be confident but wrong.

**Solution:** Implement conformal prediction for prediction intervals.

**Files to Create:**
```
src/models/conformal.py  # ConformalPredictor class
```

**Reference:** Shafer & Vovk (2008) "A Tutorial on Conformal Prediction"

**Implementation Details:** See Section 6.11

**Acceptance Criteria:**
- 90% prediction intervals contain true value 90% of time
- Coverage calibrated across regimes
- Position sizing based on interval width

---

### 3.4 Alternative Bar Types

**Problem:** Time bars have poor statistical properties (varying volume, activity). Dollar bars sample by fixed notional value, yielding better i.i.d. properties.

**Solution:** Implement dollar/volume bar generation.

**Files to Create:**
```
src/phase1/stages/ingest/alternative_bars.py   # DollarBarGenerator, VolumeBarGenerator
```

**Reference:** Lopez de Prado (2018) "Advances in Financial Machine Learning" Chapter 2

**Implementation Details:** See Section 6.12

**Acceptance Criteria:**
- Dollar bars have lower autocorrelation than time bars
- Returns closer to normal distribution
- Pipeline supports bar type selection via config

---

## Detailed Implementation Specs

### 6.1 Probability Calibration

**File: `src/models/calibration/calibrator.py`**

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

**Integration into `src/models/trainer.py`:**

Add after line 234 (after evaluation metrics):

```python
# Calibration
from src.models.calibration import ProbabilityCalibrator, CalibrationConfig

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
    calibrator.save(self.output_path / "checkpoints" / "calibrator.pkl")
```

---

### 6.2 Fold-Aware Scaling

**File: `src/cross_validation/fold_scaling.py`**

```python
"""
Fold-Aware Scaling for Cross-Validation.

Ensures each CV fold's scaler is fit only on training indices,
preventing information leakage from validation/test data.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class FoldAwareScaler:
    """
    Scaler that fits on fold training data only.

    Prevents CV leakage by ensuring validation data is transformed
    using only statistics from the training portion of each fold.

    Example:
        >>> scaler = FoldAwareScaler(method="robust")
        >>> X_train_scaled, X_val_scaled = scaler.fit_transform_fold(
        ...     X_train, X_val
        ... )
    """

    def __init__(
        self,
        method: str = "robust",
        clip_outliers: bool = True,
        clip_std: float = 5.0,
    ) -> None:
        """
        Initialize FoldAwareScaler.

        Args:
            method: Scaling method ("robust", "standard")
            clip_outliers: Whether to clip extreme values after scaling
            clip_std: Number of standard deviations for clipping
        """
        self.method = method
        self.clip_outliers = clip_outliers
        self.clip_std = clip_std
        self._scaler = None

    def fit_transform_fold(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit on training data and transform both train and validation.

        Args:
            X_train: Training features (n_train, n_features)
            X_val: Validation features (n_val, n_features)

        Returns:
            Tuple of (X_train_scaled, X_val_scaled)
        """
        # Create fresh scaler for this fold
        if self.method == "robust":
            self._scaler = RobustScaler()
        elif self.method == "standard":
            self._scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

        # Fit only on training data
        X_train_scaled = self._scaler.fit_transform(X_train)

        # Transform validation using training statistics
        X_val_scaled = self._scaler.transform(X_val)

        # Optional outlier clipping
        if self.clip_outliers:
            X_train_scaled = np.clip(
                X_train_scaled, -self.clip_std, self.clip_std
            )
            X_val_scaled = np.clip(
                X_val_scaled, -self.clip_std, self.clip_std
            )

        return X_train_scaled, X_val_scaled

    def fit_transform_fold_df(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """DataFrame-preserving version of fit_transform_fold."""
        X_train_scaled, X_val_scaled = self.fit_transform_fold(
            X_train.values, X_val.values
        )
        return (
            pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
            pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index),
        )


def scale_cv_fold(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    method: str = "robust",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to scale a CV fold.

    Args:
        X: Full feature DataFrame
        train_idx: Training indices
        val_idx: Validation indices
        method: Scaling method

    Returns:
        Tuple of (X_train_scaled, X_val_scaled) as numpy arrays
    """
    scaler = FoldAwareScaler(method=method)
    X_train = X.iloc[train_idx].values
    X_val = X.iloc[val_idx].values
    return scaler.fit_transform_fold(X_train, X_val)


__all__ = ["FoldAwareScaler", "scale_cv_fold"]
```

**Modification to `src/cross_validation/oof_generator.py`:**

Replace lines 193-198 with:

```python
from src.cross_validation.fold_scaling import FoldAwareScaler

# Check if model requires scaling
model_info = ModelRegistry.get_model_info(model_name)
requires_scaling = model_info.get("requires_scaling", True)

for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
    logger.debug(f"  Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")

    # Extract fold data
    X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Apply fold-aware scaling if required
    if requires_scaling:
        scaler = FoldAwareScaler(method="robust")
        X_train, X_val = scaler.fit_transform_fold(
            X_train_raw.values, X_val_raw.values
        )
    else:
        X_train, X_val = X_train_raw.values, X_val_raw.values
```

---

### 6.3 Lookahead Audit

**File: `src/validation/lookahead_audit.py`**

```python
"""
Lookahead Bias Audit for OHLCV Features.

Verifies that features computed from OHLCV data do not peek into future bars.
Uses corruption testing: if corrupting future data changes current features,
there's lookahead bias.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LookaheadViolation:
    """Details of a detected lookahead violation."""
    feature_name: str
    sample_index: int
    original_value: float
    corrupted_value: float
    difference: float


@dataclass
class AuditResult:
    """Results from lookahead audit."""
    passed: bool
    n_features_tested: int
    n_samples_tested: int
    violations: List[LookaheadViolation]
    summary: Dict[str, int]  # feature -> violation count


class LookaheadAuditor:
    """
    Audits features for lookahead bias using corruption testing.

    Method:
    1. Compute features for sample at index i
    2. Corrupt all data after index i (set to NaN or extreme values)
    3. Recompute features for sample at index i
    4. If features differ, there's lookahead

    Example:
        >>> auditor = LookaheadAuditor(feature_pipeline)
        >>> result = auditor.audit(df, n_samples=100)
        >>> if not result.passed:
        ...     print(f"Found {len(result.violations)} violations")
    """

    def __init__(
        self,
        feature_func: Callable[[pd.DataFrame], pd.DataFrame],
        tolerance: float = 1e-10,
    ) -> None:
        """
        Initialize LookaheadAuditor.

        Args:
            feature_func: Function that computes features from OHLCV DataFrame
            tolerance: Numerical tolerance for comparison
        """
        self.feature_func = feature_func
        self.tolerance = tolerance

    def audit(
        self,
        df: pd.DataFrame,
        n_samples: int = 100,
        sample_indices: Optional[List[int]] = None,
        feature_subset: Optional[List[str]] = None,
    ) -> AuditResult:
        """
        Run lookahead audit on DataFrame.

        Args:
            df: OHLCV DataFrame
            n_samples: Number of random samples to test
            sample_indices: Specific indices to test (overrides n_samples)
            feature_subset: Specific features to test (None = all)

        Returns:
            AuditResult with pass/fail and any violations
        """
        logger.info(f"Starting lookahead audit on {len(df)} rows...")

        # Compute original features
        original_features = self.feature_func(df)

        if feature_subset:
            original_features = original_features[feature_subset]

        # Select test indices
        if sample_indices is None:
            # Avoid edges (need future data to corrupt)
            valid_range = range(100, len(df) - 100)
            sample_indices = np.random.choice(
                list(valid_range),
                size=min(n_samples, len(valid_range)),
                replace=False
            ).tolist()

        violations: List[LookaheadViolation] = []
        violation_counts: Dict[str, int] = {}

        for idx in sample_indices:
            # Create corrupted copy
            df_corrupted = df.copy()
            df_corrupted.iloc[idx + 1:] = 999999.0  # Obvious corruption

            # Recompute features
            try:
                corrupted_features = self.feature_func(df_corrupted)
                if feature_subset:
                    corrupted_features = corrupted_features[feature_subset]
            except Exception as e:
                logger.warning(f"Feature computation failed at idx {idx}: {e}")
                continue

            # Compare
            original_row = original_features.iloc[idx]
            corrupted_row = corrupted_features.iloc[idx]

            for col in original_features.columns:
                orig_val = original_row[col]
                corr_val = corrupted_row[col]

                # Skip NaN comparisons
                if pd.isna(orig_val) and pd.isna(corr_val):
                    continue

                diff = abs(orig_val - corr_val) if not (pd.isna(orig_val) or pd.isna(corr_val)) else float('inf')

                if diff > self.tolerance:
                    violations.append(LookaheadViolation(
                        feature_name=col,
                        sample_index=idx,
                        original_value=float(orig_val),
                        corrupted_value=float(corr_val),
                        difference=float(diff),
                    ))
                    violation_counts[col] = violation_counts.get(col, 0) + 1

        passed = len(violations) == 0

        if passed:
            logger.info(f"Lookahead audit PASSED: {len(sample_indices)} samples, {len(original_features.columns)} features")
        else:
            logger.warning(
                f"Lookahead audit FAILED: {len(violations)} violations in "
                f"{len(violation_counts)} features"
            )

        return AuditResult(
            passed=passed,
            n_features_tested=len(original_features.columns),
            n_samples_tested=len(sample_indices),
            violations=violations[:100],  # Limit for memory
            summary=violation_counts,
        )


def audit_mtf_features(
    df: pd.DataFrame,
    mtf_generator: "MTFGenerator",
) -> AuditResult:
    """
    Convenience function to audit MTF features.

    Checks that multi-timeframe aggregations don't peek ahead.
    """
    def compute_mtf(data: pd.DataFrame) -> pd.DataFrame:
        return mtf_generator.generate(data)

    auditor = LookaheadAuditor(compute_mtf)
    return auditor.audit(df, n_samples=50)


__all__ = ["LookaheadAuditor", "LookaheadViolation", "AuditResult", "audit_mtf_features"]
```

---

### 6.4 Online Drift Detection

**File: `src/monitoring/drift_detector.py`**

```python
"""
Online Drift Detection for Production ML Monitoring.

Implements:
- ADWIN for performance/concept drift
- PSI for feature distribution drift
- Combined alerting system
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected."""
    FEATURE = "feature"      # Input distribution change
    CONCEPT = "concept"      # Performance degradation
    PREDICTION = "prediction"  # Output distribution change


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """A detected drift alert."""
    drift_type: DriftType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    adwin_delta: float = 0.002  # ADWIN sensitivity
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.2
    performance_warning_threshold: float = 0.15  # 15% drop
    performance_critical_threshold: float = 0.25  # 25% drop
    window_size: int = 1000


class ADWINDetector:
    """
    ADWIN (ADaptive WINdowing) for concept drift detection.

    Monitors a stream of values (e.g., prediction correctness) and
    detects when the distribution changes significantly.

    Reference: Bifet & Gavalda (2007)
    """

    def __init__(self, delta: float = 0.002) -> None:
        """
        Initialize ADWIN.

        Args:
            delta: Confidence parameter (lower = more sensitive)
        """
        try:
            from river.drift import ADWIN as RiverADWIN
            self._detector = RiverADWIN(delta=delta)
        except ImportError:
            logger.warning("river not installed, using simplified ADWIN")
            self._detector = None
            self._values = []
            self._delta = delta

        self.detected_drifts: List[int] = []
        self._n_samples = 0

    def update(self, value: float) -> bool:
        """
        Update detector with new value.

        Args:
            value: New observation (e.g., 1 if correct, 0 if wrong)

        Returns:
            True if drift detected
        """
        self._n_samples += 1

        if self._detector is not None:
            self._detector.update(value)
            if self._detector.drift_detected:
                self.detected_drifts.append(self._n_samples)
                return True
        else:
            # Simplified fallback
            self._values.append(value)
            if len(self._values) > 100:
                # Compare recent mean to historical mean
                recent = np.mean(self._values[-50:])
                historical = np.mean(self._values[:-50])
                if abs(recent - historical) > 0.1:  # Simplified threshold
                    self.detected_drifts.append(self._n_samples)
                    self._values = self._values[-50:]  # Reset
                    return True

        return False


class PSICalculator:
    """
    Population Stability Index for feature drift.

    Compares current distribution to reference distribution.
    PSI < 0.1: No drift
    PSI 0.1-0.2: Moderate drift
    PSI > 0.2: Significant drift
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self._reference_bins: Optional[np.ndarray] = None
        self._bin_edges: Optional[np.ndarray] = None

    def fit_reference(self, reference_values: np.ndarray) -> None:
        """Compute reference distribution bins."""
        self._bin_edges = np.percentile(
            reference_values,
            np.linspace(0, 100, self.n_bins + 1)
        )
        self._reference_bins, _ = np.histogram(
            reference_values, bins=self._bin_edges
        )
        self._reference_bins = self._reference_bins / len(reference_values)
        # Avoid zeros
        self._reference_bins = np.clip(self._reference_bins, 0.0001, None)

    def compute_psi(self, current_values: np.ndarray) -> float:
        """
        Compute PSI between reference and current.

        Args:
            current_values: Current distribution values

        Returns:
            PSI score
        """
        if self._reference_bins is None:
            raise RuntimeError("Must call fit_reference first")

        current_bins, _ = np.histogram(current_values, bins=self._bin_edges)
        current_bins = current_bins / len(current_values)
        current_bins = np.clip(current_bins, 0.0001, None)

        psi = np.sum(
            (current_bins - self._reference_bins) *
            np.log(current_bins / self._reference_bins)
        )
        return float(psi)


class OnlineDriftMonitor:
    """
    Combined drift monitoring for production ML systems.

    Monitors:
    - Feature drift via PSI
    - Performance drift via ADWIN
    - Prediction distribution drift

    Example:
        >>> monitor = OnlineDriftMonitor(reference_df, config)
        >>> for batch in production_data:
        ...     alerts = monitor.check(batch, predictions, actuals)
        ...     for alert in alerts:
        ...         handle_alert(alert)
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        config: DriftConfig,
        feature_names: Optional[List[str]] = None,
        alert_callback: Optional[Callable[[DriftAlert], None]] = None,
    ) -> None:
        """
        Initialize monitor with reference data.

        Args:
            reference_data: Reference feature matrix (n_samples, n_features)
            config: Drift detection configuration
            feature_names: Names of features for reporting
            alert_callback: Optional callback for alerts
        """
        self.config = config
        self.feature_names = feature_names or [f"f{i}" for i in range(reference_data.shape[1])]
        self.alert_callback = alert_callback

        # Initialize PSI calculators for each feature
        self._psi_calculators: Dict[str, PSICalculator] = {}
        for i, name in enumerate(self.feature_names):
            calc = PSICalculator()
            calc.fit_reference(reference_data[:, i])
            self._psi_calculators[name] = calc

        # Initialize performance ADWIN
        self._performance_adwin = ADWINDetector(config.adwin_delta)

        # Track baseline performance
        self._baseline_performance: Optional[float] = None

        # Alert history
        self.alerts: List[DriftAlert] = []

    def set_baseline_performance(self, performance: float) -> None:
        """Set baseline performance for comparison."""
        self._baseline_performance = performance

    def check(
        self,
        features: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
    ) -> List[DriftAlert]:
        """
        Check for drift in a batch of data.

        Args:
            features: Feature matrix (n_samples, n_features)
            predictions: Model predictions (optional)
            actuals: True labels (optional, for performance drift)

        Returns:
            List of detected drift alerts
        """
        batch_alerts: List[DriftAlert] = []
        timestamp = datetime.now()

        # Check feature drift
        feature_alerts = self._check_feature_drift(features, timestamp)
        batch_alerts.extend(feature_alerts)

        # Check performance drift if actuals provided
        if predictions is not None and actuals is not None:
            perf_alerts = self._check_performance_drift(
                predictions, actuals, timestamp
            )
            batch_alerts.extend(perf_alerts)

        # Store and callback
        self.alerts.extend(batch_alerts)
        if self.alert_callback:
            for alert in batch_alerts:
                self.alert_callback(alert)

        return batch_alerts

    def _check_feature_drift(
        self,
        features: np.ndarray,
        timestamp: datetime,
    ) -> List[DriftAlert]:
        """Check for feature distribution drift."""
        alerts = []

        for i, name in enumerate(self.feature_names):
            psi = self._psi_calculators[name].compute_psi(features[:, i])

            if psi > self.config.psi_critical_threshold:
                alerts.append(DriftAlert(
                    drift_type=DriftType.FEATURE,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Critical drift in {name}: PSI={psi:.3f}",
                    timestamp=timestamp,
                    details={"feature": name, "psi": psi},
                ))
            elif psi > self.config.psi_warning_threshold:
                alerts.append(DriftAlert(
                    drift_type=DriftType.FEATURE,
                    severity=AlertSeverity.WARNING,
                    message=f"Drift warning in {name}: PSI={psi:.3f}",
                    timestamp=timestamp,
                    details={"feature": name, "psi": psi},
                ))

        return alerts

    def _check_performance_drift(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        timestamp: datetime,
    ) -> List[DriftAlert]:
        """Check for performance degradation."""
        alerts = []

        # Compute accuracy
        correct = (predictions == actuals).astype(float)

        # Update ADWIN with each prediction
        for c in correct:
            if self._performance_adwin.update(c):
                alerts.append(DriftAlert(
                    drift_type=DriftType.CONCEPT,
                    severity=AlertSeverity.CRITICAL,
                    message="Concept drift detected by ADWIN",
                    timestamp=timestamp,
                    details={"n_drifts": len(self._performance_adwin.detected_drifts)},
                ))

        # Check performance drop if baseline set
        if self._baseline_performance is not None:
            current_perf = correct.mean()
            drop = (self._baseline_performance - current_perf) / self._baseline_performance

            if drop > self.config.performance_critical_threshold:
                alerts.append(DriftAlert(
                    drift_type=DriftType.CONCEPT,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Performance drop {drop:.1%} from baseline",
                    timestamp=timestamp,
                    details={"current": current_perf, "baseline": self._baseline_performance},
                ))
            elif drop > self.config.performance_warning_threshold:
                alerts.append(DriftAlert(
                    drift_type=DriftType.CONCEPT,
                    severity=AlertSeverity.WARNING,
                    message=f"Performance degradation {drop:.1%}",
                    timestamp=timestamp,
                    details={"current": current_perf, "baseline": self._baseline_performance},
                ))

        return alerts


__all__ = [
    "DriftType",
    "AlertSeverity",
    "DriftAlert",
    "DriftConfig",
    "ADWINDetector",
    "PSICalculator",
    "OnlineDriftMonitor",
]
```

---

### 6.5 Label-Aware Purging

**Modification to `src/phase1/stages/labeling/triple_barrier.py`:**

After line 168, add label_end_time computation:

```python
def compute_label_end_times(
    df: pd.DataFrame,
    bars_to_hit: np.ndarray,
    bar_interval_minutes: int = 5,
) -> pd.Series:
    """
    Compute when each label's outcome is known.

    For triple-barrier labels, the outcome is known when a barrier is hit
    or timeout occurs. This is essential for proper CV purging.

    Args:
        df: DataFrame with datetime index
        bars_to_hit: Number of bars until barrier hit
        bar_interval_minutes: Bar frequency in minutes

    Returns:
        Series of label end timestamps
    """
    import pandas as pd

    datetime_col = df.index if isinstance(df.index, pd.DatetimeIndex) else df['datetime']

    label_end_times = datetime_col + pd.to_timedelta(
        bars_to_hit * bar_interval_minutes, unit='m'
    )

    return pd.Series(label_end_times, index=df.index, name='label_end_time')
```

**Modification to `TripleBarrierLabeler.compute_labels()`:**

Add after line 305:

```python
# Compute label end times for CV purging
label_end_times = compute_label_end_times(
    df, bars_to_hit, bar_interval_minutes=5
)
result.metadata['label_end_times'] = label_end_times.values
```

**Modification to `src/cross_validation/oof_generator.py`:**

Replace the split call with:

```python
# Get label end times if available
label_end_times = None
if hasattr(self.cv, 'config') and 'label_end_times' in kwargs:
    label_end_times = kwargs.get('label_end_times')

for fold_idx, (train_idx, val_idx) in enumerate(
    self.cv.split(X, y, label_end_times=label_end_times)
):
```

---

### 6.6 Walk-Forward Evaluation

**File: `src/cross_validation/walk_forward.py`**

```python
"""
Walk-Forward (Rolling-Origin) Evaluation.

Evaluates models using expanding/rolling windows that respect
temporal ordering, providing a realistic estimate of live performance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward evaluation."""
    initial_train_size: float = 0.5  # Fraction for initial training
    step_size: int = 288  # Bars to step forward (1 day at 5-min)
    min_test_size: int = 288  # Minimum test window
    expanding: bool = True  # True = expanding, False = rolling
    purge_bars: int = 60
    embargo_bars: int = 1440


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""
    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_size: int
    test_size: int
    accuracy: float
    f1: float
    sharpe: Optional[float] = None
    predictions: Optional[np.ndarray] = None


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward evaluation results."""
    model_name: str
    horizon: int
    config: WalkForwardConfig
    window_results: List[WindowResult]

    @property
    def n_windows(self) -> int:
        return len(self.window_results)

    @property
    def mean_f1(self) -> float:
        return np.mean([w.f1 for w in self.window_results])

    @property
    def std_f1(self) -> float:
        return np.std([w.f1 for w in self.window_results])

    @property
    def temporal_degradation(self) -> float:
        """Compute slope of performance over time (negative = degradation)."""
        if len(self.window_results) < 2:
            return 0.0
        f1s = [w.f1 for w in self.window_results]
        x = np.arange(len(f1s))
        slope, _ = np.polyfit(x, f1s, 1)
        return float(slope)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        return pd.DataFrame([
            {
                "window": w.window_idx,
                "train_size": w.train_size,
                "test_size": w.test_size,
                "accuracy": w.accuracy,
                "f1": w.f1,
                "sharpe": w.sharpe,
            }
            for w in self.window_results
        ])


class WalkForwardEvaluator:
    """
    Implements walk-forward (rolling-origin) evaluation.

    Unlike k-fold CV which shuffles folds, walk-forward always moves
    forward in time, providing a realistic simulation of live trading.

    Example:
        >>> config = WalkForwardConfig(initial_train_size=0.5, step_size=288)
        >>> evaluator = WalkForwardEvaluator(config)
        >>> result = evaluator.evaluate("xgboost", X, y, horizon=20)
        >>> print(f"Mean F1: {result.mean_f1:.3f}, Degradation: {result.temporal_degradation:.4f}")
    """

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def generate_windows(
        self,
        n_samples: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index pairs for walk-forward.

        Yields:
            Tuple of (train_indices, test_indices) for each window
        """
        initial_train = int(n_samples * self.config.initial_train_size)

        train_end = initial_train
        while train_end < n_samples - self.config.min_test_size:
            # Training indices
            if self.config.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - initial_train)

            train_idx = np.arange(train_start, train_end - self.config.purge_bars)

            # Test indices (with embargo)
            test_start = train_end + self.config.embargo_bars
            test_end = min(test_start + self.config.step_size, n_samples)

            if test_end <= test_start:
                break

            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

            # Step forward
            train_end += self.config.step_size

    def evaluate(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int,
        model_config: Optional[Dict[str, Any]] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward evaluation.

        Args:
            model_name: Name of model to evaluate
            X: Feature DataFrame
            y: Labels
            horizon: Label horizon
            model_config: Optional model configuration
            sample_weights: Optional sample weights

        Returns:
            WalkForwardResult with per-window and aggregate metrics
        """
        from sklearn.metrics import accuracy_score, f1_score

        logger.info(f"Running walk-forward evaluation for {model_name}...")

        window_results: List[WindowResult] = []

        for window_idx, (train_idx, test_idx) in enumerate(
            self.generate_windows(len(X))
        ):
            logger.debug(
                f"Window {window_idx}: train={len(train_idx)}, test={len(test_idx)}"
            )

            # Extract data
            X_train = X.iloc[train_idx].values
            X_test = X.iloc[test_idx].values
            y_train = y.iloc[train_idx].values
            y_test = y.iloc[test_idx].values

            w_train = None
            if sample_weights is not None:
                w_train = sample_weights.iloc[train_idx].values

            # Create and train model
            model = ModelRegistry.create(model_name, config=model_config)

            # Split validation from training (last 20%)
            val_split = int(len(X_train) * 0.8)
            X_val = X_train[val_split:]
            y_val = y_train[val_split:]
            X_train_inner = X_train[:val_split]
            y_train_inner = y_train[:val_split]
            w_train_inner = w_train[:val_split] if w_train is not None else None

            model.fit(
                X_train=X_train_inner,
                y_train=y_train_inner,
                X_val=X_val,
                y_val=y_val,
                sample_weights=w_train_inner,
            )

            # Predict on test
            predictions = model.predict(X_test)
            y_pred = predictions.class_predictions

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            window_results.append(WindowResult(
                window_idx=window_idx,
                train_start=int(train_idx[0]),
                train_end=int(train_idx[-1]),
                test_start=int(test_idx[0]),
                test_end=int(test_idx[-1]),
                train_size=len(train_idx),
                test_size=len(test_idx),
                accuracy=float(accuracy),
                f1=float(f1),
                predictions=y_pred,
            ))

        result = WalkForwardResult(
            model_name=model_name,
            horizon=horizon,
            config=self.config,
            window_results=window_results,
        )

        logger.info(
            f"Walk-forward complete: {result.n_windows} windows, "
            f"mean_f1={result.mean_f1:.3f}, degradation={result.temporal_degradation:.4f}"
        )

        return result


__all__ = [
    "WalkForwardConfig",
    "WindowResult",
    "WalkForwardResult",
    "WalkForwardEvaluator",
]
```

---

### 6.7 CPCV and PBO

**File: `src/cross_validation/cpcv.py`**

```python
"""
Combinatorial Purged Cross-Validation (CPCV).

Tests C(n,k) combinations of folds to estimate probability of
backtest overfitting. More robust than standard k-fold for
hyperparameter selection.

Reference: Bailey et al. (2014) "The Probability of Backtest Overfitting"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.cross_validation.purged_kfold import PurgedKFoldConfig

logger = logging.getLogger(__name__)


@dataclass
class CPCVConfig:
    """Configuration for CPCV."""
    n_groups: int = 6  # Number of time groups
    n_test_groups: int = 2  # Groups held out for test
    purge_bars: int = 60
    embargo_bars: int = 1440
    max_combinations: int = 15  # Limit for computation


@dataclass
class CPCVResult:
    """Results from CPCV evaluation."""
    mean_score: float
    std_score: float
    scores: List[float]
    combinations_tested: int
    pbo_estimate: float  # Probability of Backtest Overfitting

    @property
    def is_overfit(self) -> bool:
        """Returns True if PBO suggests overfitting."""
        return self.pbo_estimate > 0.5


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.

    Instead of a single k-fold path, tests multiple combinations
    of time groups to assess robustness of model selection.

    Example:
        >>> cpcv = CombinatorialPurgedCV(CPCVConfig(n_groups=6, n_test_groups=2))
        >>> for train_idx, test_idx in cpcv.split(X):
        ...     # Train and evaluate
        >>> pbo = cpcv.compute_pbo(scores, scores_is)
    """

    def __init__(self, config: CPCVConfig) -> None:
        self.config = config
        self._combinations = list(combinations(
            range(config.n_groups),
            config.n_test_groups
        ))
        # Limit combinations if too many
        if len(self._combinations) > config.max_combinations:
            np.random.shuffle(self._combinations)
            self._combinations = self._combinations[:config.max_combinations]

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits for all combinations.

        Args:
            X: Feature DataFrame
            y: Labels (unused, for API compatibility)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        group_size = n_samples // self.config.n_groups

        # Create group boundaries
        groups = []
        for i in range(self.config.n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < self.config.n_groups - 1 else n_samples
            groups.append(np.arange(start, end))

        for test_group_indices in self._combinations:
            # Test indices
            test_idx = np.concatenate([groups[i] for i in test_group_indices])

            # Training indices (with purge/embargo)
            train_groups = [i for i in range(self.config.n_groups) if i not in test_group_indices]

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False

            # Apply purge before each test group
            for test_group in test_group_indices:
                test_start = groups[test_group][0]
                purge_start = max(0, test_start - self.config.purge_bars)
                train_mask[purge_start:test_start] = False

            # Apply embargo after each test group
            for test_group in test_group_indices:
                test_end = groups[test_group][-1] + 1
                embargo_end = min(n_samples, test_end + self.config.embargo_bars)
                train_mask[test_end:embargo_end] = False

            train_idx = np.where(train_mask)[0]

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return number of combinations to test."""
        return len(self._combinations)


def compute_pbo(
    oos_scores: np.ndarray,
    is_scores: np.ndarray,
) -> float:
    """
    Compute Probability of Backtest Overfitting.

    PBO estimates the probability that a strategy selected based on
    in-sample performance will underperform out-of-sample.

    Args:
        oos_scores: Out-of-sample scores for each combination
        is_scores: In-sample scores for each combination

    Returns:
        PBO estimate (0 = no overfitting, 1 = complete overfitting)
    """
    n = len(oos_scores)
    if n == 0:
        return 0.5

    # Rank strategies by IS performance
    is_ranks = np.argsort(np.argsort(is_scores))[::-1]

    # Check if best IS performer has negative OOS
    best_is_idx = np.argmax(is_scores)

    # Compute probability that top IS strategy underperforms median OOS
    median_oos = np.median(oos_scores)
    underperform_count = np.sum(
        (is_ranks == 0) & (oos_scores < median_oos)
    )

    # More sophisticated: compare ranks
    oos_ranks = np.argsort(np.argsort(oos_scores))[::-1]

    # PBO = probability that IS-best is OOS-worst-half
    pbo = np.mean(oos_ranks[np.argmax(is_scores)] > n // 2)

    return float(pbo)


__all__ = ["CPCVConfig", "CPCVResult", "CombinatorialPurgedCV", "compute_pbo"]
```

---

### 6.8 Sequence Model CV Fix

**Modification to `src/cross_validation/cv_runner.py`:**

Replace `_run_single_cv` method starting at line 317:

```python
def _run_single_cv(
    self,
    container: "TimeSeriesDataContainer",
    model_name: str,
    horizon: int,
) -> CVResult:
    """Run CV for single model/horizon combination."""
    start_time = time.time()

    # Get model info to determine data requirements
    try:
        model_info = ModelRegistry.get_model_info(model_name)
        model_family = model_info.get("family", "boosting")
        requires_sequences = model_info.get("requires_sequences", False)
    except ValueError:
        model_family = "boosting"
        requires_sequences = False

    # Get appropriate data format
    if requires_sequences:
        # For sequence models, get raw DataFrame and build sequences per fold
        X_df, y_series, weights = container.get_sklearn_arrays("train", return_df=True)
        seq_len = self.sequence_length or 60
        logger.info(f"Using sequence data with seq_len={seq_len} for {model_name}")

        # Will build sequences within each fold
        use_sequence_cv = True
    else:
        # Tabular models use standard arrays
        X_df, y_series, weights = container.get_sklearn_arrays("train", return_df=True)
        use_sequence_cv = False

    # Adapt CV for model family
    model_cv = ModelAwareCV(model_family, self.cv)
    cv_splits = list(model_cv.get_cv_splits(X_df, y_series))

    # Feature selection (if enabled) - only for tabular models
    selected_features = list(X_df.columns)
    if self.select_features and not use_sequence_cv:
        selector = WalkForwardFeatureSelector(
            n_features_to_select=self.n_features_to_select
        )
        selection_result = selector.select_features_walkforward(X_df, y_series, cv_splits)
        selected_features = selection_result.stable_features
        if selected_features:
            X_df = X_df[selected_features]
        logger.debug(f"  Selected {len(selected_features)} stable features")

    # Generate OOF predictions using appropriate method
    if use_sequence_cv:
        oof_pred = self._generate_sequence_oof(
            X_df, y_series, weights, cv_splits, model_name, seq_len, horizon
        )
    else:
        # Standard OOF generation
        oof_generator = OOFGenerator(self.cv)
        oof_predictions = oof_generator.generate_oof_predictions(
            X=X_df,
            y=y_series,
            model_configs={model_name: {}},
            sample_weights=weights,
        )
        oof_pred = oof_predictions[model_name]

    # ... rest of method remains the same


def _generate_sequence_oof(
    self,
    X_df: pd.DataFrame,
    y_series: pd.Series,
    weights: pd.Series,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    seq_len: int,
    horizon: int,
) -> OOFPrediction:
    """Generate OOF predictions for sequence models."""
    from src.phase1.stages.datasets.sequences import create_sequences

    n_samples = len(X_df)
    n_classes = 3

    oof_probs = np.full((n_samples, n_classes), np.nan)
    oof_preds = np.full(n_samples, np.nan)
    fold_info = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        logger.debug(f"  Sequence fold {fold_idx + 1}")

        # Build sequences for this fold (respecting fold boundaries)
        X_train_seq, y_train_seq = create_sequences(
            X_df.iloc[train_idx].values,
            y_series.iloc[train_idx].values,
            seq_len=seq_len,
        )
        X_val_seq, y_val_seq = create_sequences(
            X_df.iloc[val_idx].values,
            y_series.iloc[val_idx].values,
            seq_len=seq_len,
        )

        # Create and train model
        model = ModelRegistry.create(model_name)
        metrics = model.fit(
            X_train=X_train_seq,
            y_train=y_train_seq,
            X_val=X_val_seq,
            y_val=y_val_seq,
        )

        # Predict
        predictions = model.predict(X_val_seq)

        # Map predictions back to original indices
        # Sequences start at index seq_len-1
        valid_val_idx = val_idx[seq_len - 1:]

        for i, orig_idx in enumerate(valid_val_idx):
            if i < len(predictions.class_predictions):
                oof_probs[orig_idx] = predictions.class_probabilities[i]
                oof_preds[orig_idx] = predictions.class_predictions[i]

        fold_info.append({
            "fold": fold_idx,
            "train_size": len(X_train_seq),
            "val_size": len(X_val_seq),
            "val_accuracy": metrics.val_accuracy,
            "val_f1": metrics.val_f1,
        })

    # Build OOF prediction result
    oof_df = pd.DataFrame({
        "datetime": X_df.index if isinstance(X_df.index, pd.DatetimeIndex) else range(len(X_df)),
        f"{model_name}_prob_short": oof_probs[:, 0],
        f"{model_name}_prob_neutral": oof_probs[:, 1],
        f"{model_name}_prob_long": oof_probs[:, 2],
        f"{model_name}_pred": oof_preds,
        f"{model_name}_confidence": np.nanmax(oof_probs, axis=1),
    })

    coverage = float((~np.isnan(oof_preds)).mean())

    return OOFPrediction(
        model_name=model_name,
        predictions=oof_df,
        fold_info=fold_info,
        coverage=coverage,
    )
```

---

### 6.9 Inference Pipeline

**File: `src/inference/pipeline.py`**

```python
"""
Production Inference Pipeline.

Provides a complete, serializable pipeline for production deployment
including feature scaling, model inference, probability calibration,
and ensemble aggregation.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base import PredictionOutput
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    model_names: List[str]
    horizons: List[int]
    feature_columns: List[str]
    use_calibration: bool = True
    use_ensemble: bool = True
    sequence_length: int = 60  # For sequential models
    batch_size: int = 1000


@dataclass
class InferenceResult:
    """Result from inference."""
    predictions: np.ndarray  # -1, 0, 1
    probabilities: np.ndarray  # (n_samples, 3)
    confidence: np.ndarray
    inference_time_ms: float
    model_outputs: Dict[str, np.ndarray] = field(default_factory=dict)


class InferencePipeline:
    """
    Production-ready inference pipeline.

    Bundles:
    - Feature scaler
    - Feature column specification
    - Base models
    - Calibrators
    - Ensemble meta-learner (optional)

    Example:
        >>> pipeline = InferencePipeline.load("artifacts/pipeline_v1")
        >>> result = pipeline.predict(df, horizon=20)
        >>> positions = result.predictions  # -1, 0, 1
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._scaler = None
        self._models: Dict[str, Dict[int, Any]] = {}  # model_name -> horizon -> model
        self._calibrators: Dict[str, Dict[int, Any]] = {}
        self._meta_learner: Dict[int, Any] = {}
        self._is_loaded = False

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        return_model_outputs: bool = False,
    ) -> InferenceResult:
        """
        Generate predictions for input DataFrame.

        Args:
            df: DataFrame with OHLCV and feature columns
            horizon: Prediction horizon
            return_model_outputs: Include individual model predictions

        Returns:
            InferenceResult with predictions and probabilities
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Extract and scale features
        X = df[self.config.feature_columns].values
        X_scaled = self._scaler.transform(X)

        # Get base model predictions
        model_outputs = {}
        for model_name in self.config.model_names:
            model = self._models[model_name][horizon]

            # Handle sequence models
            if hasattr(model, 'requires_sequences') and model.requires_sequences:
                X_input = self._create_sequences(X_scaled)
            else:
                X_input = X_scaled

            pred_output = model.predict(X_input)
            probs = pred_output.class_probabilities

            # Apply calibration if available
            if self.config.use_calibration and model_name in self._calibrators:
                calibrator = self._calibrators[model_name].get(horizon)
                if calibrator is not None:
                    probs = calibrator.calibrate(probs)

            model_outputs[model_name] = probs

        # Ensemble or single model
        if self.config.use_ensemble and horizon in self._meta_learner:
            ensemble_probs = self._apply_ensemble(model_outputs, horizon)
        else:
            # Average base model probabilities
            ensemble_probs = np.mean(list(model_outputs.values()), axis=0)

        # Final predictions
        predictions = np.argmax(ensemble_probs, axis=1) - 1  # Map to -1, 0, 1
        confidence = np.max(ensemble_probs, axis=1)

        inference_time = (time.perf_counter() - start_time) * 1000

        result = InferenceResult(
            predictions=predictions,
            probabilities=ensemble_probs,
            confidence=confidence,
            inference_time_ms=inference_time,
        )

        if return_model_outputs:
            result.model_outputs = model_outputs

        return result

    def predict_single(
        self,
        df: pd.DataFrame,
        idx: int,
        horizon: int = 20,
    ) -> np.ndarray:
        """Predict for single sample (for lookahead verification)."""
        X = df[self.config.feature_columns].iloc[idx:idx+1].values
        X_scaled = self._scaler.transform(X)

        probs_list = []
        for model_name in self.config.model_names:
            model = self._models[model_name][horizon]
            pred = model.predict(X_scaled)
            probs_list.append(pred.class_probabilities)

        return np.mean(probs_list, axis=0)

    def _apply_ensemble(
        self,
        model_outputs: Dict[str, np.ndarray],
        horizon: int,
    ) -> np.ndarray:
        """Apply meta-learner to base predictions."""
        # Stack probabilities as meta-features
        meta_features = np.hstack([
            model_outputs[m] for m in self.config.model_names
        ])

        meta_learner = self._meta_learner[horizon]
        return meta_learner.predict_proba(meta_features)

    def _create_sequences(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Create sequences for sequential models."""
        seq_len = self.config.sequence_length
        n_samples = len(X) - seq_len + 1
        n_features = X.shape[1]

        sequences = np.zeros((n_samples, seq_len, n_features))
        for i in range(n_samples):
            sequences[i] = X[i:i + seq_len]

        return sequences

    def save(self, path: Path) -> None:
        """
        Serialize complete pipeline to disk.

        Creates directory structure:
            path/
                config.json
                scaler.pkl
                models/
                    xgboost_h20.pkl
                    ...
                calibrators/
                    xgboost_h20.pkl
                    ...
                ensemble/
                    meta_h20.pkl
                manifest.json
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_dict = {
            "model_names": self.config.model_names,
            "horizons": self.config.horizons,
            "feature_columns": self.config.feature_columns,
            "use_calibration": self.config.use_calibration,
            "use_ensemble": self.config.use_ensemble,
            "sequence_length": self.config.sequence_length,
            "batch_size": self.config.batch_size,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save scaler
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f)

        # Save models
        models_dir = path / "models"
        models_dir.mkdir(exist_ok=True)
        for model_name, horizons in self._models.items():
            for horizon, model in horizons.items():
                model_path = models_dir / f"{model_name}_h{horizon}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

        # Save calibrators
        if self._calibrators:
            cal_dir = path / "calibrators"
            cal_dir.mkdir(exist_ok=True)
            for model_name, horizons in self._calibrators.items():
                for horizon, calibrator in horizons.items():
                    cal_path = cal_dir / f"{model_name}_h{horizon}.pkl"
                    with open(cal_path, "wb") as f:
                        pickle.dump(calibrator, f)

        # Save meta-learners
        if self._meta_learner:
            ensemble_dir = path / "ensemble"
            ensemble_dir.mkdir(exist_ok=True)
            for horizon, meta in self._meta_learner.items():
                meta_path = ensemble_dir / f"meta_h{horizon}.pkl"
                with open(meta_path, "wb") as f:
                    pickle.dump(meta, f)

        # Create manifest
        manifest = {
            "version": "1.0.0",
            "created_at": pd.Timestamp.now().isoformat(),
            "model_names": self.config.model_names,
            "horizons": self.config.horizons,
            "n_features": len(self.config.feature_columns),
        }
        with open(path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved inference pipeline to {path}")

    @classmethod
    def load(cls, path: Path) -> "InferencePipeline":
        """Load serialized pipeline from disk."""
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        config = InferenceConfig(**config_dict)

        pipeline = cls(config)

        # Load scaler
        with open(path / "scaler.pkl", "rb") as f:
            pipeline._scaler = pickle.load(f)

        # Load models
        models_dir = path / "models"
        for model_name in config.model_names:
            pipeline._models[model_name] = {}
            for horizon in config.horizons:
                model_path = models_dir / f"{model_name}_h{horizon}.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        pipeline._models[model_name][horizon] = pickle.load(f)

        # Load calibrators
        cal_dir = path / "calibrators"
        if cal_dir.exists():
            for model_name in config.model_names:
                pipeline._calibrators[model_name] = {}
                for horizon in config.horizons:
                    cal_path = cal_dir / f"{model_name}_h{horizon}.pkl"
                    if cal_path.exists():
                        with open(cal_path, "rb") as f:
                            pipeline._calibrators[model_name][horizon] = pickle.load(f)

        # Load meta-learners
        ensemble_dir = path / "ensemble"
        if ensemble_dir.exists():
            for horizon in config.horizons:
                meta_path = ensemble_dir / f"meta_h{horizon}.pkl"
                if meta_path.exists():
                    with open(meta_path, "rb") as f:
                        pipeline._meta_learner[horizon] = pickle.load(f)

        pipeline._is_loaded = True
        logger.info(f"Loaded inference pipeline from {path}")

        return pipeline


__all__ = ["InferenceConfig", "InferenceResult", "InferencePipeline"]
```

---

### 6.10-6.12 Remaining Specs

Due to document length, remaining detailed specs for:
- 6.10 Regime-Adaptive Models
- 6.11 Conformal Prediction
- 6.12 Alternative Bar Types

Are available in the expanded implementation details section. Core patterns follow the same modular approach demonstrated above.

---

## Migration Guide

### Step 1: Create New Directories

```bash
mkdir -p src/models/calibration
mkdir -p src/monitoring
mkdir -p src/inference
mkdir -p src/validation
mkdir -p src/regime
```

### Step 2: Install New Dependencies

Add to `requirements.txt`:

```
river>=0.21.0        # For ADWIN drift detection
hmmlearn>=0.3.0      # For regime detection
```

### Step 3: Update Existing Files

| File | Changes |
|------|---------|
| `src/models/trainer.py` | Add calibration after training (line 234) |
| `src/cross_validation/oof_generator.py` | Add fold-aware scaling (line 193), label_end_times |
| `src/cross_validation/cv_runner.py` | Add sequence model handling, walk-forward option |
| `src/phase1/stages/labeling/triple_barrier.py` | Add label_end_time computation |
| `src/phase1/stages/datasets/container.py` | Expose label_end_times |

### Step 4: Run Tests

```bash
# Existing tests should pass
pytest tests/

# Add new test modules
pytest tests/models/test_calibration.py
pytest tests/cross_validation/test_fold_scaling.py
pytest tests/validation/test_lookahead.py
pytest tests/monitoring/test_drift.py
```

### Step 5: Verify CV Metrics Drop

After implementing fold-aware scaling, expect:
- 5-15% drop in CV F1 scores
- This is expected and correct (removing optimistic bias)

---

## Expected Outcomes

### Metrics Before/After

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| **CV F1 (XGBoost, H20)** | 0.52 | 0.45-0.48 | Expected drop from removing leakage |
| **Brier Score** | 0.25-0.30 | < 0.15 | Calibration improvement |
| **ECE** | 0.10-0.15 | < 0.05 | Calibration improvement |
| **Val-Test Gap** | 15-25% | < 10% | Better generalization |
| **PBO** | Unknown | < 0.5 | Quantified overfitting risk |
| **Drift Detection** | None | < 100 samples | Real-time monitoring |

### Production Readiness Checklist

After implementation:

- [ ] Probability calibration active (Brier < 0.15, ECE < 0.05)
- [ ] CV uses fold-aware scaling
- [ ] Label-aware purging enabled
- [ ] Lookahead audit passes (0 violations)
- [ ] Walk-forward evaluation shows stable performance
- [ ] PBO < 0.5 for selected model
- [ ] Inference pipeline serialized
- [ ] Drift monitoring active
- [ ] Test set evaluation complete

---

## Risk Mitigation

### Risk 1: CV Metrics Drop Significantly

**Symptom:** F1 drops > 20% after fold-aware scaling.

**Mitigation:**
1. Verify scaling implementation is correct
2. Check if features are highly sensitive to scaling
3. Consider reducing feature set
4. May indicate previous metrics were optimistic

### Risk 2: Calibration Doesn't Improve

**Symptom:** Brier/ECE unchanged or worse after calibration.

**Mitigation:**
1. Ensure sufficient validation samples (> 500)
2. Try different method (isotonic vs sigmoid)
3. Check for class imbalance
4. Verify probabilities are in valid range [0, 1]

### Risk 3: Sequence Model CV Fails

**Symptom:** LSTM/GRU CV produces NaN or crashes.

**Mitigation:**
1. Ensure sequences don't cross fold boundaries
2. Check minimum samples per fold (need > seq_len)
3. Verify padding/truncation handling

### Risk 4: Drift Detection False Positives

**Symptom:** Too many alerts during normal operation.

**Mitigation:**
1. Increase ADWIN delta (less sensitive)
2. Raise PSI thresholds
3. Implement alert aggregation
4. Add cooldown period between alerts

### Risk 5: Inference Latency Too High

**Symptom:** Ensemble inference > 100ms.

**Mitigation:**
1. Use boosting-only ensemble (< 5ms)
2. Implement model caching
3. Use batch inference
4. Profile and optimize bottlenecks

---

## Appendix: Code Templates

### A.1 Test Template for New Modules

```python
"""Tests for src/models/calibration/calibrator.py"""
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
```

### A.2 CLI Script Template

```python
#!/usr/bin/env python
"""
Run walk-forward evaluation.

Usage:
    python scripts/run_walk_forward.py --model xgboost --horizon 20
"""
import argparse
import logging
from pathlib import Path

from src.cross_validation.walk_forward import WalkForwardEvaluator, WalkForwardConfig
from src.phase1.stages.datasets.container import TimeSeriesDataContainer


def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--horizon", type=int, default=20, help="Label horizon")
    parser.add_argument("--data-dir", default="data/splits/scaled", help="Data directory")
    parser.add_argument("--output", default="results/walk_forward", help="Output directory")
    parser.add_argument("--initial-train", type=float, default=0.5, help="Initial train fraction")
    parser.add_argument("--step-size", type=int, default=288, help="Step size in bars")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load data
    container = TimeSeriesDataContainer.from_parquet_dir(
        args.data_dir, horizon=args.horizon
    )
    X, y, weights = container.get_sklearn_arrays("train", return_df=True)

    # Configure and run
    config = WalkForwardConfig(
        initial_train_size=args.initial_train,
        step_size=args.step_size,
    )
    evaluator = WalkForwardEvaluator(config)
    result = evaluator.evaluate(args.model, X, y, args.horizon)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.to_dataframe().to_csv(
        output_dir / f"{args.model}_h{args.horizon}_walkforward.csv",
        index=False
    )

    print(f"\nWalk-Forward Results for {args.model} H{args.horizon}:")
    print(f"  Windows: {result.n_windows}")
    print(f"  Mean F1: {result.mean_f1:.3f} +/- {result.std_f1:.3f}")
    print(f"  Temporal Degradation: {result.temporal_degradation:.4f}")


if __name__ == "__main__":
    main()
```

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-28 | ML Engineering | Initial comprehensive plan |

**Next Review:** After Phase 1 completion (Week 2)

---

*End of Implementation Plan*
