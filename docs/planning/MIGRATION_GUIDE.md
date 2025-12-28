# MIGRATION GUIDE: OHLCV ML Trading Pipeline

**Version:** 1.0.0
**Date:** 2025-12-28
**Status:** Ready for Implementation

---

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Migration](#step-by-step-migration)
4. [Breaking Changes](#breaking-changes)
5. [Risk Mitigation](#risk-mitigation)
6. [Testing Strategy](#testing-strategy)
7. [Code Templates](#code-templates)

---

## Migration Overview

This guide walks through implementing the 12 improvements identified in the gaps analysis. The migration is designed to be **incremental** - each phase can be completed independently without breaking existing functionality.

**Migration Phases:**
- **Phase 1 (Week 1-2):** Critical fixes (calibration, CV scaling, lookahead audit)
- **Phase 2 (Week 3-4):** Production safety (drift detection, label-aware purging, walk-forward)
- **Phase 3 (Week 5-8):** Performance upgrades (inference pipeline, regime models)

**Backward Compatibility:**
- Existing trained models remain usable
- Existing datasets remain valid
- Pipeline can run in "legacy mode" during transition

---

## Prerequisites

### 1. Install New Dependencies

Add to `requirements.txt`:

```txt
river>=0.21.0        # For ADWIN drift detection
hmmlearn>=0.3.0      # For regime detection (optional, Phase 3)
```

Install:
```bash
pip install -r requirements.txt
```

### 2. Create New Directories

```bash
mkdir -p src/models/calibration
mkdir -p src/monitoring
mkdir -p src/inference
mkdir -p src/validation
mkdir -p src/regime  # Optional for Phase 3
```

### 3. Backup Current State

```bash
# Backup trained models
cp -r experiments/runs backup/runs_$(date +%Y%m%d)

# Backup configuration
cp -r config backup/config_$(date +%Y%m%d)

# Backup data splits
cp -r data/splits backup/splits_$(date +%Y%m%d)
```

---

## Step-by-Step Migration

### Phase 1: Critical Foundation (Week 1-2)

#### Step 1.1: Implement Probability Calibration

**Create new files:**

1. `src/models/calibration/__init__.py`
2. `src/models/calibration/calibrator.py` (see [specs/probability_calibration.md](specs/probability_calibration.md))
3. `src/models/calibration/metrics.py`

**Modify existing files:**

**File:** `src/models/trainer.py`
**Location:** After line 234 (after evaluation metrics)

```python
# Add import at top
from src.models.calibration import ProbabilityCalibrator, CalibrationConfig

# Add after evaluation (line 234)
# Calibration
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

**Test:**
```bash
python scripts/train_model.py --model xgboost --horizon 20
# Check logs for "Calibration complete: Brier X -> Y, ECE A -> B"
# Verify calibrator.pkl exists in checkpoints/
```

---

#### Step 1.2: Implement Fold-Aware Scaling

**Create new files:**

1. `src/cross_validation/fold_scaling.py` (see [specs/cv_improvements.md](specs/cv_improvements.md#fold-aware-scaling))

**Modify existing files:**

**File:** `src/cross_validation/oof_generator.py`
**Location:** Lines 193-198

Replace:
```python
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
```

With:
```python
from src.cross_validation.fold_scaling import FoldAwareScaler

# Extract fold data
X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# Apply fold-aware scaling
scaler = FoldAwareScaler(method="robust")
X_train, X_val = scaler.fit_transform_fold(
    X_train_raw.values, X_val_raw.values
)
```

**Expected Impact:**
- CV metrics will **drop 5-15%** (this is correct - removing optimism)
- Smaller val-test gap

**Test:**
```bash
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5
# Compare F1 before and after (should drop)
```

---

#### Step 1.3: Implement Lookahead Audit

**Create new files:**

1. `src/validation/lookahead_audit.py` (see [specs/cv_validation_methods.md](specs/cv_validation_methods.md#lookahead-audit))
2. `tests/validation/test_lookahead.py`

**Run audit:**

```python
# scripts/audit_lookahead.py
from src.validation.lookahead_audit import audit_mtf_features
from src.phase1.stages.mtf.generator import MTFGenerator
import pandas as pd

# Load data
df = pd.read_parquet("data/raw/MES_1m.parquet")

# Create MTF generator
mtf_gen = MTFGenerator(...)

# Run audit
result = audit_mtf_features(df, mtf_gen)

if result.passed:
    print(f"✓ Lookahead audit PASSED: {result.n_features_tested} features, {result.n_samples_tested} samples")
else:
    print(f"✗ Lookahead audit FAILED: {len(result.violations)} violations")
    for violation in result.violations[:10]:
        print(f"  - {violation.feature_name} at index {violation.sample_index}")
```

**Test:**
```bash
python scripts/audit_lookahead.py
# Should report 0 violations
```

---

### Phase 2: Production Safety (Week 3-4)

#### Step 2.1: Implement Drift Detection

**Create new files:**

1. `src/monitoring/__init__.py`
2. `src/monitoring/drift_detector.py` (see [specs/drift_detection.md](specs/drift_detection.md))
3. `src/monitoring/performance_monitor.py`
4. `src/monitoring/alert_handlers.py`

**Integration example:**

```python
from src.monitoring.drift_detector import OnlineDriftMonitor, DriftConfig

# During training, save reference data
reference_data = X_train[:5000].values  # Representative sample
np.save("artifacts/reference_features.npy", reference_data)

# In production
monitor = OnlineDriftMonitor(
    reference_data=reference_data,
    config=DriftConfig(psi_warning_threshold=0.1, psi_critical_threshold=0.2),
    feature_names=feature_columns,
)

# For each batch
alerts = monitor.check(features=X_batch, predictions=y_pred, actuals=y_true)
for alert in alerts:
    if alert.severity == AlertSeverity.CRITICAL:
        send_slack_alert(alert.message)
```

**Test:**
```python
# Inject synthetic drift
X_drifted = X_batch * 1.5  # Obvious shift
alerts = monitor.check(X_drifted)
assert len(alerts) > 0, "Should detect drift"
```

---

#### Step 2.2: Implement Label-Aware Purging

**Modify existing files:**

**File:** `src/phase1/stages/labeling/triple_barrier.py`
**Location:** After line 168 (after computing bars_to_hit)

Add:
```python
def compute_label_end_times(
    df: pd.DataFrame,
    bars_to_hit: np.ndarray,
    bar_interval_minutes: int = 5,
) -> pd.Series:
    """Compute when each label's outcome is known."""
    datetime_col = df.index if isinstance(df.index, pd.DatetimeIndex) else df['datetime']
    label_end_times = datetime_col + pd.to_timedelta(
        bars_to_hit * bar_interval_minutes, unit='m'
    )
    return pd.Series(label_end_times, index=df.index, name='label_end_time')

# In TripleBarrierLabeler.compute_labels(), after line 305:
label_end_times = compute_label_end_times(df, bars_to_hit, bar_interval_minutes=5)
result.metadata['label_end_times'] = label_end_times.values
```

**File:** `src/cross_validation/oof_generator.py`
**Location:** Before CV split loop

Add:
```python
# Get label end times if available
label_end_times = kwargs.get('label_end_times', None)

for fold_idx, (train_idx, val_idx) in enumerate(
    self.cv.split(X, y, label_end_times=label_end_times)
):
    ...
```

**Test:**
```bash
# Re-run Phase 1 to generate label_end_times
./pipeline run --symbols MES

# Verify in output
import pandas as pd
df = pd.read_parquet("data/splits/datasets/core_full/h20/train.parquet")
assert 'label_end_time_h20' in df.columns, "Missing label_end_time column"
```

---

#### Step 2.3: Implement Walk-Forward Evaluation

**Create new files:**

1. `src/cross_validation/walk_forward.py` (see [specs/cv_validation_methods.md](specs/cv_validation_methods.md#walk-forward-evaluation))
2. `scripts/run_walk_forward.py`

**Usage:**

```bash
python scripts/run_walk_forward.py --model xgboost --horizon 20 --step-size 288
```

**Test:**
Verify output shows temporal degradation metrics and per-window F1 scores.

---

#### Step 2.4: Implement CPCV

**Create new files:**

1. `src/cross_validation/cpcv.py` (see [specs/cv_cpcv.md](specs/cv_cpcv.md))
2. `scripts/run_cpcv.py`

**Usage:**

```bash
python scripts/run_cpcv.py --model xgboost --horizon 20 --n-groups 6 --n-test-groups 2
```

**Test:**
Verify PBO estimate is computed and warnings appear if PBO > 0.5.

---

#### Step 2.5: Fix Sequence Model CV

**Modify existing files:**

**File:** `src/cross_validation/cv_runner.py`
**Location:** Line 317 (in `_run_single_cv` method)

Replace data loading section with:
```python
# Get model info to determine data requirements
model_info = ModelRegistry.get_model_info(model_name)
requires_sequences = model_info.get("requires_sequences", False)

if requires_sequences:
    # For sequence models, get raw DataFrame and build sequences per fold
    X_df, y_series, weights = container.get_sklearn_arrays("train", return_df=True)
    use_sequence_cv = True
else:
    # Tabular models use standard arrays
    X_df, y_series, weights = container.get_sklearn_arrays("train", return_df=True)
    use_sequence_cv = False
```

Add method to handle sequence OOF (see [specs/cv_improvements.md](specs/cv_improvements.md#sequence-model-cv-fix)).

**Test:**
```bash
python scripts/run_cv.py --models lstm,gru --horizons 20 --n-splits 5
# Should complete without errors and show proper F1 scores
```

---

### Phase 3: Performance Upgrades (Week 5-8)

#### Step 3.1: Implement Inference Pipeline

**Create new files:**

1. `src/inference/__init__.py`
2. `src/inference/pipeline.py` (see [specs/inference_pipeline.md](specs/inference_pipeline.md))
3. `src/inference/serializer.py`
4. `scripts/serialize_pipeline.py`

**Usage:**

```python
from src.inference.pipeline import InferencePipeline, InferenceConfig

# Create and save
config = InferenceConfig(
    model_names=["xgboost", "lightgbm", "lstm"],
    horizons=[5, 10, 15, 20],
    feature_columns=feature_cols,
)
pipeline = InferencePipeline(config)
# ... load models, calibrators, etc. ...
pipeline.save("artifacts/production_pipeline_v1")

# Load and use
pipeline = InferencePipeline.load("artifacts/production_pipeline_v1")
result = pipeline.predict(df, horizon=20)
print(f"Predictions: {result.predictions}")
print(f"Confidence: {result.confidence}")
print(f"Inference time: {result.inference_time_ms:.2f}ms")
```

**Test:**
```bash
python scripts/serialize_pipeline.py --output deployment/pipeline_v1
python scripts/test_inference.py --pipeline deployment/pipeline_v1
```

---

## Breaking Changes

### 1. CV Metrics Will Drop

**Change:** Fold-aware scaling removes leakage.

**Impact:** F1 scores may drop 5-15%.

**Migration:**
- This is **expected and correct**
- Update performance targets in documentation
- Re-run hyperparameter tuning with new CV setup

### 2. PredictionOutput Structure

**Change:** Adding `calibrator` field to `PredictionOutput`.

**Impact:** Code expecting old structure may break.

**Migration:**
```python
# Old
probs = prediction_output.class_probabilities

# New (same, backward compatible)
probs = prediction_output.class_probabilities
```

### 3. Label End Times Column

**Change:** New `label_end_time_h{horizon}` column in datasets.

**Impact:** Existing parquet files don't have this column.

**Migration:**
- Re-run Phase 1 pipeline to regenerate datasets
- Or add column manually (not recommended)

### 4. Inference API

**Change:** New `InferencePipeline` class replaces manual model loading.

**Impact:** Production code using old approach needs update.

**Migration:**
```python
# Old
scaler = pickle.load("scaler.pkl")
model = pickle.load("model.pkl")
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)

# New
pipeline = InferencePipeline.load("pipeline_v1")
result = pipeline.predict(df, horizon=20)
predictions = result.predictions
```

---

## Risk Mitigation

### Risk 1: CV Metrics Drop Significantly

**Symptom:** F1 drops > 20% after fold-aware scaling.

**Diagnosis:**
1. Verify scaling implementation is correct
2. Check if features are highly sensitive to scaling
3. Compare train vs. val distributions

**Mitigation:**
1. Features may be too noisy - consider reducing feature set
2. May indicate previous metrics were severely optimistic
3. Increase training data or reduce model complexity
4. Use more robust features (e.g., rank-based instead of raw values)

**Example:**
```python
# Check feature sensitivity to scaling
from src.cross_validation.fold_scaling import FoldAwareScaler

scaler1 = FoldAwareScaler()
scaler2 = FoldAwareScaler()

# Different folds
X1_scaled, _ = scaler1.fit_transform_fold(X_train[:1000], X_val)
X2_scaled, _ = scaler2.fit_transform_fold(X_train[1000:2000], X_val)

# Compare scaling parameters
print("Fold 1 center:", scaler1._scaler.center_)
print("Fold 2 center:", scaler2._scaler.center_)
# Large differences indicate unstable features
```

---

### Risk 2: Calibration Doesn't Improve

**Symptom:** Brier/ECE unchanged or worse after calibration.

**Diagnosis:**
1. Check validation sample size (need > 500 samples)
2. Check class balance
3. Verify probabilities are in [0, 1] range

**Mitigation:**
1. Try different calibration method (isotonic vs sigmoid)
2. Increase validation data size
3. Check for outliers in probability outputs

**Example:**
```python
# Debug calibration
calibrator = ProbabilityCalibrator(CalibrationConfig(method="isotonic"))
metrics = calibrator.fit(y_val, probs_val)

print(f"Samples: {len(y_val)}")
print(f"Class distribution: {np.bincount(y_val)}")
print(f"Prob ranges: {probs_val.min(axis=0)} to {probs_val.max(axis=0)}")
print(f"Brier: {metrics.brier_before:.4f} -> {metrics.brier_after:.4f}")

# If no improvement, try sigmoid
calibrator2 = ProbabilityCalibrator(CalibrationConfig(method="sigmoid"))
metrics2 = calibrator2.fit(y_val, probs_val)
print(f"Sigmoid Brier: {metrics2.brier_before:.4f} -> {metrics2.brier_after:.4f}")
```

---

### Risk 3: Sequence Model CV Fails

**Symptom:** LSTM/GRU CV produces NaN or crashes.

**Diagnosis:**
1. Check sequence lengths cross fold boundaries
2. Verify minimum samples per fold
3. Check padding/truncation

**Mitigation:**
1. Ensure sequences don't cross fold boundaries:
   ```python
   # Each fold must have at least seq_len samples
   min_fold_size = max(train_idx.min(), seq_len)
   ```
2. Adjust CV split sizes if needed
3. Handle edge cases in sequence creation

---

### Risk 4: Drift Detection False Positives

**Symptom:** Too many alerts during normal operation.

**Diagnosis:**
1. Check alert frequency and severity
2. Review feature distributions
3. Check if reference data is representative

**Mitigation:**
1. Increase ADWIN delta (less sensitive):
   ```python
   config = DriftConfig(adwin_delta=0.005)  # Default: 0.002
   ```
2. Raise PSI thresholds:
   ```python
   config = DriftConfig(
       psi_warning_threshold=0.15,  # Default: 0.1
       psi_critical_threshold=0.25,  # Default: 0.2
   )
   ```
3. Implement alert aggregation (e.g., alert only if drift persists for N batches)
4. Add cooldown period between alerts

---

### Risk 5: Inference Latency Too High

**Symptom:** Ensemble inference > 100ms.

**Diagnosis:**
1. Profile inference path:
   ```python
   import cProfile
   cProfile.run('pipeline.predict(df, horizon=20)')
   ```
2. Check model count and complexity
3. Check sequence generation overhead

**Mitigation:**
1. Use boosting-only ensemble (< 5ms):
   ```python
   config = InferenceConfig(
       model_names=["xgboost", "lightgbm", "catboost"],  # No neural models
       use_ensemble=True,
   )
   ```
2. Implement model caching
3. Use batch inference for offline scoring
4. Optimize bottlenecks (usually scaling or sequence creation)

---

## Testing Strategy

### Unit Tests

**Test calibration:**
```bash
pytest tests/models/test_calibration.py -v
```

**Test fold scaling:**
```bash
pytest tests/cross_validation/test_fold_scaling.py -v
```

**Test lookahead audit:**
```bash
pytest tests/validation/test_lookahead.py -v
```

**Test drift detection:**
```bash
pytest tests/monitoring/test_drift.py -v
```

### Integration Tests

**Full CV pipeline:**
```bash
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 3
# Verify:
# - Fold-aware scaling is used
# - Calibration metrics are logged
# - No errors
```

**Walk-forward evaluation:**
```bash
python scripts/run_walk_forward.py --model xgboost --horizon 20
# Verify:
# - Per-window metrics reported
# - Temporal degradation computed
```

**Inference pipeline:**
```bash
python scripts/serialize_pipeline.py --models xgboost,lightgbm --output /tmp/test_pipeline
python scripts/test_inference.py --pipeline /tmp/test_pipeline --data data/splits/scaled/test_scaled.parquet
# Verify:
# - Predictions match standalone model
# - Latency < 100ms
# - No errors
```

### Regression Tests

**Compare old vs. new:**
```bash
# Old CV (before migration)
python scripts/run_cv.py --models xgboost --horizons 20 --legacy-mode

# New CV (after migration)
python scripts/run_cv.py --models xgboost --horizons 20

# Compare metrics (expect 5-15% drop, smaller val-test gap)
```

---

## Code Templates

### Template 1: Test for New Modules

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

---

### Template 2: CLI Script

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
| 1.0.0 | 2025-12-28 | ML Engineering | Initial migration guide extracted from IMPLEMENTATION_PLAN.md |

**Next Review:** After Phase 1 completion (Week 2)
