# Inference Pipeline Specification

**Version:** 1.0.0
**Date:** 2025-12-28
**Priority:** P0 (Blocks Deployment)

---

## Overview

### Problem Statement

Phase 5 (Inference) is documented but `src/inference/` doesn't exist. No way to:
- Serialize complete pipeline for deployment
- Load and make predictions on new data
- Ensure production preprocessing matches training
- Monitor inference performance

**Impact:** Cannot deploy models to production.

### Solution

Complete inference pipeline that bundles:
- Feature scaler
- Feature column specification
- Trained models (base + ensemble)
- Calibrators
- Meta-learner

---

## Architecture

### Single Artifact Deployment

```
deployment/pipeline_v1/
├── config.json             # Pipeline configuration
├── scaler.pkl              # RobustScaler (fit on train only)
├── models/
│   ├── xgboost_h20.pkl
│   ├── lightgbm_h20.pkl
│   └── lstm_h20.pkl
├── calibrators/
│   ├── xgboost_h20.pkl
│   └── lightgbm_h20.pkl
├── ensemble/
│   └── meta_h20.pkl        # Logistic meta-learner
└── manifest.json           # Version, timestamp, metadata
```

### Inference Flow

```
Raw OHLCV DataFrame
    ↓
[Extract Features]
    ↓
[Scale with scaler.pkl]
    ↓
[Base Model Predictions] → xgboost, lightgbm, lstm
    ↓
[Calibrate Probabilities]
    ↓
[Ensemble Meta-Learner]
    ↓
Final Predictions (-1, 0, 1) + Confidence
```

---

## Implementation Outline

### Core Classes

**`InferenceConfig`:**
```python
@dataclass
class InferenceConfig:
    model_names: List[str]           # ["xgboost", "lightgbm", "lstm"]
    horizons: List[int]              # [5, 10, 15, 20]
    feature_columns: List[str]       # Feature list
    use_calibration: bool = True
    use_ensemble: bool = True
    sequence_length: int = 60
    batch_size: int = 1000
```

**`InferenceResult`:**
```python
@dataclass
class InferenceResult:
    predictions: np.ndarray          # -1, 0, 1
    probabilities: np.ndarray        # (n_samples, 3)
    confidence: np.ndarray           # Max probability
    inference_time_ms: float
    model_outputs: Dict[str, np.ndarray]  # Optional individual model outputs
```

**`InferencePipeline`:**
- `predict(df, horizon)` → InferenceResult
- `predict_single(df, idx, horizon)` → np.ndarray (for lookahead testing)
- `save(path)` → None
- `load(path)` → InferencePipeline

---

## Usage Examples

### Create and Save Pipeline

```python
from src.inference.pipeline import InferencePipeline, InferenceConfig

# Configuration
config = InferenceConfig(
    model_names=["xgboost", "lightgbm", "lstm"],
    horizons=[20],
    feature_columns=feature_cols,
    use_calibration=True,
    use_ensemble=True,
)

# Create pipeline
pipeline = InferencePipeline(config)

# Load artifacts
pipeline._scaler = load_scaler("data/splits/scaled/feature_scaler.pkl")
pipeline._models = load_trained_models("experiments/runs/best_run")
pipeline._calibrators = load_calibrators("experiments/runs/best_run")
pipeline._meta_learner = load_meta_learner("data/stacking/h20/meta.pkl")

# Save as single artifact
pipeline.save("deployment/production_pipeline_v1")
```

### Load and Predict

```python
# Load pipeline
pipeline = InferencePipeline.load("deployment/production_pipeline_v1")

# Predict on new data
df_new = pd.read_parquet("data/live/latest.parquet")
result = pipeline.predict(df_new, horizon=20)

print(f"Predictions: {result.predictions}")          # [-1, 0, 1, 1, 0, ...]
print(f"Confidence: {result.confidence}")            # [0.65, 0.72, 0.58, ...]
print(f"Latency: {result.inference_time_ms:.2f}ms")  # 45.23ms
```

### Batch Inference

```python
# Process large dataset in batches
from src.inference.batch import BatchInference

batch_inf = BatchInference(pipeline, batch_size=1000)
results = batch_inf.process_file(
    "data/backtest/2024_q4.parquet",
    horizon=20,
    output_path="results/backtest_predictions.parquet"
)

print(f"Processed {results.n_samples} samples in {results.total_time:.2f}s")
print(f"Avg latency: {results.avg_latency_ms:.2f}ms")
```

---

## Full Implementation

**File: `src/inference/pipeline.py`**

See IMPLEMENTATION_PLAN.md section 6.9 for complete 332-line implementation.

Key components:
- `InferencePipeline` class (~250 lines)
- `save()` method - Serializes all artifacts
- `load()` classmethod - Deserializes pipeline
- `predict()` method - Main inference entry point
- `_apply_ensemble()` - Meta-learner application
- `_create_sequences()` - Sequence creation for LSTM/GRU/TCN

---

## Integration with Drift Monitoring

```python
from src.inference.pipeline import InferencePipeline
from src.monitoring.drift_detector import OnlineDriftMonitor, DriftConfig

# Load pipeline
pipeline = InferencePipeline.load("deployment/pipeline_v1")

# Setup drift monitoring
monitor = OnlineDriftMonitor(
    reference_data=reference_features,
    config=DriftConfig(),
    feature_names=pipeline.config.feature_columns,
)

# Production loop
for batch in live_data_stream:
    # Predict
    result = pipeline.predict(batch, horizon=20)
    
    # Monitor for drift
    alerts = monitor.check(
        features=batch[pipeline.config.feature_columns].values,
        predictions=result.predictions,
        actuals=batch['label'] if 'label' in batch else None,
    )
    
    # Handle alerts
    if any(a.severity == AlertSeverity.CRITICAL for a in alerts):
        trigger_retraining()
```

---

## Deployment Profiles

### Profile 1: Low-Latency Production

```python
config = InferenceConfig(
    model_names=["xgboost", "lightgbm", "catboost"],  # Boosting only
    horizons=[20],
    use_calibration=True,
    use_ensemble=True,  # Voting ensemble
)
# Expected latency: < 5ms
```

### Profile 2: Maximum Accuracy

```python
config = InferenceConfig(
    model_names=["xgboost", "lightgbm", "lstm", "gru"],
    horizons=[20],
    use_calibration=True,
    use_ensemble=True,  # Stacking meta-learner
)
# Expected latency: 50-100ms
```

### Profile 3: Multi-Horizon

```python
config = InferenceConfig(
    model_names=["xgboost", "lightgbm"],
    horizons=[5, 10, 15, 20],  # All horizons
    use_calibration=True,
    use_ensemble=False,  # Individual models
)
# Returns predictions for all horizons
```

---

## Testing

### Lookahead Verification

```python
from src.validation.lookahead_audit import LookaheadAuditor

def verify_no_lookahead(pipeline, df):
    """Verify pipeline doesn't use future data."""
    
    def predict_func(data):
        result = pipeline.predict(data, horizon=20)
        return pd.DataFrame({
            'prediction': result.predictions,
            'confidence': result.confidence,
        })
    
    auditor = LookaheadAuditor(predict_func)
    result = auditor.audit(df, n_samples=50)
    
    assert result.passed, f"Lookahead violations: {len(result.violations)}"
```

### Latency Benchmark

```python
import time

pipeline = InferencePipeline.load("deployment/pipeline_v1")
df_test = pd.read_parquet("data/splits/scaled/test_scaled.parquet")

# Warm-up
_ = pipeline.predict(df_test[:100], horizon=20)

# Benchmark
latencies = []
for i in range(100):
    batch = df_test[i*100:(i+1)*100]
    result = pipeline.predict(batch, horizon=20)
    latencies.append(result.inference_time_ms)

print(f"Median latency: {np.median(latencies):.2f}ms")
print(f"P95 latency: {np.percentile(latencies, 95):.2f}ms")
assert np.median(latencies) < 100, "Latency too high"
```

---

## Acceptance Criteria

- [ ] Pipeline saves and loads correctly
- [ ] Predictions match standalone model predictions
- [ ] Inference latency < 100ms for ensemble
- [ ] Lookahead verification passes
- [ ] Batch inference processes 10k samples/sec
- [ ] Drift monitoring integration works
- [ ] Multi-horizon predictions supported

---

## Cross-References

- [ROADMAP.md](../ROADMAP.md#31-inference-pipeline-phase-5) - Phase 3 overview
- [GAPS_ANALYSIS.md](../GAPS_ANALYSIS.md#gap-3-phase-5-inference-not-implemented) - Gap details
- [specs/drift_detection.md](drift_detection.md) - Monitoring integration

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-28 | ML Engineering | Initial inference pipeline spec from IMPLEMENTATION_PLAN.md |
