# PHASE 2: ADAPTER INTEGRATION - Implementation Plan

**Status:** ✅ COMPLETE (95%)
**Last Updated:** 2026-01-18
**Dependencies:** PHASE_0 (Foundation), PHASE_1 (Features)

---

## Executive Summary

PHASE_2 ensures ALL data preparation flows through the adapter system. The key principle: **One model → One adapter → One data format**. No bypass paths allowed.

---

## Current State Analysis

### Package Structure

```
src/adapters/
├── __init__.py              ✅ Complete - All exports defined
├── base.py                  ✅ Complete - BaseAdapter + AdapterResult
├── registry.py              ✅ Complete - @register decorator + get_adapter()
├── tabular.py               ✅ Complete - 2D (n_samples, n_features)
├── sequence.py              ✅ Complete - 3D (n_samples, seq_len, n_features)
├── multi_stream.py          ✅ Complete - 4D (n, n_tfs, seq_len, n_features)
├── factory.py               ✅ Complete - AdapterFactory with PipelineConfig
├── scaling.py               ✅ Complete - AdapterScaler for 2D/3D/4D
├── preparation.py           ✅ Complete - UnifiedDataPreparation
└── alignment.py             ✅ Complete - OOFAligner for heterogeneous
```

---

## Implemented Components

### 1. Adapter Registry (`registry.py`)

```python
# Key exports:
AdapterRegistry    # Singleton with @register decorator
get_adapter()      # Get adapter by model_name or adapter_id

# Registered adapters:
- "tabular": TabularAdapter    → 2D output
- "sequence": SequenceAdapter  → 3D output
- "multi_stream": MultiStreamAdapter → 4D output
```

### 2. TabularAdapter (`tabular.py`)

```python
# For: boosting (xgb, lgbm, catboost), classical (rf, logistic, svm), meta-learners
# Output shape: (n_samples, n_features)

adapter = TabularAdapter(feature_columns, label_column="label")
result = adapter.transform(df)
# result.data.shape = (10000, 100)  # 2D
```

### 3. SequenceAdapter (`sequence.py`)

```python
# For: lstm, gru, tcn, transformer, tft, nbeats, inceptiontime, resnet1d
# Output shape: (n_sequences, sequence_length, n_features)

adapter = SequenceAdapter(feature_columns, sequence_length=60)
result = adapter.transform(df)
# result.data.shape = (9941, 60, 80)  # 3D
# Loses first (seq_len - 1) = 59 samples
```

### 4. MultiStreamAdapter (`multi_stream.py`)

```python
# For: patchtst, itransformer
# Output shape: (n_sequences, n_timeframes, sequence_length, n_features)

adapter = MultiStreamAdapter(
    feature_columns=["open", "high", "low", "close", "volume"],
    sequence_length=60,
    timeframes=["1min", "5min", "15min"]
)
result = adapter.transform({"1min": df_1m, "5min": df_5m, "15min": df_15m})
# result.data.shape = (9941, 3, 60, 5)  # 4D
```

### 5. AdapterFactory (`factory.py`)

```python
# Key exports:
AdapterFactory         # Config-driven adapter creation
create_adapter_factory # Convenience constructor

# Usage:
factory = AdapterFactory(config)
result = factory.prepare_data("xgboost", df)

# Heterogeneous:
results = factory.prepare_heterogeneous(
    models=["xgboost", "lstm", "patchtst"],
    df=df,
    additional_dfs={"5min": df_5min}
)
```

### 6. AdapterScaler (`scaling.py`)

```python
# Key exports:
AdapterScaler    # Fit-on-train scaler for 2D/3D/4D
ScalerConfig     # Configuration dataclass
create_scaler    # Factory function

# Scaling methods: robust, standard, minmax, none
# Handles reshaping 3D/4D → 2D for sklearn scalers

scaler = AdapterScaler(ScalerConfig(method="robust", clip_value=5.0))
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_val_scaled = scaler.transform(X_val)          # Transform only
X_test_scaled = scaler.transform(X_test)        # Transform only
```

### 7. UnifiedDataPreparation (`preparation.py`)

```python
# Key exports:
UnifiedDataPreparation  # Complete pipeline (split + transform + scale)
PreparedData            # Result dataclass

# Usage:
prep = UnifiedDataPreparation(config)
data = prep.prepare(df, model_name="xgboost")

# data contains:
#   X_train, X_val, X_test (scaled)
#   y_train, y_val, y_test
#   train_indices, val_indices, test_indices
#   scaler (fitted, for inference)
```

### 8. OOFAligner (`alignment.py`)

```python
# Key exports:
OOFAligner          # Aligns heterogeneous OOF predictions
AlignedOOFResult    # Aligned predictions with metadata
align_oof_predictions  # Convenience function

# Handles different coverage:
# - Tabular: 100% coverage (offset=0)
# - Sequence: ~98% coverage (offset=seq_len-1)

aligner = OOFAligner()
aligned = aligner.align([oof_xgb, oof_lstm, oof_patch])
# aligned.stacking_features includes:
#   - All probability columns
#   - Mean confidence (derived)
#   - Prediction agreement (derived)
```

---

## Model-to-Adapter Mapping

| Model | Adapter | Data Rank | Shape |
|-------|---------|-----------|-------|
| xgboost | tabular | 2D | (n, f) |
| lightgbm | tabular | 2D | (n, f) |
| catboost | tabular | 2D | (n, f) |
| random_forest | tabular | 2D | (n, f) |
| logistic | tabular | 2D | (n, f) |
| svm | tabular | 2D | (n, f) |
| lstm | sequence | 3D | (n, s, f) |
| gru | sequence | 3D | (n, s, f) |
| tcn | sequence | 3D | (n, s, f) |
| transformer | sequence | 3D | (n, s, f) |
| tft | sequence | 3D | (n, s, f) |
| nbeats | sequence | 3D | (n, s, f) |
| inceptiontime | sequence | 3D | (n, s, f) |
| resnet1d | sequence | 3D | (n, s, f) |
| patchtst | multi_stream | 4D | (n, t, s, f) |
| itransformer | multi_stream | 4D | (n, t, s, f) |
| ridge_meta | tabular | 2D | (n, f) |
| mlp_meta | tabular | 2D | (n, f) |
| xgboost_meta | tabular | 2D | (n, f) |
| calibrated_meta | tabular | 2D | (n, f) |
| voting | tabular | 2D | (n, f) |
| stacking | tabular | 2D | (n, f) |
| blending | tabular | 2D | (n, f) |

---

## Remaining Tasks

### Task 2.1: Verify NO BYPASS ⚠️

**Gap:** Need to audit all training code paths to ensure adapters are used.

**Action Items:**
- [ ] Grep for direct `df.values` or `df.to_numpy()` in training code
- [ ] Add assertion in Trainer that data comes from AdapterResult
- [ ] Add logging to track adapter usage

### Task 2.2: Trainer Integration Validation

**Gap:** Need to verify trainer uses `_prepare_data()` consistently.

**Action Items:**
- [ ] Add integration test: Trainer.train() → AdapterResult path
- [ ] Verify OOF generation uses adapter indices
- [ ] Test heterogeneous ensemble path

---

## Data Flow Diagram

```
DataFrame (features)
       │
       ▼
┌─────────────────────────────────────┐
│  MODEL_ADAPTER_MAP[model_name]      │
│  "xgboost" → "tabular"              │
│  "lstm" → "sequence"                │
│  "patchtst" → "multi_stream"        │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  AdapterRegistry.get(adapter_name)  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  adapter.transform(df)              │
│  Returns: AdapterResult             │
│    - data: 2D/3D/4D array           │
│    - original_indices: for OOF      │
│    - feature_names: columns used    │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  AdapterScaler.fit_transform()      │
│  (Fit on train, transform val/test) │
└─────────────────────────────────────┘
       │
       ▼
     Model.fit(X, y)
```

---

## Integration Points

| Downstream Phase | Consumes |
|------------------|----------|
| PHASE_3 | `AdapterFactory.prepare_data()`, `PreparedData` |
| PHASE_4 | `OOFAligner`, `AlignedOOFResult` |
| PHASE_5 | `AdapterScaler` (serialized in bundle) |

---

## Usage Examples

### Example 1: Single Model Preparation
```python
from src.adapters import UnifiedDataPreparation
from src.core import PipelineConfig

config = PipelineConfig(
    symbol="MES",
    data_path="./data",
    output_dir="./out",
    sequence_length=60,
)

prep = UnifiedDataPreparation(config)
data = prep.prepare(df, model_name="lstm")

print(data.X_train.shape)  # (7000, 60, 80) - 3D
print(data.scaler)         # Fitted AdapterScaler
```

### Example 2: Heterogeneous Ensemble
```python
from src.adapters import AdapterFactory

factory = AdapterFactory(config)
results = factory.prepare_heterogeneous(
    models=["xgboost", "lstm", "patchtst"],
    df=df_1min,
    additional_dfs={"5min": df_5min, "15min": df_15min}
)

print(results["xgboost"].data.shape)   # (10000, 100) - 2D
print(results["lstm"].data.shape)      # (9941, 60, 80) - 3D
print(results["patchtst"].data.shape)  # (9941, 3, 60, 5) - 4D
```

### Example 3: OOF Alignment
```python
from src.adapters import align_oof_predictions

aligned = align_oof_predictions([oof_xgb, oof_lstm, oof_patch])

print(f"Common samples: {aligned.n_common}")
print(f"Stacking features shape: {aligned.stacking_features.shape}")
print(f"Coverage: {aligned.coverage}")
```

---

## Sign-off Criteria

- [x] AdapterRegistry with @register decorator
- [x] TabularAdapter (2D) implemented
- [x] SequenceAdapter (3D) implemented
- [x] MultiStreamAdapter (4D) implemented
- [x] AdapterFactory with PipelineConfig
- [x] AdapterScaler for 2D/3D/4D
- [x] UnifiedDataPreparation pipeline
- [x] OOFAligner for heterogeneous models
- [ ] Audit for NO BYPASS paths
- [ ] Integration tests for trainer

**PHASE_2 Status: READY FOR PHASE_3**
