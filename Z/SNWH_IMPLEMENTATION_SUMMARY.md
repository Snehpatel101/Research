# SNwH Implementation Summary

## Unified Multi-Timeframe Model Factory

This document summarizes the complete SNwH (Unified Multi-Timeframe Model Factory) implementation across Phases 0-6.

---

## Implementation Status

| Phase | Name | Status | New Files | Modified Files |
|:-----:|------|:------:|-----------|----------------|
| 0 | Canonical Contracts | Designed | 4 | 0 |
| 1 | Configuration Layer | Designed | 1 | 3 |
| 2 | Adapter Architecture | Designed | 5 | 0 |
| 3 | Timeframe Coordination | Designed | 2 | 1 |
| 4 | OOF Integrity | Designed | 1 | 3 |
| 5 | Feature Strategy Integration | Designed | 2 | 3 |
| 6 | Single Config System | Designed | 3 | 0 (delete 10+) |

**Total: 18 new files, 10 modified files, ~5400 lines deleted**

---

## File Summary

### New Files (15)

| File | Phase | Purpose |
|------|:-----:|---------|
| `src/contracts/__init__.py` | 0 | Package init |
| `src/contracts/data_contract.py` | 0 | DataContract, DataRank, FeatureMode, MTFMode |
| `src/contracts/model_contract.py` | 0 | ModelContract, MODEL_CONTRACTS for 23 models |
| `src/contracts/artifact_manifest.py` | 0 | ArtifactManifest for reproducibility |
| `src/models/config/per_model_config.py` | 1 | PerModelConfig, EnsemblePlan |
| `src/adapters/__init__.py` | 2 | Package init |
| `src/adapters/base.py` | 2 | BaseAdapter, AdapterResult |
| `src/adapters/registry.py` | 2 | AdapterRegistry, get_adapter() |
| `src/adapters/tabular.py` | 2 | TabularAdapter (2D) |
| `src/adapters/sequence.py` | 2 | SequenceAdapter (3D) |
| `src/adapters/multi_stream.py` | 2 | MultiStreamAdapter (4D) |
| `src/coordination/__init__.py` | 3 | Package init |
| `src/coordination/timeframe_coordinator.py` | 3 | TimeframeCoordinator |
| `src/coordination/alignment.py` | 3 | Temporal alignment utilities |
| `src/cross_validation/oof_alignment.py` | 4 | OOFAlignmentValidator |
| `src/features/strategy_manager.py` | 5 | FeatureStrategyManager |
| `src/features/optimization.py` | 5 | FeatureOptimizer |

### Modified Files (10)

| File | Phase | Changes |
|------|:-----:|---------|
| `src/models/config/trainer_config.py` | 1 | +8 fields, +from_model_contract() |
| `src/models/config/data_requirements.py` | 1 | +6 fields to ModelDataRequirements |
| `src/config/unified.py` | 1 | +ModelConfigSection, +get_trainer_config_for_model() |
| `src/models/config/__init__.py` | 1 | Export PerModelConfig, EnsemblePlan |
| `src/models/training/trainer.py` | 3 | +_load_data_for_model(), +_load_heterogeneous_data() |
| `src/cross_validation/oof_sequence.py` | 4 | +strict_validation, +original_indices |
| `src/cross_validation/oof_stacking.py` | 4 | +HeterogeneousStackingBuilder |
| `src/cross_validation/oof_core.py` | 4 | +alignment metadata to OOFPrediction |
| `src/models/training/features.py` | 5 | +_get_strategy_features() |
| `src/features/__init__.py` | 5 | Export new classes |

---

## Dependency Graph

```
Phase 0: Canonical Contracts
    |
    +-- DataContract, ModelContract, ArtifactManifest
    |
    v
Phase 1: Configuration Layer
    |
    +-- TrainerConfig extensions
    +-- PerModelConfig, EnsemblePlan
    |
    v
Phase 2: Adapter Architecture
    |
    +-- AdapterRegistry
    +-- TabularAdapter, SequenceAdapter, MultiStreamAdapter
    |
    v
Phase 3: Timeframe Coordination
    |
    +-- TimeframeCoordinator
    +-- Temporal alignment
    |
    v
Phase 4: OOF Integrity
    |
    +-- OOFAlignmentValidator
    +-- HeterogeneousStackingBuilder
    |
    v
Phase 5: Feature Strategy Integration
    |
    +-- FeatureStrategyManager
    +-- FeatureOptimizer
```

---

## Critical Changes by Location

### trainer.py (lines 314-317) - CRITICAL FIX

**Before:**
```python
X_train_df, y_train_series, w_train_series = container.get_sklearn_arrays(
    "train", return_df=True
)
```

**After:**
```python
# SNwH: Load data based on model contract
train_df, val_df = self._load_data_for_model(container)
feature_names = self._get_feature_columns(train_df)  # Strategy-aware
X_train_df = train_df[feature_names]
```

### oof_sequence.py (lines 194-206) - CRITICAL FIX

**Before:**
```python
# Warning only, no validation
if coverage_shortfall > COVERAGE_WARNING_THRESHOLD:
    logger.warning(...)
```

**After:**
```python
# Strict validation with alignment metadata
if strict_validation and coverage_shortfall > COVERAGE_WARNING_THRESHOLD:
    raise ValueError(...)
# Store original indices for alignment
return OOFPrediction(..., original_indices=valid_indices)
```

### TrainerConfig - NEW FIELDS

```python
# NEW fields for SNwH
primary_timeframe: str = "5min"
mtf_mode: str = "indicators"
mtf_timeframes: list[str] = []
feature_mode: str = "engineered"
adapter_id: str | None = None
input_rank: int = 2
```

---

## Integration Points

### 1. Model Registration (contracts/model_contract.py)

Every model now has a `ModelContract` that declares:
- `input_rank`: 2 (tabular), 3 (sequence), or 4 (multi-stream)
- `primary_timeframe`: Default training timeframe
- `mtf_mode`: none, indicators, or multi_stream
- `feature_mode`: engineered, raw, or hybrid

### 2. Adapter Routing (adapters/registry.py)

```python
# Automatic adapter selection based on contract
adapter = get_adapter(model_name="lstm")  # Returns SequenceAdapter
adapter = get_adapter(model_name="xgboost")  # Returns TabularAdapter
adapter = get_adapter(model_name="patchtst")  # Returns MultiStreamAdapter
```

### 3. Timeframe Data Loading (coordination/timeframe_coordinator.py)

```python
# Load correct timeframe for each model
coordinator = TimeframeCoordinator(data_dir="data/splits/scaled")
xgb_df = coordinator.get_data_for_model("xgboost")  # Gets 15min data
lstm_df = coordinator.get_data_for_model("lstm")  # Gets 5min data
```

### 4. OOF Alignment (cross_validation/oof_alignment.py)

```python
# Ensure aligned stacking
builder = HeterogeneousStackingBuilder()
X_stack, y_aligned, w_aligned = builder.build_stacking_dataset(
    oof_results={"xgboost": xgb_oof, "lstm": lstm_oof},
    y=y_train,
)
```

### 5. Feature Selection (features/strategy_manager.py)

```python
# Strategy-based feature selection
manager = FeatureStrategyManager(df=train_df)
xgb_features = manager.get_features_for_model("xgboost")  # ~100 features
lstm_features = manager.get_features_for_model("lstm")  # ~80 features
patchtst_features = manager.get_features_for_model("patchtst")  # 5 features (OHLCV)
```

---

## Backward Compatibility

All changes maintain backward compatibility:

1. **TrainerConfig**: New fields have defaults, old constructors work
2. **Trainer.run()**: Falls back to container data if no contract
3. **Feature selection**: Falls back to all features if strategy fails
4. **OOF validation**: strict_validation=False disables new validation

---

## Migration Path

### Existing Single-Model Training

No changes required. Works as before.

### Adding SNwH Support

1. **Use contract-based config:**
```python
config = TrainerConfig.from_model_contract("xgboost", horizon=20)
```

2. **Use adapter for data transformation:**
```python
adapter = get_adapter(model_name="lstm", sequence_length=60)
result = adapter.transform(df)
```

3. **Use coordinator for multi-TF:**
```python
coordinator = TimeframeCoordinator(data_dir)
coordinator.load_timeframes(["5min", "15min", "60min"])
```

### Heterogeneous Ensembles

```python
from src.models.config import EnsemblePlan, PerModelConfig

plan = EnsemblePlan(
    base_models=[
        PerModelConfig(name="xgboost", timeframe="15min"),
        PerModelConfig(name="lstm", timeframe="5min"),
        PerModelConfig(name="patchtst", timeframe="1min"),
    ],
    meta_learner="ridge_meta",
)
```

---

## Verification Checklist

After implementation, verify:

- [ ] `ModelContract` exists for all 23 models
- [ ] `TrainerConfig.from_model_contract()` works for all models
- [ ] `AdapterRegistry.get_for_model()` returns correct adapter type
- [ ] `TimeframeCoordinator` loads correct timeframe per model
- [ ] `OOFAlignmentValidator` detects coverage mismatches
- [ ] `FeatureStrategyManager` filters to correct baseline features
- [ ] Heterogeneous ensemble stacking produces aligned datasets
- [ ] All existing tests pass (backward compatibility)

---

## Documentation Files

| File | Purpose |
|------|---------|
| `SNWH_IMPLEMENTATION_PHASE_0.md` | Canonical Contracts |
| `SNWH_IMPLEMENTATION_PHASE_1.md` | Configuration Layer |
| `SNWH_IMPLEMENTATION_PHASE_2.md` | Adapter Architecture |
| `SNWH_IMPLEMENTATION_PHASE_3.md` | Timeframe Coordination |
| `SNWH_IMPLEMENTATION_PHASE_4.md` | OOF Integrity |
| `SNWH_IMPLEMENTATION_PHASE_5.md` | Feature Strategy Integration |
| `SNWH_IMPLEMENTATION_SUMMARY.md` | This summary |

---

## Next Steps (Phases 6-8)

After Phases 0-5 are implemented:

- **Phase 6**: Validation - End-to-end tests for heterogeneous ensembles
- **Phase 7**: Testing - Unit tests for all new components
- **Phase 8**: Documentation - Update CLAUDE.md and user guides
