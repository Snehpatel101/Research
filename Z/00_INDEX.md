# SNwH Implementation Plan - Master Index

**Generated:** 2026-01-16
**Purpose:** Comprehensive implementation plan to unify the ML Model Factory pipeline
**Goal:** Every model works with every timeframe, heterogeneous ensembles work by default

---

## Quick Start

1. Read `SNWH_ARCHITECTURE_SYNTHESIS.md` for the complete gap analysis
2. Follow phases in order: 0 → 1 → 2 → 3 → 4 → 5
3. Use `SNWH_TESTING_STRATEGY.md` to validate each phase
4. Reference `SNWH_IMPLEMENTATION_SUMMARY.md` for file listings

---

## Document Overview

| Document | Purpose | Lines | Priority |
|----------|---------|-------|----------|
| **00_INDEX.md** | This master index | - | Start here |
| **SNWH_ARCHITECTURE_SYNTHESIS.md** | Gap analysis, dependency graph, priority matrix | 559 | Read first |
| **SNWH_IMPLEMENTATION_PHASE_0.md** | Canonical Contracts (DataContract, ModelContract, ArtifactManifest) | ~1000 | Implement first |
| **SNWH_IMPLEMENTATION_PHASE_1.md** | Configuration Layer (TrainerConfig, PerModelConfig, EnsemblePlan) | ~600 | After Phase 0 |
| **SNWH_IMPLEMENTATION_PHASE_2.md** | Adapter Architecture (TabularAdapter, SequenceAdapter, MultiStreamAdapter) | ~900 | After Phase 1 |
| **SNWH_IMPLEMENTATION_PHASE_3.md** | Timeframe Coordination (TimeframeCoordinator, alignment) | ~700 | After Phase 2 |
| **SNWH_IMPLEMENTATION_PHASE_4.md** | OOF Integrity (OOFAlignmentValidator, coverage fixes) | ~550 | After Phase 3 |
| **SNWH_IMPLEMENTATION_PHASE_5.md** | Feature Strategy Integration (FeatureStrategyManager, optimization) | ~650 | After Phase 4 |
| **SNWH_IMPLEMENTATION_SUMMARY.md** | Complete file listings, migration path, verification checklist | 296 | Reference |
| **SNWH_TESTING_STRATEGY.md** | 28 test files, 82 test classes, 305 test methods | ~1700 | Validate each phase |
| **SNWH_IMPLEMENTATION_PHASE_6.md** | ML for Dummies Config System (smart defaults, per-model configs) | ~500 | ✅ **IMPLEMENTED** |

---

## Critical Gaps (Must Fix)

These 3 gaps block heterogeneous ensembles from working:

| Gap | File | Lines | Issue | Fix Phase |
|-----|------|-------|-------|-----------|
| **GAP-001** | `src/models/training/trainer.py` | 314-317 | Loads SINGLE timeframe for ALL models | Phase 3 |
| **GAP-002** | `src/cross_validation/oof_sequence.py` | 194-206 | OOF coverage mismatch breaks stacking | Phase 4 |
| **GAP-003** | `src/models/config/trainer_config.py` | 27-102 | Missing primary_timeframe, mtf_mode, adapter_id | Phase 1 |

---

## Implementation Phases

### Phase 0: Canonical Contracts (Foundation)

**New Files:**
- `src/contracts/__init__.py`
- `src/contracts/data_contract.py` - DataContract, DataRank, FeatureMode, MTFMode
- `src/contracts/model_contract.py` - ModelContract, MODEL_CONTRACTS for 23 models
- `src/contracts/artifact_manifest.py` - ArtifactManifest for reproducibility

**Key Classes:**
```python
@dataclass(frozen=True)
class ModelContract:
    model_name: str
    input_rank: int  # 2, 3, or 4
    feature_mode: FeatureMode  # engineered, raw, hybrid
    mtf_mode: MTFMode  # none, indicators, multi_stream
    primary_timeframe: str  # "5min", "15min", "1min"
    sequence_length: int | None
    min_features: int
    max_features: int
```

### Phase 1: Configuration Layer

**Modified Files:**
- `src/models/config/trainer_config.py` - Add 8 new fields
- `src/models/config/data_requirements.py` - Add 6 new fields
- `src/config/unified.py` - Add ModelConfigSection

**New Files:**
- `src/models/config/per_model_config.py` - PerModelConfig, EnsemblePlan

**Key Addition to TrainerConfig:**
```python
# NEW fields (add after line 102)
primary_timeframe: str = "5min"
mtf_mode: str = "indicators"
mtf_timeframes: list[str] = field(default_factory=list)
feature_mode: str = "engineered"
adapter_id: str | None = None
input_rank: int = 2
min_features: int = 20
max_features: int = 200
```

### Phase 2: Adapter Architecture

**New Files:**
- `src/adapters/__init__.py`
- `src/adapters/base.py` - BaseAdapter, AdapterResult
- `src/adapters/registry.py` - AdapterRegistry
- `src/adapters/tabular.py` - TabularAdapter (2D output)
- `src/adapters/sequence.py` - SequenceAdapter (3D output)
- `src/adapters/multi_stream.py` - MultiStreamAdapter (4D output)

**Usage:**
```python
from src.adapters import get_adapter

adapter = get_adapter("xgboost")  # TabularAdapter
result = adapter.transform(df, feature_columns=features)
# result.data.shape = (n_samples, n_features)

adapter = get_adapter("lstm", sequence_length=60)  # SequenceAdapter
result = adapter.transform(df, feature_columns=features)
# result.data.shape = (n_samples, 60, n_features)
```

### Phase 3: Timeframe Coordination

**New Files:**
- `src/coordination/__init__.py`
- `src/coordination/timeframe_coordinator.py` - TimeframeCoordinator
- `src/coordination/alignment.py` - Temporal alignment utilities

**Modified Files:**
- `src/models/training/trainer.py` - Add `_load_data_for_model()`, `_load_heterogeneous_data()`

**Critical Fix in trainer.py:314-317:**
```python
# BEFORE (GAP-001):
X_train_df = container.get_sklearn_arrays("train", return_df=True)

# AFTER:
train_df, val_df = self._load_data_for_model(container)
feature_names = self._get_feature_columns(train_df)
X_train_df = train_df[feature_names]
```

### Phase 4: OOF Integrity

**New Files:**
- `src/cross_validation/oof_alignment.py` - OOFAlignmentValidator

**Modified Files:**
- `src/cross_validation/oof_sequence.py` - Add strict validation, original indices
- `src/cross_validation/oof_stacking.py` - Add HeterogeneousStackingBuilder
- `src/cross_validation/oof_core.py` - Add alignment metadata

**Critical Fix in oof_sequence.py:194-206:**
```python
# BEFORE (GAP-002):
if coverage_shortfall > COVERAGE_WARNING_THRESHOLD:
    logger.warning(...)  # Just warns

# AFTER:
if strict_validation and coverage_shortfall > COVERAGE_WARNING_THRESHOLD:
    raise ValueError(f"Coverage {coverage:.1%} below threshold")
# Store original indices for alignment
return OOFPrediction(..., original_indices=valid_indices)
```

### Phase 5: Feature Strategy Integration

**New Files:**
- `src/features/strategy_manager.py` - FeatureStrategyManager
- `src/features/optimization.py` - FeatureOptimizer

**Modified Files:**
- `src/models/training/features.py` - Add `_get_strategy_features()`
- `src/features/__init__.py` - Export new classes

**Usage:**
```python
from src.features import FeatureStrategyManager

manager = FeatureStrategyManager(df=train_df)
xgb_features = manager.get_features_for_model("xgboost")  # ~100 features
lstm_features = manager.get_features_for_model("lstm")  # ~80 features
patchtst_features = manager.get_features_for_model("patchtst")  # 5 features (OHLCV)
```

### Phase 6: ML for Dummies Config ✅ IMPLEMENTED

**Status:** Phase 6.1 (smart_config.py) and Phase 6.2 (training integration) are complete.

**Files Created:**
- `src/config/smart_config.py` (~900 lines) - Main API with `train()`, `SmartConfig`, `MODEL_DEFAULTS`

**Files Modified:**
- `src/training/config.py` - Added `features` and `batch_size` to `ModelConfig`
- `src/training/feature_selector.py` - Added smart config feature sets
- `src/training/orchestrator.py` - Added per-model feature filtering

**The Simple API:**
```python
from src.config import train

# Just works with smart defaults
train("xgboost")

# Per-model timeframe, features, batch size all automatic
train(["xgboost", "lstm", "patchtst"], ensemble=True)

# Override any default
train("lstm", sequence_length=120)
```

**Helper Functions:**
```python
list_models()           # All 23 models
list_models("boosting") # Filter by family
describe_model("xgboost")  # Get description
show_defaults("lstm")   # See smart defaults
preview_config("xgboost", optimize="hyperparameters")  # Preview without training
quick_compare(["xgboost", "lstm", "patchtst"])  # Compare defaults
```

---

## File Reference Index

### Files to Modify

| File | Phase | Key Lines | Changes |
|------|-------|-----------|---------|
| `src/models/config/trainer_config.py` | 1 | 27-102 | +8 fields |
| `src/models/config/data_requirements.py` | 1 | 47-83 | +6 fields |
| `src/config/unified.py` | 1 | 446-489 | +ModelConfigSection |
| `src/models/training/trainer.py` | 3 | 253-511 | +_load_data_for_model() |
| `src/phase1/stages/mtf/constants.py` | 3 | 42-50 | 7→9 TFs |
| `src/cross_validation/oof_sequence.py` | 4 | 54-241 | +strict validation |
| `src/cross_validation/oof_stacking.py` | 4 | various | +alignment builder |
| `src/cross_validation/oof_core.py` | 4 | various | +metadata |
| `src/models/training/features.py` | 5 | various | +strategy lookup |
| `src/features/strategies.py` | 5 | 145-367 | Sync with YAMLs |

### Files to Create

| File | Phase | Purpose |
|------|-------|---------|
| `src/contracts/__init__.py` | 0 | Package init |
| `src/contracts/data_contract.py` | 0 | Data schema contract |
| `src/contracts/model_contract.py` | 0 | Model input contracts |
| `src/contracts/artifact_manifest.py` | 0 | Reproducibility manifest |
| `src/models/config/per_model_config.py` | 1 | Per-model config |
| `src/adapters/__init__.py` | 2 | Package init |
| `src/adapters/base.py` | 2 | Base adapter class |
| `src/adapters/registry.py` | 2 | Adapter routing |
| `src/adapters/tabular.py` | 2 | 2D adapter |
| `src/adapters/sequence.py` | 2 | 3D adapter |
| `src/adapters/multi_stream.py` | 2 | 4D adapter |
| `src/coordination/__init__.py` | 3 | Package init |
| `src/coordination/timeframe_coordinator.py` | 3 | TF coordination |
| `src/coordination/alignment.py` | 3 | Temporal alignment |
| `src/cross_validation/oof_alignment.py` | 4 | OOF validation |
| `src/features/strategy_manager.py` | 5 | Strategy integration |
| `src/features/optimization.py` | 5 | Feature optimization |

---

## Testing Strategy Summary

| Category | Files | Classes | Methods |
|----------|-------|---------|---------|
| Unit Tests | 15 | 45 | 180 |
| Integration Tests | 5 | 15 | 45 |
| Regression Tests | 4 | 12 | 40 |
| Property Tests | 4 | 10 | 40 |
| **Total** | **28** | **82** | **305** |

**Critical Test Scenarios:**
1. Heterogeneous stacking: XGBoost (15min, 2D) + LSTM (5min, 3D) + PatchTST (1min, 4D)
2. OOF coverage alignment: Tabular 100% vs Sequence (100%-seq_len)
3. Per-model feature selection: Different features for different models
4. MTF leakage prevention: shift(1) applied correctly
5. Adapter shape validation: Correct routing for all 23 models

---

## Sprint Plan

### Week 1: Foundation (Phases 0-1)
- Day 1-2: Implement canonical contracts
- Day 3-4: Extend TrainerConfig and UnifiedConfig
- Day 5: Create PerModelConfig and EnsemblePlan

### Week 2: Data Routing (Phases 2-3)
- Day 1-2: Implement adapter architecture
- Day 3: Fix MTF constants (7→9 TFs)
- Day 4-5: Implement TimeframeCoordinator and modify trainer.py

### Week 3: OOF & Features (Phases 4-5)
- Day 1-2: Fix OOF coverage alignment
- Day 3: Implement OOFAlignmentValidator
- Day 4-5: Wire FeatureStrategyManager

### Week 4: Validation & Testing
- Day 1-2: Implement pre-flight validation
- Day 3-4: Write integration tests
- Day 5: Documentation updates

---

## Success Criteria

| Criterion | Verification |
|-----------|--------------|
| Per-model timeframe config works | `TrainerConfig.from_model_contract()` returns correct TF |
| Adapters convert data correctly | All 23 models get correct input shape |
| Heterogeneous stacking works | XGBoost + LSTM + PatchTST trains successfully |
| Ensemble validation fails fast | Invalid combos caught BEFORE OOF generation |
| OOF integrity enforced | Coverage validated, no NaN misalignment |
| Reproducibility | Config hash matches between runs |
| Single-contract isolation | No cross-symbol data leakage |

---

## Quick Reference

### Model Contracts (23 models)

| Model | Input Rank | Primary TF | MTF Mode | Feature Mode |
|-------|------------|------------|----------|--------------|
| xgboost | 2 | 15min | indicators | engineered |
| lightgbm | 2 | 15min | indicators | engineered |
| catboost | 2 | 15min | indicators | engineered |
| random_forest | 2 | 10min | indicators | engineered |
| logistic | 2 | 15min | none | engineered |
| svm | 2 | 15min | none | engineered |
| lstm | 3 | 5min | indicators | engineered |
| gru | 3 | 5min | indicators | engineered |
| tcn | 3 | 5min | indicators | engineered |
| transformer | 3 | 5min | indicators | engineered |
| patchtst | 4 | 1min | multi_stream | raw |
| itransformer | 4 | 1min | multi_stream | raw |
| tft | 3 | 5min | indicators | engineered |
| nbeats | 3 | 5min | none | engineered |
| inceptiontime | 3 | 5min | indicators | engineered |
| resnet1d | 3 | 5min | indicators | engineered |
| voting | 2 | varies | varies | engineered |
| stacking | 2 | varies | varies | engineered |
| blending | 2 | varies | varies | engineered |
| ridge_meta | 2 | - | - | oof_probs |
| mlp_meta | 2 | - | - | oof_probs |
| calibrated_meta | 2 | - | - | oof_probs |
| xgboost_meta | 2 | - | - | oof_probs |

### Adapter Routing

```
Model → ModelContract.input_rank → Adapter
  2 → TabularAdapter  → (n_samples, n_features)
  3 → SequenceAdapter → (n_samples, seq_len, n_features)
  4 → MultiStreamAdapter → (n_samples, n_streams, seq_len, n_features)
```

---

## Contact & Support

For questions about this implementation plan:
1. Review the relevant phase document
2. Check the testing strategy for validation approaches
3. Consult the architecture synthesis for dependency information
