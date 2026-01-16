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
| **SNWH_IMPLEMENTATION_PHASE_0.md** | Canonical Contracts (DataContract, ModelContract, ArtifactManifest) | ~1000 | ✅ **IMPLEMENTED** |
| **SNWH_IMPLEMENTATION_PHASE_1.md** | Configuration Layer (TrainerConfig, PerModelConfig, EnsemblePlan) | ~600 | ✅ **IMPLEMENTED** |
| **SNWH_IMPLEMENTATION_PHASE_2.md** | Adapter Architecture (TabularAdapter, SequenceAdapter, MultiStreamAdapter) | ~900 | ✅ **IMPLEMENTED** |
| **SNWH_IMPLEMENTATION_PHASE_3.md** | Timeframe Coordination (TimeframeCoordinator, alignment) | ~700 | ✅ **IMPLEMENTED** |
| **SNWH_IMPLEMENTATION_PHASE_4.md** | OOF Integrity (OOFAlignmentValidator, coverage fixes) | ~550 | ✅ **IMPLEMENTED** |
| **SNWH_IMPLEMENTATION_PHASE_5.md** | Feature Strategy Integration (FeatureStrategyManager, optimization) | ~650 | ✅ **IMPLEMENTED** |
| **SNWH_IMPLEMENTATION_SUMMARY.md** | Complete file listings, migration path, verification checklist | 296 | Reference |
| **SNWH_TESTING_STRATEGY.md** | 28 test files, 82 test classes, 305 test methods | ~1700 | Validate each phase |
| **SNWH_IMPLEMENTATION_PHASE_6.md** | ML for Dummies Config System (smart defaults, per-model configs) | ~500 | ✅ **IMPLEMENTED** |

---

## Critical Gaps (Must Fix)

These 3 gaps block heterogeneous ensembles from working:

| Gap | File | Lines | Issue | Fix Phase |
|-----|------|-------|-------|-----------|
| **GAP-001** | `src/models/training/trainer.py` | 314-317 | Loads SINGLE timeframe for ALL models | ✅ Phase 3 (FIXED) |
| **GAP-002** | `src/cross_validation/oof_sequence.py` | 194-206 | OOF coverage mismatch breaks stacking | ✅ Phase 4 (FIXED) |
| **GAP-003** | `src/models/config/trainer_config.py` | 27-102 | Missing primary_timeframe, mtf_mode, adapter_id | ✅ Phase 1 (FIXED) |

---

## Implementation Phases

### Phase 0: Canonical Contracts (Foundation) ✅ IMPLEMENTED

**Status:** Complete (81 tests passing) - Implemented 2026-01-16

**Files Created:**
- `src/contracts/__init__.py` (65 lines)
- `src/contracts/data_contract.py` (380 lines) - DataContract, DataRank, FeatureMode, MTFMode
- `src/contracts/model_contract.py` (520 lines) - ModelContract, MODEL_CONTRACTS for 23 models
- `src/contracts/artifact_manifest.py` (340 lines) - ArtifactManifest for reproducibility

**Tests Created:**
- `tests/contracts/test_data_contract.py` (30 tests)
- `tests/contracts/test_model_contract.py` (32 tests)
- `tests/contracts/test_artifact_manifest.py` (19 tests)

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

### Phase 1: Configuration Layer ✅ IMPLEMENTED

**Status:** Complete (42 tests passing) - Implemented 2026-01-16

**Modified Files:**
- `src/models/config/trainer_config.py` - Added 8 new fields + `from_model_contract()` factory
- `src/models/config/data_requirements.py` - Added 6 new fields to all 24 models
- `src/config/unified.py` - Added ModelConfigSection for family/model overrides
- `src/models/config/__init__.py` - Added exports for PerModelConfig, EnsemblePlan

**New Files:**
- `src/models/config/per_model_config.py` (432 lines) - PerModelConfig, EnsemblePlan

**Tests Created:**
- `tests/config/test_phase1_config.py` (42 tests)

**Key Classes:**
```python
# TrainerConfig now has Phase 1 SNwH fields:
@dataclass
class TrainerConfig:
    # ... existing fields ...
    primary_timeframe: str = "5min"
    mtf_mode: str = "indicators"  # none, indicators, multi_stream
    mtf_timeframes: list[str] = field(default_factory=list)
    feature_mode: str = "engineered"  # engineered, raw, hybrid
    adapter_id: str | None = None  # auto-resolved from input_rank
    input_rank: int = 2  # 2, 3, or 4
    min_features: int = 4
    max_features: int = 200

    @classmethod
    def from_model_contract(cls, model_name: str, horizon: int, **overrides) -> "TrainerConfig":
        """Create from ModelContract with contract-based defaults."""

# PerModelConfig allows per-model settings for heterogeneous ensembles:
@dataclass
class PerModelConfig:
    name: str
    timeframe: str | None = None  # Override contract default
    feature_mode: str | None = None
    mtf_mode: str | None = None
    # ... resolved_* properties use contracts for defaults

# EnsemblePlan manages heterogeneous ensemble configuration:
@dataclass
class EnsemblePlan:
    base_models: list[PerModelConfig]
    meta_learner: str = "ridge_meta"
    # ... grouping by adapter, validation, serialization
```

### Phase 2: Adapter Architecture ✅ IMPLEMENTED

**Status:** Complete (39 tests passing) - Implemented 2026-01-16

**Files Created:**
- `src/adapters/__init__.py` (50 lines) - Package exports with auto-registration
- `src/adapters/base.py` (242 lines) - BaseAdapter, AdapterResult with validation
- `src/adapters/registry.py` (190 lines) - AdapterRegistry, get_adapter() function
- `src/adapters/tabular.py` (~250 lines) - TabularAdapter (2D: n_samples, n_features)
- `src/adapters/sequence.py` (~350 lines) - SequenceAdapter (3D: n_samples, seq_len, n_features)
- `src/adapters/multi_stream.py` (~500 lines) - MultiStreamAdapter (4D: n_samples, n_tfs, seq_len, n_features)

**Tests Created:**
- `tests/adapters/__init__.py`
- `tests/adapters/test_phase2_adapters.py` (39 tests)

**Key Classes:**
```python
@dataclass
class AdapterResult:
    X: np.ndarray  # Transformed features (2D, 3D, or 4D)
    y: np.ndarray  # Labels
    weights: np.ndarray | None  # Sample weights
    data_rank: DataRank  # TABULAR_2D, SEQUENCE_3D, MULTI_TF_4D
    sequence_length: int | None  # For 3D/4D
    n_timeframes: int | None  # For 4D
    data_contract: DataContract | None  # Auto-created

    def validate(self) -> tuple[bool, list[str]]: ...
    def to_dict(self) -> dict[str, Any]: ...

@AdapterRegistry.register("tabular")
class TabularAdapter(BaseAdapter):
    def transform(df, model_contract=None) -> AdapterResult: ...

@AdapterRegistry.register("sequence")
class SequenceAdapter(BaseAdapter):
    def __init__(sequence_length=60, stride=1, symbol_column="symbol"): ...
    def transform(df, model_contract=None) -> AdapterResult: ...
    # Builds sliding windows, enforces symbol isolation

@AdapterRegistry.register("multi_stream")
class MultiStreamAdapter(BaseAdapter):
    def __init__(sequence_length=60, timeframes=["1min","5min","15min"]): ...
    def transform(df, additional_dfs=None, model_contract=None) -> AdapterResult: ...
    # Aligns multi-timeframe data using ratio-based index mapping
```

**Usage:**
```python
from src.adapters import get_adapter

# Auto-route by model name (uses ModelContract)
adapter = get_adapter(model_name="xgboost")  # → TabularAdapter
result = adapter.transform(df)
# result.X.shape = (n_samples, n_features)

adapter = get_adapter(model_name="lstm", sequence_length=60)  # → SequenceAdapter
result = adapter.transform(df)
# result.X.shape = (n_sequences, 60, n_features)

adapter = get_adapter(model_name="patchtst", sequence_length=60)  # → MultiStreamAdapter
result = adapter.transform(df, additional_dfs={"5min": df_5min, "15min": df_15min})
# result.X.shape = (n_sequences, 3, 60, 5)

# Or get adapter directly by ID
adapter = get_adapter(adapter_id="tabular")
adapter = get_adapter(adapter_id="sequence", sequence_length=30)
adapter = get_adapter(adapter_id="multi_stream", timeframes=["1min", "5min"])
```

### Phase 3: Timeframe Coordination ✅ IMPLEMENTED

**Status:** Complete (27 tests passing) - Implemented 2026-01-16

**Files Created:**
- `src/coordination/__init__.py` (45 lines) - Package exports
- `src/coordination/timeframe_coordinator.py` (642 lines) - TimeframeCoordinator, TimeframeData
- `src/coordination/alignment.py` (481 lines) - Temporal alignment utilities

**Tests Created:**
- `tests/coordination/__init__.py`
- `tests/coordination/test_phase3_coordination.py` (27 tests)

**Modified Files:**
- `src/models/training/trainer.py` - Added `_get_coordinator()`, `_load_data_for_model()`, `_load_heterogeneous_data()`

**Key Classes:**
```python
@dataclass
class TimeframeData:
    timeframe: str
    df: pd.DataFrame
    feature_columns: list[str]
    n_samples: int  # computed
    start_time: pd.Timestamp | None  # computed
    end_time: pd.Timestamp | None  # computed

class TimeframeCoordinator:
    def __init__(data_dir, split="train", horizon=20): ...
    def load_timeframes(timeframes, feature_columns=None): ...
    def get_timeframe_data(timeframe) -> TimeframeData: ...
    def get_data_for_model(model_name, contract=None) -> pd.DataFrame: ...
    def get_multi_stream_dfs(timeframes, align_timestamps=True) -> dict[str, pd.DataFrame]: ...
    def get_required_timeframes_for_ensemble(base_models) -> set[str]: ...
    def validate_timeframe_coverage(required) -> tuple[bool, list[str]]: ...

# Alignment utilities
def align_to_anchor(anchor_df, higher_tf_df, anchor_tf, higher_tf, datetime_col): ...
def apply_mtf_lag(df, mtf_columns, shift=1): ...
def compute_sequence_offset(tabular_samples, sequence_samples, sequence_length): ...
def validate_timestamp_alignment(df1, df2, datetime_col, tolerance_minutes): ...
```

**Usage:**
```python
from src.coordination import TimeframeCoordinator

coordinator = TimeframeCoordinator(data_dir="data/splits/scaled", split="train")
coordinator.load_timeframes(["1min", "5min", "15min"])

# Get data for specific model
xgb_df = coordinator.get_data_for_model("xgboost")  # Returns 15min features

# Get aligned multi-stream data for transformers
dfs = coordinator.get_multi_stream_dfs(["1min", "5min", "15min"])

# Check ensemble requirements
required = coordinator.get_required_timeframes_for_ensemble(["xgboost", "lstm", "patchtst"])
```

**Critical Fix (GAP-001) in trainer.py:**
```python
# NEW: Trainer now uses coordinator for multi-timeframe loading
def _load_data_for_model(self, container, model_name=None):
    """Load data based on model's primary_timeframe contract."""
    contract = get_model_contract(model_name)
    if contract.primary_timeframe == current_tf:
        return container.get_split("train").df, container.get_split("val").df
    # Otherwise load from correct timeframe via coordinator
    coordinator = self._get_coordinator()
    coordinator.load_timeframes([contract.primary_timeframe])
    return train_df, val_df
```

### Phase 4: OOF Integrity ✅ IMPLEMENTED

**Status:** Complete (37 tests passing) - Implemented 2026-01-16

**Files Created:**
- `src/cross_validation/oof_alignment.py` (471 lines) - OOFAlignmentResult, OOFAlignmentValidator, ModelCoverage

**Modified Files:**
- `src/cross_validation/oof_core.py` - Added alignment metadata to OOFPrediction (original_indices, sequence_length, n_total_samples)
- `src/cross_validation/oof_sequence.py` - Added strict_validation parameter, original_indices storage
- `src/cross_validation/oof_stacking.py` - Added HeterogeneousStackingBuilder class

**Tests Created:**
- `tests/cross_validation/test_phase4_oof.py` (37 tests)

**Key Classes:**
```python
@dataclass
class OOFAlignmentResult:
    is_aligned: bool
    issues: list[str]
    sample_counts: dict[str, int]
    common_start_idx: int
    common_end_idx: int
    n_common_samples: int
    offsets: dict[str, int]

class OOFAlignmentValidator:
    def __init__(self): ...
    def register_oof(model_name, oof_indices, n_total_samples, sequence_length=None): ...
    def validate(self) -> OOFAlignmentResult: ...
    def align_oof_predictions(oof_dict, alignment) -> dict[str, np.ndarray]: ...

class HeterogeneousStackingBuilder:
    def __init__(purge_bars=60, embargo_bars=1440): ...
    def build_stacking_dataset(oof_results, y, sample_weights=None) -> tuple[X_stack, y_aligned, weights_aligned]: ...
    def get_alignment_summary(self) -> dict: ...
```

**Enhanced OOFPrediction (oof_core.py):**
```python
@dataclass
class OOFPrediction:
    # ... existing fields ...
    # Phase 4 SNwH: Alignment metadata
    original_indices: np.ndarray | None = None
    sequence_length: int | None = None
    n_total_samples: int | None = None

    @property
    def n_valid(self) -> int: ...
    @property
    def alignment_offset(self) -> int: ...
    def get_aligned_probabilities(start_idx, n_samples) -> np.ndarray: ...
```

**Critical Fix (GAP-002) in oof_sequence.py:**
```python
# BEFORE:
if coverage_shortfall > COVERAGE_WARNING_THRESHOLD:
    logger.warning(...)  # Just warns

# AFTER (Phase 4 SNwH):
if strict_validation and coverage_shortfall > COVERAGE_WARNING_THRESHOLD:
    raise ValueError(f"Coverage {coverage:.1%} below threshold")
# Store original indices for alignment
valid_indices = np.where(~np.isnan(oof_preds))[0]
return OOFPrediction(..., original_indices=valid_indices, sequence_length=seq_len, n_total_samples=n_samples)
```

**Usage:**
```python
from src.cross_validation.oof_alignment import OOFAlignmentValidator
from src.cross_validation.oof_stacking import HeterogeneousStackingBuilder

# Validate alignment between tabular and sequence models
validator = OOFAlignmentValidator()
validator.register_oof("xgboost", None, n_total_samples=1000)  # Full coverage
validator.register_oof("lstm", None, n_total_samples=1000, sequence_length=60)  # Partial coverage
result = validator.validate()
# result.common_start_idx = 59, result.n_common_samples = 941

# Build aligned stacking dataset
builder = HeterogeneousStackingBuilder()
X_stack, y_aligned, weights_aligned = builder.build_stacking_dataset(
    oof_results={"xgboost": xgb_oof, "lstm": lstm_oof},
    y=y_true,
    sample_weights=weights,
)
# X_stack.shape = (941, 6)  # 2 models * 3 classes
```

### Phase 5: Feature Strategy Integration ✅ IMPLEMENTED

**Status:** Complete (41 tests passing) - Implemented 2026-01-16

**Files Created:**
- `src/features/strategy_manager.py` (313 lines) - FeatureStrategyManager, ResolvedFeatureSet, get_features_for_model

**Files Modified:**
- `src/features/optimization.py` - Enhanced with FeatureOptimizer class
- `src/features/__init__.py` - Export new classes
- `src/models/config/trainer_config.py` - Added get_feature_strategy(), get_baseline_features(), resolve_features()
- `src/models/training/features.py` - Added _get_feature_columns(), _get_strategy_features(), _get_all_features(), _validate_features_for_model()

**Tests Created:**
- `tests/features/test_phase5_strategy.py` (41 tests)

**Key Classes:**
```python
@dataclass
class ResolvedFeatureSet:
    model_name: str
    feature_columns: list[str]
    n_features: int  # computed in __post_init__
    baseline_requested: int = 0
    baseline_available: int = 0
    optimized: bool = False
    included_families: list[str]

class FeatureStrategyManager:
    def __init__(df=None, available_features=None): ...
    def get_strategy(model_name) -> ModelFeatureStrategy: ...
    def get_features_for_model(model_name, custom_baseline=None, strict=True) -> ResolvedFeatureSet: ...
    def get_features_for_ensemble(base_models) -> dict[str, ResolvedFeatureSet]: ...
    def validate_feature_diversity(base_models, min_unique_ratio=0.3) -> tuple[bool, list[str]]: ...

class FeatureOptimizer:
    def __init__(model_name, n_trials=30, metric="f1_weighted", min_features=None, random_seed=42): ...
    def optimize(X_train, y_train, X_val, y_val, feature_names, sample_weights=None) -> OptimizationResult: ...
```

**TrainerConfig integration:**
```python
# TrainerConfig now has Phase 5 feature strategy methods:
config = TrainerConfig(model_name="xgboost", horizon=20)
strategy = config.get_feature_strategy()  # -> ModelFeatureStrategy
baseline = config.get_baseline_features()  # -> list of baseline feature names
resolved = config.resolve_features(available_features, strict=True)  # -> filtered list
```

**Usage:**
```python
from src.features import FeatureStrategyManager, FeatureOptimizer, get_features_for_model

# Simple usage
features = get_features_for_model("xgboost", df)

# With manager for multiple models
manager = FeatureStrategyManager(df=train_df)
xgb_features = manager.get_features_for_model("xgboost")  # ~100 features
lstm_features = manager.get_features_for_model("lstm")  # ~80 features
patchtst_features = manager.get_features_for_model("patchtst")  # 5 features (OHLCV)

# Validate diversity for ensemble
is_diverse, warnings = manager.validate_feature_diversity(["xgboost", "lstm", "patchtst"])

# With optimization
optimizer = FeatureOptimizer("xgboost", n_trials=30)
result = optimizer.optimize(X_train, y_train, X_val, y_val, feature_names)
optimized_features = result.optimized_features
```

**Data Flow After Integration:**
```
TrainerConfig.model_name
        |
        v
get_strategy_for_model() --> ModelFeatureStrategy
        |
        v
FeatureStrategyManager.get_features_for_model()
        |
        v
ResolvedFeatureSet (baseline features filtered to available)
        |
        v
[Optional] FeatureOptimizer.optimize()
        |
        v
Final feature list --> Adapter.transform()
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
