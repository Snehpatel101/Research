# Configuration Audit - 2026-01-15

## Executive Summary

**Problem:** Configuration is spread across multiple locations with hardcoded values, competing definitions, and unclear precedence rules. This makes it difficult to:
- Configure any timeframe dynamically
- Support arbitrary MTF combinations
- Implement robust per-model feature selection
- Scale the system reliably

**Goal:** Centralize all configuration into a single source of truth with clear hierarchy and zero hardcoded values in business logic.

---

## Current Config State

### Config Locations Found

| Location | Purpose | Lines | Issues |
|----------|---------|-------|---------|
| `src/phase1/pipeline_config.py` | Phase 1 pipeline config | ~400 | ✅ Well-structured dataclass |
| `src/models/config/trainer_config.py` | Phase 6 training config | ~200 | ✅ Well-structured dataclass |
| `src/common/timeframes.py` | Timeframe registry | ~488 | ✅ Canonical source for TFs |
| `src/common/horizon_config.py` | Horizon/purge/embargo | ~300+ | ⚠️ Has hardcoded defaults |
| `src/phase1/config/__init__.py` | Config aggregator | ~240 | ⚠️ Re-exports from many places |
| `src/models/config/data_requirements.py` | Model data needs | ~200+ | ✅ Good MODEL_DATA_REQUIREMENTS dict |
| `src/feature_selection/config.py` | Feature selection config | ? | ⚠️ Needs audit |
| `config/` (YAML files) | 36 YAML configs | Various | ⚠️ Not fully utilized |

### Hardcoded Values Found

#### 1. Split Ratios
**Location:** Multiple places
```python
# src/phase1/pipeline_config.py (lines 112-114)
train_ratio: float = 0.7  # Hardcoded default
val_ratio: float = 0.15
test_ratio: float = 0.15

# src/common/splits.py (assumed)
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
```
**Issue:** Hardcoded, but correctly using constants.
**Status:** ✅ Acceptable (uses constants)

#### 2. Purge/Embargo
**Location:** `src/common/horizon_config.py`
```python
# Lines 69-70
PURGE_MULTIPLIER = 3.0  # purge_bars = max_horizon * 3
EMBARGO_MULTIPLIER = 72.0  # embargo_bars = max_horizon * 72

# Lines 85-86
EMBARGO_TIME_MINUTES = 7200  # 5 days
DEFAULT_TIMEFRAME_MINUTES = 5

# Line 89
MIN_EMBARGO_BARS = 1440  # Deprecated but still defined
```
**Issue:** These should be configurable per experiment.
**Status:** ⚠️ Needs centralization

#### 3. Sequence Length
**Location:** Scattered across codebase
```python
# src/models/config/trainer_config.py (line 32)
sequence_length: int = 60  # Default for neural models

# Many test files use hardcoded seq_len=20, 30, 60
# src/models/training_utils.py line 91
dataset = container.get_pytorch_sequences(split, seq_len=60, ...)
```
**Issue:** Default of 60 is reasonable, but should be model-specific.
**Status:** ⚠️ Should be in MODEL_DATA_REQUIREMENTS

#### 4. Feature Selection Defaults
**Location:** `src/models/config/trainer_config.py`
```python
# Lines 50-53
use_feature_selection: bool = True
feature_selection_n_features: int = 50  # Hardcoded
feature_selection_method: str = "mda"
feature_selection_cv_splits: int = 5
```
**Issue:** Not per-model, not in central config.
**Status:** ❌ Needs per-model configuration

#### 5. Model Hyperparameters
**Location:** `config/models/*.yaml` (36 files)
```yaml
# config/models/xgboost.yaml
learning_rate: 0.01
max_depth: 6
n_estimators: 300
```
**Issue:** YAML files exist but may not be fully utilized.
**Status:** ⚠️ Need to verify YAML loading is robust

#### 6. Timeframe Defaults
**Location:** Multiple places
```python
# src/phase1/pipeline_config.py (line 68)
target_timeframe: str = "1min"  # Hardcoded default

# src/common/horizon_config.py (line 86)
DEFAULT_TIMEFRAME_MINUTES = 5  # Assumed 5-min bars

# Many places assume 5min as base timeframe
```
**Issue:** Should support ANY timeframe as primary.
**Status:** ⚠️ Works but needs better docs/validation

---

## Competing Configuration Issues

### Issue 1: Feature Set Naming Conflict
**Problem:** Two different concepts share the name "feature_set"

| Config | Field | Purpose | Values |
|--------|-------|---------|--------|
| `PipelineConfig` | `feature_generation` | What features to GENERATE in Phase 1 | "full", "minimal", "custom" |
| `TrainerConfig` | `feature_set` | What features to SELECT in Phase 6 | "boosting_optimal", "neural_optimal", "full" |

**Current State:** ✅ Fixed via CFG-002/CFG-003 comments in code
- Pipeline uses `feature_generation`
- Trainer uses `feature_set`
- Old `PipelineConfig.feature_set` deprecated

**Remaining Work:** Ensure all code uses new names

### Issue 2: MTF Configuration Spread
**Locations:**
1. `PipelineConfig.mtf_timeframes` - which TFs to generate
2. `PipelineConfig.mtf_mode` - "bars", "indicators", "both"
3. `PipelineConfig.output_timeframes` - which clean datasets to produce
4. `PipelineConfig.process_all_timeframes` - flag for 9-TF generation

**Issue:** Overlapping concerns, unclear precedence.

**Proposal:** Simplify to:
```python
# PipelineConfig
output_timeframes: list[str] = ["5min"]  # What TFs to produce
mtf_enrichment: bool = True  # Add MTF indicator features
mtf_indicator_timeframes: list[str] = ["1min", "15min", "60min"]  # Source TFs for MTF features
```

### Issue 3: Horizon Configuration Duality
**Problem:** Two ways to specify horizons

```python
# Method 1: Direct list
label_horizons: list[int] = [5, 10, 15, 20]

# Method 2: HorizonConfig object
horizon_config: HorizonConfig | None = None
# If set, overrides label_horizons
```

**Issue:** Redundant, confusing precedence.

**Proposal:** Keep both, but make precedence clear in docs.

---

## Missing Capabilities

### 1. Per-Model Feature Selection
**Current:** `TrainerConfig.feature_set` is global per training run.
**Needed:** Each model should specify its own feature requirements.

**Proposal:** Extend `MODEL_DATA_REQUIREMENTS`:
```python
MODEL_DATA_REQUIREMENTS = {
    "xgboost": ModelDataRequirements(
        family=ModelFamily.TABULAR,
        features="boosting_optimal",  # NEW: reference to feature set
        n_features=50,  # NEW: max features to select
        feature_selection_method="mda",  # NEW: selection method
        ...
    ),
    "lstm": ModelDataRequirements(
        family=ModelFamily.SEQUENCE,
        features="neural_optimal",
        n_features=43,
        feature_selection_method="mdi",
        sequence_length=60,  # MOVE: from TrainerConfig
        ...
    ),
}
```

### 2. Feature Optimization
**Current:** No genetic algorithm or Optuna-based feature optimization.
**Needed:** Automated feature selection optimization per model family.

**Proposal:**
```python
# New module: src/feature_selection/optimization.py
class FeatureOptimizer:
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_family: ModelFamily,
        method: str = "optuna",  # or "genetic"
        n_trials: int = 100,
    ) -> list[str]:
        """Return optimized feature list for model family."""
        ...
```

### 3. Dynamic Timeframe Validation
**Current:** Timeframes are validated but not dynamically scaled.
**Needed:** Support ANY valid timeframe with auto-scaling of horizons/purge/embargo.

**Already exists but needs better docs:**
- `get_scaled_horizons(horizons, source_tf, target_tf)`
- `compute_embargo_bars(embargo_time_minutes, target_timeframe)`

**Proposal:** Add validation helper:
```python
def validate_timeframe_config(
    target_tf: str,
    output_tfs: list[str],
    label_horizons: list[int],
) -> dict[str, Any]:
    """Validate timeframe config and return scaled parameters."""
    # Scale horizons if needed
    # Compute purge/embargo in bars for target_tf
    # Validate output_tfs are all >= target_tf
    ...
```

---

## Proposed Architecture

### Centralized Config Hierarchy

```
config/
├── global.yaml                 # Global defaults (NEW)
│   ├── timeframes:
│   │   └── default_primary: "5min"
│   │   └── canonical_ladder: [1m, 5m, 10m, ...]
│   ├── splits:
│   │   └── train: 0.7
│   │   └── val: 0.15
│   │   └── test: 0.15
│   ├── purge_embargo:
│   │   └── purge_multiplier: 3.0
│   │   └── embargo_time_minutes: 7200
│   └── random_seed: 42
│
├── features/
│   ├── feature_sets.yaml       # Feature set definitions
│   ├── selection_methods.yaml  # Feature selection config
│   └── optimization.yaml       # Feature optimization config (NEW)
│
├── models/
│   ├── {model_name}.yaml       # Per-model hyperparameters (EXISTS)
│   └── requirements.yaml       # MODEL_DATA_REQUIREMENTS in YAML (NEW)
│
├── pipeline/
│   ├── training.yaml           # TrainerConfig defaults (EXISTS)
│   └── cv.yaml                 # Cross-validation config (EXISTS)
│
└── experiments/
    ├── baseline_experiment.yaml (EXISTS)
    └── full_benchmark.yaml      (EXISTS)
```

### Loading Order (Precedence)

1. **Global defaults** (`config/global.yaml`)
2. **Model-specific** (`config/models/{model}.yaml`)
3. **Experiment config** (if provided via `--config`)
4. **CLI arguments** (highest priority)

### Python Config Classes

```python
# src/config/unified.py (NEW)
@dataclass
class GlobalConfig:
    """Global defaults for all experiments."""
    timeframes: TimeframeConfig
    splits: SplitConfig
    purge_embargo: PurgeEmbargoConfig
    random_seed: int = 42

@dataclass
class UnifiedConfig:
    """Unified configuration object loaded from hierarchy."""
    global_config: GlobalConfig
    pipeline: PipelineConfig
    trainer: TrainerConfig
    models: dict[str, ModelConfig]  # Per-model configs

    @classmethod
    def from_files(
        cls,
        global_path: Path,
        experiment_path: Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> "UnifiedConfig":
        """Load config hierarchy and merge with precedence."""
        ...
```

---

## Implementation Plan

### Phase 1: Audit & Document (Current)
- [x] Audit all config locations
- [ ] Document all hardcoded values
- [ ] Identify competing configurations
- [ ] Create this document

### Phase 2: Create Global Config (2-3 hours)
- [ ] Create `config/global.yaml`
- [ ] Create `src/config/global_config.py` dataclass
- [ ] Add loader: `load_global_config()`
- [ ] Add tests

### Phase 3: Centralize Defaults (4-5 hours)
- [ ] Move all hardcoded constants to `config/global.yaml`
- [ ] Update `horizon_config.py` to read from global
- [ ] Update `pipeline_config.py` to use global defaults
- [ ] Update `trainer_config.py` to use global defaults
- [ ] Ensure backward compatibility

### Phase 4: Per-Model Feature Selection (6-8 hours)
- [ ] Extend `ModelDataRequirements` with feature config
- [ ] Create `config/models/requirements.yaml`
- [ ] Add feature selection per model in `Trainer`
- [ ] Add tests for per-model feature selection
- [ ] Update docs

### Phase 5: Feature Optimization (8-10 hours)
- [ ] Create `src/feature_selection/optimization.py`
- [ ] Implement Optuna-based feature optimizer
- [ ] Implement genetic algorithm feature optimizer
- [ ] Add `config/features/optimization.yaml`
- [ ] Integrate with training pipeline
- [ ] Add tests

### Phase 6: Config Validation & Docs (4-5 hours)
- [ ] Create `src/config/validation.py` for unified validation
- [ ] Add config schema validation (pydantic or jsonschema)
- [ ] Update all documentation
- [ ] Add config examples for common use cases
- [ ] Add troubleshooting guide

### Phase 7: Refactor & Test (6-8 hours)
- [ ] Refactor all config loading to use unified hierarchy
- [ ] Remove deprecated config fields
- [ ] Update all tests to use new config system
- [ ] Run full test suite
- [ ] Fix any breakages

---

## Effort Estimate

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1: Audit & Document | 2-3 hours | P0 (current) |
| Phase 2: Global Config | 2-3 hours | P0 |
| Phase 3: Centralize Defaults | 4-5 hours | P0 |
| Phase 4: Per-Model Features | 6-8 hours | P1 |
| Phase 5: Feature Optimization | 8-10 hours | P1 |
| Phase 6: Validation & Docs | 4-5 hours | P1 |
| Phase 7: Refactor & Test | 6-8 hours | P0 |
| **Total** | **32-42 hours** | **4-5 days** |

---

## Success Criteria

### P0 (Must Have)
- [ ] Zero hardcoded values in business logic (all in config/)
- [ ] Single source of truth for each parameter
- [ ] Clear precedence rules documented
- [ ] All existing functionality preserved
- [ ] Full test suite passing

### P1 (Should Have)
- [ ] Per-model feature selection working
- [ ] Feature optimization framework in place
- [ ] Config validation catches errors early
- [ ] Comprehensive documentation updated

### P2 (Nice to Have)
- [ ] Config schema validation (pydantic/jsonschema)
- [ ] Config migration tool for old experiments
- [ ] Config diff tool for comparing experiments
- [ ] Config templates for common scenarios

---

**Next Steps:**
1. Review this audit with user
2. Get approval on proposed architecture
3. Start Phase 2: Create global config
