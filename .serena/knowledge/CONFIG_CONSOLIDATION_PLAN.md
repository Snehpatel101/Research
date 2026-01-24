# Config Consolidation Plan

## Executive Summary

**Current State:** 81+ config classes across 5 major systems
**Target State:** 1 canonical `PipelineConfig` at `src/core/config.py`

### Critical Issue: Name Collision

There are **TWO** classes named `PipelineConfig`:
1. **Canonical** (KEEP): `/Users/sneh/research/src/core/config.py`
2. **Legacy** (RENAME): `/Users/sneh/research/src/phase1/pipeline_config.py`

---

## 1. Canonical PipelineConfig Analysis

**Location:** `/Users/sneh/research/src/core/config.py`

### Field Categories

#### Required Fields (3)
| Field | Type | Description |
|-------|------|-------------|
| `symbol` | `str` | Trading symbol (e.g., "MES", "ES", "NQ") |
| `data_path` | `Union[str, Path]` | Path to 1-min OHLCV parquet |
| `output_dir` | `Union[str, Path]` | Where to save everything |

#### Model Configuration (3)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `models` | `List[str]` | `["xgboost", "lightgbm"]` | Models to train |
| `horizons` | `List[int]` | `[20]` | Prediction horizons |
| `build_ensemble` | `bool` | `True` | Build stacking ensemble |

#### Training Configuration (9)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training_mode` | `str` | `"standard"` | standard/walk_forward/regime_aware/meta_labeling |
| `cv_method` | `str` | `"purged_kfold"` | purged_kfold/cpcv/walk_forward/pbo |
| `n_splits` | `int` | `5` | CV folds |
| `purge_bars` | `int` | `60` | Gap before test |
| `embargo_bars` | `int` | `1440` | Embargo after test |
| `train_ratio` | `float` | `0.70` | Train split ratio |
| `val_ratio` | `float` | `0.15` | Validation split ratio |
| `test_ratio` | `float` | `0.15` | Test split ratio |
| `meta_learner` | `str` | `"ridge_meta"` | Ensemble meta-learner |

#### Meta-Labeling Configuration (3)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `meta_labeling_primary_model` | `str` | `"xgboost"` | Primary model for direction |
| `meta_labeling_meta_model` | `str` | `"logistic"` | Meta-model for bet sizing |
| `meta_labeling_threshold` | `float` | `0.5` | Min probability to trade |

#### Feature Configuration (3)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `feature_families` | `Union[str, List[str]]` | `"auto"` | Feature family selection |
| `mtf_timeframes` | `List[str]` | `["5min", "15min", "60min"]` | MTF timeframes |
| `sequence_length` | `int` | `60` | For neural models |

#### Labeling Configuration (4)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `labeling_method` | `str` | `"triple_barrier"` | Label strategy |
| `upper_mult` | `float` | `2.0` | Upper barrier = ATR * upper_mult |
| `lower_mult` | `float` | `2.0` | Lower barrier = ATR * lower_mult |
| `atr_period` | `int` | `14` | ATR calculation period |

#### Optuna Optimization (9)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimize_labels` | `bool` | `True` | Optimize labeling params |
| `label_optimization_trials` | `int` | `100` | Label optimization trials |
| `target_class_distribution` | `Optional[Dict]` | `None` | Target class balance |
| `optimize_features` | `bool` | `True` | Optimize features |
| `feature_selection_trials` | `int` | `100` | Feature selection trials |
| `feature_pruning_trials` | `int` | `50` | Feature pruning trials |
| `min_features` | `int` | `20` | Minimum features |
| `optimize_hyperparams` | `bool` | `True` | Optimize hyperparams |
| `hyperparam_trials` | `int` | `100` | Hyperparam trials |
| `optuna_random_state` | `int` | `42` | Optuna random seed |

#### Regime-Aware Configuration (6)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `regime_detection_method` | `str` | `"volatility_percentile"` | Regime detection method |
| `regime_lookback` | `int` | `60` | Lookback period |
| `n_regimes` | `int` | `3` | Number of regimes |
| `regime_volatility_window` | `int` | `20` | Volatility rolling window |
| `regime_adx_threshold` | `float` | `25.0` | ADX trend threshold |
| `regime_min_samples` | `int` | `100` | Min samples per regime |
| `train_separate_regime_models` | `bool` | `True` | Separate models per regime |

#### Neural Network Configuration (4)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `256` | Training batch size |
| `max_epochs` | `int` | `100` | Maximum training epochs |
| `learning_rate` | `float` | `0.001` | Learning rate |
| `early_stopping_patience` | `int` | `10` | Early stopping patience |

#### Output Configuration (4)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_bundle` | `bool` | `True` | Save inference bundle |
| `save_oof` | `bool` | `True` | Save OOF predictions |
| `save_models` | `bool` | `True` | Save model artifacts |
| `save_metrics` | `bool` | `True` | Save metrics JSON |

#### Misc Configuration (5)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `random_state` | `int` | `42` | Global random seed |
| `n_jobs` | `int` | `1` | Parallelism |
| `verbose` | `int` | `1` | Logging verbosity |
| `created_at` | `str` | `datetime.now()` | Creation timestamp |
| `version` | `str` | `"1.0.0"` | Config version |

**Total Canonical Fields: ~53**

---

## 2. Competing Config Systems Analysis

### 2.1 UnifiedConfig (`src/config/unified.py`)

**Purpose:** YAML-based configuration consolidation with nested sections
**Overlap Level:** HIGH

#### Unique Fields (should add to PipelineConfig):
| Field | Source Section | Rationale |
|-------|----------------|-----------|
| `run_id` | root | Unique experiment identifier |
| `description` | root | Run description |
| `start_date` / `end_date` | root | Date range filtering |
| `symbols` | root | List for batch processing |
| `device` | TrainingSection | auto/cpu/cuda/mps |
| `mixed_precision` | TrainingSection | Enable AMP |
| `num_workers` | TrainingSection | DataLoader workers |
| `pin_memory` | TrainingSection | DataLoader pin_memory |
| `calibration.enabled` | CalibrationSection | Enable calibration |
| `calibration.method` | CalibrationSection | auto/isotonic/sigmoid |
| `ga.*` | GASection | Genetic algorithm params |
| `oom_recovery.*` | OOMRecoverySection | OOM recovery settings |
| `tracking.*` | TrackingSection | MLflow/W&B settings |

#### Overlapping Fields:
- `random_seed` -> `random_state` (rename needed)
- `horizons.active` -> `horizons`
- `splits.*` -> `train_ratio`, `val_ratio`, `test_ratio`
- `purge_embargo.*` -> `purge_bars`, `embargo_bars`
- `training.sequence_length` -> `sequence_length`
- `training.batch_size` -> `batch_size`
- `training.max_epochs` -> `max_epochs`
- `training.early_stopping_patience` -> `early_stopping_patience`
- `mtf.default_timeframes` -> `mtf_timeframes`

**Recommendation:** DEPRECATE after migration. Keep YAML loading capability in PipelineConfig.

---

### 2.2 TrainerConfig (`src/models/config/trainer_config.py`)

**Purpose:** Model-specific training configuration
**Overlap Level:** MEDIUM-HIGH

#### Unique Fields (should add to PipelineConfig):
| Field | Type | Default | Rationale |
|-------|------|---------|-----------|
| `pipeline_run_id` | `str | None` | Link to pipeline run |
| `experiment_name` | `str | None` | Experiment tracking |
| `model_config` | `dict` | `{}` | Per-model hyperparams |
| `evaluate_test_set` | `bool` | `True` | Run test evaluation |
| `feature_selection_n_features` | `int` | `50` | Feature selection count |
| `deterministic_mode` | `bool` | `False` | Reproducibility mode |
| `nan_check_raise_error` | `bool` | `True` | NaN handling |
| `checkpoint_interval` | `int` | `10` | Checkpoint frequency |
| `keep_n_checkpoints` | `int` | `3` | Checkpoint retention |
| `checkpoint_dir` | `str | None` | Checkpoint directory |
| `primary_timeframe` | `str` | `"5min"` | Model's timeframe |
| `mtf_mode` | `str` | `"indicators"` | MTF mode |
| `feature_mode` | `str` | `"engineered"` | Feature mode |
| `adapter_id` | `str | None` | Input adapter |
| `input_rank` | `int` | `2` | Tensor rank |
| `min_features` / `max_features` | `int` | `4` / `200` | Feature bounds |

#### Usage Pattern:
- Created **per model** during training
- Gets values from `UnifiedConfig.to_trainer_config()`
- Has `from_model_contract()` factory

**Recommendation:** Keep as internal adapter. TrainerConfig should be DERIVED from PipelineConfig, not a separate entity.

---

### 2.3 MLConfig (`src/ml_pipeline/config.py`)

**Purpose:** Unified ML pipeline configuration
**Overlap Level:** VERY HIGH (near duplicate)

#### Unique Fields:
| Field | Type | Default | Rationale |
|-------|------|---------|-----------|
| `output_timeframes` | `list[str] | None` | Output TF list |
| `process_all_timeframes` | `bool` | `False` | MTF-P1-002 flag |
| `feature_mode` | `str` | `"auto"` | auto/full/minimal/hft_only |
| `enable_wavelets` | `bool` | `True` | Wavelet features |
| `enable_microstructure` | `bool` | `True` | Microstructure features |
| `enable_volume` | `bool` | `True` | Volume features |
| `enable_volatility` | `bool` | `True` | Volatility features |
| `enable_mtf` | `bool` | `True` | Enable MTF |
| `k_up` / `k_down` | `float | None` | Barrier params |
| `max_bars` | `int | None` | Max bars for barrier |
| `optimize_labeling` | `bool` | `True` | Optimize labeling |
| `labeling_optimization_trials` | `int` | `100` | Labeling trials |
| `walk_forward_*` | various | | Walk-forward params |
| `meta_labeling_base_model` | `str | None` | Meta-labeling base |
| `evaluation_methods` | `list[str]` | `["cv"]` | Evaluation methods |
| `cpcv_pbo_*` | various | | CPCV-PBO params |
| `global_feature_optimization` | `bool` | `False` | Global feat opt |
| `global_hyperparam_optimization` | `bool` | `False` | Global HP opt |
| `parallel_training` | `bool` | `False` | Parallel training |
| `validate_pipeline_output` | `bool` | `True` | Output validation |

**Recommendation:** DEPRECATE. This is essentially a duplicate of PipelineConfig with some extensions that should be merged.

---

### 2.4 Phase1 PipelineConfig (`src/phase1/pipeline_config.py`) - NAME COLLISION

**Purpose:** Phase 1 data pipeline configuration
**Overlap Level:** HIGH (different purpose, same name)

#### Unique Fields:
| Field | Type | Default | Rationale |
|-------|------|---------|-----------|
| `bar_resolution` | `str` | `None` | Legacy alias |
| `feature_generation` | `str` | `"full"` | Feature generation mode |
| `sma_periods` / `ema_periods` / `atr_periods` | `list[int]` | various | Indicator periods |
| `rsi_period` | `int` | `14` | RSI period |
| `macd_params` | `dict` | `{fast, slow, signal}` | MACD settings |
| `bb_period` / `bb_std` | various | `20` / `2.0` | Bollinger params |
| `max_bars_ahead` | `int` | `50` | Max lookahead |
| `auto_scale_purge_embargo` | `bool` | `True` | Auto-scale splits |
| `ga_*` | various | | GA params |
| `allow_batch_symbols` | `bool` | `False` | Multi-symbol batch |
| `feature_toggles` | `dict | None` | Feature toggles |
| `barrier_overrides` | `dict | None` | Barrier overrides |
| `scaler_type` | `str` | `"robust"` | Scaler type |
| `project_root` | `Path` | auto | Project root path |
| `horizon_config` | `HorizonConfig | None` | Horizon config object |

#### Special Behavior:
- Mixes in `PipelinePathMixin` and `PipelinePersistenceMixin`
- Has `__deepcopy__` and `__getstate__`/`__setstate__` for run_id preservation
- Uses `process_all_timeframes` for MTF-P1-002

**Recommendation:** RENAME to `Phase1Config` or `DataPipelineConfig` to resolve name collision.

---

### 2.5 GlobalConfig (`src/config/global_config.py`)

**Purpose:** YAML-loaded global defaults
**Contains:** 17 nested dataclasses

#### Structure:
```
GlobalConfig
├── TimeframeConfig
├── SplitConfig
├── PurgeEmbargoConfig
├── HorizonsConfig
├── FeaturesConfig
│   ├── FeatureSelectionConfig
│   └── FeatureGenerationConfig
├── MTFConfig
├── TrainingConfig
├── CalibrationConfig
├── OptimizationConfig
│   ├── GAConfig
│   └── OptunaConfig
├── CrossValidationConfig
├── ProcessingConfig
├── ScalerConfig
├── TrackingConfig
└── OOMRecoveryConfig
```

**Recommendation:** KEEP as YAML source. PipelineConfig should read defaults from GlobalConfig but be self-contained.

---

## 3. Additional Config Classes Found

| File | Config Classes | Purpose | Action |
|------|----------------|---------|--------|
| `src/training/config.py` | `ModelConfig`, `ExperimentConfig` | Training experiment | DEPRECATE |
| `src/feature_selection/config.py` | `FeatureSelectionConfig`, `FeatureSelectorConfig`, `ModelFamilyDefaults` | Feature selection | MERGE |
| `src/common/horizon_config.py` | `HorizonConfig` | Horizon management | EMBED |
| `src/phase1/config/barriers_config.py` | (module-level dicts) | Barrier params | MERGE |
| `src/phase1/config/labeling_config.py` | `LabelingStrategyType` | Labeling strategies | MERGE |
| `src/phase1/config/regime_config.py` | (regime configs) | Regime settings | MERGE |
| `src/models/config/per_model_config.py` | `PerModelConfig`, `EnsemblePlan` | Heterogeneous ensemble | KEEP separate |

---

## 4. Field Consolidation Matrix

### Fields to ADD to Canonical PipelineConfig:

| Field | Source | Priority | Notes |
|-------|--------|----------|-------|
| `run_id` | UnifiedConfig | P0 | Auto-generated unique ID |
| `description` | UnifiedConfig | P1 | Run description |
| `start_date` / `end_date` | Phase1Config | P1 | Date filtering |
| `symbols` | Phase1Config | P1 | Batch symbol list |
| `device` | TrainerConfig | P1 | Training device |
| `mixed_precision` | TrainerConfig | P1 | AMP support |
| `num_workers` | TrainerConfig | P2 | DataLoader workers |
| `pin_memory` | TrainerConfig | P2 | DataLoader option |
| `feature_generation` | Phase1Config | P1 | full/minimal mode |
| `scaler_type` | Phase1Config | P1 | Scaler choice |
| `calibration_enabled` | UnifiedConfig | P1 | Enable calibration |
| `calibration_method` | UnifiedConfig | P2 | Calibration method |
| `tracking_enabled` | UnifiedConfig | P1 | Enable tracking |
| `tracking_backend` | UnifiedConfig | P2 | mlflow/wandb/local |
| `tracking_uri` | TrainerConfig | P2 | Tracking server URI |
| `tracking_tags` | TrainerConfig | P3 | Custom tags |
| `oom_recovery_enabled` | UnifiedConfig | P2 | OOM recovery |
| `oom_max_retries` | UnifiedConfig | P2 | OOM retry count |
| `checkpoint_interval` | TrainerConfig | P2 | Checkpoint frequency |
| `keep_n_checkpoints` | TrainerConfig | P3 | Checkpoint retention |
| `deterministic_mode` | TrainerConfig | P2 | Reproducibility |
| `parallel_training` | MLConfig | P2 | Parallel execution |
| `evaluate_test_set` | TrainerConfig | P2 | Test evaluation |
| `walk_forward_window_size` | MLConfig | P2 | Walk-forward param |
| `walk_forward_step_size` | MLConfig | P2 | Walk-forward param |
| `evaluation_methods` | MLConfig | P2 | ["cv", "walk_forward", "cpcv_pbo"] |
| `allow_batch_symbols` | Phase1Config | P3 | Multi-symbol flag |

### Fields to RENAME:

| Old Field | New Field | Source |
|-----------|-----------|--------|
| `random_seed` (UnifiedConfig) | `random_state` | Standardize |
| `learning_rate` | `initial_learning_rate` | Clarity |

---

## 5. Configs to Deprecate

### Immediate Deprecation (After Migration):

| Config | Location | Replacement |
|--------|----------|-------------|
| `UnifiedConfig` | `src/config/unified.py` | `PipelineConfig` + YAML loader |
| `MLConfig` | `src/ml_pipeline/config.py` | `PipelineConfig` |
| `ExperimentConfig` | `src/training/config.py` | `PipelineConfig` |

### Requires Rename:

| Config | Location | New Name |
|--------|----------|----------|
| `PipelineConfig` | `src/phase1/pipeline_config.py` | `Phase1DataConfig` |

### Keep But Derive From PipelineConfig:

| Config | Location | Usage |
|--------|----------|-------|
| `TrainerConfig` | `src/models/config/trainer_config.py` | Per-model adapter |
| `PerModelConfig` | `src/models/config/per_model_config.py` | Ensemble planning |
| `EnsemblePlan` | `src/models/config/per_model_config.py` | Ensemble planning |
| `HorizonConfig` | `src/common/horizon_config.py` | Horizon utilities |

### Keep as Static Configuration:

| Config | Location | Usage |
|--------|----------|-------|
| `GlobalConfig` | `src/config/global_config.py` | YAML defaults |
| `FeatureSelectionConfig` | `src/feature_selection/config.py` | Feature selection |
| barrier dicts | `src/phase1/config/barriers_config.py` | Symbol-specific |
| labeling enums | `src/phase1/config/labeling_config.py` | Strategy types |

---

## 6. Name Collision Resolution

### Problem:
Two classes named `PipelineConfig`:
1. `src/core/config.py` - Canonical (new, intended unified config)
2. `src/phase1/pipeline_config.py` - Legacy Phase 1 config

### Solution:

**Step 1: Rename Legacy**
```python
# src/phase1/pipeline_config.py
# BEFORE
class PipelineConfig(PipelinePathMixin, PipelinePersistenceMixin):

# AFTER
class Phase1DataConfig(PipelinePathMixin, PipelinePersistenceMixin):
    """Configuration for Phase 1 data pipeline."""

# Add deprecation alias
PipelineConfig = Phase1DataConfig  # Deprecated, use Phase1DataConfig

def __getattr__(name):
    if name == "PipelineConfig":
        import warnings
        warnings.warn(
            "PipelineConfig from src.phase1 is deprecated. "
            "Use Phase1DataConfig or src.core.config.PipelineConfig",
            DeprecationWarning,
            stacklevel=2,
        )
        return Phase1DataConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Step 2: Update Imports**
Search and replace across codebase:
```python
# Find
from src.phase1.pipeline_config import PipelineConfig

# Replace with
from src.phase1.pipeline_config import Phase1DataConfig
# OR (for new code)
from src.core.config import PipelineConfig
```

**Step 3: Import Resolution**
Add explicit re-export in `src/__init__.py`:
```python
# Canonical config
from src.core.config import PipelineConfig

# Legacy (deprecated)
from src.phase1.pipeline_config import Phase1DataConfig
```

---

## 7. Migration Steps

### Phase 1: Preparation (No Breaking Changes)
1. [ ] Add missing fields to canonical `PipelineConfig` (P0 and P1 priority)
2. [ ] Add `run_id` auto-generation to canonical `PipelineConfig`
3. [ ] Add YAML `load()` classmethod to canonical `PipelineConfig`
4. [ ] Add conversion methods: `to_trainer_config()`, `to_phase1_config()`

### Phase 2: Name Collision Resolution
1. [ ] Rename `src/phase1/pipeline_config.py::PipelineConfig` to `Phase1DataConfig`
2. [ ] Add deprecation warning for old import
3. [ ] Update all Phase 1 internal imports
4. [ ] Run tests, fix failures

### Phase 3: Config Delegation
1. [ ] Update `MLFactory` to accept only canonical `PipelineConfig`
2. [ ] Update `UnifiedConfig.to_pipeline_config()` to return canonical version
3. [ ] Add `PipelineConfig.from_unified()` factory method
4. [ ] Add deprecation warnings to competing configs

### Phase 4: Deprecation Period
1. [ ] Add deprecation warnings to `UnifiedConfig`, `MLConfig`, `ExperimentConfig`
2. [ ] Update documentation to use canonical `PipelineConfig`
3. [ ] Run full test suite, fix warnings

### Phase 5: Removal (Future)
1. [ ] Remove deprecated config classes
2. [ ] Remove conversion methods
3. [ ] Final cleanup

---

## 8. Target State

### Single Entry Point:
```python
from src.core.config import PipelineConfig

config = PipelineConfig(
    symbol="MES",
    data_path="./data/mes.parquet",
    output_dir="./experiments/exp_001",
    models=["xgboost", "lightgbm", "lstm"],
    horizons=[20],
    training_mode="walk_forward",
    build_ensemble=True,
    optimize_labels=True,
    optimize_features=True,
    optimize_hyperparams=True,
    device="auto",
    tracking_enabled=True,
)

# Factory handles everything
pipeline = MLFactory(config)
results = pipeline.run()
```

### Derived Configs (Internal Only):
```python
# Internal: Created by MLFactory as needed
trainer_config = config.to_trainer_config(model_name="xgboost")
phase1_config = config.to_phase1_config()
```

### YAML Support Preserved:
```python
# Load from YAML (for backwards compatibility)
config = PipelineConfig.from_yaml("config/global.yaml", overrides={
    "symbol": "MES",
    "models": ["xgboost"],
})
```

---

## 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing notebooks | HIGH | Deprecation period with warnings |
| Import path confusion | MEDIUM | Clear documentation, __init__.py exports |
| Missing fields during migration | MEDIUM | Comprehensive field audit (this doc) |
| Test failures | LOW-MEDIUM | Incremental migration with CI |
| Serialization incompatibility | LOW | Version field in config JSON |

---

## 10. Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Preparation | 2-3 days | None |
| Phase 2: Name Resolution | 1-2 days | Phase 1 |
| Phase 3: Config Delegation | 2-3 days | Phase 2 |
| Phase 4: Deprecation Period | 2+ weeks | Phase 3 |
| Phase 5: Removal | 1 day | Phase 4 complete |

**Total:** ~3-4 weeks including deprecation period

---

## Appendix A: Full Field Inventory by Config Class

### Canonical PipelineConfig (53 fields)
- Required: symbol, data_path, output_dir
- Model: models, horizons, build_ensemble, meta_learner
- Training: training_mode, cv_method, n_splits, purge_bars, embargo_bars, train_ratio, val_ratio, test_ratio
- Meta-labeling: meta_labeling_primary_model, meta_labeling_meta_model, meta_labeling_threshold
- Features: feature_families, mtf_timeframes, sequence_length
- Labeling: labeling_method, upper_mult, lower_mult, atr_period
- Optuna: optimize_labels, label_optimization_trials, target_class_distribution, optimize_features, feature_selection_trials, feature_pruning_trials, min_features, optimize_hyperparams, hyperparam_trials, optuna_random_state
- Regime: regime_detection_method, regime_lookback, n_regimes, regime_volatility_window, regime_adx_threshold, regime_min_samples, train_separate_regime_models
- Neural: batch_size, max_epochs, learning_rate, early_stopping_patience
- Output: save_bundle, save_oof, save_models, save_metrics
- Misc: random_state, n_jobs, verbose, created_at, version

### UnifiedConfig (est. 80+ fields via sections)
### TrainerConfig (43 fields)
### MLConfig (55 fields)
### Phase1 PipelineConfig (38 fields)
### GlobalConfig (17 nested configs, ~50 leaf fields)

---

*Document created: 2026-01-20*
*Author: Config Consolidation Analysis*
*Updated: 2026-01-23 (Phase 1 Analysis integration)*

---

## Phase 1 Configuration System Analysis (2026-01-23)

### Configuration Hierarchy (From Phase 1 Agent #3)

The configuration system has a 5-level hierarchy:
1. **Level 1:** GlobalConfig (YAML-based, application-wide)
2. **Level 2:** UnifiedConfig (high-level API with sections)
3. **Level 3:** PipelineConfig (user-facing, Phase 0)
4. **Level 4:** DataConfig (internal Phase 1)
5. **Level 5:** Stage-specific (barriers, feature sets, regimes)

### Key Modules Identified

| Module | Purpose |
|--------|---------|
| `runtime.py` | Paths, symbols, split ratios |
| `barriers_config.py` | Symbol-specific barrier defaults |
| `feature_sets/` | 50+ model-specific feature definitions |
| `regime_config.py` | Volatility/trend adaptive adjustments |
| `adaptive_costs.py` | Time/volume transaction costs |

### Feature Sets (50+ Aliases)

- `core_min` - 30-50 base features
- `boosting_optimal` - XGBoost/LightGBM optimized
- `neural_optimal` - LSTM/GRU normalized
- `transformer_raw` - Minimal for foundation models

### Critical Issues Confirmed

1. **CFG-001:** Dual hierarchy confusion (UnifiedConfig vs PipelineConfig)
2. **CFG-002:** 71+ duplicated `_get_global_or_default()` patterns
3. **CFG-005:** No cross-config validation
4. **CFG-010:** Constants scattered across multiple locations

### Phase 1 Recommendations

1. Migrate all `_get_global_or_default()` to `get_config_value()`
2. Add cross-config validation (CompositeValidator)
3. Clarify config hierarchy in documentation
4. Create feature sets reference guide
