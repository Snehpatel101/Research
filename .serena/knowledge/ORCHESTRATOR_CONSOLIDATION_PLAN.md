# Orchestrator Consolidation Plan

**Created:** 2026-01-20
**Status:** Analysis Complete
**Target State:** 5 orchestrators -> 3 orchestrators (Training, Ensemble, Inference)

---

## Executive Summary

The ML Factory currently has **5 orchestrators** managing different aspects of the pipeline. This document analyzes each orchestrator and recommends a consolidation path to reach the target of **3 orchestrators**.

### Target Architecture

| Orchestrator | Purpose | Status |
|-------------|---------|--------|
| **UnifiedTrainingOrchestrator** | All training modes (standard, walk-forward, regime-aware, meta-labeling) | KEEP - Primary |
| **EnsembleOrchestrator** | Ensemble/stacking training with OOF alignment | KEEP |
| **InferenceOrchestrator** | All inference operations | KEEP |

### Orchestrators to Deprecate/Merge

| Orchestrator | Action | Merge Into |
|-------------|--------|------------|
| **TrainingOrchestrator** (Legacy) | DEPRECATE | UnifiedTrainingOrchestrator |
| **CVOrchestrator** | DEPRECATE | Use underlying CV classes directly |

---

## Detailed Analysis

### 1. TrainingOrchestrator (Legacy)

**File:** `/Users/sneh/research/src/training/orchestrator.py`
**Class:** `TrainingOrchestrator`
**Lines of Code:** ~612

#### Capabilities

- Data loading from parquet directories
- Feature selection (mode-based filtering)
- Single model training via `Trainer`
- Multi-model training (iterating over model configs)
- Hyperparameter optimization (Optuna via `TimeSeriesOptunaTuner`)
- Feature optimization
- Ensemble building (stacking)
- Results saving (JSON + model pickle)

#### Configuration Interface

Accepts **two configuration formats**:
1. `ExperimentConfig` dataclass (newer)
2. Dict-based config (legacy YAML structure)

```python
# ExperimentConfig usage
config = ExperimentConfig(
    symbol="MES",
    horizons=[20],
    models=[ModelConfig(name="xgboost")],
    ...
)
orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()

# Dict-based usage
config = {
    'experiment': {'name': 'test'},
    'data': {'horizons': [20], 'data_dir': '...'},
    'models': {'model_list': [{'name': 'xgboost'}]},
    'features': {'mode': 'full'},
    'ensemble': {'enabled': True, 'method': 'stacking'}
}
orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()
```

#### Key Dependencies

- `src.models.Trainer`, `TrainerConfig`
- `src.phase1.stages.datasets.TimeSeriesDataContainer`
- `src.cross_validation.cv_tuner.TimeSeriesOptunaTuner`
- `src.features.optimization.optimize_features_for_model`

#### Files Importing This Orchestrator

```
docs/implementation/UNIFIED_PIPELINE_ARCHITECTURE.md
docs/implementation/UNIFIED_TRAINING_SYSTEM.md
src/factory.py
src/__init__.py
src/training/__init__.py
src/training/unified_orchestrator.py
MIGRATION_PLANS/00_MASTER_SUMMARY.md
MIGRATION_PLANS/PHASE_5_IMPLEMENTATION.md
MIGRATION_PLANS/PHASE_4_IMPLEMENTATION.md
MIGRATION_PLANS/PHASE_3_IMPLEMENTATION.md
scripts/phase3_validation.py
Not done yet/PHASE_4_META_LEARNERS.md
REFACTORING_PROPOSAL.md
src/ml_pipeline/unified.py
src/config/smart_config.py
notebooks/unified_training_colab.ipynb
```

#### Recommendation: **DEPRECATE**

**Rationale:**
- `UnifiedTrainingOrchestrator` is the PHASE_3 recommended replacement
- All functionality is duplicated in the newer orchestrator
- Legacy config format creates maintenance burden
- The `__init__.py` already marks this as "legacy"

---

### 2. UnifiedTrainingOrchestrator (PHASE_3 - RECOMMENDED)

**File:** `/Users/sneh/research/src/training/unified_orchestrator.py`
**Class:** `UnifiedTrainingOrchestrator`
**Lines of Code:** ~1600

#### Capabilities

- **All training modes supported:**
  - `standard`: PurgedKFold CV with OOF generation
  - `walk_forward`: Expanding/rolling window training
  - `regime_aware`: Separate models per market regime
  - `meta_labeling`: Lopez de Prado 2018 bet sizing

- **Full PHASE_2 adapter integration:**
  - `UnifiedDataPreparation` for model-specific data prep
  - Handles 2D (tabular), 3D (sequence), and 4D (multi-stream) data

- **OOF prediction generation and alignment:**
  - `OOFGenerator` integration
  - `OOFAligner` for heterogeneous ensembles

- **Ensemble building:**
  - Integrated with PHASE_4 EnsembleOrchestrator
  - Stacking dataset creation

- **Comprehensive results:**
  - `TrainingRunResult` dataclass with all outputs
  - JSON serialization, model saving

#### Configuration Interface

Uses **PipelineConfig ONLY** from `src/core`:

```python
from src.core import PipelineConfig
from src.training import UnifiedTrainingOrchestrator

config = PipelineConfig(
    symbol="MES",
    data_path="./data/mes.parquet",
    output_dir="./experiments/exp_001",
    models=["xgboost", "lightgbm", "lstm"],
    training_mode="standard",  # or "walk_forward", "regime_aware", "meta_labeling"
    build_ensemble=True,
)

orchestrator = UnifiedTrainingOrchestrator(config)
result = orchestrator.train(df)
```

#### Key Dependencies

- `src.core.PipelineConfig`, `TrainingMode`, `CVMethod`
- `src.adapters.UnifiedDataPreparation`, `OOFAligner`, `PreparedData`
- `src.cross_validation.OOFGenerator`, `PurgedKFold`, `TimeSeriesOptunaTuner`
- `src.models.Trainer`, `TrainerConfig`
- `src.training.modes.WalkForwardTrainer`
- `src.training.regime_trainer.RegimeAwareTrainer`

#### Files Importing This Orchestrator

```
src/factory.py
src/__init__.py
src/training/__init__.py
src/training/unified_orchestrator.py
src/models/ensemble/orchestrator.py
MIGRATION_PLANS/00_MASTER_SUMMARY.md
MIGRATION_PLANS/PHASE_5_IMPLEMENTATION.md
MIGRATION_PLANS/PHASE_4_IMPLEMENTATION.md
MIGRATION_PLANS/PHASE_3_IMPLEMENTATION.md
scripts/phase3_validation.py
scripts/validate_phase5_inference.py
src/inference/builder.py
```

#### Recommendation: **KEEP - Primary Training Orchestrator**

**Rationale:**
- Most comprehensive training implementation
- Unified PipelineConfig interface (single source of truth)
- Full adapter integration (PHASE_2)
- Supports all training modes
- Properly marked as PHASE_3 recommended

---

### 3. CVOrchestrator (CV Specialized)

**File:** `/Users/sneh/research/src/cross_validation/cv_orchestrator.py`
**Class:** `CVOrchestrator`
**Lines of Code:** ~625

#### Capabilities

- Unified interface to all CV methods:
  - `PurgedKFold`: Standard purged k-fold
  - `CPCV`: Combinatorial Purged Cross-Validation
  - `WalkForward`: Expanding/rolling window
  - `PBO`: Probability of Backtest Overfitting

- Split generation (`split()` method)
- Split info reporting (`get_split_info()`)
- PBO computation (`compute_pbo()`)
- Coverage validation (`validate_coverage()`)

#### Configuration Interface

Can be created from `PipelineConfig` or directly:

```python
from src.core import PipelineConfig
from src.cross_validation import CVOrchestrator

# From config
config = PipelineConfig(...)
cv_orch = CVOrchestrator.from_config(config)

# Direct instantiation
cv_orch = CVOrchestrator(
    cv_method=CVMethod.PURGED_KFOLD,
    n_splits=5,
    purge_bars=60,
    embargo_bars=1440,
)

# Iterate splits
for train_idx, test_idx in cv_orch.split(X, y):
    model.fit(X[train_idx], y[train_idx])
```

#### Key Dependencies

- `src.core.CVMethod`, `PipelineConfig`, `MODEL_DATA_RANKS`
- `src.cross_validation.purged_kfold.PurgedKFold`
- `src.cross_validation.cpcv.CombinatorialPurgedCV`
- `src.cross_validation.walk_forward.WalkForwardEvaluator`
- `src.cross_validation.pbo.compute_pbo`

#### Files Importing This Orchestrator

```
TEST_COVERAGE_ANALYSIS.md
src/cross_validation/__init__.py
src/cross_validation/cv_orchestrator.py
```

#### Recommendation: **DEPRECATE**

**Rationale:**
- **Very low adoption:** Only 3 files reference it (one is itself)
- **Thin wrapper:** The underlying CV classes (`PurgedKFold`, `CPCV`, `WalkForwardEvaluator`) are already well-designed and directly usable
- **Redundant with UnifiedTrainingOrchestrator:** The unified orchestrator already creates CV internally via `_create_cv()`
- **Not actually orchestrating:** This is really a CV factory/adapter, not an orchestrator

**Alternative Approach:**
- Keep the `get_cv_for_model()` and `create_cv_orchestrator()` factory functions as standalone utilities
- Deprecate the `CVOrchestrator` class itself
- Direct users to use `PurgedKFold`, `CombinatorialPurgedCV`, or `WalkForwardEvaluator` directly

---

### 4. InferenceOrchestrator (PHASE_5 - RECOMMENDED)

**File:** `/Users/sneh/research/src/inference/orchestrator.py`
**Class:** `InferenceOrchestrator`
**Lines of Code:** ~842

#### Capabilities

- **Bundle loading:**
  - From experiment directory (`from_experiment()`)
  - From single bundle (`from_bundle()`)
  - From multiple bundles (`from_bundles()`)
  - From TrainingRunResult (`from_training_result()`)

- **Prediction methods:**
  - Single model: `predict(X, model_name="xgboost")`
  - Ensemble: `predict(X)` when ensemble loaded
  - All models: `predict_all(X)`
  - Raw OHLCV: `predict_from_raw(raw_df)`
  - Batch: `predict_batch(data, batch_size=10000)`
  - With uncertainty: `predict_with_uncertainty(X)`

- **PreprocessingGraph integration:**
  - End-to-end inference from raw OHLCV data

#### Configuration Interface

```python
from src.core import PipelineConfig
from src.inference import InferenceOrchestrator

# From experiment
config = PipelineConfig.load("./experiments/exp_001/config.json")
orchestrator = InferenceOrchestrator.from_experiment(config)
result = orchestrator.predict(X_new)

# From bundle
orchestrator = InferenceOrchestrator.from_bundle("./bundles/xgb_h20")
result = orchestrator.predict(X_new)

# End-to-end from raw OHLCV
result = orchestrator.predict_from_raw(raw_ohlcv_df)
```

#### Key Dependencies

- `src.core.PipelineConfig`
- `src.inference.bundle.ModelBundle`
- `src.inference.ensemble_bundle.EnsembleBundle`
- `src.models.base.PredictionOutput`
- `src.models.ensemble.*` meta-learners

#### Files Importing This Orchestrator

```
src/__init__.py
MIGRATION_PLANS/00_MASTER_SUMMARY.md
MIGRATION_PLANS/PHASE_5_IMPLEMENTATION.md
scripts/validate_phase5_inference.py
src/inference/__init__.py
src/inference/orchestrator.py
```

#### Recommendation: **KEEP**

**Rationale:**
- Only inference orchestrator in the system
- Clean, well-designed interface
- Proper integration with bundles, ensembles, and preprocessing
- Marked as PHASE_5 recommended

---

### 5. EnsembleOrchestrator (PHASE_4 - RECOMMENDED)

**File:** `/Users/sneh/research/src/models/ensemble/orchestrator.py`
**Class:** `EnsembleOrchestrator`
**Lines of Code:** ~733

#### Capabilities

- **OOF-based ensemble training:**
  - Converts OOFPrediction to OOFResult format
  - Uses OOFAligner for heterogeneous model alignment
  - Builds stacking dataset from aligned predictions

- **Meta-learner support:**
  - `ridge_meta`: Ridge regression
  - `mlp_meta`: Multi-layer perceptron
  - `xgboost_meta`: XGBoost gradient boosting
  - `calibrated_meta`: Isotonic/Platt calibration

- **Integration with PHASE_3:**
  - `train_from_training_result()` method

- **Prediction methods:**
  - `predict(base_predictions)`
  - `predict_proba(base_predictions)`

#### Configuration Interface

```python
from src.core import PipelineConfig
from src.models.ensemble import EnsembleOrchestrator

config = PipelineConfig(
    symbol="MES",
    data_path="./data/mes.parquet",
    output_dir="./experiments/exp_001",
    models=["xgboost", "lightgbm", "lstm"],
    build_ensemble=True,
    meta_learner="ridge_meta",
)

orchestrator = EnsembleOrchestrator(config)
result = orchestrator.train(oof_predictions, y_train)

# Or from training result
result = orchestrator.train_from_training_result(training_run_result)
```

#### Key Dependencies

- `src.core.PipelineConfig`, `OOFResult`, `MODEL_DATA_RANKS`
- `src.adapters.OOFAligner`, `AlignedOOFResult`
- `src.cross_validation.OOFPrediction`, `StackingDataset`
- `src.models.ensemble.*` meta-learners

#### Files Importing This Orchestrator

```
src/models/ensemble/orchestrator.py
MIGRATION_PLANS/PHASE_4_IMPLEMENTATION.md
scripts/validate_phase5_inference.py
tests/phase4_validation.py
src/inference/ensemble_bundle.py
src/inference/builder.py
```

#### Recommendation: **KEEP**

**Rationale:**
- Specialized ensemble logic deserves its own orchestrator
- Complex OOF alignment and stacking dataset building
- Different lifecycle from training (trains AFTER base models)
- Clean separation of concerns

---

## Consolidation Plan

### Phase 1: Deprecation Warnings (Week 1-2)

1. **Add deprecation warnings to TrainingOrchestrator:**
   ```python
   import warnings

   class TrainingOrchestrator:
       def __init__(self, config):
           warnings.warn(
               "TrainingOrchestrator is deprecated and will be removed in v2.0. "
               "Use UnifiedTrainingOrchestrator from src.training instead.",
               DeprecationWarning,
               stacklevel=2
           )
   ```

2. **Add deprecation warnings to CVOrchestrator:**
   ```python
   class CVOrchestrator:
       def __init__(self, ...):
           warnings.warn(
               "CVOrchestrator is deprecated. Use PurgedKFold, CombinatorialPurgedCV, "
               "or WalkForwardEvaluator directly from src.cross_validation.",
               DeprecationWarning,
               stacklevel=2
           )
   ```

3. **Update documentation** to point to recommended orchestrators.

### Phase 2: Migration (Week 2-4)

1. **Migrate all usages in production code:**
   - `src/factory.py` - Update to use only UnifiedTrainingOrchestrator
   - `src/ml_pipeline/unified.py` - Update imports
   - `src/config/smart_config.py` - Update to generate PipelineConfig

2. **Update notebooks:**
   - `notebooks/unified_training_colab.ipynb` - Use new API

3. **Update validation scripts:**
   - `scripts/phase3_validation.py` - Remove legacy references

### Phase 3: Removal (Week 4-6)

1. **Remove deprecated code:**
   - Move `TrainingOrchestrator` to `src/training/_deprecated/`
   - Move `CVOrchestrator` to `src/cross_validation/_deprecated/`

2. **Update `__init__.py` exports:**
   - Remove from `__all__` lists
   - Keep imports but with deprecation wrapper

3. **Update migration guides:**
   - Document API differences
   - Provide code transformation examples

---

## Migration Guide

### From TrainingOrchestrator to UnifiedTrainingOrchestrator

#### Before (Legacy):
```python
from src.training import TrainingOrchestrator

config = {
    'experiment': {'name': 'my_experiment'},
    'data': {
        'data_dir': './data/mes/5min',
        'horizons': [20, 60]
    },
    'models': {
        'model_list': [
            {'name': 'xgboost', 'optimize_hyperparams': True},
            {'name': 'lightgbm', 'optimize_hyperparams': True}
        ]
    },
    'features': {'mode': 'full'},
    'ensemble': {'enabled': True, 'method': 'stacking', 'meta_learner': 'ridge_meta'}
}

orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()
```

#### After (Recommended):
```python
from src.core import PipelineConfig
from src.training import UnifiedTrainingOrchestrator

config = PipelineConfig(
    symbol="MES",
    data_path="./data/mes_1min.parquet",
    output_dir="./experiments/my_experiment",
    horizons=[20, 60],
    models=["xgboost", "lightgbm"],
    training_mode="standard",
    optimize_hyperparams=True,
    build_ensemble=True,
    meta_learner="ridge_meta",
)

orchestrator = UnifiedTrainingOrchestrator(config)
result = orchestrator.train(df)

# Access results
print(result.best_model)
print(result.get_metrics_summary())
```

### From CVOrchestrator to Direct CV Usage

#### Before:
```python
from src.cross_validation import CVOrchestrator
from src.core import CVMethod

cv_orch = CVOrchestrator(
    cv_method=CVMethod.PURGED_KFOLD,
    n_splits=5,
    purge_bars=60,
    embargo_bars=1440,
)

for train_idx, test_idx in cv_orch.split(X, y):
    model.fit(X[train_idx], y[train_idx])
```

#### After (Direct CV usage):
```python
from src.cross_validation import PurgedKFold, PurgedKFoldConfig

cv_config = PurgedKFoldConfig(
    n_splits=5,
    purge_bars=60,
    embargo_bars=1440,
)
cv = PurgedKFold(cv_config)

for train_idx, test_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
```

#### After (Model-aware CV):
```python
from src.cross_validation import get_cv_for_model
from src.core import PipelineConfig

config = PipelineConfig(...)

# Automatically adjusts n_splits for sequence models
cv = get_cv_for_model("lstm", config)
```

---

## Summary Table

| Orchestrator | File | Lines | Recommendation | Justification |
|-------------|------|-------|----------------|---------------|
| TrainingOrchestrator | `/src/training/orchestrator.py` | ~612 | **DEPRECATE** | Superseded by UnifiedTrainingOrchestrator |
| UnifiedTrainingOrchestrator | `/src/training/unified_orchestrator.py` | ~1600 | **KEEP** | PHASE_3 recommended, full feature set |
| CVOrchestrator | `/src/cross_validation/cv_orchestrator.py` | ~625 | **DEPRECATE** | Thin wrapper, low adoption, use CV classes directly |
| InferenceOrchestrator | `/src/inference/orchestrator.py` | ~842 | **KEEP** | PHASE_5 recommended, only inference orchestrator |
| EnsembleOrchestrator | `/src/models/ensemble/orchestrator.py` | ~733 | **KEEP** | Specialized ensemble logic, clean separation |

---

## Final Target State

```
src/
  training/
    unified_orchestrator.py      # UnifiedTrainingOrchestrator (KEEP)
    orchestrator.py              # TrainingOrchestrator (DEPRECATE -> _deprecated/)

  cross_validation/
    cv_orchestrator.py           # CVOrchestrator (DEPRECATE -> _deprecated/)
    purged_kfold.py              # PurgedKFold (USE DIRECTLY)
    cpcv.py                      # CombinatorialPurgedCV (USE DIRECTLY)
    walk_forward.py              # WalkForwardEvaluator (USE DIRECTLY)

  inference/
    orchestrator.py              # InferenceOrchestrator (KEEP)

  models/ensemble/
    orchestrator.py              # EnsembleOrchestrator (KEEP)
```

**Result: 5 orchestrators -> 3 orchestrators (Training, Ensemble, Inference)**
