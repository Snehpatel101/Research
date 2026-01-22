# Simplified Vision: ONE Config, ONE Orchestrator

**Date:** 2026-01-21
**Status:** IMPLEMENTED

---

## The User's Question

> "Why can't we have ONE SINGLE config inside SRC that we can edit that has all the configs? MLPipeline should ORCHESTRATE every single thing inside SRC."

## The Answer: We CAN. And now it's DONE.

---

## IMPLEMENTATION COMPLETE

The simplified architecture has been implemented:

| Component | File | Status |
|-----------|------|--------|
| **THE ONE CONFIG** | `src/pipeline_config.py` | DONE |
| **THE ONE ORCHESTRATOR** | `src/orchestrator.py` | DONE |
| **Updated Exports** | `src/__init__.py` | DONE |

---

## The Simple API

```python
from src import MLPipeline, PipelineConfig

# ONE config with everything
config = PipelineConfig(
    symbol="MES",
    data_path="./data/mes_1min.parquet",
    output_dir="./experiments",

    # Models
    models=["xgboost", "lightgbm", "lstm"],
    horizons=[5, 10, 15, 20],
    build_ensemble=True,

    # Training
    training_mode="standard",
    cv_method="purged_kfold",
    n_splits=5,

    # Optimization
    optimize_labels=True,
    optimize_features=True,
    optimize_hyperparams=True,
)

# ONE orchestrator that does EVERYTHING
result = MLPipeline(config).run()

print(f"Best model: {result.best_model}")
print(result.summary())
```

---

## What PipelineConfig Contains (50+ fields)

```
REQUIRED:
  symbol              # Trading symbol (e.g., "MES")

DATA:
  data_path           # Path to parquet file
  output_dir          # Where to save outputs

MODELS:
  models              # List of models to train
  horizons            # Prediction horizons

ENSEMBLE:
  build_ensemble      # Whether to build ensemble
  ensemble_method     # stacking, voting, blending
  meta_learner        # ridge, mlp, xgboost

TRAINING:
  training_mode       # standard, walk_forward, regime_aware, meta_labeling
  batch_size          # Batch size for neural nets
  max_epochs          # Max training epochs
  device              # cpu, cuda, mps, auto

CROSS-VALIDATION:
  cv_method           # purged_kfold, cpcv, walk_forward
  n_splits            # Number of CV folds
  purge_bars          # Bars to purge
  embargo_bars        # Bars to embargo

LABELING:
  labeling_method     # triple_barrier, directional, threshold
  upper_mult          # Upper barrier multiplier
  lower_mult          # Lower barrier multiplier

FEATURES:
  feature_mode        # full, minimal, custom
  feature_families    # Which feature families to use
  compute_mtf_features # Multi-timeframe features

OPTIMIZATION:
  optimize_labels     # Run label optimization
  optimize_features   # Run feature selection
  optimize_hyperparams # Run HPO
```

---

## What MLPipeline Does (9 Phases)

```
Phase 1: DATA_PREP     - Load and clean data
Phase 2: FEATURES      - Generate 180+ features
Phase 3: LABELING      - Triple-barrier labels
Phase 4: SPLITS        - Train/val/test with purge/embargo
Phase 5: TRAINING      - Train all models
Phase 6: ENSEMBLE      - Build stacking ensemble
Phase 7: EVALUATION    - CV, walk-forward, CPCV-PBO
Phase 8: BACKTEST      - Strategy backtesting
Phase 9: BUNDLING      - Production packaging
```

---

## Preset Configurations

```python
from src import quick_config, production_config, research_config

# Fast iteration (no optimization)
config = quick_config("MES")

# Full production (all optimizations)
config = production_config("MES")

# Research mode (many models)
config = research_config("MES")
```

---

## Backward Compatibility

Old imports still work (with deprecation warnings):

```python
# OLD (deprecated)
from src import MLFactory
factory = MLFactory(config)
result = factory.run(df)

# NEW (use this)
from src import MLPipeline, PipelineConfig
result = MLPipeline(config).run()
```

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/pipeline_config.py` | ~350 | THE ONE config with 50+ fields |
| `src/orchestrator.py` | ~600 | THE ONE orchestrator with 9 phases |

---

## What This Replaces

The old complexity:
- 50+ config classes across multiple files
- 4 different entry points
- Confusing imports

The new simplicity:
- 1 config class (PipelineConfig)
- 1 orchestrator (MLPipeline)
- 2 files total

---

**That's it. ONE config. ONE orchestrator. EVERYTHING runs.**
