# ML Pipeline Architecture

**Status:** Implemented
**Date:** 2026-01-22

---

## The API

```python
from src import MLPipeline, PipelineConfig

config = PipelineConfig(
    symbol="MES",
    models=["xgboost", "lstm"],
    build_ensemble=True,
)

result = MLPipeline(config).run()
```

ONE config. ONE orchestrator. Done.

---

## Files

| File | Purpose |
|------|---------|
| `src/pipeline_config.py` | PipelineConfig - all settings |
| `src/orchestrator.py` | MLPipeline - runs everything |
| `src/__init__.py` | Exports |

---

## PipelineConfig Fields

```
REQUIRED:
  symbol                    # "MES", "ES", etc.

DATA:
  data_path                 # Path to parquet
  output_dir                # Where outputs go

MODELS:
  models                    # ["xgboost", "lightgbm", "lstm", ...]
  horizons                  # [5, 10, 15, 20]
  build_ensemble            # True/False
  ensemble_method           # stacking, voting, blending
  meta_learner              # ridge, mlp, xgboost

TRAINING:
  training_mode             # standard, walk_forward, regime_aware, meta_labeling
  cv_method                 # purged_kfold, cpcv
  n_splits                  # CV folds
  batch_size, max_epochs    # Neural net params
  device                    # auto, cpu, cuda, mps

LABELING:
  labeling_method           # triple_barrier, directional
  upper_mult, lower_mult    # Barrier multipliers

OPTIMIZATION:
  optimize_labels           # Run Optuna for labels
  optimize_features         # Run feature selection
  optimize_hyperparams      # Run HPO
```

---

## MLPipeline Phases

| Phase | Method | What it does |
|-------|--------|--------------|
| 1-4 | `run_data()` | Load, clean, features, labels, splits |
| 5-6 | `run_train()` | Train models, build ensemble |
| 7 | `run_evaluate()` | CV, walk-forward, CPCV-PBO |
| 8 | `run_backtest()` | Strategy backtesting |
| 9 | `run_bundle()` | Production packaging |
| ALL | `run()` | Everything above |

---

## Presets

```python
from src import quick_config, production_config, research_config

quick_config("MES")       # Fast iteration
production_config("MES")  # Full optimization
research_config("MES")    # Many models
```

---

## Architecture

```
User Code
    │
    ▼
MLPipeline (src/orchestrator.py)
    │
    ├── Phase 1-4: Data prep
    ├── Phase 5-6: Training + Ensemble
    ├── Phase 7: Evaluation
    ├── Phase 8: Backtest
    └── Phase 9: Bundling
    │
    ▼
PipelineResult
```
