# Unified Training System - Implementation Status

**Created:** 2026-01-16
**Updated:** 2026-01-18
**Status:** READY TO USE (90% complete, needs final wiring)

---

## Optuna Optimization Integration

The unified training system integrates with the complete Optuna optimization pipeline:

| Stage | Optimization Type | Trials | Reference Doc |
|-------|------------------|--------|---------------|
| Stage 7 | Label Optimization (triple-barrier params) | 100 | `plan.md` |
| Stage 8 | Feature Selection (binary include/exclude) | 100 | `PHASE_5_ADAPTERS.md` |
| Stage 9 | Feature Pruning (importance-based) | 50 | `PHASE_5_ADAPTERS.md` |
| Stage 13 | Hyperparameter Tuning (per model) | 100/model | `PHASE_6_TRAINING.md` |
| Stage 15 | Meta-Learner Optimization | 50 | `PHASE_7_META_LEARNER_STACKING.md` |

**Total Trial Budget:** ~100 + 100 + 50 + (100 x 23 models) = **~2,550 trials**

---

## What You Already Have

### ✅ Complete
1. **Feature Strategies** (`src/features/strategies.py`)
   - `MODEL_FEATURE_STRATEGIES` for all 23 models
   - Baseline features defined per model
   - MTF modes configured
   - Feature family preferences

2. **Feature Optimization** (`src/features/optimization.py`)
   - Optuna-based feature selection (100 trials - Stage 8)
   - Optuna-based feature pruning (50 trials - Stage 9)
   - Takes baseline → finds optimal subset
   - Per-model optimization

3. **Hyperparameter Optimization** (`src/optimization/search_spaces.py`)
   - Optuna search spaces for all 23 models
   - 100 trials per model (Stage 13)
   - GPU/CPU optimization considerations

4. **Training Orchestrator** (`src/training/orchestrator.py`)
   - Already exists!
   - Handles data loading, model training, ensemble building
   - Integrates with Optuna optimization
   - Uses dict-based config (needs ExperimentConfig integration)

5. **Existing Pipeline** (`src/phase1/`)
   - Generates all ~180 features
   - MTF support (9 timeframes)
   - Triple-barrier labeling with Optuna optimization (Stage 7)

## What Needs Wiring

### 🔧 Integration Tasks

1. **Update Orchestrator to use ExperimentConfig**
   - Replace dict config with `ExperimentConfig` dataclass
   - Add per-model timeframe loading
   - Integrate feature optimization calls

2. **Update Notebook**
   - Simple interface using `ExperimentConfig`
   - Example configurations

3. **Update train_model.py**
   - Thin wrapper calling orchestrator

## Notebook Interface (What You Want)

```python
from src.training import TrainingOrchestrator, ExperimentConfig, ModelConfig
from src.optimization import OptunaConfig

config = ExperimentConfig(
    symbol="MES",
    horizons=[20],
    models=[
        ModelConfig(
            name="xgboost",
            timeframe="15min",
            optimize_features=True,
            feature_selection_trials=100,  # Stage 8: binary include/exclude
            feature_pruning_trials=50,     # Stage 9: importance-based
            hyperparam_trials=100,         # Stage 13: model hyperparameters
        ),
        ModelConfig(
            name="lstm",
            timeframe="5min",
            optimize_features=True,
            feature_selection_trials=100,
            feature_pruning_trials=50,
            hyperparam_trials=100,
            sequence_length=60,
        ),
        ModelConfig(
            name="patchtst",
            timeframe="multi_tf",  # Uses 1m+5m+15m
            optimize_features=False,  # Raw OHLCV for advanced models
            hyperparam_trials=100,
        ),
    ],
    build_ensemble=True,
    ensemble_method="stacking",
    meta_learner="ridge_meta",
    meta_learner_trials=50,  # Stage 15: meta-learner optimization
    cross_validate=True,
    cv_splits=5,
    optuna_config=OptunaConfig(
        storage="sqlite:///optuna.db",
        pruner="median",
        sampler="tpe",
    ),
)

orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()

orchestrator.display_results()
```

### Optuna Trial Budget for This Example

| Model | Feature Selection | Feature Pruning | Hyperparameters | Total |
|-------|------------------|-----------------|-----------------|-------|
| xgboost | 100 | 50 | 100 | 250 |
| lstm | 100 | 50 | 100 | 250 |
| patchtst | 0 | 0 | 100 | 100 |
| **Meta-learner** | - | - | 50 | 50 |
| **Total** | 200 | 100 | 350 | **650** |

## How It Works

### 1. Feature System
```
Pipeline Stage 3 generates ~180 features
         ↓
MODEL_FEATURE_STRATEGIES maps each model to baseline subset
         ↓
Optuna optimizes/prunes features (optional)
         ↓
Model trains on optimized feature set
```

### 2. Per-Model Timeframe Strategy
```
config.models = [
    ModelConfig(name="xgboost", timeframe="15min"),  # Loads 15min data
    ModelConfig(name="lstm", timeframe="5min"),      # Loads 5min data
    ModelConfig(name="patchtst", timeframe="multi_tf"),  # Multi-TF stream
]
         ↓
Orchestrator loads appropriate data per model
         ↓
Each model trains on its configured timeframe
```

### 3. Feature Optimization Flow (Stages 8-9)
```
1. Get baseline features from MODEL_FEATURE_STRATEGIES[model_name]
2. Load data for model's configured timeframe
3. Filter to available features (~180 total)
4. If optimize_features=True:
   Stage 8: Run 100 Optuna trials for binary feature selection
   - Each trial toggles features on/off
   - Evaluate via PurgedKFold CV
   - Select best feature subset
   Stage 9: Run 50 Optuna trials for feature pruning
   - Compute feature importances
   - Optimize importance threshold
   - Remove low-importance features
5. Stage 13: Run 100 Optuna trials for hyperparameter optimization
6. Train final model with optimized features + hyperparameters
```

## File Locations

### New Files (Created Today)
- `src/features/strategies.py` - Model feature mappings
- `src/features/optimization.py` - Optuna optimization
- `src/training/config.py` - ExperimentConfig dataclass

### Existing Files (Already Working)
- `src/training/orchestrator.py` - Main orchestrator
- `src/phase1/stages/features/` - Feature generation
- `src/models/trainer.py` - Single model training
- `src/cross_validation/` - CV infrastructure

## Next Steps

### Immediate (< 1 hour)
1. Update `orchestrator.py` to use `ExperimentConfig` instead of dict
2. Add per-model timeframe data loading
3. Integrate feature optimization calls
4. Test with 3-model ensemble

### Soon (< 2 hours)
5. Update notebook with new interface
6. Refactor `train_model.py` to thin wrapper
7. Update `MODEL_DATA_REQUIREMENTS` to reference strategies

### Later (< 4 hours)
8. Add MLflow tracking
9. Document in CLAUDE.md
10. Full end-to-end test

## Example: How XGBoost Gets Features

```python
# 1. Strategy defines baseline
MODEL_FEATURE_STRATEGIES["xgboost"] = ModelFeatureStrategy(
    baseline_features=[
        "rsi_14", "macd_line", "atr_14", "bb_width",
        "volume_ratio", "vpin", ...  # ~100 features
    ],
    mtf_mode="indicators",
    min_features=40,
    max_features=120,
)

# 2. Pipeline generates all ~180 features
# (features_15min.parquet has ALL features)

# 3. Orchestrator loads data
container = load_data(symbol="MES", horizon=20, timeframe="15min")

# 4. Filter to baseline
available = [f for f in baseline if f in container.feature_columns]

# 5. Optuna optimizes (if enabled)
if model_config.optimize_features:
    optimized = optimize_features_for_model(
        "xgboost", X_train, y_train, X_val, y_val
    )
    features = optimized.optimized_features  # ~67 features
else:
    features = available  # ~100 features

# 6. Train
model.fit(X_train[features], y_train)
```

## Key Insight

**You were right** - all features ARE important! The system:
1. Generates ALL ~180 features in Phase 1
2. Maps models to baseline subsets (per-model feature selection)
3. Optionally optimizes/prunes with Optuna
4. Each model gets tailored features

No features are lost - they're just intelligently allocated to the right models.

## Summary

**Status:** Infrastructure complete, needs final integration
**Time to working:** ~1 hour of wiring
**What you get:** Notebook interface to train any model combo with per-model timeframes and feature optimization

The hard part (feature strategies, optimization, orchestrator) is done.
The easy part (wiring it together) remains.

---

## Complete Model Inventory (23 Models)

All models receive Optuna optimization (100 trials each for hyperparameters):

### Boosting Models (3)
| # | Model | Optuna Search Space | GPU |
|---|-------|---------------------|-----|
| 1 | `xgboost` | learning_rate, max_depth, subsample, colsample | Yes |
| 2 | `lightgbm` | learning_rate, num_leaves, feature_fraction | Yes |
| 3 | `catboost` | learning_rate, depth, l2_leaf_reg | Yes |

### Classical Models (3)
| # | Model | Optuna Search Space | GPU |
|---|-------|---------------------|-----|
| 4 | `logistic` | C, solver, penalty | No |
| 5 | `random_forest` | n_estimators, max_depth, min_samples | No |
| 6 | `svm` | C, kernel, gamma | No |

### Neural Models - Basic (4)
| # | Model | Optuna Search Space | GPU |
|---|-------|---------------------|-----|
| 7 | `lstm` | hidden_size, num_layers, dropout, lr | Yes |
| 8 | `gru` | hidden_size, num_layers, dropout, lr | Yes |
| 9 | `tcn` | num_channels, kernel_size, dropout, lr | Yes |
| 10 | `transformer` | d_model, n_heads, num_layers, dropout | Yes |

### Neural Models - Advanced (6)
| # | Model | Optuna Search Space | GPU |
|---|-------|---------------------|-----|
| 11 | `patchtst` | patch_len, d_model, n_heads, num_layers | Yes |
| 12 | `itransformer` | d_model, n_heads, num_layers, d_ff | Yes |
| 13 | `tft` | hidden_size, attention_heads, num_lstm_layers | Yes |
| 14 | `nbeats` | stack_types, n_blocks, n_layers, layer_width | Yes |
| 15 | `inceptiontime` | n_filters, depth, use_residual | Yes |
| 16 | `resnet1d` | n_blocks, n_filters, kernel_size | Yes |

### Ensemble Models (3)
| # | Model | Optuna Search Space | GPU |
|---|-------|---------------------|-----|
| 17 | `voting` | weights | N/A |
| 18 | `stacking` | meta_learner_type | N/A |
| 19 | `blending` | blend_alpha | N/A |

### Meta-Learners (4)
| # | Model | Optuna Search Space | GPU |
|---|-------|---------------------|-----|
| 20 | `ridge_meta` | alpha | No |
| 21 | `mlp_meta` | hidden_sizes, dropout, lr | Yes |
| 22 | `calibrated_meta` | method, cv | No |
| 23 | `xgboost_meta` | learning_rate, max_depth, n_estimators | Yes |

---

## Cross-References

- **Optuna Search Spaces:** See `PHASE_6_TRAINING.md` for complete search space definitions
- **Feature Selection:** See `PHASE_5_ADAPTERS.md` for per-model feature optimization
- **Meta-Learner Stacking:** See `PHASE_7_META_LEARNER_STACKING.md` for Stage 15 details
- **Pipeline Stages:** See `plan.md` for 16-stage pipeline overview
