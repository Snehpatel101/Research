# Unified Training System - Implementation Status

**Created:** 2026-01-16  
**Status:** READY TO USE (90% complete, needs final wiring)

## What You Already Have

### ✅ Complete
1. **Feature Strategies** (`src/features/strategies.py`)
   - `MODEL_FEATURE_STRATEGIES` for all 23 models
   - Baseline features defined per model
   - MTF modes configured
   - Feature family preferences

2. **Feature Optimization** (`src/features/optimization.py`)
   - Optuna-based feature pruning
   - Takes baseline → finds optimal subset
   - Per-model optimization

3. **Training Orchestrator** (`src/training/orchestrator.py`)
   - Already exists!
   - Handles data loading, model training, ensemble building
   - Uses dict-based config (needs ExperimentConfig integration)

4. **Existing Pipeline** (`src/phase1/`)
   - Generates all ~180 features
   - MTF support (9 timeframes)
   - Triple-barrier labeling

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

config = ExperimentConfig(
    symbol="MES",
    horizons=[20],
    models=[
        ModelConfig(
            name="xgboost",
            timeframe="15min",
            optimize_features=True,
            feature_opt_trials=50,
        ),
        ModelConfig(
            name="lstm",
            timeframe="5min",
            optimize_features=True,
            sequence_length=60,
        ),
        ModelConfig(
            name="patchtst",
            timeframe="multi_tf",  # Uses 1m+5m+15m
            optimize_features=False,  # Raw OHLCV
        ),
    ],
    build_ensemble=True,
    ensemble_method="stacking",
    meta_learner="ridge_meta",
    cross_validate=True,
    cv_splits=5,
)

orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()

orchestrator.display_results()
```

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

### 3. Feature Optimization Flow
```
1. Get baseline features from MODEL_FEATURE_STRATEGIES[model_name]
2. Load data for model's configured timeframe
3. Filter to available features
4. If optimize_features=True:
   - Run Optuna to prune features
   - Find optimal subset via CV
5. Train on optimized features
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
