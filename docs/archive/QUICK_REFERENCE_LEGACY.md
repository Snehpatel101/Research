# ARCHIVED (Legacy)

This file is preserved for historical context and does not reflect the current pipeline.
Use `docs/QUICK_REFERENCE.md`.

---

# Quick Reference Guide

**Status:** Production Ready | **Tests:** 1592 passing | **Models:** 12 | **Single-Contract Architecture**

---

## Quick Start (5 Minutes)

```bash
# 1. Run Phase 1 pipeline (data preparation)
./pipeline run --symbols MES

# 2. Train a model (Phase 2)
python scripts/train_model.py --model xgboost --horizon 20

# 3. View results
cat experiments/runs/latest/metrics/training_metrics.json
```

---

## Phase 1: Data Pipeline

### Run Full Pipeline

```bash
# Basic run (MES contract, all horizons)
./pipeline run --symbols MES

# Custom configuration
./pipeline run --symbols MGC \
  --horizons 5,10,15,20 \
  --timeframe 5min \
  --enable-wavelets \
  --mtf-mode both
```

### Key Parameters

```python
SYMBOL = "MES"                    # or "MGC", "SI", "GC" (one at a time)
LABEL_HORIZONS = [5, 10, 15, 20]  # bars ahead to predict
TRAIN/VAL/TEST = 70/15/15          # percentage split
PURGE_BARS = 60                    # 3× max horizon (prevents label leakage)
EMBARGO_BARS = 1440                # ~5 days at 5min (breaks serial correlation)
```

### Output Files

```
data/splits/scaled/
├── train_scaled.parquet       # 24,711 samples × 221 features
├── val_scaled.parquet         # 3,808 samples
├── test_scaled.parquet        # 3,869 samples
├── feature_scaler.pkl         # RobustScaler (fitted on train only)
└── split_config.json

runs/{run_id}/artifacts/
├── manifest.json
├── pipeline_state.json
└── dataset_manifest.json
```

---

## Phase 2: Model Training

### Train Single Model

```bash
# Boosting models (fast, accurate)
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lightgbm --horizon 20
python scripts/train_model.py --model catboost --horizon 20

# Neural models (requires GPU)
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60
python scripts/train_model.py --model gru --horizon 20
python scripts/train_model.py --model tcn --horizon 20

# Classical models (fast baselines)
python scripts/train_model.py --model random_forest --horizon 20
python scripts/train_model.py --model logistic --horizon 20
python scripts/train_model.py --model svm --horizon 20
```

### List Available Models

```bash
# CLI
python scripts/train_model.py --list-models

# Python
from src.models import ModelRegistry
print(ModelRegistry.list_all())
# Output: ['xgboost', 'lightgbm', 'catboost', 'lstm', 'gru', 'tcn',
#          'random_forest', 'logistic', 'svm', 'voting', 'stacking', 'blending']
```

### Model Family Comparison

| Family | Models | GPU | Speed | Best For |
|--------|--------|-----|-------|----------|
| Boosting | xgboost, lightgbm, catboost | Optional | 1-3 min | Best single-model accuracy |
| Neural | lstm, gru, tcn | Required | 5-30 min | Complex temporal patterns |
| Classical | random_forest, logistic, svm | No | < 1 min | Fast baselines |
| Ensemble | voting, stacking, blending | Mixed | Variable | Maximum accuracy |

---

## Phase 3: Cross-Validation

### Run CV with Hyperparameter Tuning

```bash
# Single model, single horizon
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Multiple models, multiple horizons
python scripts/run_cv.py --models xgboost,lightgbm,lstm \
  --horizons 5,10,15,20 \
  --n-splits 5 \
  --tune \
  --n-trials 50

# All models (long runtime)
python scripts/run_cv.py --models all --horizons all --tune
```

### CV Configuration

```python
# CV parameters by model family
BOOSTING_FOLDS = 5      # XGBoost, LightGBM, CatBoost
NEURAL_FOLDS = 3        # LSTM, GRU, TCN (slow to train)
TRIALS = 50             # Optuna hyperparameter search trials
```

### Output Files

```
data/stacking/
├── stacking_dataset_h5.parquet    # OOF predictions for H5
├── stacking_dataset_h10.parquet
├── stacking_dataset_h15.parquet
└── stacking_dataset_h20.parquet

data/cv_results/
├── xgboost_h20_cv_results.json
├── best_params_xgboost_h20.json
└── feature_importance_xgboost_h20.csv
```

---

## Phase 4: Ensemble Training

### Train Ensemble Models

```bash
# Voting ensemble (simple averaging)
python scripts/train_model.py --model voting \
  --base-models xgboost,lightgbm,catboost \
  --horizon 20

# Stacking ensemble (meta-learner on OOF predictions)
python scripts/train_model.py --model stacking \
  --base-models xgboost,lightgbm,lstm \
  --meta-learner logistic \
  --horizon 20

# Blending ensemble (meta-learner on holdout)
python scripts/train_model.py --model blending \
  --base-models xgboost,lightgbm \
  --horizon 20
```

### Recommended Configurations

| Use Case | Models | Method | Expected Sharpe |
|----------|--------|--------|-----------------|
| Low Latency | xgboost, lightgbm, catboost | voting | 0.8-1.0 |
| Balanced | xgboost, lightgbm, random_forest | blending | 0.9-1.1 |
| Neural Focus | lstm, gru, tcn | stacking | 0.7-1.0 |
| Maximum Accuracy | All 12 models | stacking | 1.0-1.3 |

---

## Configuration Management

### Load Model Config

```python
from src.models.config import load_model_config

# Load default config
config = load_model_config("xgboost")

# Override parameters
config["training"]["n_estimators"] = 500
config["training"]["learning_rate"] = 0.01

# Use in training
from src.models import ModelRegistry
model = ModelRegistry.get("xgboost")
trainer = ModelTrainer(model, config)
```

### Pipeline Config

```python
from src.phase1.pipeline_config import PipelineConfig

config = PipelineConfig(
    symbol="MES",
    label_horizons=[5, 10, 15, 20],
    train_pct=0.70,
    val_pct=0.15,
    test_pct=0.15,
    purge_bars=60,
    embargo_bars=1440,
    allow_batch_symbols=False  # Enforce single-contract
)
```

---

## File Locations

### Data Paths

```
data/raw/{symbol}_1m.csv               # Raw OHLCV input
data/raw/{symbol}_1m.parquet           # Converted to parquet
data/clean/{symbol}_5m.parquet         # Resampled data
data/features/{symbol}_features.parquet  # Feature-engineered
data/final/{symbol}_labeled.parquet    # Triple-barrier labels
data/splits/scaled/train_scaled.parquet  # Model-ready training data
```

### Experiment Paths

```
experiments/runs/{run_id}/
├── checkpoints/
│   ├── model.pkl                      # Boosting/classical models
│   └── model.pth                      # Neural models
├── metrics/
│   ├── training_metrics.json
│   └── evaluation_metrics.json
└── config/
    └── model_config.yaml
```

---

## Common Operations

### Check GPU Availability

```python
from src.models.device import DeviceManager

dm = DeviceManager()
print(f"Device: {dm.device_str}")
print(f"GPU: {dm.gpu_info.name if dm.gpu_info else 'CPU'}")
print(f"Mixed Precision: {dm.amp_dtype}")
```

### Load Trained Model

```python
from src.models import ModelRegistry

# Load model
model = ModelRegistry.load("xgboost", "experiments/runs/latest/checkpoints/model.pkl")

# Make predictions
predictions = model.predict(X_test)
```

### Validate Pipeline Output

```bash
./pipeline validate --run-id latest
```

### Rerun Failed Stage

```bash
./pipeline rerun --stage feature_engineering --run-id latest
```

---

## Troubleshooting

### Pipeline Issues

```bash
# Check pipeline status
./pipeline status --run-id latest

# Validate data quality
./pipeline validate --run-id latest

# View logs
cat runs/{run_id}/logs/pipeline.log
```

### Model Training Issues

```python
# Verify data availability
from pathlib import Path
assert Path("data/splits/scaled/train_scaled.parquet").exists()

# Check GPU
from src.models.device import print_gpu_info
print_gpu_info()

# Test model registration
from src.models import ModelRegistry
print(len(ModelRegistry.list_all()))  # Should print 12
```

### Test Failures

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/models/test_boosting_models.py -v

# Run with detailed output
python -m pytest tests/ -v --tb=short
```

---

## Performance Expectations

### Training Times (H20, ~40k samples)

- **Random Forest:** 30-60 sec (CPU)
- **XGBoost:** 1-3 min (GPU) / 3-5 min (CPU)
- **LightGBM:** 1-2 min (GPU) / 2-4 min (CPU)
- **LSTM:** 5-15 min (GPU) / 30+ min (CPU)

### Typical Metrics (H20)

| Metric | Range | Target |
|--------|-------|--------|
| Macro F1 | 0.40-0.65 | > 0.50 |
| Accuracy | 0.50-0.65 | > 0.55 |
| Sharpe Ratio | 0.5-1.2 | > 0.8 |
| Win Rate | 48-55% | > 50% |

---

## Best Practices

### 1. Start Simple

```bash
# Begin with fast baseline
python scripts/train_model.py --model random_forest --horizon 20

# Move to boosting for accuracy
python scripts/train_model.py --model xgboost --horizon 20

# Try neural if needed
python scripts/train_model.py --model lstm --horizon 20 --device cuda
```

### 2. Use Proper CV

```python
# Always use PurgedKFold for time series
from src.cross_validation import PurgedKFold

cv = PurgedKFold(n_splits=5, embargo_bars=1440)
```

### 3. Monitor GPU Memory

```python
# Check batch size fits in VRAM
from src.models.device import DeviceManager

dm = DeviceManager()
settings = dm.get_optimal_settings("lstm")
print(f"Recommended batch size: {settings['batch_size']}")
```

### 4. Track Experiments

```bash
# Use run IDs for experiment tracking
RUN_ID=$(date +%Y%m%d_%H%M%S)
python scripts/train_model.py --model xgboost --horizon 20 --run-id $RUN_ID
```

---

## Advanced Usage

### Custom Feature Sets

```python
from src.phase1.config.feature_sets import FeatureSetRegistry

# List available feature sets
print(FeatureSetRegistry.list_available())

# Use custom feature set
./pipeline run --symbols MES --feature-set minimal
```

### Walk-Forward Feature Selection

```python
from src.cross_validation import WalkForwardFeatureSelector

selector = WalkForwardFeatureSelector(
    model_type="xgboost",
    n_features=30,
    method="mda"  # or "mdi"
)
selected_features = selector.fit_transform(X_train, y_train)
```

### Custom Ensemble Weights

```python
from src.models.ensemble import VotingEnsemble

ensemble = VotingEnsemble(
    base_models=[xgb_model, lgb_model, lstm_model],
    weights=[0.4, 0.4, 0.2]  # Favor boosting models
)
```

---

## Key Commands Summary

```bash
# Phase 1: Data Pipeline
./pipeline run --symbols MES

# Phase 2: Single Model
python scripts/train_model.py --model xgboost --horizon 20

# Phase 3: Cross-Validation
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5 --tune

# Phase 4: Ensemble
python scripts/train_model.py --model voting --base-models xgboost,lightgbm

# Utilities
./pipeline status --run-id latest
./pipeline validate --run-id latest
python scripts/train_model.py --list-models
```

---

## Further Reading

- **Architecture:** `/home/user/Research/docs/reference/ARCHITECTURE.md`
- **Phase 1 Guide:** `/home/user/Research/docs/phases/PHASE_1.md`
- **Phase 2 Guide:** `/home/user/Research/docs/phases/PHASE_2.md`
- **CLAUDE.md:** `/home/user/Research/CLAUDE.md` (project instructions)
- **Complete Overview:** `/home/user/Research/PIPELINE_READY.md`

---

**Note:** This is a single-contract ML factory. Each contract (MES, MGC, SI, etc.) trains in complete isolation with its own pipeline run and trained models. Multi-symbol processing is blocked by default (`allow_batch_symbols=False`).
