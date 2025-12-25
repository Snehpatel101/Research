# ML Model Factory - Quick Reference

**Status:** PRODUCTION READY | **Tests:** 1592 passing | **Models:** 12

---

## Train a Model (One Command)

```bash
# Fast baseline (< 1 min)
python scripts/train_model.py --model random_forest --horizon 20

# Best accuracy (GPU recommended)
python scripts/train_model.py --model xgboost --horizon 20

# Deep learning (requires GPU)
python scripts/train_model.py --model lstm --horizon 20
```

---

## Model Families

| Family | Models | GPU? | Speed | Best For |
|--------|--------|------|-------|----------|
| **Classical** | random_forest, logistic, svm | No | Fast (< 1 min) | Baselines, quick iteration |
| **Boosting** | xgboost, lightgbm, catboost | Optional | Medium (1-3 min) | Best single-model accuracy |
| **Neural** | lstm, gru, tcn | Yes | Slow (5-30 min) | Complex patterns, long sequences |
| **Ensemble** | voting, stacking, blending | Mixed | Slow | Maximum accuracy |

---

## Quick Start Workflows

### 1. First Model (5 minutes)
```bash
# Open notebook
jupyter notebook notebooks/01_quick_start.ipynb

# Or use script
python scripts/train_model.py --model random_forest --horizon 20
```

### 2. Benchmark All Models (30 minutes)
```bash
# Open notebook
jupyter notebook notebooks/02_model_comparison.ipynb

# Or use script loop
for model in random_forest xgboost lightgbm lstm; do
  python scripts/train_model.py --model $model --horizon 20
done
```

### 3. Build Ensemble (40 minutes)
```bash
# Train base models
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lightgbm --horizon 20
python scripts/train_model.py --model lstm --horizon 20

# Create ensemble
python scripts/train_ensemble.py --ensemble-type stacking \
  --base-models xgboost,lightgbm,lstm \
  --horizon 20
```

---

## Configuration

### Load Config
```python
from src.models.config import load_model_config

config = load_model_config("xgboost")
```

### Override Parameters
```python
config["training"]["n_estimators"] = 500
config["training"]["learning_rate"] = 0.01
```

### Available Configs
```python
from src.models.config import list_available_models

print(list_available_models())
# ['xgboost', 'lightgbm', 'catboost', 'lstm', 'gru', 'tcn',
#  'random_forest', 'logistic', 'svm', 'voting', 'stacking', 'blending']
```

---

## GPU Detection

### Check GPU
```python
from src.models.device import DeviceManager

dm = DeviceManager()
print(f"Device: {dm.device_str}")
print(f"GPU: {dm.gpu_info.name if dm.gpu_info else 'CPU'}")
print(f"Mixed Precision: {dm.amp_dtype}")
```

### Get Optimal Settings
```python
settings = dm.get_optimal_settings("lstm")
print(f"Batch size: {settings['batch_size']}")
print(f"Mixed precision: {settings['mixed_precision']}")
```

---

## Jupyter/Colab

### Setup Notebook
```python
from src.utils.notebook import setup_notebook

env = setup_notebook(log_level="INFO", seed=42)
print(f"GPU: {env['gpu_name']}")
```

### Display Results
```python
from src.utils.notebook import display_metrics

display_metrics(results, title="XGBoost Results")
```

### Plot Comparison
```python
from src.utils.notebook import plot_model_comparison

results = {
    "XGBoost": xgb_results,
    "LightGBM": lgb_results,
    "LSTM": lstm_results,
}
plot_model_comparison(results, metric="macro_f1")
```

---

## Cross-Validation

### Purged K-Fold
```python
from src.cross_validation import PurgedKFold

cv = PurgedKFold(n_splits=5, embargo_bars=60)

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # Train model
```

### Walk-Forward Feature Selection
```python
from src.cross_validation import WalkForwardFeatureSelector

selector = WalkForwardFeatureSelector(
    model_type="xgboost",
    n_features=30,
    direction="forward"
)
selected_features = selector.fit_transform(X_train, y_train)
```

### Out-of-Fold Predictions
```python
from src.cross_validation import OOFGenerator

oof_gen = OOFGenerator(model_type="xgboost", n_splits=5)
oof_preds, oof_scores = oof_gen.generate(X, y)
```

---

## Troubleshooting

### Tests Failing
```bash
python -m pytest tests/ -v --tb=short
```

### GPU Not Detected
```python
from src.models.device import print_gpu_info
print_gpu_info()
```

### Import Errors
```bash
pip install -r requirements.txt
```

### Config Issues
```python
from src.models.config import list_available_models
print(list_available_models())
```

---

## Performance Expectations

### Training Times (H20, ~40k samples)
- **Random Forest:** 30s (CPU)
- **XGBoost:** 1min (GPU) / 3min (CPU)
- **LSTM:** 5min (GPU) / 30min (CPU)

### Typical Metrics (H20)
- **Macro F1:** 0.40-0.65
- **Accuracy:** 0.50-0.65
- **Sharpe Ratio:** 0.5-1.2
- **Win Rate:** 48-55%

---

## File Locations

### Documentation
- `/home/jake/Desktop/Research/PIPELINE_READY.md` - Full overview
- `/home/jake/Desktop/Research/docs/README.md` - Quick start guide
- `/home/jake/Desktop/Research/docs/phases/` - Phase guides

### Code
- `/home/jake/Desktop/Research/src/models/` - Model implementations
- `/home/jake/Desktop/Research/configs/models/` - YAML configs
- `/home/jake/Desktop/Research/notebooks/` - Jupyter notebooks

### Scripts
- `/home/jake/Desktop/Research/scripts/train_model.py` - Train single model
- `/home/jake/Desktop/Research/scripts/train_ensemble.py` - Train ensemble
- `/home/jake/Desktop/Research/scripts/compare_models.py` - Compare results

---

## Best Practices

1. **Start with Random Forest** - Fast baseline for feature validation
2. **Use GPU for neural models** - CPU training is prohibitively slow
3. **Tune XGBoost first** - Usually best single-model performance
4. **Build ensembles last** - After individual model optimization
5. **Monitor memory** - 150+ features can be memory-intensive
6. **Use purged CV** - Prevents lookahead bias in time series

---

## Common Patterns

### Train and Evaluate
```python
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator

# Train
trainer = ModelTrainer(config)
trainer.train(X_train, y_train, X_val, y_val)

# Evaluate
evaluator = ModelEvaluator()
results = evaluator.evaluate(trainer.model, X_test, y_test)
```

### Save and Load
```python
# Save
trainer.save("runs/my_model/")

# Load
from src.models import ModelRegistry
model = ModelRegistry.load("xgboost", "runs/my_model/model.pkl")
predictions = model.predict(X_new)
```

### Custom Ensemble
```python
from src.models.ensemble import StackingEnsemble

# Define base models
base_models = [
    ("xgb", load_model("xgboost", "runs/xgb/model.pkl")),
    ("lgb", load_model("lightgbm", "runs/lgb/model.pkl")),
    ("lstm", load_model("lstm", "runs/lstm/model.pth")),
]

# Create stacking ensemble
ensemble = StackingEnsemble(base_models=base_models, meta_learner="logistic")
ensemble.train(X_train, y_train, X_val, y_val)
```

---

**Need Help?** See `/home/jake/Desktop/Research/PIPELINE_READY.md` for comprehensive guide.
