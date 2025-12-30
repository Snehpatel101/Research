# XGBoost Training Guide

## Model Overview

**Family:** Boosting
**Type:** Gradient Boosted Decision Trees
**GPU Support:** Yes (CUDA)
**Input Shape:** 2D `(n_samples, n_features)`
**Output:** 3-class predictions (short=-1, neutral=0, long=1)

## Hardware Requirements

### Minimum
- **CPU:** 4 cores
- **RAM:** 8GB
- **GPU:** None (CPU-only mode)
- **Training Time:** 10-20 minutes (CPU)

### Recommended
- **CPU:** 8+ cores
- **RAM:** 16GB
- **GPU:** Any NVIDIA GPU (GTX 1060+, 6GB VRAM)
- **Training Time:** 2-5 minutes (GPU)

### Optimal
- **CPU:** 16+ cores
- **RAM:** 32GB
- **GPU:** RTX 3080/4080 (10-16GB VRAM)
- **Training Time:** 1-2 minutes (GPU)

## Hyperparameters

### Default Configuration

```yaml
# config/models/xgboost.yaml
n_estimators: 500
max_depth: 6
min_child_weight: 10
subsample: 0.8
colsample_bytree: 0.8
learning_rate: 0.05
gamma: 0.1
reg_alpha: 0.1          # L1 regularization
reg_lambda: 1.0         # L2 regularization
tree_method: hist       # Fast histogram algorithm (required for GPU)
early_stopping_rounds: 50
```

### Hyperparameter Ranges (for tuning)

```python
# src/cross_validation/param_spaces.py
{
    "n_estimators": [100, 300, 500, 1000],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 5, 10, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "gamma": [0.0, 0.1, 0.3, 0.5],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.1, 1.0, 10.0]
}
```

## Training Configuration

### Quick Start (CPU)

```bash
python scripts/train_model.py \
    --model xgboost \
    --horizon 20 \
    --config config/models/xgboost.yaml
```

### GPU Training

```bash
python scripts/train_model.py \
    --model xgboost \
    --horizon 20 \
    --use-gpu \
    --config config/models/xgboost.yaml
```

### Custom Configuration

```bash
python scripts/train_model.py \
    --model xgboost \
    --horizon 20 \
    --use-gpu \
    --n-estimators 1000 \
    --max-depth 8 \
    --learning-rate 0.03
```

## Training Time Estimates

| Dataset Size | CPU (8 cores) | GPU (RTX 3080) | GPU (A100) |
|--------------|---------------|----------------|------------|
| 50K samples  | 5 min         | 1 min          | 30 sec     |
| 100K samples | 10 min        | 2 min          | 1 min      |
| 500K samples | 40 min        | 8 min          | 3 min      |
| 1M samples   | 90 min        | 15 min         | 6 min      |

## Memory Requirements

### CPU Mode
- **Base Memory:** 2GB
- **Per 100K samples:** +500MB
- **Formula:** `memory_gb = 2 + (n_samples / 100000) * 0.5`

### GPU Mode
- **GPU VRAM:** 2-4GB (fixed, independent of dataset size)
- **System RAM:** Same as CPU mode
- **Note:** XGBoost uses GPU only for tree building, dataset stays in RAM

## Batch Size Recommendations

**N/A** - XGBoost processes entire dataset at once (not minibatch-based)

## Learning Rate Schedule

XGBoost uses fixed learning rate (no scheduling). Adjust via:
- **High learning rate (0.1):** Fast convergence, risk of overfitting
- **Medium learning rate (0.05):** Default, balanced
- **Low learning rate (0.01):** Slow but stable, requires more estimators

## Early Stopping

```python
# Configured in YAML
early_stopping_rounds: 50  # Stop if no improvement for 50 rounds
```

**How it works:**
1. Model evaluates on validation set every round
2. If validation loss doesn't improve for 50 consecutive rounds, stop
3. Best iteration is saved automatically

## Validation Strategy

```python
# Built-in validation
- Training set: 70%
- Validation set: 15%  # Used for early stopping
- Test set: 15%        # Held out for final evaluation
```

## Feature Importance

XGBoost provides multiple importance metrics:

```python
from src.models import ModelRegistry

model = ModelRegistry.create("xgboost")
model.fit(X_train, y_train, X_val, y_val)

# Get feature importance (gain-based)
importance = model.get_feature_importance()

# Returns: {'rsi_14': 0.25, 'macd': 0.18, ...}
```

## GPU Configuration

### Enable GPU

```python
# In config YAML
device:
  use_gpu: true
  default: auto  # Auto-detect GPU

# Or via CLI
--use-gpu
```

### GPU Memory Optimization

XGBoost GPU mode uses fixed VRAM (~2-4GB). If GPU OOM:
1. Reduce `max_bin` (default 256): `--max-bin 128`
2. Use CPU mode
3. Process data in chunks (custom implementation)

## Performance Optimization Tips

### For Speed
1. **Enable GPU:** 5-10x speedup
2. **Reduce n_estimators:** 500 → 300
3. **Increase learning_rate:** 0.05 → 0.1
4. **Use smaller max_depth:** 6 → 4

### For Accuracy
1. **Increase n_estimators:** 500 → 1000
2. **Tune max_depth:** Try 6, 7, 8
3. **Enable class weighting:** `use_class_weights: true`
4. **Reduce learning_rate:** 0.05 → 0.03

### For Memory Efficiency
1. **Use hist tree method** (already default)
2. **Reduce max_depth:** 6 → 4
3. **Subsample data:** `subsample: 0.7`

## Common Issues

### Issue: GPU not detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA toolkit or use CPU mode
```

### Issue: Training too slow

```bash
# Enable GPU
--use-gpu

# Or reduce n_estimators
--n-estimators 300
```

### Issue: Overfitting

```python
# Increase regularization
reg_alpha: 1.0    # L1
reg_lambda: 10.0  # L2

# Reduce depth
max_depth: 4

# Increase min_child_weight
min_child_weight: 20
```

## Example Training Output

```
Training XGBoost: n_estimators=500, max_depth=6, gpu=on
[0]     train-mlogloss:1.09156  val-mlogloss:1.09234
[50]    train-mlogloss:0.87543  val-mlogloss:0.88912
[100]   train-mlogloss:0.78234  val-mlogloss:0.81456
[150]   train-mlogloss:0.72145  val-mlogloss:0.78234
[200]   train-mlogloss:0.68234  val-mlogloss:0.77123
[250]   train-mlogloss:0.65123  val-mlogloss:0.76890
[300]   train-mlogloss:0.62890  val-mlogloss:0.76912  # Early stop triggered
Training complete: epochs=300, val_f1=0.6234, time=98.2s
```

## Cross-Validation

```bash
# Run 5-fold time-series CV
python scripts/run_cv.py \
    --models xgboost \
    --horizons 20 \
    --n-splits 5

# With hyperparameter tuning (Optuna)
python scripts/run_cv.py \
    --models xgboost \
    --horizons 20 \
    --n-splits 5 \
    --tune \
    --n-trials 50
```

## Model Files

After training, model artifacts are saved:

```
experiments/runs/{run_id}/
├── model.json              # XGBoost model (JSON format)
├── metadata.pkl            # Config + feature names
├── training_metrics.json   # Training history
└── predictions.csv         # Validation predictions
```

## Integration with Ensembles

XGBoost can be used in ensembles with other tabular models:

```bash
# Valid: XGBoost + LightGBM + CatBoost
python scripts/train_model.py \
    --model voting \
    --base-models xgboost,lightgbm,catboost \
    --horizon 20

# INVALID: XGBoost + LSTM (mixed 2D/3D inputs)
# This will raise EnsembleCompatibilityError
```

## References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- GPU Support: https://xgboost.readthedocs.io/en/latest/gpu/index.html
- Parameter Tuning: https://xgboost.readthedocs.io/en/latest/parameter.html
