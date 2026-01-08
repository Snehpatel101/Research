# Pipeline Configurations

Global configuration files for training pipeline, cross-validation, and data processing.

## Configuration Files

| File | Purpose | Updated By | Lines |
|------|---------|-----------|-------|
| [training.yaml](training.yaml) | Global training settings, device config, experiment tracking | Manual | 134 |
| [cv.yaml](cv.yaml) | Cross-validation settings, hyperparameter tuning | Manual | 106 |
| scaling_stats.json | Scaling statistics from training data | Auto-generated | Varies |

## training.yaml

Global training configuration used by all models unless overridden.

### Key Sections

**Data Settings:**
```yaml
data:
  horizon: 20                # Default label horizon
  train_ratio: 0.70          # 70% train
  val_ratio: 0.15            # 15% validation
  test_ratio: 0.15           # 15% test
  purge_bars: 60             # Leakage prevention
  embargo_bars: 1440         # Serial correlation handling
  default_symbols: [MES, MGC]
```

**Training Defaults:**
```yaml
training:
  batch_size: 256
  max_epochs: 100
  early_stopping_patience: 15
  random_seed: 42
```

**Device Settings:**
```yaml
device:
  default: auto              # auto, cuda, or cpu
  mixed_precision: true      # FP16 for neural models
  max_memory_fraction: 0.9
```

**Environment Overrides:**
```yaml
environments:
  colab:                     # Google Colab (T4/A100)
    training:
      batch_size: 512
  local_cpu:                 # CPU-only development
    device:
      default: cpu
  local_gpu:                 # Local GPU (RTX, etc.)
    training:
      batch_size: 512
```

### Usage

**Load configuration:**
```python
from src.models.config.loaders import load_training_config

config = load_training_config()
print(config['training']['batch_size'])  # 256
```

**Override with environment:**
```bash
export ML_ENV=colab
python scripts/train_model.py --model xgboost --horizon 20
# Uses batch_size=512 from colab environment
```

**Override specific values:**
```bash
python scripts/train_model.py \
  --model xgboost \
  --horizon 20 \
  --override "training.batch_size=1024"
```

## cv.yaml

Cross-validation and hyperparameter tuning configuration.

### Key Sections

**Time-Series CV:**
```yaml
time_series_cv:
  n_splits: 5
  min_train_samples: 10000
  gap: 60                    # Purge bars between folds
  expanding: true            # Expanding vs sliding window
```

**Purged K-Fold:**
```yaml
purged_kfold:
  n_splits: 5
  purge_overlap: 60          # Purge samples between folds
  embargo: 60                # Embargo at fold boundaries
```

**Walk-Forward Validation:**
```yaml
walk_forward:
  enabled: true
  retrain_every_days: 30
  min_train_days: 180
  validation_days: 30
```

**Hyperparameter Tuning:**
```yaml
hyperparameter_tuning:
  framework: optuna
  n_trials: 100
  timeout: 3600              # 1 hour
  pruning: true
  sampler: tpe
```

**Metrics:**
```yaml
metrics:
  primary: f1_macro          # Primary metric for model selection

  additional:
    - accuracy
    - precision_macro
    - recall_macro
    - roc_auc_ovr

  trading:
    - sharpe_ratio
    - max_drawdown
    - win_rate
```

### Usage

**Load CV configuration:**
```python
from src.models.config.loaders import load_training_config

cv_config = load_training_config("cv")
print(cv_config['time_series_cv']['n_splits'])  # 5
```

**Run cross-validation:**
```bash
python scripts/run_cv.py \
  --models xgboost \
  --horizons 20 \
  --n-splits 5
```

**Hyperparameter tuning:**
```bash
python scripts/run_cv.py \
  --models xgboost \
  --horizons 20 \
  --tune \
  --n-trials 100
```

**Walk-forward validation:**
```bash
python scripts/run_walk_forward.py \
  --models xgboost \
  --horizons 20 \
  --retrain-days 30
```

## scaling_stats.json

Auto-generated scaling statistics from training data. Used to scale validation and test data with the same parameters.

**Structure:**
```json
{
  "scaler_type": "robust",
  "features": {
    "feature1": {
      "median": 0.0,
      "iqr": 1.0
    },
    "feature2": {...}
  },
  "n_features": 150,
  "n_samples": 100000,
  "generated_at": "2025-12-30T10:00:00"
}
```

**Usage:**
- Generated automatically during pipeline run (Phase 1)
- Saved to `config/pipeline/scaling_stats.json` (legacy) or `runs/{run_id}/scaling_stats.json` (recommended)
- Loaded during model training to scale validation/test data
- Should NOT be edited manually

## Configuration Precedence

Settings are applied in this order (later overrides earlier):

1. **Global defaults** (training.yaml, cv.yaml)
2. **Model-specific config** (config/models/{model}.yaml)
3. **Environment overrides** (ML_ENV variable)
4. **CLI overrides** (--override flag)

### Example

```bash
# Global: batch_size=256 (training.yaml)
# Model: batch_size=512 (lstm.yaml)
# Environment: batch_size=1024 (colab in training.yaml)
# CLI: batch_size=2048 (--override)

export ML_ENV=colab
python scripts/train_model.py \
  --model lstm \
  --horizon 20 \
  --override "training.batch_size=2048"

# Final batch_size = 2048 (CLI wins)
```

## Best Practices

### training.yaml
- Set conservative defaults that work on most hardware
- Use environment overrides for deployment-specific settings
- Don't hardcode paths (use relative paths from project root)
- Document non-obvious settings with comments

### cv.yaml
- Use n_splits >= 5 for robust CV estimates
- Ensure purge/embargo match or exceed values in training.yaml
- Set timeout high enough for large hyperparameter spaces
- Choose primary metric aligned with trading objectives

### scaling_stats.json
- Never edit manually (auto-generated)
- Commit to version control if using global path (deprecated)
- Use run-scoped path for reproducibility (recommended)
- Delete if rerunning pipeline from scratch

## Related Documentation

- [Phase 1 Ingestion](../../docs/implementation/PHASE_1_INGESTION.md) - Data pipeline
- [Phase 3 Features](../../docs/implementation/PHASE_3_FEATURES.md) - Feature engineering
- [Hyperparameter Tuning](../../docs/guides/HYPERPARAMETER_TUNING.md)

---

*Last Updated: 2025-12-30*
