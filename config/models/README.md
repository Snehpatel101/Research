# Model Configurations

This directory contains configuration files for all 13 implemented models across 4 families.

## Model Families

### Boosting Models (3 models)
Fast, interpretable gradient boosting models optimized for tabular data.

| Model | File | GPU Support | Training Time | Memory | Best For |
|-------|------|-------------|---------------|--------|----------|
| **XGBoost** | [xgboost.yaml](xgboost.yaml) | Yes (CUDA) | ~10 min | 2-4 GB | General purpose, feature interactions |
| **LightGBM** | [lightgbm.yaml](lightgbm.yaml) | Yes (CUDA) | ~8 min | 2-4 GB | Large datasets, speed |
| **CatBoost** | [catboost.yaml](catboost.yaml) | Yes (CUDA) | ~12 min | 2-4 GB | Categorical features, robust |

### Neural Models (4 models)
Deep learning models for temporal dependencies and sequential patterns.

| Model | File | GPU Support | Training Time | Memory | Best For |
|-------|------|-------------|---------------|--------|----------|
| **LSTM** | [lstm.yaml](lstm.yaml) | Yes (CUDA + FP16) | ~60 min | 4-8 GB | Long-term dependencies |
| **GRU** | [gru.yaml](gru.yaml) | Yes (CUDA + FP16) | ~50 min | 3-6 GB | Simpler RNN, faster than LSTM |
| **TCN** | [tcn.yaml](tcn.yaml) | Yes (CUDA + FP16) | ~40 min | 4-8 GB | Long sequences, parallelizable |
| **Transformer** | [transformer.yaml](transformer.yaml) | Yes (CUDA + FP16) | ~90 min | 6-12 GB | Attention mechanisms |

### Classical Models (3 models)
Robust baseline models with interpretable predictions.

| Model | File | GPU Support | Training Time | Memory | Best For |
|-------|------|-------------|---------------|--------|----------|
| **Random Forest** | [random_forest.yaml](random_forest.yaml) | No (CPU only) | ~15 min | 3-6 GB | Robust baseline, feature importance |
| **Logistic Regression** | [logistic.yaml](logistic.yaml) | No (CPU only) | ~5 min | 1-2 GB | Linear baseline, interpretability |
| **SVM** | [svm.yaml](svm.yaml) | No (CPU only) | ~20 min | 2-4 GB | Non-linear boundaries |

### Ensemble Models (3 models)
Meta-learning models that combine multiple base models.

| Model | File | GPU Support | Training Time | Memory | Best For |
|-------|------|-------------|---------------|--------|----------|
| **Voting** | [voting.yaml](voting.yaml) | Depends on base | Varies | Varies | Simple averaging, fast |
| **Stacking** | [stacking.yaml](stacking.yaml) | Depends on base | +30 min | +2 GB | Meta-learning, OOF predictions |
| **Blending** | [blending.yaml](blending.yaml) | Depends on base | +20 min | +2 GB | Holdout-based meta-learning |

*Training times are approximate for horizon=20 on RTX 4070 Ti with MES symbol.*

## Configuration Structure

All model configs follow the same template:

```yaml
# Model identification
model:
  name: {model_name}
  family: {boosting | neural | classical | ensemble}
  description: {description}

# Default hyperparameters
defaults:
  # Model-specific parameters

# Training settings
training:
  feature_set: {boosting_optimal | neural_optimal | classical_optimal}
  random_seed: 42

# Device settings
device:
  default: auto
  mixed_precision: true  # For neural models
```

## Hyperparameter Optimization with Optuna

Each model family has a defined search space for Optuna-based hyperparameter tuning.

### Optuna Trial Configuration

| Model Family | Trials | Timeout | Early Stopping | Pruner |
|--------------|--------|---------|----------------|--------|
| Boosting | 100 | 2 hours | 50 rounds | Hyperband |
| Neural | 100 | 4 hours | 20 epochs | Hyperband |
| Classical | 50 | 1 hour | N/A | Median |

**Configuration:** See `config/optimization/hyperparameter.yaml`

### Boosting Model Search Spaces

#### XGBoost
```yaml
xgboost_search_space:
  n_estimators:
    type: int
    low: 100
    high: 2000
    step: 100
  max_depth:
    type: int
    low: 3
    high: 12
  learning_rate:
    type: float
    low: 0.001
    high: 0.3
    log: true
  subsample:
    type: float
    low: 0.5
    high: 1.0
  colsample_bytree:
    type: float
    low: 0.5
    high: 1.0
  gamma:
    type: float
    low: 0.0
    high: 5.0
  reg_alpha:
    type: float
    low: 1e-8
    high: 10.0
    log: true
  reg_lambda:
    type: float
    low: 1e-8
    high: 10.0
    log: true
  min_child_weight:
    type: int
    low: 1
    high: 10
```

#### LightGBM
```yaml
lightgbm_search_space:
  n_estimators:
    type: int
    low: 100
    high: 2000
    step: 100
  max_depth:
    type: int
    low: 3
    high: 12
  learning_rate:
    type: float
    low: 0.001
    high: 0.3
    log: true
  num_leaves:
    type: int
    low: 20
    high: 300
  subsample:
    type: float
    low: 0.5
    high: 1.0
  colsample_bytree:
    type: float
    low: 0.5
    high: 1.0
  reg_alpha:
    type: float
    low: 1e-8
    high: 10.0
    log: true
  reg_lambda:
    type: float
    low: 1e-8
    high: 10.0
    log: true
  min_child_samples:
    type: int
    low: 5
    high: 100
```

#### CatBoost
```yaml
catboost_search_space:
  iterations:
    type: int
    low: 100
    high: 2000
    step: 100
  depth:
    type: int
    low: 4
    high: 10
  learning_rate:
    type: float
    low: 0.001
    high: 0.3
    log: true
  l2_leaf_reg:
    type: float
    low: 1.0
    high: 10.0
  bagging_temperature:
    type: float
    low: 0.0
    high: 1.0
  random_strength:
    type: float
    low: 0.0
    high: 10.0
  border_count:
    type: categorical
    choices: [32, 64, 128, 254]
```

### Neural Model Search Spaces

#### LSTM / GRU
```yaml
rnn_search_space:
  hidden_size:
    type: categorical
    choices: [64, 128, 256, 512]
  num_layers:
    type: int
    low: 1
    high: 4
  dropout:
    type: float
    low: 0.0
    high: 0.5
  bidirectional:
    type: categorical
    choices: [true, false]
  learning_rate:
    type: float
    low: 1e-5
    high: 1e-2
    log: true
  batch_size:
    type: categorical
    choices: [32, 64, 128, 256]
  weight_decay:
    type: float
    low: 1e-6
    high: 1e-2
    log: true
  gradient_clip:
    type: float
    low: 0.5
    high: 5.0
```

#### TCN
```yaml
tcn_search_space:
  num_channels:
    type: categorical
    choices:
      - [32, 64]
      - [64, 128]
      - [64, 128, 256]
      - [128, 256, 512]
  kernel_size:
    type: int
    low: 2
    high: 7
  dropout:
    type: float
    low: 0.0
    high: 0.5
  learning_rate:
    type: float
    low: 1e-5
    high: 1e-2
    log: true
  batch_size:
    type: categorical
    choices: [32, 64, 128, 256]
```

#### Transformer
```yaml
transformer_search_space:
  d_model:
    type: categorical
    choices: [64, 128, 256, 512]
  nhead:
    type: categorical
    choices: [2, 4, 8]
  num_layers:
    type: int
    low: 2
    high: 6
  dim_feedforward:
    type: categorical
    choices: [256, 512, 1024, 2048]
  dropout:
    type: float
    low: 0.0
    high: 0.5
  learning_rate:
    type: float
    low: 1e-5
    high: 1e-3
    log: true
  batch_size:
    type: categorical
    choices: [32, 64, 128]
  weight_decay:
    type: float
    low: 1e-6
    high: 1e-2
    log: true
```

### Classical Model Search Spaces

#### Random Forest
```yaml
random_forest_search_space:
  n_estimators:
    type: int
    low: 100
    high: 1000
    step: 100
  max_depth:
    type: int
    low: 5
    high: 30
  min_samples_split:
    type: int
    low: 2
    high: 20
  min_samples_leaf:
    type: int
    low: 1
    high: 10
  max_features:
    type: categorical
    choices: [sqrt, log2, 0.3, 0.5, 0.7]
  bootstrap:
    type: categorical
    choices: [true, false]
```

#### SVM
```yaml
svm_search_space:
  C:
    type: float
    low: 1e-3
    high: 1e3
    log: true
  kernel:
    type: categorical
    choices: [rbf, poly, sigmoid]
  gamma:
    type: categorical
    choices: [scale, auto]
  degree:  # Only for poly kernel
    type: int
    low: 2
    high: 5
```

#### Logistic Regression
```yaml
logistic_search_space:
  C:
    type: float
    low: 1e-4
    high: 1e2
    log: true
  penalty:
    type: categorical
    choices: [l1, l2, elasticnet]
  solver:
    type: categorical
    choices: [saga, lbfgs]
  l1_ratio:  # Only for elasticnet
    type: float
    low: 0.0
    high: 1.0
```

## Early Stopping Integration

### Boosting Models
Early stopping is integrated via native callbacks:

```python
# XGBoost
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

# LightGBM
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)]
)

# CatBoost
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False
)
```

### Neural Models
Early stopping is configured in the training loop:

```yaml
# In model config
training:
  early_stopping:
    enabled: true
    patience: 20
    min_delta: 0.001
    monitor: val_f1_macro
    mode: max
    restore_best_weights: true
```

### Optuna Pruning
Optuna prunes unpromising trials during training:

```python
# Integration with training loop
for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader)
    val_score = evaluate(model, val_loader)

    # Report to Optuna
    trial.report(val_score, epoch)

    # Handle pruning
    if trial.should_prune():
        raise optuna.TrialPruned()
```

## Quick Start

### Train a Single Model
```bash
# Use default config
python scripts/train_model.py --model xgboost --horizon 20

# Override config values
python scripts/train_model.py \
  --model xgboost \
  --horizon 20 \
  --override "defaults.n_estimators=1000" \
  --override "defaults.learning_rate=0.01"
```

### Run Hyperparameter Optimization
```bash
# Optimize single model
python scripts/tune_model.py \
    --model xgboost \
    --horizon 20 \
    --n-trials 100

# Optimize all models
python scripts/tune_model.py \
    --model all \
    --horizon 20 \
    --config config/optimization/hyperparameter.yaml
```

### List Available Models
```bash
python scripts/train_model.py --list-models
```

### Validate a Configuration
```python
from src.models.config.loaders import load_model_config

config = load_model_config("xgboost")
print(config['model']['name'])  # xgboost
print(config['model']['family'])  # boosting
```

## Model Selection Guide

### By Training Speed
1. **Fastest (< 15 min):** logistic, xgboost, lightgbm
2. **Fast (15-30 min):** catboost, random_forest, svm
3. **Medium (30-60 min):** tcn, gru, lstm
4. **Slow (> 60 min):** transformer

### By Accuracy (Typical F1 Scores)
1. **Best (> 0.52):** Ensembles, transformer, stacking
2. **Good (0.50-0.52):** xgboost, lightgbm, catboost, lstm, gru
3. **Baseline (0.48-0.50):** tcn, random_forest
4. **Simple (< 0.48):** logistic, svm

### By Memory Requirements
1. **Low (< 4 GB):** logistic, svm, xgboost, lightgbm
2. **Medium (4-8 GB):** catboost, random_forest, lstm, gru, tcn
3. **High (> 8 GB):** transformer, ensembles

### By Use Case
- **Quick baseline:** logistic, xgboost
- **Production deployment:** xgboost, lightgbm (fast inference)
- **Maximum accuracy:** stacking ensemble
- **Temporal patterns:** lstm, transformer
- **Interpretability:** random_forest, logistic

## Configuration Reference

See [config/INDEX.md](../INDEX.md) for comprehensive configuration reference including:
- All hyperparameters for each model
- Configuration validation rules
- Environment-specific overrides
- Best practices

## Related Documentation

- [Model Integration Guide](../../docs/guides/MODEL_INTEGRATION.md) - How to add new models
- [Hyperparameter Tuning](../../docs/guides/HYPERPARAMETER_TUNING.md) - Tuning strategies
- [Optimization Configuration](../optimization/README.md) - Optuna optimization configs
- [Infrastructure Requirements](../../docs/reference/INFRASTRUCTURE.md) - Hardware requirements
- [Phase 6 Training](../../docs/implementation/PHASE_6_TRAINING.md) - Model training pipeline

---

*Last Updated: 2026-01-18*
