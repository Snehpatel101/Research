# Model Configurations

This directory contains configuration files for all 13 implemented models across 4 families.

## Model Families

### Boosting Models (3 models)
Fast, interpretable gradient boosting models optimized for tabular data.

| Model | File | GPU Support | Training Time | Memory | Best For |
|-------|------|-------------|---------------|--------|----------|
| **XGBoost** | [xgboost.yaml](xgboost.yaml) | ✓ (CUDA) | ~10 min | 2-4 GB | General purpose, feature interactions |
| **LightGBM** | [lightgbm.yaml](lightgbm.yaml) | ✓ (CUDA) | ~8 min | 2-4 GB | Large datasets, speed |
| **CatBoost** | [catboost.yaml](catboost.yaml) | ✓ (CUDA) | ~12 min | 2-4 GB | Categorical features, robust |

### Neural Models (4 models)
Deep learning models for temporal dependencies and sequential patterns.

| Model | File | GPU Support | Training Time | Memory | Best For |
|-------|------|-------------|---------------|--------|----------|
| **LSTM** | [lstm.yaml](lstm.yaml) | ✓ (CUDA + FP16) | ~60 min | 4-8 GB | Long-term dependencies |
| **GRU** | [gru.yaml](gru.yaml) | ✓ (CUDA + FP16) | ~50 min | 3-6 GB | Simpler RNN, faster than LSTM |
| **TCN** | [tcn.yaml](tcn.yaml) | ✓ (CUDA + FP16) | ~40 min | 4-8 GB | Long sequences, parallelizable |
| **Transformer** | [transformer.yaml](transformer.yaml) | ✓ (CUDA + FP16) | ~90 min | 6-12 GB | Attention mechanisms |

### Classical Models (3 models)
Robust baseline models with interpretable predictions.

| Model | File | GPU Support | Training Time | Memory | Best For |
|-------|------|-------------|---------------|--------|----------|
| **Random Forest** | [random_forest.yaml](random_forest.yaml) | ✗ (CPU only) | ~15 min | 3-6 GB | Robust baseline, feature importance |
| **Logistic Regression** | [logistic.yaml](logistic.yaml) | ✗ (CPU only) | ~5 min | 1-2 GB | Linear baseline, interpretability |
| **SVM** | [svm.yaml](svm.yaml) | ✗ (CPU only) | ~20 min | 2-4 GB | Non-linear boundaries |

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

- [Model Integration Guide](../../docs/guides/MODEL_INTEGRATION_GUIDE.md) - How to add new models
- [Hyperparameter Optimization Guide](../../docs/guides/HYPERPARAMETER_OPTIMIZATION_GUIDE.md) - Tuning strategies
- [Model Infrastructure Requirements](../../docs/guides/MODEL_INFRASTRUCTURE_REQUIREMENTS.md) - Hardware requirements
- [Phase 2 Documentation](../../docs/phases/PHASE_2.md) - Model training pipeline

---

*Last Updated: 2025-12-30*
