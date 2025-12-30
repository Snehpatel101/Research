# Voting Ensemble Training Guide

## Model Overview

**Family:** Ensemble
**Type:** Weighted/Unweighted Voting
**GPU Support:** Depends on base models
**Input Shape:** 2D (tabular) or 3D (sequence) - **must be consistent**
**Output:** Averaged predictions from multiple models

## Critical Constraint: Input Shape Compatibility

**VOTING ENSEMBLES CANNOT MIX TABULAR AND SEQUENCE MODELS**

### Valid Configurations

**Tabular-Only (2D input):**
```bash
# Boosting models
--base-models xgboost,lightgbm,catboost

# Boosting + Classical
--base-models xgboost,lightgbm,random_forest

# All tabular models
--base-models xgboost,lightgbm,catboost,random_forest,logistic,svm
```

**Sequence-Only (3D input):**
```bash
# RNN variants
--base-models lstm,gru

# Temporal models
--base-models lstm,gru,tcn

# All neural models
--base-models lstm,gru,tcn,transformer
```

### Invalid Configurations

```bash
# INVALID: Mixing tabular + sequence
--base-models xgboost,lstm          # WILL FAIL
--base-models lightgbm,gru,tcn      # WILL FAIL
--base-models random_forest,transformer  # WILL FAIL
```

## Hardware Requirements

**Voting ensemble requires:**
- Hardware for largest base model (typically neural models)
- Additional RAM for storing all base model predictions

### Recommended Specs

**For Tabular Ensembles (XGBoost + LightGBM + CatBoost):**
- **CPU:** 16 cores
- **RAM:** 32GB
- **GPU:** Optional (10GB VRAM if enabled)
- **Training Time:** 10-20 minutes

**For Sequence Ensembles (LSTM + GRU + TCN):**
- **CPU:** 16 cores
- **RAM:** 64GB
- **GPU:** RTX 3090 (24GB VRAM) - **highly recommended**
- **Training Time:** 45-90 minutes

## Hyperparameters

### Default Configuration

```yaml
# config/models/voting.yaml
model_name: voting
family: ensemble
description: Weighted voting ensemble

# Base models (must all be tabular OR all be sequence)
base_model_names:
  - xgboost
  - lightgbm
  - catboost

# Voting configuration
voting_type: soft        # 'soft' (probability averaging) or 'hard' (majority vote)
weights: null            # null = equal weights, or [0.4, 0.3, 0.3]

# Training configuration
train_base_models: true  # Train base models or load pretrained
base_model_configs: {}   # Override base model configs (optional)
```

### Voting Types

**Soft Voting (Recommended):**
```python
# Average class probabilities
P_final = (w1 * P_model1 + w2 * P_model2 + w3 * P_model3) / (w1 + w2 + w3)
prediction = argmax(P_final)

# Example with equal weights:
# Model 1: [0.6, 0.3, 0.1]  (predicts class 0)
# Model 2: [0.5, 0.4, 0.1]  (predicts class 0)
# Model 3: [0.4, 0.5, 0.1]  (predicts class 1)
# Average:  [0.5, 0.4, 0.1]  → Final prediction: class 0
```

**Hard Voting:**
```python
# Majority vote on class predictions
predictions = [model1.predict(), model2.predict(), model3.predict()]
final = mode(predictions)  # Most common class

# Example:
# Model 1: class 0
# Model 2: class 0
# Model 3: class 1
# Final: class 0 (majority)
```

## Training Configuration

### Tabular Ensemble (Recommended)

```bash
# Train boosting trio ensemble
python scripts/train_model.py \
    --model voting \
    --base-models xgboost,lightgbm,catboost \
    --horizon 20 \
    --voting-type soft
```

### Sequence Ensemble

```bash
# Train temporal ensemble (requires GPU)
python scripts/train_model.py \
    --model voting \
    --base-models lstm,gru,tcn \
    --horizon 20 \
    --seq-len 60 \
    --batch-size 512 \
    --voting-type soft
```

### Custom Weights

```bash
# Weighted voting (XGBoost gets 50%, others 25% each)
python scripts/train_model.py \
    --model voting \
    --base-models xgboost,lightgbm,catboost \
    --horizon 20 \
    --weights 0.5,0.25,0.25
```

### Using Pretrained Models

```bash
# Load base models from previous runs instead of retraining
python scripts/train_model.py \
    --model voting \
    --base-models xgboost,lightgbm,catboost \
    --horizon 20 \
    --load-base-models \
    --base-model-dirs exp1,exp2,exp3
```

## Training Time Estimates

### Tabular Ensembles

| Base Models | Dataset | CPU | GPU | Notes |
|-------------|---------|-----|-----|-------|
| XGBoost + LightGBM | 100K | 15 min | 4 min | GPU for both models |
| XGB + LGB + CatBoost | 100K | 20 min | 6 min | Training in sequence |
| All 6 tabular models | 100K | 35 min | 10 min | Not recommended |

### Sequence Ensembles

| Base Models | Dataset | GPU (RTX 3080) | GPU (A100) |
|-------------|---------|----------------|------------|
| LSTM + GRU | 100K | 35 min | 15 min |
| LSTM + GRU + TCN | 100K | 60 min | 25 min |
| All 4 neural models | 100K | 90 min | 35 min |

**Note:** Sequence ensembles train base models sequentially due to GPU memory constraints.

## Memory Requirements

### Tabular Ensembles

```python
# Formula: Sum of individual model memory + prediction storage
memory_gb = sum([model_mem for model in base_models]) + (n_samples * n_models * 3 classes * 8 bytes) / (1024**3)

# Example: XGBoost (2GB) + LightGBM (2GB) + CatBoost (2.5GB) + predictions (0.5GB)
# Total: ~7GB RAM
```

### Sequence Ensembles

```python
# Base models trained sequentially (one at a time in GPU)
# GPU memory = max(model1_mem, model2_mem, model3_mem)

# Example: LSTM (4GB) + GRU (3.5GB) + TCN (5GB)
# GPU VRAM needed: 5GB (TCN is largest)
# RAM needed: 10GB (store all 3 models)
```

## Weight Optimization

### Equal Weights (Default)

```yaml
weights: null  # Each model gets weight = 1.0
```

### Manual Weights

```yaml
# Based on validation F1 scores
weights: [0.4, 0.35, 0.25]  # XGBoost, LightGBM, CatBoost
```

### Auto-optimize Weights (Not Yet Implemented)

```bash
# Future feature: optimize weights on validation set
python scripts/train_model.py \
    --model voting \
    --base-models xgboost,lightgbm,catboost \
    --horizon 20 \
    --optimize-weights
```

## Validation Strategy

```python
# Each base model uses same train/val/test split
train: 70%
val: 15%    # Used for early stopping in base models
test: 15%   # Final ensemble evaluation

# Ensemble inherits validation from base models
# No additional validation needed for voting
```

## Common Issues

### Issue: EnsembleCompatibilityError

```python
# Error: Cannot mix tabular and sequence models
EnsembleCompatibilityError: Cannot mix tabular and sequence models.
  Tabular models (2D): ['xgboost', 'lightgbm']
  Sequence models (3D): ['lstm']

# Solution: Use only tabular OR only sequence models
--base-models xgboost,lightgbm,catboost  # OK
--base-models lstm,gru,tcn               # OK
--base-models xgboost,lstm               # FAIL
```

### Issue: Out of Memory (Sequence Ensemble)

```bash
# Solution 1: Reduce batch size for neural models
--batch-size 256

# Solution 2: Reduce sequence length
--seq-len 30

# Solution 3: Use fewer base models
--base-models lstm,gru  # Remove TCN
```

### Issue: Training takes too long

```bash
# Solution 1: Use pretrained base models
--load-base-models --base-model-dirs run1,run2,run3

# Solution 2: Reduce number of base models
--base-models xgboost,lightgbm  # Remove CatBoost

# Solution 3: Reduce base model complexity
# Edit config/models/xgboost.yaml: n_estimators: 300 (from 500)
```

## Example Training Output

```
Training Voting Ensemble: 3 base models (xgboost, lightgbm, catboost)
Voting type: soft, Weights: equal

[1/3] Training xgboost...
  Completed in 3.2 minutes, val_f1=0.6234

[2/3] Training lightgbm...
  Completed in 2.8 minutes, val_f1=0.6189

[3/3] Training catboost...
  Completed in 4.1 minutes, val_f1=0.6301

Computing ensemble predictions...
  Validation F1: 0.6456  (+0.0155 vs best base model)
  Test F1: 0.6389

Ensemble training complete: 10.1 minutes
```

## Performance Expectations

### Tabular Ensembles

| Ensemble | Typical Improvement | Best Use Case |
|----------|---------------------|---------------|
| XGB + LGB | +0.01-0.02 F1 | Fast, reliable |
| XGB + LGB + CatBoost | +0.015-0.025 F1 | Recommended |
| All 6 tabular | +0.02-0.03 F1 | Diminishing returns |

### Sequence Ensembles

| Ensemble | Typical Improvement | Best Use Case |
|----------|---------------------|---------------|
| LSTM + GRU | +0.01-0.02 F1 | Simple temporal diversity |
| LSTM + GRU + TCN | +0.02-0.03 F1 | Recommended |
| All 4 neural | +0.025-0.035 F1 | High compute budget |

## Cross-Validation

```bash
# CV with voting ensemble
python scripts/run_cv.py \
    --models voting \
    --base-models xgboost,lightgbm,catboost \
    --horizons 20 \
    --n-splits 5
```

## Model Files

```
experiments/runs/{run_id}/
├── ensemble_config.json       # Ensemble configuration
├── base_models/
│   ├── xgboost/
│   │   ├── model.json
│   │   └── metadata.pkl
│   ├── lightgbm/
│   │   ├── model.txt
│   │   └── metadata.pkl
│   └── catboost/
│       ├── model.cbm
│       └── metadata.pkl
├── ensemble_predictions.csv   # Ensemble predictions
└── training_metrics.json      # Performance comparison
```

## Comparison with Other Ensembles

| Ensemble Type | Training Data | Complexity | Improvement |
|---------------|---------------|------------|-------------|
| **Voting** | Same as base models | Low | +1-2% F1 |
| **Stacking** | OOF predictions | Medium | +2-3% F1 |
| **Blending** | 20% holdout | Low-Medium | +1.5-2.5% F1 |

**When to use Voting:**
- Want simple, fast ensemble
- Have 2-3 diverse models
- Don't want to generate OOF predictions

**When NOT to use Voting:**
- Base models are too similar (XGBoost + LightGBM gives minimal gain)
- Want maximum performance (use Stacking instead)
- Have compute budget for meta-learning

## References

- Ensemble Methods: Dietterich (2000) "Ensemble Methods in Machine Learning"
- Soft Voting: https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
- Model Diversity: Kuncheva (2003) "Measures of Diversity in Classifier Ensembles"
