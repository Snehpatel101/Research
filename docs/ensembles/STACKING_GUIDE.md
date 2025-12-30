# Stacking Ensemble Guide

## Overview

Stacking uses a **two-layer architecture**: base models generate out-of-fold (OOF) predictions, then a meta-learner trains on those predictions to produce final output. More sophisticated than voting, learns optimal model weighting.

**Implementation:** `/home/user/Research/src/models/ensemble/stacking.py`
**Config:** `/home/user/Research/config/models/stacking.yaml`
**OOF Generation:** `/home/user/Research/src/cross_validation/oof_*.py`

---

## How It Works

### Two-Layer Architecture

```
Layer 1: Base Models (generate OOF predictions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
├─ Fold 1: Train on folds 2-5, predict fold 1
├─ Fold 2: Train on folds 1,3-5, predict fold 2
├─ Fold 3: Train on folds 1-2,4-5, predict fold 3
├─ Fold 4: Train on folds 1-3,5, predict fold 4
└─ Fold 5: Train on folds 1-4, predict fold 5

Result: Full training set with OOF predictions
        (each sample predicted by model that never saw it)

Layer 2: Meta-Learner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train on OOF predictions → Learn optimal model weighting
```

### Why OOF Predictions?

**In-sample predictions are overconfident:**
```
Train model on all data → predict same data → overfit
Meta-learner sees unrealistic performance → overfits
```

**OOF predictions are realistic:**
```
Train model without sample X → predict sample X → honest
Meta-learner sees true generalization → better weighting
```

### Leakage Prevention

**CRITICAL:** Stacking uses **default configs** for OOF generation:

```python
# Phase 1: Generate OOF with DEFAULT configs (prevents leakage)
use_default_configs_for_oof: true  # Default

# Phase 2: Train final models with TUNED configs (for inference)
base_model_configs:
  xgboost: {max_depth: 8}  # Only used for final models
```

**Why?** If you generate OOF with tuned configs, the meta-learner trains on optimistically biased predictions → overfit.

---

## Configuration

### Basic Configuration

```yaml
# config/models/stacking.yaml or custom config

# Base models (Layer 1)
base_model_names:
  - xgboost
  - lightgbm
  - random_forest

base_model_configs: {}          # For FINAL models only

# Meta-learner (Layer 2)
meta_learner_name: logistic     # Default: logistic regression
meta_learner_config: {}

# Cross-validation for OOF
n_folds: 5                      # 5 for boosting, 3 for neural

# Meta-learner input
use_probabilities: true         # Use probs (3*n_models features) vs predictions (n_models)
passthrough: false              # Include original features in meta input

# Leakage prevention (CRITICAL)
use_default_configs_for_oof: true  # Use defaults for OOF, tuned for final models

# PurgedKFold settings
purge_bars: 60                  # 3x max horizon (prevents label leakage)
embargo_bars: 1440              # 5 days at 5min (breaks serial correlation)
```

### Advanced Configuration

```yaml
# Use class predictions instead of probabilities
use_probabilities: false        # n_models features instead of 3*n_models

# Include original features in meta-learner
passthrough: true               # Concat [X_original, OOF_predictions]

# Reduce folds for faster training
n_folds: 3                      # Trade coverage for speed

# Custom meta-learner
meta_learner_name: xgboost      # Can use any registered model
meta_learner_config:
  max_depth: 3
  n_estimators: 50
```

---

## Recommended Configurations

### 1. Boosting Stack (Fast, Strong Baseline)

**Best for:** Quick improvement over single models

```bash
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,catboost
```

**Config:**
```yaml
base_model_names: [xgboost, lightgbm, catboost]
meta_learner_name: logistic
n_folds: 5
use_probabilities: true
```

**Expected Performance:**
- Training time: 15-25 minutes (5 folds × 3 models × ~1 min)
- Val F1: +0.02-0.04 over voting
- Meta-learner learns optimal model weighting

### 2. Boosting + Classical (Maximum Diversity)

**Best for:** Large datasets with rich feature sets

```bash
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,random_forest
```

**Expected Performance:**
- Training time: 20-30 minutes
- More stable (RF adds bagging diversity)
- Good for noisy labels

### 3. RNN Stack (Temporal Patterns)

**Best for:** Sequential data with temporal dependencies

```bash
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models lstm,gru,tcn --seq-len 60
```

**Config:**
```yaml
base_model_names: [lstm, gru, tcn]
meta_learner_name: logistic
n_folds: 3                      # Fewer folds for neural models
use_probabilities: true
```

**Expected Performance:**
- Training time: 45-90 minutes (GPU) / 3-6 hours (CPU)
- Val F1: +0.03-0.06 over voting
- Best for trend-following

### 4. All Tabular (High Diversity, High Variance)

**Best for:** Very large datasets (>50k samples), exploratory analysis

```bash
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,catboost,random_forest,logistic,svm
```

**Expected Performance:**
- Training time: 40-60 minutes
- High variance (may overfit)
- Requires careful validation

---

## OOF Generation Details

### PurgedKFold Cross-Validation

Stacking uses `PurgedKFold` to prevent information leakage:

```python
from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig

config = PurgedKFoldConfig(
    n_splits=5,
    purge_bars=60,      # Remove 60 bars before test set (3x max horizon=20)
    embargo_bars=1440,  # Skip 1440 bars after test set (5 days at 5min)
)

cv = PurgedKFold(config)
```

**Purge:** Removes samples whose labels overlap with test set start time
**Embargo:** Skips samples after test set to break serial correlation

### Coverage Validation

After OOF generation, check coverage:

```python
from src.cross_validation.oof_validation import OOFValidator

validator = OOFValidator()
results = validator.validate_coverage(oof_predictions)

print(f"Coverage: {results['coverage_pct']:.1f}%")  # Should be ~100%
print(f"Missing samples: {results['n_missing']}")   # Should be 0
```

**Good coverage:** 99-100% (a few samples may be dropped due to purging/embargo)
**Bad coverage:** <95% (indicates configuration issues)

---

## Meta-Learner Selection

### Default: Logistic Regression

**Why logistic?**
- Fast (trains in <1 second on OOF predictions)
- Calibrated probabilities
- Regularization prevents overfitting
- Interpretable coefficients (model weights)

**When to use:**
- Most use cases (recommended default)
- When you need calibrated probabilities
- When meta-learner training time matters

### Alternative: XGBoost

```yaml
meta_learner_name: xgboost
meta_learner_config:
  max_depth: 3
  n_estimators: 50
  learning_rate: 0.1
```

**When to use:**
- Complex base model interactions
- Non-linear weighting needed
- Large OOF datasets (>10k samples)

**Trade-off:** Slower, may overfit

### Alternative: Neural Network

```yaml
meta_learner_name: logistic  # Still recommended for most cases
```

**Not recommended:** Neural networks as meta-learners tend to overfit on small OOF datasets.

---

## Passthrough Features

Include original features alongside OOF predictions:

```yaml
passthrough: true
```

**Meta-learner input:**
```
Without passthrough: [OOF_xgb, OOF_lgb, OOF_cat]  # 9 features (3 models × 3 classes)
With passthrough:    [X_original, OOF_xgb, OOF_lgb, OOF_cat]  # 9 + n_features
```

**When to use:**
- Base models miss important features
- Meta-learner can learn residual patterns
- Large feature sets (>100 features)

**Trade-off:** Higher meta-learner complexity, longer training

---

## Training from Scratch

```python
from src.models import ModelRegistry
from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig
import pandas as pd

# Create stacking ensemble
ensemble = ModelRegistry.create("stacking", config={
    "base_model_names": ["xgboost", "lightgbm", "random_forest"],
    "meta_learner_name": "logistic",
    "n_folds": 5,
    "use_probabilities": True,
    "passthrough": False,
})

# Prepare label_end_times for purging overlapping labels
label_end_times = pd.Series(...)  # When each label is resolved

# Train ensemble
metrics = ensemble.fit(
    X_train, y_train,
    X_val, y_val,
    label_end_times=label_end_times,  # Enable overlapping label purging
)

# Predict
output = ensemble.predict(X_test)
```

---

## Compatibility Requirements

**CRITICAL:** All base models must have the same input shape.

### Valid Combinations

✅ **Tabular-only** (2D input):
- `xgboost`, `lightgbm`, `catboost`, `random_forest`, `logistic`, `svm`

✅ **Sequence-only** (3D input):
- `lstm`, `gru`, `tcn`, `transformer`

### Invalid Combinations

❌ **Mixed tabular + sequence** (WILL FAIL):
```bash
# WILL FAIL
python scripts/train_model.py --model stacking \
  --base-models xgboost,lstm
```

See `/home/user/Research/docs/ensembles/ENSEMBLE_COMPATIBILITY.md` for details.

---

## CLI Usage

### Basic Training

```bash
# Tabular stacking
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,random_forest

# Sequence stacking
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models lstm,gru,tcn --seq-len 60
```

### Custom Configuration

```bash
# Use XGBoost meta-learner
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm \
  --config '{"meta_learner_name": "xgboost"}'

# Reduce folds for faster training
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,catboost \
  --config '{"n_folds": 3}'

# Enable passthrough features
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm \
  --config '{"passthrough": true}'
```

---

## Troubleshooting

### Issue: OOF coverage < 95%

**Possible causes:**
1. Purge/embargo settings too aggressive
2. Small dataset (<5k samples)
3. Label end times missing or incorrect

**Solutions:**
- Reduce `purge_bars` or `embargo_bars`
- Use blending instead (faster, no folds)
- Check `label_end_times` are provided correctly

### Issue: Meta-learner overfits

**Symptoms:**
- High train F1, low val F1
- Meta-learner outperforms base models on train but not val

**Solutions:**
- Use logistic regression (default, more regularized)
- Reduce `n_folds` (less OOF data)
- Enable `use_default_configs_for_oof=True` (prevents leakage)

### Issue: Slow training

**Solutions:**
- Reduce `n_folds` (5→3)
- Use fewer base models
- Use blending instead (no folds)
- For neural models, use GPU

### Issue: No improvement over voting

**Possible causes:**
1. Base models too similar (meta-learner can't learn better weights)
2. Dataset too small (<5k samples)
3. Meta-learner too simple

**Solutions:**
- Use diverse base models (boosting + classical)
- Try XGBoost meta-learner
- Enable `passthrough=True`

---

## Performance Expectations

| Configuration | Models | Training Time | Val F1 Lift vs Voting | Use Case |
|--------------|---------|---------------|---------------------|----------|
| Boosting Stack | XGB+LGB+Cat | 15-25 min | +0.02-0.04 | Fast baseline |
| Boosting+Classical | XGB+LGB+RF | 20-30 min | +0.02-0.03 | Stable |
| RNN Stack | LSTM+GRU+TCN | 45-90 min (GPU) | +0.03-0.06 | Temporal |
| All Tabular | 6 models | 40-60 min | +0.03-0.05 | High diversity |

**Note:** Performance varies by dataset, horizon, and base model quality.

---

## Comparing to Voting

| Aspect | Voting | Stacking |
|--------|--------|----------|
| **Training Time** | Fast (single pass) | Slower (k-fold) |
| **Complexity** | Simple averaging | Learned weighting |
| **Overfitting Risk** | Lower | Higher (if misconfigured) |
| **Performance** | Good baseline | +0.02-0.04 F1 improvement |
| **Interpretability** | High (clear weights) | Medium (meta-learner) |
| **Use Case** | Quick baseline | Production models |

**Recommendation:** Start with voting, upgrade to stacking if you need the extra performance.

---

## Next Steps

- **Blending:** For faster alternative to stacking, see `BLENDING_GUIDE.md`
- **Voting:** For simpler baseline, see `VOTING_GUIDE.md`
- **Compatibility:** Check model combinations in `ENSEMBLE_COMPATIBILITY.md`
- **OOF Validation:** See `/home/user/Research/src/cross_validation/oof_validation.py`
- **Cross-Validation:** Run `scripts/run_cv.py` for ensemble validation
