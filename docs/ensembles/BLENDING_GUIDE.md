# Blending Ensemble Guide

## Overview

Blending is a **faster alternative to stacking** that uses a holdout validation set instead of k-fold cross-validation. Trades some statistical rigor for speed and simplicity.

**Implementation:** `/home/user/Research/src/models/ensemble/blending.py`
**Config:** `/home/user/Research/config/models/blending.yaml`

---

## How It Works

### Two-Phase Training

```
Phase 1: Train Base Models on Subset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Data Split (time-based):
├─ Blend Train (80%): Train base models
└─ Holdout (20%):     Generate predictions for meta-learner

Base models train on 80% → predict on 20% holdout

Phase 2: Train Meta-Learner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta-learner trains on:
- Input: Holdout predictions from base models
- Target: Holdout labels

Phase 3: Retrain Base Models (Optional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retrain base models on FULL training data
(Meta-learner already trained, doesn't change)
```

### Time-Based Split (CRITICAL)

**Blending uses the LAST `holdout_fraction` of data as holdout:**

```
Training Data (chronological):
[─────────────────── Blend Train 80% ─────────────────][── Holdout 20% ──]
  ↑                                                     ↑
  Train base models here                               Generate predictions here
```

**Why last portion?**
- Preserves temporal ordering (no future leakage)
- Holdout represents most recent data (realistic validation)
- Meta-learner trains on current market regime

**BAD (random split):** Would leak future data into past training
**GOOD (time-based):** Past trains on past, predicts future

---

## Configuration

### Basic Configuration

```yaml
# config/models/blending.yaml or custom config

# Base models (Layer 1)
base_model_names:
  - xgboost
  - lightgbm

base_model_configs: {}

# Meta-learner (Layer 2)
meta_learner_name: logistic
meta_learner_config: {}

# Holdout split
holdout_fraction: 0.2           # 20% for meta-learner training

# Meta-learner input
use_probabilities: true         # Use probs vs predictions
passthrough: false              # Include original features

# Retraining
retrain_on_full: true           # Retrain base models on full data after meta-learner
```

### Advanced Configuration

```yaml
# Larger holdout for small datasets
holdout_fraction: 0.3           # 30% holdout

# Skip retraining (faster, uses less data)
retrain_on_full: false          # Base models trained on 80% only

# Use class predictions instead of probabilities
use_probabilities: false        # Faster, less meta-learner input

# Include original features
passthrough: true               # Concat [X_original, base_predictions]
```

---

## Recommended Configurations

### 1. Boosting Blend (Fast Baseline)

**Best for:** Quick stacking alternative, large datasets (>10k samples)

```bash
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm,catboost
```

**Config:**
```yaml
base_model_names: [xgboost, lightgbm, catboost]
meta_learner_name: logistic
holdout_fraction: 0.2
retrain_on_full: true
```

**Expected Performance:**
- Training time: 5-10 minutes (much faster than stacking)
- Val F1: +0.01-0.03 over voting (slightly less than stacking)
- Good for rapid prototyping

### 2. Boosting Pair (Minimal)

**Best for:** Speed-critical applications, limited compute

```bash
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm
```

**Expected Performance:**
- Training time: 3-5 minutes
- Val F1: +0.01-0.02 over voting
- Minimal overhead, fast inference

### 3. RNN Blend (Temporal Patterns)

**Best for:** Sequential data, faster than RNN stacking

```bash
python scripts/train_model.py --model blending --horizon 20 \
  --base-models lstm,gru --seq-len 60
```

**Config:**
```yaml
base_model_names: [lstm, gru]
meta_learner_name: logistic
holdout_fraction: 0.2
n_folds: 3
```

**Expected Performance:**
- Training time: 20-40 minutes (GPU) / 1-2 hours (CPU)
- Val F1: +0.02-0.04 over voting
- 2-3x faster than stacking

### 4. Large Holdout (Small Datasets)

**Best for:** Datasets <5k samples where k-fold is overkill

```bash
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm \
  --config '{"holdout_fraction": 0.3}'
```

**Expected Performance:**
- More data for meta-learner (better estimates)
- Less data for base models (may underfit)
- Best when meta-learner needs robust training

---

## Blending vs Stacking

| Aspect | Blending | Stacking |
|--------|----------|----------|
| **Training Time** | Fast (1 pass) | Slow (k-fold) |
| **Base Model Training** | 80% of data | 80% per fold (averaged) |
| **Meta-Learner Data** | 20% holdout | 100% OOF |
| **Variance** | Higher | Lower |
| **Complexity** | Simple | Complex |
| **Best For** | Large datasets, prototyping | Production, small datasets |

**Use blending when:**
- Dataset is large (>10k samples)
- Training time matters
- Prototyping ensembles
- Base models train slowly (neural networks)

**Use stacking when:**
- Dataset is small (<10k samples)
- Maximum performance needed
- Production deployment
- Statistical rigor matters

---

## Holdout Fraction Selection

### Recommended Settings

| Dataset Size | Holdout Fraction | Reasoning |
|-------------|------------------|-----------|
| <5k samples | 0.3 (30%) | More data for meta-learner |
| 5k-20k samples | 0.2 (20%) | Balanced (default) |
| >20k samples | 0.15-0.2 (15-20%) | More data for base models |

### Trade-offs

**Larger holdout (0.3-0.4):**
- ✅ More meta-learner training data
- ✅ Better meta-learner estimates
- ❌ Less base model training data
- ❌ Base models may underfit

**Smaller holdout (0.1-0.15):**
- ✅ More base model training data
- ✅ Stronger base models
- ❌ Less meta-learner training data
- ❌ Meta-learner may overfit

**Rule of thumb:** Holdout should have at least 1000 samples for stable meta-learner training.

---

## Retrain on Full Data

### Default: Retrain Enabled

```yaml
retrain_on_full: true
```

**Training flow:**
1. Train base models on 80% blend_train
2. Generate predictions on 20% holdout
3. Train meta-learner on holdout predictions
4. **Retrain base models on 100% training data**
5. Use retrained base models + meta-learner for inference

**Benefits:**
- Base models use all available data
- Better inference performance
- Meta-learner sees realistic holdout predictions

**Cost:** Extra training time (trains base models twice)

### Alternative: No Retrain

```yaml
retrain_on_full: false
```

**Training flow:**
1. Train base models on 80% blend_train
2. Generate predictions on 20% holdout
3. Train meta-learner on holdout predictions
4. **Use blend_train models** for inference

**Benefits:**
- Faster (no retraining)
- Simpler training flow

**Cost:**
- Base models trained on 80% only
- May underfit if dataset is small

**When to use:**
- Very large datasets (>50k samples)
- Training time is critical
- Base models train slowly (neural networks)

---

## Training from Scratch

```python
from src.models import ModelRegistry

# Create blending ensemble
ensemble = ModelRegistry.create("blending", config={
    "base_model_names": ["xgboost", "lightgbm"],
    "meta_learner_name": "logistic",
    "holdout_fraction": 0.2,
    "use_probabilities": True,
    "retrain_on_full": True,
})

# Train ensemble
metrics = ensemble.fit(
    X_train, y_train,
    X_val, y_val,
    sample_weights=sample_weights,
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
python scripts/train_model.py --model blending \
  --base-models xgboost,lstm
```

See `/home/user/Research/docs/ensembles/ENSEMBLE_COMPATIBILITY.md` for details.

---

## CLI Usage

### Basic Training

```bash
# Tabular blending
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Sequence blending
python scripts/train_model.py --model blending --horizon 20 \
  --base-models lstm,gru --seq-len 60
```

### Custom Configuration

```bash
# Larger holdout
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm \
  --config '{"holdout_fraction": 0.3}'

# Skip retraining (faster)
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm,catboost \
  --config '{"retrain_on_full": false}'

# Use XGBoost meta-learner
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm \
  --config '{"meta_learner_name": "xgboost"}'
```

---

## Troubleshooting

### Issue: Small holdout set warning

**Warning:**
```
Holdout set is small (95 samples). Consider using Stacking for small datasets.
```

**Solutions:**
1. Increase `holdout_fraction` (0.2 → 0.3)
2. Use stacking instead (better for <5k samples)
3. Collect more data

### Issue: Base models underfit

**Symptoms:**
- Low base model performance on blend_train
- Ensemble performance worse than single models

**Solutions:**
- Reduce `holdout_fraction` (0.3 → 0.2)
- Enable `retrain_on_full=True`
- Collect more training data

### Issue: Meta-learner overfits

**Symptoms:**
- High holdout performance, low validation performance
- Meta-learner outperforms base models on holdout but not validation

**Solutions:**
- Use logistic regression (default, more regularized)
- Increase `holdout_fraction` (more meta-learner data)
- Simplify meta-learner config

### Issue: Slower than expected

**Solutions:**
- Disable retraining: `retrain_on_full=false`
- Use fewer base models
- Use voting instead (no meta-learner)

### Issue: No improvement over voting

**Possible causes:**
1. Base models too similar
2. Holdout too small (<500 samples)
3. Meta-learner too simple

**Solutions:**
- Use diverse base models
- Increase `holdout_fraction`
- Try XGBoost meta-learner
- Enable `passthrough=True`

---

## Performance Expectations

| Configuration | Models | Training Time | Val F1 Lift vs Voting | Use Case |
|--------------|---------|---------------|---------------------|----------|
| Boosting Blend | XGB+LGB+Cat | 5-10 min | +0.01-0.03 | Fast baseline |
| Boosting Pair | XGB+LGB | 3-5 min | +0.01-0.02 | Speed-critical |
| RNN Blend | LSTM+GRU | 20-40 min (GPU) | +0.02-0.04 | Temporal, fast |
| No Retrain | XGB+LGB+Cat | 3-5 min | +0.01-0.02 | Fastest |

**Note:** Blending is typically 2-3x faster than stacking with comparable performance (slightly lower).

---

## When to Use Blending

### ✅ Use Blending When:

1. **Large datasets** (>10k samples)
   - Holdout set will be large enough (>1k samples)
   - Base models won't suffer from reduced training data

2. **Training time matters**
   - Need quick prototypes
   - Iterating on ensemble configurations
   - Base models train slowly (neural networks)

3. **Simple deployment**
   - Prefer simpler training pipeline
   - Fewer moving parts than stacking

### ❌ Use Stacking Instead When:

1. **Small datasets** (<5k samples)
   - Need maximum data efficiency
   - k-fold provides better coverage

2. **Maximum performance needed**
   - Production deployment
   - Every 0.01 F1 matters
   - Statistical rigor important

3. **Base models train fast**
   - Boosting models (<1 min each)
   - k-fold overhead is acceptable

---

## Next Steps

- **Stacking:** For maximum performance, see `STACKING_GUIDE.md`
- **Voting:** For simpler baseline, see `VOTING_GUIDE.md`
- **Compatibility:** Check model combinations in `ENSEMBLE_COMPATIBILITY.md`
- **Cross-Validation:** Run `scripts/run_cv.py` for ensemble validation
