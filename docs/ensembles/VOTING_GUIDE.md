# Voting Ensemble Guide

## Overview

Voting ensembles combine predictions from multiple base models by averaging (soft voting) or majority vote (hard voting). Simple, fast, and effective for reducing variance.

**Implementation:** `/home/user/Research/src/models/ensemble/voting.py`
**Config:** `/home/user/Research/config/models/voting.yaml`

---

## How It Works

### Soft Voting (Recommended)

Average class probabilities from all base models:

```
Model 1 probs: [0.2, 0.3, 0.5]  (predicts class 2)
Model 2 probs: [0.1, 0.6, 0.3]  (predicts class 1)
Model 3 probs: [0.3, 0.4, 0.3]  (predicts class 1)

Average probs:  [0.2, 0.43, 0.37]
Final prediction: Class 1 (argmax of average)
```

**Benefits:**
- Uses full probability distributions
- More robust to overconfident models
- Smoother decision boundaries

### Hard Voting

Majority vote of class predictions:

```
Model 1: Class 2
Model 2: Class 1
Model 3: Class 1

Final prediction: Class 1 (majority)
```

**Benefits:**
- Simple and interpretable
- Fast (no probability calculations)
- Works when probability calibration differs across models

---

## Configuration

### Basic Configuration

```yaml
# config/models/voting.yaml or custom config

voting: soft                    # "soft" or "hard"
weights: null                   # Equal weights (or [0.4, 0.3, 0.3])

base_model_names:
  - xgboost
  - lightgbm
  - catboost

base_model_configs: {}          # Use defaults

# Low-latency options
parallel: true                  # Run predictions in parallel
n_workers: 3                    # Thread pool size
```

### Weighted Voting

Give more weight to stronger models:

```yaml
voting: soft
weights: [0.5, 0.3, 0.2]        # Must match number of base models

base_model_names:
  - xgboost                     # 50% weight
  - lightgbm                    # 30% weight
  - catboost                    # 20% weight
```

**Weights are auto-normalized** to sum to 1.0.

### Custom Base Model Configs

Override hyperparameters for individual base models:

```yaml
base_model_names:
  - xgboost
  - lightgbm

base_model_configs:
  xgboost:
    max_depth: 8
    n_estimators: 200
  lightgbm:
    num_leaves: 64
    learning_rate: 0.05
```

---

## Recommended Configurations

### 1. Boosting Trio (Fast Baseline)

**Best for:** Quick, strong baseline with minimal tuning

```bash
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost
```

**Config:**
```yaml
voting: soft
base_model_names: [xgboost, lightgbm, catboost]
```

**Expected Performance:**
- Training time: 3-5 minutes (3 models × ~1-2 min each)
- Val F1: 0.01-0.03 improvement over single models
- Good diversity (different tree-building algorithms)

### 2. Boosting + Random Forest (Balanced)

**Best for:** Tree diversity with bagging

```bash
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,random_forest
```

**Expected Performance:**
- Training time: 4-6 minutes
- More stable predictions (RF adds bagging)
- Good for high-noise data

### 3. RNN Variants (Temporal Patterns)

**Best for:** Sequential pattern diversity

```bash
python scripts/train_model.py --model voting --horizon 20 \
  --base-models lstm,gru --seq-len 60
```

**Config:**
```yaml
voting: soft
base_model_names: [lstm, gru]
```

**Expected Performance:**
- Training time: 15-30 minutes (GPU) / 1-2 hours (CPU)
- Better for trend-following strategies
- Requires 3D sequence data

### 4. All Neural (Maximum Temporal Diversity)

**Best for:** Complex temporal patterns, resource-rich environments

```bash
python scripts/train_model.py --model voting --horizon 20 \
  --base-models lstm,gru,tcn,transformer --seq-len 60
```

**Expected Performance:**
- Training time: 30-60 minutes (GPU) / 2-4 hours (CPU)
- High variance, may overfit
- Best with large datasets (>50k samples)

---

## Low-Latency Mode (Parallel Predictions)

For production inference, enable parallel base model predictions:

```yaml
parallel: true                  # Enable parallel execution
n_workers: 3                    # Match number of base models
```

**Performance (3-model boosting ensemble):**
- Sequential: ~15ms (sum of individual latencies)
- Parallel: ~6ms (max of individual latencies + overhead)

**Why it works:** Boosting models (XGBoost, LightGBM, CatBoost) release the GIL during prediction, making ThreadPoolExecutor effective.

**Important:** Parallel mode is best for boosting models. Neural models on GPU may not benefit due to GPU serialization.

---

## Training from Scratch vs Pre-Trained Models

### Option 1: Train from Scratch

```python
from src.models import ModelRegistry

ensemble = ModelRegistry.create("voting", config={
    "voting": "soft",
    "base_model_names": ["xgboost", "lightgbm", "catboost"],
})

metrics = ensemble.fit(X_train, y_train, X_val, y_val)
```

### Option 2: Use Pre-Trained Models

```python
from src.models import ModelRegistry

# Train base models separately
xgb = ModelRegistry.create("xgboost")
lgb = ModelRegistry.create("lightgbm")
cat = ModelRegistry.create("catboost")

xgb.fit(X_train, y_train, X_val, y_val)
lgb.fit(X_train, y_train, X_val, y_val)
cat.fit(X_train, y_train, X_val, y_val)

# Create ensemble with pre-trained models
ensemble = ModelRegistry.create("voting")
ensemble.set_base_models([xgb, lgb, cat], weights=[0.5, 0.3, 0.2])

output = ensemble.predict(X_test)
```

---

## Compatibility Requirements

**CRITICAL:** All base models must have the same input shape.

### Valid Combinations

✅ **Tabular-only** (2D input: `(n_samples, n_features)`):
- `xgboost`, `lightgbm`, `catboost`, `random_forest`, `logistic`, `svm`

✅ **Sequence-only** (3D input: `(n_samples, seq_len, n_features)`):
- `lstm`, `gru`, `tcn`, `transformer`

### Invalid Combinations

❌ **Mixed tabular + sequence** (WILL FAIL):
```bash
# WILL FAIL: Cannot mix xgboost (2D) with lstm (3D)
python scripts/train_model.py --model voting \
  --base-models xgboost,lstm
```

**Error:**
```
EnsembleCompatibilityError: Cannot mix tabular and sequence models.
Tabular models (2D): ['xgboost']
Sequence models (3D): ['lstm']
```

See `/home/user/Research/docs/ensembles/ENSEMBLE_COMPATIBILITY.md` for details.

---

## CLI Usage

### Basic Training

```bash
# Tabular ensemble
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Sequence ensemble
python scripts/train_model.py --model voting --horizon 20 \
  --base-models lstm,gru,tcn --seq-len 60
```

### Custom Configuration

```bash
# Hard voting with custom weights
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm \
  --config '{"voting": "hard", "weights": [0.6, 0.4]}'

# Disable parallel prediction
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost \
  --config '{"parallel": false}'
```

---

## Feature Importance

Voting ensembles average feature importances from base models:

```python
ensemble = ModelRegistry.create("voting", config={
    "base_model_names": ["xgboost", "lightgbm", "random_forest"],
})

ensemble.fit(X_train, y_train, X_val, y_val)

# Get averaged feature importances
importance = ensemble.get_feature_importance()
```

**Aggregation:** Mean of importances from all base models (models without importance support are skipped).

---

## Troubleshooting

### Issue: Models have different input shapes

**Error:**
```
EnsembleCompatibilityError: Cannot mix tabular and sequence models
```

**Solution:** Use only tabular models OR only sequence models (see Compatibility section).

### Issue: Weights don't sum to 1.0

**Not an issue:** Weights are auto-normalized during training.

```python
ensemble = VotingEnsemble(config={"weights": [2, 3, 5]})
# Auto-normalized to [0.2, 0.3, 0.5]
```

### Issue: Slow prediction latency

**Solution:** Enable parallel predictions for boosting models:

```yaml
parallel: true
n_workers: 3
```

### Issue: Low ensemble improvement

**Possible causes:**
1. Base models are too similar (use diverse families)
2. Base models are overfit (try regularization)
3. Dataset is too small (need >5k samples per model)

**Solutions:**
- Mix boosting + classical models
- Tune base model hyperparameters
- Use stacking instead (learns optimal weighting)

---

## Performance Expectations

| Configuration | Models | Training Time | Val F1 Lift | Use Case |
|--------------|---------|---------------|-------------|----------|
| Boosting Trio | XGB+LGB+Cat | 3-5 min | +0.01-0.03 | Fast baseline |
| Boosting+Forest | XGB+LGB+RF | 4-6 min | +0.01-0.02 | Balanced |
| RNN Variants | LSTM+GRU | 15-30 min (GPU) | +0.02-0.04 | Temporal |
| All Neural | LSTM+GRU+TCN+Trans | 30-60 min (GPU) | +0.03-0.05 | Complex patterns |

**Note:** Performance varies by dataset, horizon, and base model quality. These are rough estimates based on internal testing.

---

## Next Steps

- **Stacking:** For more sophisticated meta-learning, see `STACKING_GUIDE.md`
- **Blending:** For faster alternative to stacking, see `BLENDING_GUIDE.md`
- **Compatibility:** Check model combinations in `ENSEMBLE_COMPATIBILITY.md`
- **Cross-Validation:** Run `scripts/run_cv.py` for ensemble validation
