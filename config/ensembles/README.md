# Ensemble Configurations

Pre-configured ensemble templates for common model combinations.

## Available Ensembles

| Ensemble | Base Models | Method | Input Type | Training Time | F1 Baseline |
|----------|-------------|--------|------------|---------------|-------------|
| [Boosting Trio](boosting_trio.yaml) | XGB + LGB + CAT | Voting | Tabular (2D) | ~45 min | 0.52 |
| [Temporal Stack](temporal_stack.yaml) | LSTM + GRU + TCN | Stacking | Sequence (3D) | ~180 min | 0.54 |

## Ensemble Methods

### Voting
Combines predictions via weighted or unweighted averaging.

**Pros:**
- Simple and fast
- No additional training required
- Works well when base models are diverse

**Cons:**
- Cannot learn which model to trust in which situation
- Equal treatment of all base models (unless manually weighted)

**Example:**
```bash
python scripts/train_model.py \
  --model voting \
  --base-models xgboost,lightgbm,catboost \
  --horizon 20
```

### Stacking
Trains a meta-learner on out-of-fold (OOF) predictions from base models.

**Pros:**
- Learns when to trust each base model
- Can capture model interactions
- Generally highest accuracy

**Cons:**
- Requires additional training (meta-learner)
- More complex, higher risk of overfitting
- Needs proper OOF generation (no leakage)

**Example:**
```bash
python scripts/train_model.py \
  --model stacking \
  --base-models xgboost,lightgbm,random_forest \
  --horizon 20
```

### Blending
Similar to stacking but uses holdout set instead of OOF predictions.

**Pros:**
- Simpler than stacking (no CV required)
- Faster to train

**Cons:**
- Uses less data for meta-learner training
- Can be less robust than stacking

**Example:**
```bash
python scripts/train_model.py \
  --model blending \
  --base-models xgboost,lightgbm,catboost \
  --horizon 20
```

## Ensemble Compatibility

**CRITICAL:** All base models in an ensemble must have the same input shape.

### Valid Combinations

**Tabular-Only Ensembles (2D input):**
- ✓ xgboost + lightgbm + catboost
- ✓ xgboost + lightgbm + random_forest
- ✓ random_forest + logistic + svm
- ✓ All tabular models (6 total)

**Sequence-Only Ensembles (3D input):**
- ✓ lstm + gru
- ✓ lstm + gru + tcn
- ✓ All neural models (4 total)

### Invalid Combinations

**Mixed Ensembles (WILL FAIL):**
- ✗ xgboost + lstm (mixing 2D + 3D)
- ✗ lightgbm + gru + tcn (mixing 2D + 3D)
- ✗ random_forest + transformer (mixing 2D + 3D)

**Error:**
```
EnsembleCompatibilityError: Cannot mix tabular and sequence models.
Tabular models: ['xgboost'], Sequence models: ['lstm']
```

## Configuration Structure

### Voting Ensemble
```yaml
ensemble:
  name: {ensemble_name}
  method: voting
  description: {description}

base_models:
  - model1
  - model2
  - model3

voting:
  weights: null              # null = equal weights, or [1.0, 1.0, 1.0]
  strategy: soft             # soft (probability avg) or hard (majority vote)

training:
  train_base_models: true    # Train from scratch or load existing
  use_cv: true               # Use CV for base model training
  n_splits: 5

data:
  horizon: 20
  feature_set: {boosting_optimal | neural_optimal}

expected:
  training_time_minutes: {time}
  memory_gb: {memory}
  baseline_f1: {f1_score}
```

### Stacking Ensemble
```yaml
ensemble:
  name: {ensemble_name}
  method: stacking
  description: {description}

base_models:
  - model1
  - model2
  - model3

stacking:
  meta_learner: logistic     # Must be tabular model
  use_oof: true              # Use OOF predictions (recommended)
  passthrough: false         # Include original features
  cv_splits: 5
  cv_purge: 60
  cv_embargo: 60

training:
  train_base_models: true
  sequence_length: 60        # For sequence ensembles

data:
  horizon: 20
  feature_set: {boosting_optimal | neural_optimal}
```

## Usage Examples

### Boosting Trio (Fast Baseline)

**Config:** [boosting_trio.yaml](boosting_trio.yaml)

**Description:** Fast voting ensemble combining three gradient boosting models (XGBoost + LightGBM + CatBoost).

**Use Case:** Quick baseline with good accuracy.

**Train:**
```bash
# Using config file
python scripts/train_model.py \
  --model voting \
  --base-models xgboost,lightgbm,catboost \
  --horizon 20

# With custom weights
python scripts/train_model.py \
  --model voting \
  --base-models xgboost,lightgbm,catboost \
  --horizon 20 \
  --override "voting.weights=[1.5,1.0,1.0]"  # Favor XGBoost
```

**Expected:**
- Training time: ~45 minutes (RTX 4070 Ti)
- Memory: ~8 GB
- Baseline F1: 0.52

### Temporal Stack (Best Accuracy)

**Config:** [temporal_stack.yaml](temporal_stack.yaml)

**Description:** Stacking ensemble for sequential patterns (LSTM + GRU + TCN with logistic meta-learner).

**Use Case:** Maximum accuracy for temporal dependencies.

**Train:**
```bash
# Using config file
python scripts/train_model.py \
  --model stacking \
  --base-models lstm,gru,tcn \
  --horizon 20 \
  --seq-len 60

# With different meta-learner
python scripts/train_model.py \
  --model stacking \
  --base-models lstm,gru,tcn \
  --horizon 20 \
  --seq-len 60 \
  --override "stacking.meta_learner=random_forest"
```

**Expected:**
- Training time: ~180 minutes (RTX 4070 Ti)
- Memory: ~12 GB
- Baseline F1: 0.54

## Creating Custom Ensembles

### Step 1: Choose Base Models
Ensure all models have the same input shape:
```bash
# Check model input types
python -c "
from src.models import ModelRegistry
for name in ['xgboost', 'lstm', 'random_forest']:
    model_class = ModelRegistry.get(name)
    print(f'{name}: {model_class.input_type}')
"
# Output:
# xgboost: tabular
# lstm: sequence
# random_forest: tabular
```

### Step 2: Create Config File
```yaml
# config/ensembles/my_custom_ensemble.yaml
ensemble:
  name: my_custom_ensemble
  method: voting
  description: Custom ensemble description

base_models:
  - xgboost
  - lightgbm
  - random_forest

voting:
  weights: [1.5, 1.0, 0.8]   # Custom weights
  strategy: soft

training:
  train_base_models: true
  use_cv: true
  n_splits: 5

data:
  horizon: 20
  feature_set: boosting_optimal
```

### Step 3: Train
```bash
python scripts/train_model.py \
  --model voting \
  --base-models xgboost,lightgbm,random_forest \
  --horizon 20
```

### Step 4: Evaluate
```bash
# Cross-validation
python scripts/run_cv.py \
  --models voting \
  --base-models xgboost,lightgbm,random_forest \
  --horizons 20 \
  --n-splits 5

# Walk-forward validation
python scripts/run_walk_forward.py \
  --models voting \
  --base-models xgboost,lightgbm,random_forest \
  --horizons 20
```

## Recommended Configurations

### For Speed (< 1 hour)
**Boosting Duo:**
```yaml
base_models: [xgboost, lightgbm]
method: voting
```

### For Accuracy (Best F1)
**All Boosting Stack:**
```yaml
base_models: [xgboost, lightgbm, catboost, random_forest]
method: stacking
meta_learner: xgboost
```

### For Temporal Patterns
**RNN Variants:**
```yaml
base_models: [lstm, gru]
method: voting
```

### For Maximum Diversity
**All Tabular Stack:**
```yaml
base_models: [xgboost, lightgbm, catboost, random_forest, logistic, svm]
method: stacking
meta_learner: xgboost
```

## Troubleshooting

### Issue: Ensemble Compatibility Error
```
EnsembleCompatibilityError: Cannot mix tabular and sequence models
```
**Solution:** Ensure all base models have same input type. Check with:
```python
from src.models import ModelRegistry
models = ['xgboost', 'lstm']
for name in models:
    print(f"{name}: {ModelRegistry.get(name).input_type}")
```

### Issue: OOF Generation Fails
```
ValueError: Not enough samples for CV with n_splits=5
```
**Solution:** Reduce `cv_splits` or use more training data.

### Issue: Meta-Learner Training Error
```
ValueError: Meta-learner must be a tabular model
```
**Solution:** Choose a tabular meta-learner (xgboost, lightgbm, logistic, random_forest, etc.)

## Related Documentation

- [Phase 7 Meta-Learner Stacking](../../docs/implementation/PHASE_7_META_LEARNER_STACKING.md) - Ensemble architecture
- [Model Integration Guide](../../docs/guides/MODEL_INTEGRATION.md) - Adding models
- [Meta-Learner Stacking Guide](../../docs/guides/META_LEARNER_STACKING.md) - Heterogeneous ensemble training

---

*Last Updated: 2025-12-30*
