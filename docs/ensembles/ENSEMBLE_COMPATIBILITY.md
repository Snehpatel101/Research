# Ensemble Model Compatibility Guide

## Overview

**CRITICAL LIMITATION:** Ensembles **CANNOT** mix tabular and sequence models due to incompatible input shapes. This guide explains why, what works, and how to validate configurations.

**Validation:** `/home/user/Research/src/models/ensemble/validator.py`

---

## The Problem: Input Shape Mismatch

### Tabular Models (2D Input)

```python
X_tabular.shape = (n_samples, n_features)
# Example: (10000, 150) - 10k samples, 150 features
```

**Models:**
- `xgboost`, `lightgbm`, `catboost` (boosting)
- `random_forest`, `logistic`, `svm` (classical)

### Sequence Models (3D Input)

```python
X_sequence.shape = (n_samples, seq_len, n_features)
# Example: (10000, 60, 150) - 10k samples, 60 timesteps, 150 features per timestep
```

**Models:**
- `lstm`, `gru`, `tcn`, `transformer` (neural)

### Why You Can't Mix Them

```python
# Ensemble tries to train on same data
ensemble = VotingEnsemble(base_model_names=["xgboost", "lstm"])

# During fit()
ensemble.fit(X_train, y_train, X_val, y_val)

# What happens:
# 1. Trainer detects LSTM needs sequences
# 2. Prepares X_train as 3D: (10000, 60, 150)
# 3. Passes to XGBoost → FAILS (expects 2D)
#
# OR:
# 1. Trainer detects XGBoost needs tabular
# 2. Prepares X_train as 2D: (10000, 150)
# 3. Passes to LSTM → FAILS (expects 3D)
#
# No way to satisfy both!
```

**Result:** `EnsembleCompatibilityError` raised at training time.

---

## Valid Ensemble Configurations

### Tabular-Only Ensembles ✅

All models expect 2D input `(n_samples, n_features)`:

| Family | Models | Example Config |
|--------|--------|----------------|
| **Boosting** | `xgboost`, `lightgbm`, `catboost` | `["xgboost", "lightgbm", "catboost"]` |
| **Classical** | `random_forest`, `logistic`, `svm` | `["random_forest", "logistic", "svm"]` |
| **Mixed Tabular** | Any combination of above | `["xgboost", "lightgbm", "random_forest"]` |

**Training:**
```bash
# Boosting trio
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Boosting + Classical
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,random_forest

# All tabular (6 models)
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm,catboost,random_forest,logistic,svm
```

### Sequence-Only Ensembles ✅

All models expect 3D input `(n_samples, seq_len, n_features)`:

| Family | Models | Example Config |
|--------|--------|----------------|
| **RNN** | `lstm`, `gru` | `["lstm", "gru"]` |
| **Temporal** | `lstm`, `gru`, `tcn` | `["lstm", "gru", "tcn"]` |
| **All Neural** | `lstm`, `gru`, `tcn`, `transformer` | `["lstm", "gru", "tcn", "transformer"]` |

**Training:**
```bash
# RNN variants
python scripts/train_model.py --model voting --horizon 20 \
  --base-models lstm,gru --seq-len 60

# Temporal stack
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models lstm,gru,tcn --seq-len 60

# All neural
python scripts/train_model.py --model blending --horizon 20 \
  --base-models lstm,gru,tcn,transformer --seq-len 60
```

---

## Invalid Configurations ❌

### Common Mistakes

#### 1. Mixing Boosting + Neural

```bash
# WILL FAIL
python scripts/train_model.py --model voting \
  --base-models xgboost,lightgbm,lstm

# Error: Cannot mix tabular (xgboost, lightgbm) and sequence (lstm) models
```

#### 2. Mixing Classical + Neural

```bash
# WILL FAIL
python scripts/train_model.py --model stacking \
  --base-models random_forest,logistic,gru

# Error: Cannot mix tabular (random_forest, logistic) and sequence (gru) models
```

#### 3. Single Neural in Tabular Ensemble

```bash
# WILL FAIL
python scripts/train_model.py --model blending \
  --base-models xgboost,lightgbm,catboost,lstm

# Error: Cannot mix tabular (3 models) and sequence (1 model) models
```

#### 4. Single Tabular in Neural Ensemble

```bash
# WILL FAIL
python scripts/train_model.py --model voting \
  --base-models lstm,gru,tcn,xgboost

# Error: Cannot mix tabular (xgboost) and sequence (3 models) models
```

---

## Validation Behavior

### Automatic Validation

All ensemble classes validate compatibility automatically:

```python
from src.models import ModelRegistry

# Validation happens during fit()
ensemble = ModelRegistry.create("voting", config={
    "base_model_names": ["xgboost", "lstm"],  # Invalid!
})

try:
    ensemble.fit(X_train, y_train, X_val, y_val)
except EnsembleCompatibilityError as e:
    print(e)
    # Shows detailed error with suggestions
```

### Error Message Format

```
Ensemble Compatibility Error: Cannot mix tabular and sequence models.

REASON:
  - Tabular models expect 2D input: (n_samples, n_features)
  - Sequence models expect 3D input: (n_samples, seq_len, n_features)
  - Mixed ensembles would cause shape mismatches during training/prediction

YOUR CONFIGURATION:
  Tabular models (2D): ['xgboost', 'lightgbm']
  Sequence models (3D): ['lstm']

SUPPORTED ENSEMBLE CONFIGURATIONS:

✅ All Tabular Models:
  - Boosting: xgboost, lightgbm, catboost
  - Classical: random_forest, logistic, svm
  - Example: base_model_names=['xgboost', 'lightgbm', 'random_forest']

✅ All Sequence Models:
  - Neural: lstm, gru, tcn, transformer
  - Example: base_model_names=['lstm', 'gru', 'tcn']

❌ Mixed Models (NOT SUPPORTED):
  - Example: base_model_names=['xgboost', 'lstm']  # WILL FAIL

RECOMMENDATIONS:
  - Use only tabular models: ['xgboost', 'lightgbm']
  - Use only sequence models: ['lstm']

For more information, see docs/phases/PHASE_4.md
```

---

## Validation Utilities

### Check Configuration Before Training

```python
from src.models.ensemble import validate_ensemble_config

# Validate a configuration
is_valid, error_msg = validate_ensemble_config(["xgboost", "lightgbm", "lstm"])

if not is_valid:
    print("Invalid configuration:")
    print(error_msg)
else:
    print("Configuration is valid!")
```

### Get Compatible Models

```python
from src.models.ensemble import get_compatible_models

# Get models compatible with xgboost (tabular)
tabular_models = get_compatible_models("xgboost")
print(tabular_models)
# ['catboost', 'lightgbm', 'logistic', 'random_forest', 'svm', 'xgboost']

# Get models compatible with lstm (sequence)
sequence_models = get_compatible_models("lstm")
print(sequence_models)
# ['gru', 'lstm', 'tcn', 'transformer']
```

### Programmatic Validation with Exceptions

```python
from src.models.ensemble import (
    validate_base_model_compatibility,
    EnsembleCompatibilityError,
)

try:
    validate_base_model_compatibility(["xgboost", "lstm"])
except EnsembleCompatibilityError as e:
    print(f"Invalid ensemble: {e}")
    # Raises exception with detailed error message
```

---

## Compatibility Matrix

### Full Compatibility Table

|  | xgboost | lightgbm | catboost | random_forest | logistic | svm | lstm | gru | tcn | transformer |
|---|---------|----------|----------|---------------|----------|-----|------|-----|-----|-------------|
| **xgboost** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **lightgbm** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **catboost** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **random_forest** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **logistic** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **svm** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **lstm** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **gru** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **tcn** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **transformer** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |

**Legend:**
- ✅ Compatible (can be combined in ensemble)
- ❌ Incompatible (cannot be combined)

---

## Recommended Ensemble Combinations

### Tabular Ensembles (By Use Case)

| Use Case | Models | Reasoning |
|----------|--------|-----------|
| **Fast Baseline** | `xgboost`, `lightgbm`, `catboost` | 3 strong boosting models, fast training |
| **Balanced** | `xgboost`, `lightgbm`, `random_forest` | Boosting + bagging diversity |
| **Maximum Diversity** | All 6 tabular models | High diversity, risk of overfitting |
| **Speed-Critical** | `xgboost`, `lightgbm` | 2 models, minimal overhead |
| **Interpretable** | `logistic`, `random_forest` | Linear + tree-based, both interpretable |

### Sequence Ensembles (By Use Case)

| Use Case | Models | Reasoning |
|----------|--------|-----------|
| **RNN Variants** | `lstm`, `gru` | Similar architectures, different gates |
| **Temporal Stack** | `lstm`, `gru`, `tcn` | RNN + convolution diversity |
| **Maximum Diversity** | `lstm`, `gru`, `tcn`, `transformer` | All neural architectures |
| **Fast Inference** | `gru`, `tcn` | GRU is faster than LSTM, TCN is parallelizable |
| **Long Sequences** | `transformer`, `tcn` | Better at long-range dependencies |

---

## Future: Hybrid Ensembles (Not Supported)

### What Would Be Needed

To support mixed tabular + sequence ensembles:

1. **Dual Data Preparation:**
   ```python
   # Prepare both formats
   X_tabular = X.reshape(n_samples, -1)  # 2D for tabular
   X_sequence = X.reshape(n_samples, seq_len, n_features)  # 3D for sequence
   ```

2. **Model-Specific Input Shaping:**
   ```python
   # During training
   for model in base_models:
       if model.requires_sequences:
           model.fit(X_sequence, y)
       else:
           model.fit(X_tabular, y)
   ```

3. **Prediction Routing:**
   ```python
   # During inference
   for model in base_models:
       if model.requires_sequences:
           pred = model.predict(X_sequence)
       else:
           pred = model.predict(X_tabular)
   ```

### Why Not Implemented

**Complexity:**
- 2x data preparation overhead
- Complex prediction routing
- Higher bug risk

**Uncertain Benefits:**
- No empirical evidence of improvement
- May increase overfitting
- Resource intensive

**Simpler Alternatives:**
- Train separate ensembles (tabular + sequence)
- Combine predictions post-hoc
- Use feature engineering to convert sequences → tabular

**Current Recommendation:** Use same-family ensembles for proven benefits and simpler architecture.

---

## Troubleshooting

### Issue: "Model 'X' is not registered"

**Error:**
```
Model 'catboost' is not registered (CatBoost is optional - install with: pip install catboost)
```

**Solution:**
```bash
pip install catboost
```

Or remove from ensemble config:
```bash
python scripts/train_model.py --model voting \
  --base-models xgboost,lightgbm  # Remove catboost
```

### Issue: "Cannot mix tabular and sequence models"

**Error:**
```
EnsembleCompatibilityError: Cannot mix tabular and sequence models.
Tabular models (2D): ['xgboost']
Sequence models (3D): ['lstm']
```

**Solutions:**
1. Use only tabular: `--base-models xgboost,lightgbm,catboost`
2. Use only sequence: `--base-models lstm,gru,tcn`

### Issue: "Need at least 2 base models for ensemble"

**Error:**
```
Need at least 2 base models for ensemble, got 1
```

**Solution:** Add at least one more compatible model:
```bash
python scripts/train_model.py --model voting \
  --base-models xgboost,lightgbm  # Add lightgbm
```

---

## Best Practices

### 1. Validate Before Training

Always check compatibility before expensive training:

```python
from src.models.ensemble import validate_ensemble_config

models = ["xgboost", "lightgbm", "lstm"]
is_valid, error = validate_ensemble_config(models)

if not is_valid:
    print(f"Invalid: {error}")
    # Fix configuration before training
```

### 2. Use Compatible Model Discovery

Let the system suggest compatible models:

```python
from src.models.ensemble import get_compatible_models

# Start with one model, find compatible ones
compatible = get_compatible_models("xgboost")
print(f"Compatible with xgboost: {compatible}")
# Use these for ensemble
```

### 3. Start Simple, Add Diversity

```python
# Step 1: Start with 2 similar models
base_models = ["xgboost", "lightgbm"]

# Step 2: Add diversity (boosting → classical)
base_models = ["xgboost", "lightgbm", "random_forest"]

# Step 3: Maximum diversity (if needed)
base_models = ["xgboost", "lightgbm", "catboost", "random_forest", "logistic", "svm"]
```

### 4. Separate Tabular and Sequence Experiments

```bash
# Experiment 1: Tabular ensemble
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Experiment 2: Sequence ensemble
python scripts/train_model.py --model voting --horizon 20 \
  --base-models lstm,gru,tcn --seq-len 60

# Compare results, pick better approach
```

---

## Summary

**Key Takeaways:**
1. ✅ Tabular-only ensembles work (all 2D models)
2. ✅ Sequence-only ensembles work (all 3D models)
3. ❌ Mixed ensembles fail (incompatible input shapes)
4. ✅ Validation happens automatically at training time
5. ✅ Use validation utilities to check before training

**Recommended Tabular Combos:**
- Fast: `xgboost`, `lightgbm`, `catboost`
- Balanced: `xgboost`, `lightgbm`, `random_forest`
- Diverse: All 6 tabular models

**Recommended Sequence Combos:**
- Fast: `lstm`, `gru`
- Balanced: `lstm`, `gru`, `tcn`
- Diverse: `lstm`, `gru`, `tcn`, `transformer`

**Next Steps:**
- See `VOTING_GUIDE.md` for voting ensemble details
- See `STACKING_GUIDE.md` for stacking ensemble details
- See `BLENDING_GUIDE.md` for blending ensemble details
