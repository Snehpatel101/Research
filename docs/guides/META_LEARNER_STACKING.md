# Meta-Learner Stacking Guide

Comprehensive guide for building heterogeneous ensembles via meta-learner stacking in the ML Model Factory.

**Last Updated:** 2026-01-01

---

## Table of Contents

1. [Overview](#overview)
2. [Heterogeneous Ensemble Architecture](#heterogeneous-ensemble-architecture)
3. [Base Model Selection](#base-model-selection)
4. [OOF Generation Protocol](#oof-generation-protocol)
5. [Meta-Learner Options](#meta-learner-options)
6. [Full Training Protocol](#full-training-protocol)
7. [Test Evaluation](#test-evaluation)
8. [CLI Reference](#cli-reference)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The ML Model Factory uses **heterogeneous ensemble stacking** where base models from different families (tabular, sequence, transformer) feed a single meta-learner.

**Key Principles:**
- **Heterogeneous > Homogeneous:** Different model families capture different patterns
- **Direct Stacking:** Meta-learner trained directly on OOF predictions from bases
- **1 Model per Family:** Select representative model from each family for diversity
- **Leakage-Free:** OOF generation with PurgedKFold prevents data leakage

**Architecture:**
```
Tabular (CatBoost) ──┐
                     │
CNN/TCN (TCN) ───────┼──> OOF Predictions ──> Meta-Learner ──> Final Predictions
                     │
Transformer (PatchTST)┘
```

---

## Heterogeneous Ensemble Architecture

### Why Heterogeneous?

**Diversity of Inductive Biases:**
- Tabular models excel at feature interactions and engineered indicators
- CNN/TCN models capture local temporal patterns and multi-scale features
- Transformers capture long-range dependencies and global context

**Reduced Error Correlation:**
- Errors from diverse model families are less correlated
- Meta-learner can learn when to trust each base model
- Overall ensemble is more robust than any single family

**Comparison:**

| Ensemble Type | Base Selection | Error Correlation | Diversity |
|---------------|----------------|-------------------|-----------|
| **Homogeneous** | Same family (XGB+LGB+Cat) | High | Low |
| **Heterogeneous** | Different families (Cat+TCN+PatchTST) | Low | High |

### Architectural Flow

```
Phase 1: Base Model OOF Generation
─────────────────────────────────────────────────────────
For each base model (e.g., CatBoost, TCN, PatchTST):
  - Run PurgedKFold (5 folds, purge=60, embargo=1440)
  - Generate OOF predictions for full training set
  - Output: (n_samples, n_classes) per model

Phase 2: Meta-Learner Training
─────────────────────────────────────────────────────────
  - Stack OOF predictions: (n_samples, n_models * n_classes)
  - Train meta-learner (Logistic/Ridge/MLP) on stacked OOF
  - Output: Trained meta-learner

Phase 3: Base Model Full Retrain
─────────────────────────────────────────────────────────
  - Retrain all base models on FULL training set
  - Use same hyperparameters from OOF phase
  - Output: Final base models

Phase 4: Test Evaluation
─────────────────────────────────────────────────────────
  - Base models predict on test set
  - Stack base predictions: (n_test, n_models * n_classes)
  - Meta-learner combines for final predictions
```

---

## Base Model Selection

### Selection Criteria

**One model per family for maximum diversity:**

| Family | Recommended | Alternatives | Strengths |
|--------|-------------|--------------|-----------|
| **Tabular** | CatBoost | LightGBM, XGBoost | Feature interactions, fast training |
| **CNN/TCN** | TCN | 1D ResNet, InceptionTime | Local temporal patterns, multi-scale |
| **Transformer** | PatchTST | iTransformer, TFT | Long-range dependencies |
| **Linear (optional)** | Ridge | Logistic | Baseline diversity, regularization |

### Recommended Configurations

**3-Base Standard:**
```python
base_models = ["catboost", "tcn", "patchtst"]
meta_learner = "logistic"
```
Best for: Balanced diversity, moderate training time

**4-Base Maximum:**
```python
base_models = ["lightgbm", "tcn", "tft", "ridge"]
meta_learner = "ridge"
```
Best for: Maximum diversity, longer training time

**2-Base Minimal:**
```python
base_models = ["xgboost", "lstm"]
meta_learner = "logistic"
```
Best for: Fast prototyping, quick experiments

### Model Compatibility

All base models must produce compatible prediction outputs:
- **Classification:** `(n_samples, n_classes)` probability matrix
- **Regression:** `(n_samples, 1)` prediction vector

The meta-learner receives concatenated predictions from all bases.

---

## OOF Generation Protocol

### PurgedKFold Configuration

```python
from src.cross_validation.purged_kfold import PurgedKFold

kfold = PurgedKFold(
    n_splits=5,           # 5 folds for robust OOF
    purge_bars=60,        # 3x max horizon (prevents label leakage)
    embargo_bars=1440,    # 5 days at 5min (prevents serial correlation)
)
```

### OOF Generation Process

```python
from src.cross_validation.oof_generator import generate_oof_predictions

# Generate OOF for each base model
oof_catboost = generate_oof_predictions(
    model_class=CatBoostModel,
    X_train=X_train,
    y_train=y_train,
    kfold=kfold,
    model_config=catboost_config
)
# Shape: (n_samples, n_classes) e.g., (15000, 3)

oof_tcn = generate_oof_predictions(
    model_class=TCNModel,
    X_train=X_train_seq,  # 3D for sequence models
    y_train=y_train,
    kfold=kfold,
    model_config=tcn_config
)

oof_patchtst = generate_oof_predictions(
    model_class=PatchTSTModel,
    X_train=X_train_4d,  # 4D for advanced models
    y_train=y_train,
    kfold=kfold,
    model_config=patchtst_config
)
```

### OOF Stacking

```python
import numpy as np

# Stack OOF predictions as meta-features
stacked_oof = np.hstack([
    oof_catboost,   # (n_samples, 3)
    oof_tcn,        # (n_samples, 3)
    oof_patchtst,   # (n_samples, 3)
])
# Shape: (n_samples, 9) for 3 models * 3 classes
```

### OOF Validation

```python
from src.cross_validation.oof_validation import validate_oof_coverage

# Ensure OOF covers full training set
coverage = validate_oof_coverage(stacked_oof, y_train)
assert coverage > 0.95, f"OOF coverage too low: {coverage}"
```

---

## Meta-Learner Options

### Logistic Regression (Default)

```python
from sklearn.linear_model import LogisticRegression

meta_learner = LogisticRegression(
    C=1.0,                    # Regularization strength
    penalty='l2',             # L2 regularization
    solver='lbfgs',           # Efficient solver
    max_iter=1000,
    multi_class='multinomial',
    random_state=42
)

meta_learner.fit(stacked_oof, y_train)
```

**Best for:** Calibrated probabilities, interpretable weights

### Ridge Regression

```python
from sklearn.linear_model import RidgeClassifier

meta_learner = RidgeClassifier(
    alpha=1.0,               # Regularization strength
    random_state=42
)

meta_learner.fit(stacked_oof, y_train)
```

**Best for:** Continuous predictions, strong regularization

### Small MLP

```python
from sklearn.neural_network import MLPClassifier

meta_learner = MLPClassifier(
    hidden_layer_sizes=(32, 16),  # 2 small hidden layers
    activation='relu',
    solver='adam',
    alpha=0.01,                    # L2 regularization
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

meta_learner.fit(stacked_oof, y_train)
```

**Best for:** Non-linear blending, complex interactions

### Calibrated Blender

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

base_meta = LogisticRegression(C=1.0, max_iter=1000)
meta_learner = CalibratedClassifierCV(
    base_meta,
    method='isotonic',  # or 'sigmoid'
    cv=3
)

meta_learner.fit(stacked_oof, y_train)
```

**Best for:** Calibrated confidence scores, probability calibration

### Meta-Learner Comparison

| Meta-Learner | Training Time | Overfitting Risk | Interpretable | Best For |
|--------------|---------------|------------------|---------------|----------|
| **Logistic** | <1 sec | Low | Yes (weights) | Default choice |
| **Ridge** | <1 sec | Low | Yes (weights) | Strong regularization |
| **Small MLP** | 5-30 sec | Medium | No | Complex patterns |
| **Calibrated** | 2-5 sec | Low | Yes | Probability calibration |

---

## Full Training Protocol

### Complete Training Pipeline

```python
from src.ensemble.heterogeneous_stacker import HeterogeneousStacker

# Initialize stacker
stacker = HeterogeneousStacker(
    base_model_names=["catboost", "tcn", "patchtst"],
    meta_learner_name="logistic",
    n_folds=5,
    purge_bars=60,
    embargo_bars=1440
)

# Phase 1: Generate OOF predictions
stacker.generate_oof(X_train, y_train)

# Phase 2: Train meta-learner
stacker.train_meta_learner()

# Phase 3: Full retrain base models
stacker.retrain_bases_full(X_train, y_train)

# Save ensemble
stacker.save("experiments/runs/run_001/ensemble/")
```

### Step-by-Step Protocol

**Step 1: Data Preparation**
```python
from src.models.data_preparation import load_data_container

container = load_data_container("data/splits/scaled/")
X_train, y_train, weights_train = container.get_train_data()
X_test, y_test, weights_test = container.get_test_data()
```

**Step 2: OOF Generation**
```python
# For each base model
oof_predictions = {}
for model_name in base_model_names:
    model_class = ModelRegistry.get(model_name)
    oof = generate_oof_predictions(model_class, X_train, y_train, kfold)
    oof_predictions[model_name] = oof
```

**Step 3: Meta-Learner Training**
```python
# Stack OOF predictions
stacked_oof = np.hstack([oof_predictions[name] for name in base_model_names])

# Train meta-learner
meta_learner.fit(stacked_oof, y_train)
```

**Step 4: Full Retrain**
```python
# Retrain base models on full training set
final_models = {}
for model_name in base_model_names:
    model_class = ModelRegistry.get(model_name)
    model = model_class()
    model.fit(X_train, y_train, X_val, y_val)  # Use val for early stopping
    final_models[model_name] = model
```

**Step 5: Save Ensemble**
```python
# Save all components
for name, model in final_models.items():
    model.save(f"experiments/runs/run_001/models/{name}.pkl")
joblib.dump(meta_learner, "experiments/runs/run_001/ensemble/meta_learner.pkl")
```

---

## Test Evaluation

### Evaluation Protocol

```python
def evaluate_ensemble(final_models, meta_learner, X_test, y_test):
    """Evaluate heterogeneous ensemble on test set."""

    # Get base model predictions
    base_predictions = []
    for model_name, model in final_models.items():
        pred = model.predict(X_test)  # Returns PredictionOutput
        base_predictions.append(pred.probabilities)  # (n_test, n_classes)

    # Stack predictions
    stacked_test = np.hstack(base_predictions)  # (n_test, n_models * n_classes)

    # Meta-learner combines
    final_probs = meta_learner.predict_proba(stacked_test)
    final_preds = np.argmax(final_probs, axis=1)

    # Compute metrics
    from src.models.metrics import compute_classification_metrics
    metrics = compute_classification_metrics(y_test, final_preds, final_probs)

    return metrics
```

### Metrics Computed

```python
metrics = {
    'accuracy': 0.68,
    'f1_macro': 0.65,
    'f1_weighted': 0.67,
    'precision': 0.66,
    'recall': 0.65,
    'confusion_matrix': [[...], [...], [...]],
    'per_class_f1': [0.62, 0.68, 0.65],  # [long, neutral, short]
}
```

---

## CLI Reference

### Basic Usage

```bash
# Train 3-base heterogeneous ensemble
python scripts/train_ensemble.py \
  --base-models catboost,tcn,patchtst \
  --meta-learner logistic \
  --horizon 20

# Train 4-base ensemble with Ridge meta-learner
python scripts/train_ensemble.py \
  --base-models lightgbm,tcn,tft,ridge \
  --meta-learner ridge \
  --horizon 20

# Fast 2-base ensemble
python scripts/train_ensemble.py \
  --base-models xgboost,lstm \
  --meta-learner logistic \
  --horizon 20
```

### Advanced Options

```bash
# Custom OOF folds
python scripts/train_ensemble.py \
  --base-models catboost,tcn,patchtst \
  --meta-learner logistic \
  --n-folds 3 \
  --horizon 20

# Custom purge/embargo
python scripts/train_ensemble.py \
  --base-models catboost,tcn,patchtst \
  --meta-learner logistic \
  --purge-bars 90 \
  --embargo-bars 2000 \
  --horizon 20

# MLP meta-learner with custom config
python scripts/train_ensemble.py \
  --base-models catboost,tcn,patchtst \
  --meta-learner mlp \
  --meta-config '{"hidden_layer_sizes": [64, 32], "alpha": 0.01}' \
  --horizon 20
```

### Output Structure

```
experiments/runs/{run_id}/
  models/
    catboost.pkl
    tcn.pt
    patchtst.pt
  ensemble/
    meta_learner.pkl
    oof_predictions.npz
    config.yaml
  metrics/
    ensemble_metrics.json
    base_model_metrics.json
```

---

## Best Practices

### Base Model Selection

1. **Select 1 model per family:** Maximize diversity, avoid redundancy
2. **Start with proven models:** CatBoost, TCN, PatchTST are solid defaults
3. **Consider training time:** Balance diversity vs. compute budget
4. **Include a linear baseline:** Ridge adds regularization and diversity

### OOF Generation

1. **Use 5 folds:** Robust OOF estimates with acceptable compute
2. **Maintain purge/embargo:** Prevent leakage (60 purge, 1440 embargo)
3. **Validate OOF coverage:** Ensure >95% coverage before meta-learner training
4. **Save OOF predictions:** Allows retraining meta-learner without rerunning OOF

### Meta-Learner Training

1. **Start with Logistic:** Simple, fast, calibrated probabilities
2. **Use regularization:** Prevent overfitting on small OOF sets
3. **Cross-validate meta-learner:** Use held-out fold to tune meta-learner hyperparams
4. **Consider Calibrated Blender:** If probability calibration is important

### Full Retrain

1. **Always retrain bases:** OOF models were trained on partial data
2. **Use same hyperparameters:** Don't re-tune during full retrain
3. **Use validation for early stopping:** Prevent overfitting during retrain

---

## Troubleshooting

### Issue: OOF coverage < 95%

**Causes:**
- Purge/embargo too aggressive
- Small dataset
- Label end times missing

**Solutions:**
- Reduce purge/embargo (minimum: purge=30, embargo=480)
- Use fewer folds (3 instead of 5)
- Ensure label_end_time column exists

### Issue: Meta-learner overfits

**Symptoms:**
- High OOF accuracy, low test accuracy

**Solutions:**
- Increase regularization (higher C for Logistic, higher alpha for Ridge)
- Use simpler meta-learner (Logistic instead of MLP)
- Reduce base model count (3 instead of 4)

### Issue: Base model predictions not aligned

**Symptoms:**
- Shape mismatch when stacking predictions

**Solutions:**
- Ensure all base models output same n_classes
- Verify sequence models return predictions for valid samples only
- Check for NaN/inf in OOF predictions

### Issue: Slow OOF generation

**Solutions:**
- Use fewer folds (3 instead of 5)
- Use faster base models (XGBoost instead of CatBoost)
- Enable GPU for neural/transformer models
- Parallelize OOF generation across models

---

## References

- **Architecture:** `docs/ARCHITECTURE.md` (Ensemble Architecture section)
- **OOF Generation:** `src/cross_validation/oof_generator.py`
- **Stacking Implementation:** `src/ensemble/heterogeneous_stacker.py`
- **Model Integration:** `docs/guides/MODEL_INTEGRATION.md`
