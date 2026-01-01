# Heterogeneous Ensemble Architecture

## Overview

The ML Model Factory uses **heterogeneous ensemble stacking** where base models from different families (tabular, sequence, transformer) feed a single meta-learner via out-of-fold (OOF) predictions.

**Key Principle:** Diversity of inductive biases produces lower correlated errors and more robust predictions than homogeneous ensembles.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BASE MODEL SELECTION (1 PER FAMILY)                   │
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐ │
│  │   Tabular    │   │   CNN/TCN    │   │ Transformer  │   │  Linear  │ │
│  │  (CatBoost)  │   │    (TCN)     │   │  (PatchTST)  │   │  (Ridge) │ │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └────┬─────┘ │
│         │                  │                  │                 │       │
│         └──────────────────┴──────────────────┴─────────────────┘       │
│                                    ↓                                     │
└────────────────────────────────────┼────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    OOF GENERATION (PURGEDKFOLD)                         │
│                                                                          │
│  For each base model:                                                   │
│  - Run PurgedKFold (5 folds, purge=60, embargo=1440)                   │
│  - Generate OOF predictions: (n_samples, n_classes)                    │
│                                                                          │
│  Output: OOF predictions per model                                      │
└────────────────────────────────────┼────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    OOF STACKING                                         │
│                                                                          │
│  Stack OOF predictions horizontally:                                    │
│  stacked_oof = [oof_catboost, oof_tcn, oof_patchtst]                   │
│  Shape: (n_samples, n_models * n_classes)                              │
│                                                                          │
│  Example: 3 models * 3 classes = 9 meta-features                       │
└────────────────────────────────────┼────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    META-LEARNER TRAINING                                │
│                                                                          │
│  Train meta-learner on stacked OOF:                                    │
│  meta_learner.fit(stacked_oof, y_train)                                │
│                                                                          │
│  Options: Logistic, Ridge, Small MLP, Calibrated Blender               │
└────────────────────────────────────┼────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    FULL RETRAIN BASE MODELS                             │
│                                                                          │
│  Retrain all base models on FULL training set:                         │
│  - Use same hyperparameters from OOF phase                             │
│  - Output: Final base models ready for inference                       │
└────────────────────────────────────┼────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    TEST EVALUATION                                      │
│                                                                          │
│  1. Base models predict on test set                                    │
│  2. Stack base predictions: (n_test, n_models * n_classes)            │
│  3. Meta-learner combines for final predictions                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Base Model Selection

### Selection Criteria

Select **1 model per family** to maximize diversity:

| Family | Recommended | Alternatives | Why |
|--------|-------------|--------------|-----|
| **Tabular** | CatBoost | LightGBM, XGBoost | Feature interactions, engineered indicators |
| **CNN/TCN** | TCN | InceptionTime, 1D ResNet | Local temporal patterns, multi-scale |
| **Transformer** | PatchTST | iTransformer, TFT | Long-range dependencies |
| **Linear** (optional) | Ridge | Logistic | Baseline diversity, regularization |

### Recommended Configurations

**3-Base Standard (Recommended):**
```python
base_models = ["catboost", "tcn", "patchtst"]
meta_learner = "logistic"
```

**4-Base Maximum Diversity:**
```python
base_models = ["lightgbm", "tcn", "tft", "ridge"]
meta_learner = "ridge"
```

**2-Base Minimal (Prototyping):**
```python
base_models = ["xgboost", "lstm"]
meta_learner = "logistic"
```

---

## Why Heterogeneous > Homogeneous

### Diversity of Inductive Biases

| Model Family | Inductive Bias | Strengths |
|--------------|----------------|-----------|
| **Tabular (Boosting)** | Feature interactions | Engineered indicators, categorical features |
| **CNN/TCN** | Local patterns | Multi-scale temporal patterns |
| **Transformer** | Global context | Long-range dependencies, attention |
| **Linear** | Regularization | Stable baseline, interpretable |

### Diversity of Feature Representations

**Critical:** Each model family receives DIFFERENT features from the same canonical OHLCV source:

| Model | Primary TF | MTF Strategy | Features | Input Shape | Feature Type |
|-------|-----------|--------------|----------|-------------|--------------|
| **CatBoost** | 15min | MTF Indicators | ~200 | (N, 200) | Engineered + MTF |
| **TCN** | 5min | Single-TF | ~150 | (N, 60, 150) | Base indicators |
| **PatchTST** | 1min | MTF Ingestion | Raw OHLCV | (N, 3, 60, 4) | Multi-stream raw |

**Why This Matters:**
- Different feature sets → models learn different aspects of market dynamics
- Reduced feature overlap → lower error correlation
- Complementary representations → meta-learner combines diverse signals

**See:** `.serena/knowledge/per_model_feature_selection.md` for detailed feature strategies

### Error Correlation Comparison

| Ensemble Type | Base Selection | Feature Diversity | Error Correlation | Diversity | Robustness |
|---------------|----------------|-------------------|-------------------|-----------|------------|
| **Homogeneous** | Same family (XGB+LGB+Cat) | Low (same features) | High | Low | Low |
| **Heterogeneous** | Different families (Cat+TCN+PatchTST) | High (different features) | Low | High | High |

### Mathematical Intuition

For ensembles, the variance of the average prediction decreases with:
1. **Lower individual variance** (good base models)
2. **Lower covariance between predictions** (heterogeneous bases with diverse features)

Heterogeneous ensembles minimize covariance by using:
- **Different inductive biases** (boosting vs CNN vs transformer)
- **Different feature representations** (engineered vs raw, MTF vs single-TF)
- **Different primary timeframes** (15min vs 5min vs 1min)

---

## Meta-Learner Options

### Logistic Regression (Default)

```python
from sklearn.linear_model import LogisticRegression

meta_learner = LogisticRegression(
    C=1.0,                    # Regularization strength
    penalty='l2',             # L2 regularization
    solver='lbfgs',
    max_iter=1000,
    multi_class='multinomial',
    random_state=42
)
```

**Best for:** Calibrated probabilities, interpretable weights

### Ridge Regression

```python
from sklearn.linear_model import RidgeClassifier

meta_learner = RidgeClassifier(
    alpha=1.0,               # Regularization strength
    random_state=42
)
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
```

**Best for:** Calibrated confidence scores, probability calibration

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

# For each base model
oof_predictions = {}
for model_name in base_model_names:
    model_class = ModelRegistry.get(model_name)
    oof = generate_oof_predictions(
        model_class=model_class,
        X_train=X_train,
        y_train=y_train,
        kfold=kfold,
        model_config=configs[model_name]
    )
    oof_predictions[model_name] = oof
    # Shape: (n_samples, n_classes) e.g., (15000, 3)
```

### OOF Stacking

```python
import numpy as np

# Stack OOF predictions as meta-features
stacked_oof = np.hstack([
    oof_predictions["catboost"],   # (n_samples, 3)
    oof_predictions["tcn"],        # (n_samples, 3)
    oof_predictions["patchtst"],   # (n_samples, 3)
])
# Shape: (n_samples, 9) for 3 models * 3 classes
```

---

## Full Training Protocol

### Complete Workflow

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

# Phase 4: Evaluate on test set
metrics = stacker.evaluate(X_test, y_test)
```

---

## Test Evaluation

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

---

## CLI Usage

```bash
# 3-base heterogeneous ensemble
python scripts/train_ensemble.py \
  --base-models catboost,tcn,patchtst \
  --meta-learner logistic \
  --horizon 20

# 4-base ensemble with Ridge meta-learner
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

### Full Retrain
1. **Always retrain bases:** OOF models were trained on partial data
2. **Use same hyperparameters:** Don't re-tune during full retrain
3. **Use validation for early stopping:** Prevent overfitting during retrain

---

## Files Reference

- **Stacker implementation:** `src/ensemble/heterogeneous_stacker.py`
- **OOF generation:** `src/cross_validation/oof_generator.py`
- **PurgedKFold:** `src/cross_validation/purged_kfold.py`
- **Training script:** `scripts/train_ensemble.py`
- **Guide:** `docs/guides/META_LEARNER_STACKING.md`

---

**Last Updated:** 2026-01-01
