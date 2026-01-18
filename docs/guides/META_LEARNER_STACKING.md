# Meta-Learner Stacking Guide

Comprehensive guide for building heterogeneous ensembles via meta-learner stacking in the ML Model Factory.

**Last Updated:** 2026-01-18
**Pipeline Stage:** Stage 15 (Stacking) in 16-Stage Pipeline

---

## Table of Contents

1. [Overview](#overview)
2. [Integration with 16-Stage Pipeline](#integration-with-16-stage-pipeline)
3. [Heterogeneous Ensemble Architecture](#heterogeneous-ensemble-architecture)
4. [Base Model Selection](#base-model-selection)
5. [OOF Generation Protocol](#oof-generation-protocol)
6. [Meta-Learner Options](#meta-learner-options)
7. [Optuna Optimization for Meta-Learners](#optuna-optimization-for-meta-learners)
8. [Full Training Protocol](#full-training-protocol)
9. [Test Evaluation](#test-evaluation)
10. [CLI Reference](#cli-reference)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The ML Model Factory uses **heterogeneous ensemble stacking** where base models from different families (tabular, sequence, transformer) feed a single meta-learner.

**Key Principles:**
- **Heterogeneous > Homogeneous:** Different model families capture different patterns
- **Direct Stacking:** Meta-learner trained directly on OOF predictions from bases
- **1 Model per Family:** Select representative model from each family for diversity
- **Leakage-Free:** OOF generation with PurgedKFold prevents data leakage
- **Optuna Optimized:** Meta-learner hyperparameters and OOF selection tuned via Optuna

**Architecture:**
```
Tabular (CatBoost) ──┐
                     │
CNN/TCN (TCN) ───────┼──> OOF Predictions ──> Meta-Learner ──> Final Predictions
                     │                              ↑
Transformer (PatchTST)┘                    Optuna Optimization
                                           (50 trials)
```

---

## Integration with 16-Stage Pipeline

Meta-learner stacking is **Stage 15** in the unified 16-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    16-STAGE PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│ Stages 1-6:   Data Ingestion & Preprocessing                   │
│ Stage 7:      Triple-Barrier Labeling + Optuna (100 trials)    │
│ Stage 8:      Feature Selection + Optuna (100 trials)          │
│ Stage 9:      Feature Pruning + Optuna (50 trials)             │
│ Stages 10-12: Splits, Scaling, Dataset Building                │
│ Stage 13:     Per-Model Hyperparameter Optimization (100/model)│
│ Stage 14:     Base Model Training                               │
│ ─────────────────────────────────────────────────────────────── │
│ Stage 15:     STACKING (this guide)                            │
│               ├── OOF Generation for base models                │
│               ├── OOF Feature Selection (Optuna)                │
│               ├── Meta-Learner Hyperparameter Tuning (Optuna)   │
│               └── Final Meta-Learner Training                   │
│ ─────────────────────────────────────────────────────────────── │
│ Stage 16:     Final Evaluation & Reporting                     │
└─────────────────────────────────────────────────────────────────┘
```

### Dependencies from Earlier Stages

Stage 15 uses outputs from earlier Optuna optimization stages:

| Earlier Stage | Output | Used By Stage 15 |
|---------------|--------|------------------|
| Stage 7 (Labeling) | Optimized barrier params | Labels for OOF |
| Stage 8 (Feature Selection) | Selected features | Base model inputs |
| Stage 9 (Feature Pruning) | Pruned feature set | Final features |
| Stage 13 (Hyperparams) | Per-model best params | Base model configs |
| Stage 14 (Base Training) | Trained base models | OOF prediction sources |

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

The ML Factory provides **4 meta-learner types** for stacking:

### ridge_meta - L2-Regularized (Default)

```python
from src.models.ensemble import RidgeMetaLearner

meta_learner = RidgeMetaLearner(config={
    "alpha": 1.0,           # L2 regularization strength
    "fit_intercept": True,
    "class_weight": "balanced",
    "solver": "auto",
    "scale_features": True,
})

meta_learner.fit(stacked_oof_train, y_train, stacked_oof_val, y_val)
```

**Best for:** Default choice - fast, interpretable weights, robust to multicollinearity

### mlp_meta - MLP Meta-Learner

```python
from src.models.ensemble import MLPMetaLearner

meta_learner = MLPMetaLearner(config={
    "hidden_layer_sizes": (32, 16),  # 2 small hidden layers
    "activation": "relu",
    "alpha": 0.01,                   # L2 regularization
    "learning_rate_init": 0.001,
    "max_iter": 500,
    "early_stopping": True,
    "batch_size": 32,
})

meta_learner.fit(stacked_oof_train, y_train, stacked_oof_val, y_val)
```

**Best for:** Non-linear blending, complex interactions between base predictions

### xgboost_meta - XGBoost Meta-Learner

```python
from src.models.ensemble import XGBoostMeta

meta_learner = XGBoostMeta(config={
    "learning_rate": 0.1,
    "max_depth": 4,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 20,
})

meta_learner.fit(stacked_oof_train, y_train, stacked_oof_val, y_val)
```

**Best for:** Maximum predictive power, implicit base model selection

### calibrated_meta - Calibrated Meta-Learner

```python
from src.models.ensemble import CalibratedMetaLearner

meta_learner = CalibratedMetaLearner(config={
    "method": "isotonic",    # or "sigmoid"
    "cv": 3,
    "base_estimator": "logistic",
    "base_C": 1.0,
})

meta_learner.fit(stacked_oof_train, y_train, stacked_oof_val, y_val)
```

**Best for:** Well-calibrated probabilities for risk management

### Meta-Learner Comparison

| Meta-Learner | Training Time | Overfitting Risk | Interpretable | Best For |
|--------------|---------------|------------------|---------------|----------|
| **ridge_meta** | <1 sec | Low | Yes (weights) | Default choice |
| **mlp_meta** | 5-30 sec | Medium | No | Complex patterns |
| **xgboost_meta** | 10-60 sec | Medium | Partial (importance) | Maximum power |
| **calibrated_meta** | 2-5 sec | Low | Yes | Probability calibration |

---

## Optuna Optimization for Meta-Learners

Stage 15 includes **Optuna optimization** for both meta-learner hyperparameters and OOF feature selection.

### Optuna Search Spaces by Meta-Learner

#### ridge_meta Optuna Search Space

```python
def ridge_meta_search_space(trial: optuna.Trial) -> dict:
    return {
        'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'scale_features': trial.suggest_categorical('scale_features', [True, False]),
    }
```

| Parameter | Default | Optuna Range | Description |
|-----------|---------|--------------|-------------|
| `alpha` | 1.0 | [0.001, 100] log | L2 regularization |
| `fit_intercept` | True | [True, False] | Intercept term |
| `solver` | 'auto' | ['auto', 'svd', 'cholesky', 'lsqr'] | Solver |
| `class_weight` | 'balanced' | ['balanced', None] | Weighting |

#### mlp_meta Optuna Search Space

```python
def mlp_meta_search_space(trial: optuna.Trial) -> dict:
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_sizes = [trial.suggest_int(f'hidden_{i}', 8, 128, log=True) for i in range(n_layers)]
    return {
        'hidden_layer_sizes': tuple(hidden_sizes),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
        'learning_rate_init': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
    }
```

| Parameter | Default | Optuna Range | Description |
|-----------|---------|--------------|-------------|
| `hidden_layer_sizes` | (32, 16) | 1-3 layers, [8-128] | Architecture |
| `activation` | 'relu' | ['relu', 'tanh'] | Activation |
| `alpha` | 0.01 | [1e-6, 1e-2] log | L2 reg |
| `learning_rate_init` | 0.001 | [1e-5, 1e-2] log | Learning rate |

#### xgboost_meta Optuna Search Space

```python
def xgboost_meta_search_space(trial: optuna.Trial) -> dict:
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
    }
```

| Parameter | Default | Optuna Range | Description |
|-----------|---------|--------------|-------------|
| `learning_rate` | 0.1 | [0.01, 0.3] log | Step size |
| `max_depth` | 4 | [2, 8] | Tree depth |
| `n_estimators` | 100 | [50, 300] | Boosting rounds |
| `reg_lambda` | 1.0 | [1e-8, 10] log | L2 regularization |

#### calibrated_meta Optuna Search Space

```python
def calibrated_meta_search_space(trial: optuna.Trial) -> dict:
    return {
        'method': trial.suggest_categorical('method', ['isotonic', 'sigmoid']),
        'cv': trial.suggest_int('cv', 3, 7),
        'base_C': trial.suggest_float('base_C', 0.01, 100.0, log=True),
        'base_solver': trial.suggest_categorical('base_solver', ['lbfgs', 'saga']),
    }
```

### OOF Feature Selection Optimization

Optuna also optimizes **which base model OOF predictions to include**:

```python
def oof_selection_objective(trial: optuna.Trial, available_bases: list[str]):
    """Binary selection for each base model's OOF predictions."""
    selected = []
    for base_name in available_bases:
        if trial.suggest_categorical(f'include_{base_name}', [True, False]):
            selected.append(base_name)

    if len(selected) < 2:
        return float('inf')  # Need at least 2 bases

    # Stack and evaluate
    stacked_oof = stack_selected_bases(selected)
    meta = RidgeMetaLearner()
    meta.fit(stacked_oof_train, y_train, stacked_oof_val, y_val)
    return meta.evaluate(stacked_oof_val, y_val)['loss']
```

**Example output:**
```yaml
oof_selection_result:
  selected_bases: [catboost, tcn, patchtst]
  excluded_bases: [lstm, xgboost]  # Redundant with catboost/tcn
  improvement_vs_all: 2.3%
```

### Running Optuna Optimization

```bash
# Optimize meta-learner hyperparameters (50 trials)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,tcn,patchtst \
  --meta-learner ridge_meta \
  --optimize-meta-learner --n-trials 50

# Combined OOF selection + hyperparameter optimization
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,lstm,tcn,patchtst,xgboost \
  --meta-learner ridge_meta \
  --optimize-meta-learner --optimize-oof-selection --n-trials 50
```

### Optuna Budget Summary

| Optimization | Trials | Time | Scope |
|--------------|--------|------|-------|
| OOF selection | 20-30 | 5-10 min | Which bases to include |
| Meta-learner hyperparams | 50 | 10-30 min | Per meta-learner type |
| Combined | 50 | 15-40 min | Both together |

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
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,tcn,patchtst \
  --meta-learner ridge_meta

# Train 4-base ensemble with XGBoost meta-learner
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models lightgbm,tcn,tft,nbeats \
  --meta-learner xgboost_meta

# Fast 2-base ensemble
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lstm \
  --meta-learner ridge_meta
```

### Optuna Optimization Options

```bash
# Optimize meta-learner hyperparameters (50 trials)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,tcn,patchtst \
  --meta-learner ridge_meta \
  --optimize-meta-learner --n-trials 50

# Optimize OOF feature selection (which bases to include)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,lstm,tcn,patchtst,xgboost \
  --meta-learner ridge_meta \
  --optimize-oof-selection --n-trials 30

# Combined optimization (hyperparams + OOF selection)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,lstm,tcn,patchtst,xgboost \
  --meta-learner ridge_meta \
  --optimize-meta-learner --optimize-oof-selection --n-trials 50

# Set Optuna timeout
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,tcn,patchtst \
  --meta-learner mlp_meta \
  --optimize-meta-learner --n-trials 100 --optuna-timeout 3600
```

### Advanced Options

```bash
# Custom OOF folds
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,tcn,patchtst \
  --meta-learner ridge_meta \
  --n-folds 3

# Custom purge/embargo
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,tcn,patchtst \
  --meta-learner ridge_meta \
  --purge-bars 90 \
  --embargo-bars 2000

# MLP meta-learner with custom config
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,tcn,patchtst \
  --meta-learner mlp_meta \
  --meta-config '{"hidden_layer_sizes": [64, 32], "alpha": 0.01}'
```

### CLI Parameters Summary

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--base-models` | Comma-separated base model names | Required |
| `--meta-learner` | Meta-learner type | ridge_meta |
| `--optimize-meta-learner` | Enable Optuna for meta-learner | False |
| `--optimize-oof-selection` | Enable Optuna for base selection | False |
| `--n-trials` | Optuna trial count | 50 |
| `--optuna-timeout` | Max optimization time (seconds) | 1800 |
| `--n-folds` | OOF cross-validation folds | 5 |
| `--purge-bars` | Purge gap between folds | 60 |
| `--embargo-bars` | Embargo after each fold | 1440 |

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
  optuna/                     # New: Optuna artifacts
    meta_learner_study.db     # Optuna study database
    best_params.yaml          # Best hyperparameters
    oof_selection.yaml        # Selected base models
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
5. **Let Optuna select:** Use `--optimize-oof-selection` for data-driven base selection

### OOF Generation

1. **Use 5 folds:** Robust OOF estimates with acceptable compute
2. **Maintain purge/embargo:** Prevent leakage (60 purge, 1440 embargo)
3. **Validate OOF coverage:** Ensure >95% coverage before meta-learner training
4. **Save OOF predictions:** Allows retraining meta-learner without rerunning OOF

### Meta-Learner Training

1. **Start with ridge_meta:** Simple, fast, robust - best default choice
2. **Use Optuna optimization:** 50 trials typically sufficient for meta-learners
3. **Cross-validate meta-learner:** Optuna uses internal CV for hyperparameter tuning
4. **Consider calibrated_meta:** If probability calibration is critical for trading

### Optuna Optimization

1. **Budget 50 trials for meta-learner:** Diminishing returns beyond 50
2. **Use median pruner:** Stops unpromising trials early, saves 30-50% time
3. **Run OOF selection first:** Identify optimal base set before hyperparameter tuning
4. **Cache Optuna studies:** Save to SQLite for resumption and analysis
5. **Monitor Pareto front:** For multi-objective (Sharpe vs. drawdown) optimization

### Full Retrain

1. **Always retrain bases:** OOF models were trained on partial data
2. **Use optimized hyperparameters:** Apply Optuna-tuned params from Stage 13
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

**Documentation:**
- **Architecture:** `docs/ARCHITECTURE.md` (Ensemble Architecture section)
- **Implementation Details:** `docs/implementation/PHASE_7_META_LEARNER_STACKING.md`
- **Hyperparameter Tuning:** `docs/guides/HYPERPARAMETER_TUNING.md`
- **Model Integration:** `docs/guides/MODEL_INTEGRATION.md`
- **Pipeline Stages:** `docs/reference/PIPELINE_STAGES.md`

**Source Code:**
- **Meta-Learners:** `src/models/ensemble/ridge_meta.py`, `mlp_meta.py`, `xgboost_meta.py`, `calibrated_meta.py`
- **OOF Stacking:** `src/cross_validation/oof_stacking.py`
- **OOF Generator:** `src/cross_validation/oof_generator.py`
- **Stacking Orchestrator:** `src/models/ensemble/heterogeneous_stacking.py`
- **Optuna Spaces:** `src/optimization/meta_learner_spaces.py`

**Configuration:**
- **Meta-Learner Configs:** `config/models/ridge_meta.yaml`, `mlp_meta.yaml`, `xgboost_meta.yaml`, `calibrated_meta.yaml`
- **Ensemble Config:** `config/ensembles/heterogeneous_stack.yaml`

**Scripts:**
- **Training:** `scripts/train_model.py --model stacking`
- **Ensemble Training:** `scripts/train_ensemble.py`
