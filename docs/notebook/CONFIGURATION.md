# Configuration Reference

Complete reference for all 45+ configuration parameters in Section 1 of the ML Pipeline notebook.

---

## Table of Contents

1. [Data Configuration](#data-configuration)
2. [Pipeline Configuration](#pipeline-configuration)
3. [Model Selection](#model-selection)
4. [Neural Network Settings](#neural-network-settings)
5. [Transformer Settings](#transformer-settings)
6. [Boosting Settings](#boosting-settings)
7. [Ensemble Configuration](#ensemble-configuration)
8. [Class Balancing](#class-balancing)
9. [Cross-Validation](#cross-validation)
10. [Execution Options](#execution-options)
11. [Model Default Parameters](#model-default-parameters)

---

## Data Configuration

| Parameter | Type | Default | Valid Values | Description |
|-----------|------|---------|--------------|-------------|
| `SYMBOL` | string | "SI" | SI, MES, MGC, ES, GC, NQ, CL, HG, ZB, ZN | Futures contract symbol |
| `DATE_RANGE` | string | "2019-2024" | 2019-2024, 2020-2024, 2021-2024, etc. | Date range filter |
| `DRIVE_DATA_PATH` | string | "research/data/raw" | Any valid path | Google Drive path (Colab only) |
| `CUSTOM_DATA_FILE` | string | "" | Any filename | Override auto-detection |

**Notes:**
- `SYMBOL`: Must match filename in `data/raw/{SYMBOL}_1m.parquet` or `.csv`
- `DATE_RANGE`: Used for display only; actual filtering happens in Phase 1
- `DRIVE_DATA_PATH`: Only used when running on Google Colab
- `CUSTOM_DATA_FILE`: Use when filename doesn't follow standard pattern

---

## Pipeline Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `HORIZONS` | string | "5,10,15,20" | Comma-separated ints | Prediction horizons (bars forward) |
| `TRAIN_RATIO` | float | 0.70 | 0.0-1.0 | Training set proportion |
| `VAL_RATIO` | float | 0.15 | 0.0-1.0 | Validation set proportion |
| `TEST_RATIO` | float | 0.15 | 0.0-1.0 | Test set proportion |
| `PURGE_BARS` | int | 60 | ≥0 | Bars to purge at split boundaries |
| `EMBARGO_BARS` | int | 1440 | ≥0 | Embargo period (~5 days at 5-min) |
| `TRAINING_HORIZON` | int | 20 | 5, 10, 15, 20 | Which horizon to train on |

**Notes:**
- Ratios must sum to 1.0
- `PURGE_BARS = 60`: 3x max horizon (20), prevents label leakage
- `EMBARGO_BARS = 1440`: ~5 days at 5-min bars, prevents serial correlation
- `TRAINING_HORIZON`: Must be in the list of `HORIZONS` generated in Phase 1

---

## Model Selection

Boolean toggles to enable/disable each of the 13 models:

### Boosting Models

| Parameter | Default | Model | GPU Required | Training Time (H20) |
|-----------|---------|-------|--------------|---------------------|
| `TRAIN_XGBOOST` | True | XGBoost | No | ~1 min |
| `TRAIN_LIGHTGBM` | True | LightGBM | No | ~30 sec |
| `TRAIN_CATBOOST` | True | CatBoost | Optional | ~2 min |

**Strengths:** Fast, accurate, feature importance, handles missing data
**Best for:** Production systems, interpretability, tabular data

### Neural Models

| Parameter | Default | Model | GPU Required | Training Time (H20) |
|-----------|---------|-------|--------------|---------------------|
| `TRAIN_LSTM` | False | LSTM | Yes | ~10 min |
| `TRAIN_GRU` | False | GRU | Yes | ~8 min |
| `TRAIN_TCN` | False | TCN | Yes | ~12 min |
| `TRAIN_TRANSFORMER` | False | Transformer | Yes | ~15 min |

**Strengths:** Sequential pattern learning, long-term dependencies
**Best for:** Time series with complex patterns, when GPU available

### Classical Models

| Parameter | Default | Model | GPU Required | Training Time (H20) |
|-----------|---------|-------|--------------|---------------------|
| `TRAIN_RANDOM_FOREST` | False | Random Forest | No | ~1 min |
| `TRAIN_LOGISTIC` | False | Logistic Regression | No | ~10 sec |
| `TRAIN_SVM` | False | Support Vector Machine | No | ~5 min |

**Strengths:** Simple, robust baselines, fast training
**Best for:** Baselines, interpretability, low-latency systems

### Ensemble Models

| Parameter | Default | Model | GPU Required | Training Time (H20) |
|-----------|---------|-------|--------------|---------------------|
| `TRAIN_VOTING` | False | Voting Ensemble | No | ~1 min |
| `TRAIN_STACKING` | False | Stacking Ensemble | No | ~5 min |
| `TRAIN_BLENDING` | False | Blending Ensemble | No | ~3 min |

**Strengths:** Combines diverse models, often best performance
**Best for:** Maximum accuracy, production systems with ensemble diversity

**Notes:**
- At least one model must be enabled
- Ensemble models require base models to be trained first
- GPU recommendations based on Tesla T4 hardware

---

## Neural Network Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `SEQUENCE_LENGTH` | int | 60 | 30-120 | Input sequence length (LSTM/GRU/TCN) |
| `BATCH_SIZE` | int | 256 | 64-1024 | Training batch size |
| `MAX_EPOCHS` | int | 50 | ≥1 | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | int | 10 | ≥1 | Epochs without improvement before stopping |

**Tuning Guidelines:**
- **SEQUENCE_LENGTH:** Longer = more context but slower training and higher memory
  - 30: Minimal context (~2.5 hours at 5-min)
  - 60: Balanced (default, ~5 hours)
  - 120: Maximum context (~10 hours)
- **BATCH_SIZE:** Larger = faster training but more memory
  - 64: Low memory GPUs
  - 256: Standard (T4, A100)
  - 512-1024: High-end GPUs only
- **MAX_EPOCHS:** More = better convergence but risk of overfitting
  - Early stopping typically triggers before max
- **EARLY_STOPPING_PATIENCE:** Higher = more tolerance for plateau
  - 5: Aggressive early stopping
  - 10: Balanced (default)
  - 20: Conservative, allows longer plateaus

---

## Transformer Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `TRANSFORMER_SEQUENCE_LENGTH` | int | 128 | ≥32 | Transformer input length |
| `TRANSFORMER_N_HEADS` | int | 8 | 4, 8, 16 | Number of attention heads |
| `TRANSFORMER_N_LAYERS` | int | 3 | 2-6 | Number of encoder layers |
| `TRANSFORMER_D_MODEL` | int | 256 | 128, 256, 512 | Model dimension |

**Tuning Guidelines:**
- **SEQUENCE_LENGTH:** Longer than RNN (transformers handle longer sequences better)
  - 64: Minimal (~5 hours)
  - 128: Balanced (default)
  - 256: Long-term patterns (~21 hours)
- **N_HEADS:** Must divide `D_MODEL` evenly
  - 4: Faster, less expressive
  - 8: Balanced (default)
  - 16: More expressive, slower
- **N_LAYERS:** More = better but diminishing returns
  - 2: Fast baseline
  - 3: Balanced (default)
  - 6: Deep model, requires more data
- **D_MODEL:** Larger = more capacity
  - 128: Small, fast
  - 256: Balanced (default)
  - 512: Large, requires more memory

---

## Boosting Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `N_ESTIMATORS` | int | 500 | ≥1 | Number of boosting rounds/trees |
| `BOOSTING_EARLY_STOPPING` | int | 50 | ≥1 | Early stopping rounds |

**Tuning Guidelines:**
- **N_ESTIMATORS:** More trees = better fit but risk overfitting
  - 100: Fast baseline
  - 500: Balanced (default)
  - 1000+: Requires careful early stopping
- **EARLY_STOPPING:** Higher = more tolerance
  - 20: Aggressive
  - 50: Balanced (default)
  - 100: Conservative

**Model-Specific Notes:**
- XGBoost: Uses `early_stopping_rounds` with validation set
- LightGBM: Uses `early_stopping_rounds` with validation set
- CatBoost: Uses `early_stopping_rounds` with validation set
- See [Model Default Parameters](#model-default-parameters) for other params

---

## Ensemble Configuration

### Voting Ensemble

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `VOTING_BASE_MODELS` | string | "xgboost,lightgbm,catboost" | Comma-separated base model names |
| `VOTING_WEIGHTS` | string | "" | Comma-separated weights (empty = equal weighting) |

**Example Configurations:**
```python
# Equal weights (default)
VOTING_BASE_MODELS = "xgboost,lightgbm,catboost"
VOTING_WEIGHTS = ""

# Custom weights (proportional to validation F1)
VOTING_BASE_MODELS = "xgboost,lightgbm,lstm"
VOTING_WEIGHTS = "0.52,0.51,0.54"  # Sum doesn't need to be 1.0
```

### Stacking Ensemble

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `STACKING_BASE_MODELS` | string | "xgboost,lightgbm,lstm" | - | Comma-separated base model names |
| `STACKING_META_LEARNER` | string | "logistic" | logistic, xgboost, random_forest | Meta-learner type |
| `STACKING_N_FOLDS` | int | 5 | ≥2 | CV folds for OOF predictions |

**Meta-Learner Selection:**
- **logistic:** Fast, interpretable, good default
- **xgboost:** More flexible, captures non-linear patterns
- **random_forest:** Robust, handles overfitting well

### Blending Ensemble

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `BLENDING_BASE_MODELS` | string | "xgboost,lightgbm,random_forest" | - | Comma-separated base model names |
| `BLENDING_META_LEARNER` | string | "logistic" | logistic, xgboost, random_forest | Meta-learner type |
| `BLENDING_HOLDOUT_RATIO` | float | 0.2 | 0.0-1.0 | Holdout set ratio for blending |

**Blending vs Stacking:**
- **Blending:** Simpler, uses holdout set, less prone to overfitting
- **Stacking:** More data-efficient, uses k-fold CV, slightly better performance

---

## Class Balancing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_CLASS_WEIGHTS` | bool | True | Use balanced class weights for Long/Neutral/Short |
| `USE_SAMPLE_WEIGHTS` | bool | True | Use pipeline quality-based sample weights |

**Class Weights:**
- Automatically computed as `n_samples / (n_classes * class_count)`
- Prevents majority class bias (e.g., if Neutral >> Long/Short)
- Recommended: Keep enabled unless intentionally biasing

**Sample Weights:**
- From Phase 1 pipeline (0.5x - 1.5x based on label quality)
- Downweights low-quality labels (near barrier boundaries)
- Recommended: Keep enabled for better generalization

**Both Enabled (default):**
- Model loss function uses: `loss * class_weight * sample_weight`
- Balances both class distribution and sample quality

---

## Cross-Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RUN_CROSS_VALIDATION` | bool | False | Enable CV phase (Section 5) |
| `CV_N_SPLITS` | int | 5 | Number of CV folds (purged k-fold) |
| `CV_TUNE_HYPERPARAMS` | bool | False | Enable Optuna hyperparameter tuning |
| `CV_N_TRIALS` | int | 20 | Number of Optuna trials per model |
| `CV_USE_PRESCALED` | bool | True | Use pre-scaled data (faster) vs per-fold scaling |

**Tuning Guidelines:**
- **N_SPLITS:** More = more robust but slower
  - 3: Fast baseline
  - 5: Balanced (default)
  - 10: Thorough but slow
- **N_TRIALS:** More = better hyperparams but slower
  - 10: Quick search
  - 20: Balanced (default)
  - 50-100: Thorough search (1-2 hours per model)
- **USE_PRESCALED:** Use `True` unless testing scaler robustness

**Expected Runtime (5 splits, 20 trials):**
- XGBoost: 30-60 min
- LightGBM: 20-40 min
- LSTM: 2-3 hours (GPU required)

---

## Execution Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RUN_DATA_PIPELINE` | bool | True | Run Phase 1 (data pipeline) |
| `RUN_MODEL_TRAINING` | bool | True | Run Phase 2 (model training) |
| `SAFE_MODE` | bool | False | Low-memory mode (clears memory aggressively) |
| `RANDOM_SEED` | int | 42 | Seed for reproducibility (0 = random seed) |

**Use Cases:**
- **Skip Phase 1:** Set `RUN_DATA_PIPELINE = False` if data already processed
- **Skip Training:** Set `RUN_MODEL_TRAINING = False` to only process data
- **Safe Mode:** Enable if running into OOM errors (slower but more stable)
- **Random Seed:** Set to 0 for non-deterministic training (slightly faster)

---

## Model Default Parameters

### XGBoost

```python
{
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss"
}
```

### LightGBM

```python
{
    "n_estimators": 500,
    "max_depth": -1,  # Unlimited
    "num_leaves": 31,
    "learning_rate": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "early_stopping_rounds": 50
}
```

### CatBoost

```python
{
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "task_type": "CPU",  # or "GPU"
    "early_stopping_rounds": 50
}
```

### Random Forest

```python
{
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced"
}
```

### Logistic Regression

```python
{
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": "balanced"
}
```

### SVM

```python
{
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "probability": True,
    "class_weight": "balanced"
}
```

### LSTM / GRU

```python
{
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": False,
    "batch_size": 256,
    "learning_rate": 0.001,
    "max_epochs": 50,
    "early_stopping_patience": 10
}
```

### TCN

```python
{
    "num_channels": [64, 128, 256],
    "kernel_size": 3,
    "dropout": 0.2,
    "batch_size": 256,
    "learning_rate": 0.001
}
```

### Transformer

```python
{
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 3,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_seq_len": 128,
    "batch_size": 256,
    "learning_rate": 0.0001
}
```

---

## Quick Reference Table

| Category | Parameters | Total |
|----------|------------|-------|
| Data | SYMBOL, DATE_RANGE, DRIVE_DATA_PATH, CUSTOM_DATA_FILE | 4 |
| Pipeline | HORIZONS, ratios, PURGE_BARS, EMBARGO_BARS, TRAINING_HORIZON | 7 |
| Model Selection | 13 boolean toggles (TRAIN_*) | 13 |
| Neural Settings | SEQUENCE_LENGTH, BATCH_SIZE, MAX_EPOCHS, EARLY_STOPPING_PATIENCE | 4 |
| Transformer | TRANSFORMER_* (4 params) | 4 |
| Boosting | N_ESTIMATORS, BOOSTING_EARLY_STOPPING | 2 |
| Ensemble | 9 params across voting/stacking/blending | 9 |
| Class Balance | USE_CLASS_WEIGHTS, USE_SAMPLE_WEIGHTS | 2 |
| CV | RUN_CROSS_VALIDATION, CV_* (4 params) | 5 |
| Execution | RUN_DATA_PIPELINE, RUN_MODEL_TRAINING, SAFE_MODE, RANDOM_SEED | 4 |
| **TOTAL** | | **54** |

---

**Last Updated:** 2025-12-28
