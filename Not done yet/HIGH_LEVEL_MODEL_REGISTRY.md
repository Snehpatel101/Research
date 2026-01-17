# ML Factory: Model Registry Reference

**Version:** 1.0
**Purpose:** Complete model catalog for AI agent ingestion
**Scope:** All 22-23 models with specifications

---

## 1. Model Overview

```
MODEL REGISTRY (22-23 Models)
|
+-- BOOSTING (3) -----> 2D Input
|   +-- xgboost
|   +-- lightgbm
|   +-- catboost
|
+-- CLASSICAL (3) ----> 2D Input
|   +-- random_forest
|   +-- logistic
|   +-- svm
|
+-- NEURAL (10) ------> 3D/4D Input
|   +-- lstm
|   +-- gru
|   +-- tcn
|   +-- transformer
|   +-- patchtst
|   +-- itransformer
|   +-- tft
|   +-- nbeats
|   +-- inceptiontime
|   +-- resnet1d
|
+-- ENSEMBLE (3) -----> Varies
|   +-- voting
|   +-- stacking
|   +-- blending
|
+-- META-LEARNER (4) -> 2D Input (OOF probs)
    +-- ridge_meta
    +-- mlp_meta
    +-- xgboost_meta
    +-- calibrated_meta
```

---

## 2. Boosting Models (2D)

### 2.1 XGBoost

```
Name: xgboost
Family: boosting
Input Rank: 2D (n_samples, n_features)
Output: 3-class probabilities

Default Config:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  early_stopping_rounds: 50
  eval_metric: mlogloss

Feature Strategy:
  Baseline: [momentum, moving_average, volatility, volume, trend]
  Optional: [price, microstructure, regime]
  MTF: enabled
```

### 2.2 LightGBM

```
Name: lightgbm
Family: boosting
Input Rank: 2D

Default Config:
  n_estimators: 500
  max_depth: -1
  num_leaves: 31
  learning_rate: 0.05
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 5
  early_stopping_rounds: 50

Feature Strategy: Same as XGBoost
```

### 2.3 CatBoost

```
Name: catboost
Family: boosting
Input Rank: 2D

Default Config:
  iterations: 500
  depth: 6
  learning_rate: 0.05
  early_stopping_rounds: 50
  task_type: GPU (if available)

Feature Strategy: Same as XGBoost
Note: Optional - only if catboost installed
```

---

## 3. Classical Models (2D)

### 3.1 Random Forest

```
Name: random_forest
Family: classical
Input Rank: 2D

Default Config:
  n_estimators: 200
  max_depth: 10
  min_samples_split: 5
  min_samples_leaf: 2
  max_features: sqrt
  n_jobs: -1

Feature Strategy:
  Baseline: [momentum, volatility, volume]
  Reduced features to avoid overfitting
```

### 3.2 Logistic Regression

```
Name: logistic
Family: classical
Input Rank: 2D

Default Config:
  C: 1.0
  penalty: l2
  solver: lbfgs
  max_iter: 1000
  multi_class: multinomial

Feature Strategy:
  Baseline: [momentum, moving_average]
  Minimal features - interpretable model
```

### 3.3 SVM

```
Name: svm
Family: classical
Input Rank: 2D

Default Config:
  C: 1.0
  kernel: rbf
  probability: true
  class_weight: balanced

Feature Strategy:
  Baseline: [momentum, volatility]
  Limited features due to scaling sensitivity
```

---

## 4. Neural Models - RNN (3D)

### 4.1 LSTM

```
Name: lstm
Family: neural
Input Rank: 3D (n_samples, seq_len, n_features)

Default Config:
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  bidirectional: false
  sequence_length: 60

Feature Strategy:
  Baseline: [momentum, volatility, volume, price]
  Temporal features important
  MTF: enabled for multi-scale patterns
```

### 4.2 GRU

```
Name: gru
Family: neural
Input Rank: 3D

Default Config:
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  bidirectional: false
  sequence_length: 60

Feature Strategy: Same as LSTM
Note: Faster training than LSTM, fewer parameters
```

---

## 5. Neural Models - CNN (3D)

### 5.1 TCN (Temporal Convolutional Network)

```
Name: tcn
Family: neural
Input Rank: 3D

Default Config:
  num_channels: [64, 64, 64]
  kernel_size: 3
  dropout: 0.2
  sequence_length: 60

Feature Strategy:
  Baseline: [momentum, volatility, volume]
  Good for multi-scale temporal patterns
```

### 5.2 InceptionTime

```
Name: inceptiontime
Family: neural
Input Rank: 3D

Default Config:
  num_blocks: 2
  num_filters: 32
  bottleneck_size: 32
  sequence_length: 60

Feature Strategy:
  Baseline: [raw, momentum, volatility]
  Multi-scale convolutions capture various periodicities
```

### 5.3 ResNet1D

```
Name: resnet1d
Family: neural
Input Rank: 3D

Default Config:
  num_blocks: 3
  num_filters: 64
  sequence_length: 60

Feature Strategy:
  Baseline: [momentum, volatility]
  Deep residual learning for time series
```

---

## 6. Transformer Models (3D/4D)

### 6.1 Transformer (Vanilla)

```
Name: transformer
Family: neural
Input Rank: 3D

Default Config:
  d_model: 64
  n_heads: 4
  n_layers: 2
  d_ff: 256
  dropout: 0.1
  sequence_length: 60

Feature Strategy:
  Baseline: [momentum, volatility, volume, temporal]
  Attention captures long-range dependencies
```

### 6.2 PatchTST

```
Name: patchtst
Family: neural
Input Rank: 4D (n_samples, n_timeframes, seq_len, n_features)

Default Config:
  patch_len: 16
  stride: 8
  d_model: 64
  n_heads: 4
  n_layers: 2
  sequence_length: 60

Feature Strategy:
  Baseline: [raw] (OHLCV only)
  Designed for multi-variate time series
  Multi-timeframe input via MultiStreamAdapter
```

### 6.3 iTransformer

```
Name: itransformer
Family: neural
Input Rank: 4D

Default Config:
  d_model: 64
  n_heads: 4
  n_layers: 2
  sequence_length: 60

Feature Strategy:
  Baseline: [raw]
  Inverted attention - features as tokens
  Multi-timeframe input
```

### 6.4 TFT (Temporal Fusion Transformer)

```
Name: tft
Family: neural
Input Rank: 3D

Default Config:
  hidden_size: 64
  attention_heads: 4
  num_lstm_layers: 1
  dropout: 0.1
  sequence_length: 60

Feature Strategy:
  Baseline: [momentum, volatility, volume, temporal, regime]
  Interpretable attention weights
  Variable selection networks
```

### 6.5 N-BEATS

```
Name: nbeats
Family: neural
Input Rank: 3D

Default Config:
  num_stacks: 2
  num_blocks: 3
  expansion_coefficient_dim: 5
  sequence_length: 60

Feature Strategy:
  Baseline: [raw, momentum]
  Designed for univariate, adapted for multivariate
```

---

## 7. Ensemble Models

### 7.1 Voting

```
Name: voting
Family: ensemble
Input Rank: Varies (per base model)

Config:
  voting: soft (probability averaging)
  weights: optional per-model weights

Usage:
  Combine predictions from 2+ base models
  No additional training required
```

### 7.2 Stacking

```
Name: stacking
Family: ensemble
Input Rank: 2D (OOF probabilities)

Config:
  base_models: list of model names
  meta_learner: ridge_meta | mlp_meta | xgboost_meta
  use_oof: true (always)

Usage:
  Train base models with OOF
  Stack OOF predictions
  Train meta-learner on stack
```

### 7.3 Blending

```
Name: blending
Family: ensemble
Input Rank: 2D

Config:
  base_models: list of model names
  holdout_fraction: 0.2

Usage:
  Train base on train-holdout
  Blend predictions on holdout
  Simpler than stacking, less data efficient
```

---

## 8. Meta-Learners (2D)

### 8.1 Ridge Meta

```
Name: ridge_meta
Family: meta_learner
Input Rank: 2D (stacked OOF probs)

Config:
  alphas: [0.1, 1.0, 10.0, 100.0]
  cv: 3

Characteristics:
  - Fast training
  - Robust baseline
  - L2 regularization prevents overfitting
```

### 8.2 MLP Meta

```
Name: mlp_meta
Family: meta_learner
Input Rank: 2D

Config:
  hidden_layers: (64, 32)
  dropout: 0.2
  learning_rate: 0.001
  max_iter: 500

Characteristics:
  - Captures non-linear interactions
  - More parameters than ridge
  - Early stopping for regularization
```

### 8.3 XGBoost Meta

```
Name: xgboost_meta
Family: meta_learner
Input Rank: 2D

Config:
  n_estimators: 100
  max_depth: 3
  learning_rate: 0.1
  calibrate: true

Characteristics:
  - Strong with diverse base models
  - Optional isotonic calibration
  - Handles feature interactions
```

### 8.4 Calibrated Meta

```
Name: calibrated_meta
Family: meta_learner
Input Rank: 2D

Config:
  base_estimator: logistic
  method: isotonic
  cv: 3

Characteristics:
  - Probability calibration focus
  - Better confidence estimates
  - Use when probability accuracy matters
```

---

## 9. Model-Adapter Mapping

```python
MODEL_ADAPTER_MAP = {
    # Boosting -> Tabular (2D)
    "xgboost": "tabular",
    "lightgbm": "tabular",
    "catboost": "tabular",

    # Classical -> Tabular (2D)
    "random_forest": "tabular",
    "logistic": "tabular",
    "svm": "tabular",

    # Neural -> Sequence (3D)
    "lstm": "sequence",
    "gru": "sequence",
    "tcn": "sequence",
    "transformer": "sequence",
    "nbeats": "sequence",
    "inceptiontime": "sequence",
    "resnet1d": "sequence",
    "tft": "sequence",

    # Advanced Neural -> Multi-Stream (4D)
    "patchtst": "multi_stream",
    "itransformer": "multi_stream",

    # Meta-Learners -> Tabular (2D on OOF)
    "ridge_meta": "tabular",
    "mlp_meta": "tabular",
    "xgboost_meta": "tabular",
    "calibrated_meta": "tabular",
}
```

---

## 10. Feature Strategy Summary

| Model Type | Feature Families | MTF | Typical Feature Count |
|------------|------------------|-----|----------------------|
| Boosting | Full (9 families) | Yes | 80-120 |
| Classical | Reduced (3-4 families) | Optional | 30-50 |
| RNN | Temporal-focused (5 families) | Yes | 50-80 |
| CNN | Multi-scale (4 families) | Yes | 40-60 |
| Transformer | Raw + temporal (3 families) | Multi-input | 20-40 |
| Meta-learner | OOF probs only | No | 9-15 |

---

## 11. Document Metadata

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2026-01-17 |
| Purpose | Model catalog for AI agents |
| Related Docs | HIGH_LEVEL_ARCHITECTURE.md, PHASE_1.md |
