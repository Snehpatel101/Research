# ML Model Factory Architecture

**Last Updated:** 2026-01-15

---

## Overview

This is a **single-pipeline ML model factory** for training, evaluating, and deploying machine learning models on OHLCV time series data. The factory processes one futures contract at a time through a unified pipeline, then uses **model-family adapters** to serve data in the appropriate format (2D tabular, 3D sequences, 4D multi-resolution tensors) for any model type.

**Key Principle:** One canonical dataset → Deterministic adapters → Model-specific training

---

## Architecture Diagram (16-Stage Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION (Stages 1-6)                     │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 1: INGESTION                                                  │
│  data/raw/{SYMBOL}_1m.parquet  →  [Validate]  →  Load              │
│                                                                     │
│  Stage 2: CLEANING                                                  │
│  [Resample]  →  [Gap Handling]  →  [Validation]                    │
│                                                                     │
│  Stage 3: SESSIONS                                                  │
│  [Trading Hours Filter]  →  RTH/ETH separation                      │
│                                                                     │
│  Stage 4: MTF UPSCALING                                             │
│  1-min OHLCV  →  9 Intraday Timeframes                             │
│  (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)                        │
│                                                                     │
│  Stage 5: FEATURES (162 indicators, 12 families)                    │
│  Momentum (23) | MA (16) | Volatility (25) | Volume (15) |         │
│  Trend (6) | Price (12) | Microstructure (15) | Entropy (12) |     │
│  Wavelets (15) | Temporal (9) | Regime (9) | MTF (30+)             │
│                                                                     │
│  Stage 6: REGIME DETECTION                                          │
│  [Volatility Regime]  +  [Trend Regime]  →  Composite Regime       │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
  [Checkpoint: data/features/{symbol}_features.parquet]
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│               OPTUNA OPTIMIZATION (Stages 7-9)                       │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 7: OPTUNA LABEL OPTIMIZATION (100 trials)                    │
│  Parameters: upper_mult, lower_mult, horizon, atr_period           │
│  Objective: Maximize label quality via quick CV                     │
│                         ↓                                            │
│  Stage 8: OPTUNA FEATURE SELECTION (100 trials)                     │
│  Search: Binary include/exclude for 162 features                    │
│  Objective: Maximize F1 with minimal feature set                    │
│                         ↓                                            │
│  Stage 9: OPTUNA FEATURE PRUNING (50 trials)                        │
│  Methods: gain, split, SHAP importance                              │
│  Objective: Remove low-importance features                          │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
  [Checkpoint: data/optimized/{symbol}_optimized.parquet]
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│               PREPROCESSING (Stages 10-12)                           │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 10: SPLITS                                                    │
│  Train/Val/Test: 70/15/15 + Purge (60) + Embargo (1440)            │
│                         ↓                                            │
│  Stage 11: SCALING                                                   │
│  Train-only robust scaling → Transform all splits                   │
│                         ↓                                            │
│  Stage 12: ADAPTATION (Model-Family Adapters)                       │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │   Tabular    │   │   Sequence   │   │  MultiStream │            │
│  │   Adapter    │   │   Adapter    │   │   Adapter    │            │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘            │
│         ↓                  ↓                  ↓                      │
│    2D Arrays          3D Windows         4D Tensors                 │
│    (N, ~60)           (N, T, ~60)        (N, 9, T, 4)               │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
  [Checkpoint: data/splits/scaled/{symbol}_{split}.parquet]
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│               TRAINING (Stages 13-15)                                │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 13: OPTUNA HYPERPARAMETER OPTIMIZATION (100 trials/model)    │
│  23 models × 100 trials = 2,300 hyperparameter trials               │
│  ├─ Boosting: max_depth, learning_rate, n_estimators, etc.         │
│  ├─ Neural: hidden_size, num_layers, dropout, batch_size, etc.     │
│  ├─ Transformer: d_model, n_heads, patch_len, stride, etc.         │
│  └─ Classical: n_estimators, C, kernel, penalty, etc.              │
│                         ↓                                            │
│  Stage 14: TRAINING (PurgedKFold CV, OOF Generation)                │
│  ├─ Tabular (6): XGBoost, LightGBM, CatBoost, RF, Logistic, SVM   │
│  ├─ Neural (10): LSTM, GRU, TCN, Transformer, PatchTST,            │
│  │              iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D│
│  └─ 5-fold PurgedKFold CV with OOF prediction generation           │
│                         ↓                                            │
│  Stage 15: STACKING (OOF Alignment, Meta-Learner)                   │
│  ├─ Heterogeneous base models (1 per family)                       │
│  ├─ OOF predictions aligned and validated                          │
│  └─ Meta-learner: Ridge, MLP, XGBoost, or Calibrated               │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
  [Checkpoint: experiments/runs/{run_id}/models/]
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│               DEPLOYMENT (Stage 16)                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 16: BUNDLING                                                  │
│  Model + Scaler + PreprocessingGraph  →  ModelBundle (V1.1.0)      │
│                                                                     │
│  Bundle Contents:                                                    │
│  ├─ Trained model weights (.pkl/.pt)                                │
│  ├─ Fitted scaler (RobustScaler)                                    │
│  ├─ Feature mask (from optimization)                                │
│  ├─ Barrier params (from optimization)                              │
│  ├─ Hyperparameters (from optimization)                             │
│  └─ PreprocessingGraph (feature lineage)                            │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
  [Output: experiments/runs/{run_id}/bundles/{model}_bundle.pkl]
```

---

## Optuna Optimization Summary

| Stage | Optimization | Config File | Trials | Parameters |
|-------|--------------|-------------|--------|------------|
| **7** | Triple Barrier Labels | `config/optimization/label_optimization.yaml` | 100 | upper_mult, lower_mult, horizon, atr_period |
| **8** | Feature Selection | `config/optimization/feature_selection.yaml` | 100 | binary include/exclude (162 features) |
| **9** | Feature Pruning | `config/optimization/feature_pruning.yaml` | 50 | importance_threshold, top_k, method |
| **13** | Hyperparameters | `config/optimization/hyperparameter.yaml` | 100/model | model-specific (23 models) |

**Total Trials:** 100 + 100 + 50 + (100 × 23) = **2,550 trials**

**Configuration:** All optimization configs in `config/optimization/` - see `config/optimization/README.md` for details

---

## Core Architectural Principles

### 1. Single-Contract Pipeline

**One contract at a time:**
- Pipeline processes exactly one futures contract per run (MES, MGC, etc.)
- No cross-symbol correlation or feature engineering
- Complete isolation between contracts

**Why:**
- Simpler data management (no alignment across symbols)
- Prevents information leakage between contracts
- Easier to reason about feature engineering and labeling

**Configuration:**
```bash
# Train on MES
./pipeline run --symbols MES

# Train on MGC (separate run, separate model)
./pipeline run --symbols MGC
```

### 2. Canonical Dataset with Per-Model Feature Selection

**One canonical source, different feature sets:**
- Single 1-min OHLCV source → ✅ 9 intraday timeframes derived (complete)
- Same timestamps, labels, splits for all models
- **Different features per model family** based on inductive biases:
  - Tabular models: ~200 engineered features (indicators + MTF indicators)
  - Sequence models: ~150 base features (indicators + wavelets, single-TF)
  - Advanced models: Raw multi-stream OHLCV bars (no pre-engineering)

**Why per-model feature selection:**
- **Inductive Bias Alignment:** Tabular models excel with engineered features; transformers learn from raw data
- **Diversity for Ensembles:** Different feature sets → reduced error correlation → better ensemble performance
- **Efficiency:** Sequence models have temporal memory (don't need MTF indicators)

**Adapters handle both:**
1. **Feature Selection:** Choose which features each model gets
2. **Shape Transformation:** Reshape to 2D, 3D, or 4D as needed

**Single source of truth maintained:**
- All features computed from same 1-min canonical OHLCV
- Same timestamps and labels across all models
- Deterministic feature selection (reproducible)

### 3. Model-Family Adapters

**Three adapter types:**

| Adapter | Output Shape | Model Families | Status |
|---------|--------------|----------------|--------|
| **Tabular** | 2D `(N, F)` | Boosting, Classical | ✅ Complete |
| **Sequence** | 3D `(N, T, F)` | Neural | ✅ Complete |
| **Multi-Resolution** | 4D `(N, TF, T, 4)` | Advanced (PatchTST, etc.) | ✅ Complete |

**Adapter responsibilities:**
- Read canonical dataset from `data/splits/scaled/`
- Transform to model-appropriate shape (2D, 3D, 4D)
- No feature engineering (features already computed)
- Deterministic (same input → same output)

### 4. Plugin-Based Model Registry

**Add new models trivially:**
```python
from src.models import register, BaseModel

@register(name="my_model", family="boosting")
class MyModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, ...):
        # Train model
        pass

    def predict(self, X):
        # Generate predictions
        pass
```

**Automatic discovery:**
- Models register themselves via `@register` decorator
- `ModelRegistry.list_all()` returns all available models
- CLI automatically supports new models

### 5. Leakage Prevention

**Multiple layers of protection:**

| Mechanism | Purpose | Location |
|-----------|---------|----------|
| **MTF shift(1)** | Prevent lookahead in multi-timeframe features | Stage 4 |
| **Purge (60 bars)** | Remove overlapping labels between splits | Stage 10 |
| **Embargo (1440 bars)** | Prevent serial correlation leakage | Stage 10 |
| **Train-only scaling** | Fit scaler on train only, transform all splits | Stage 11 |
| **OOF predictions** | Stacking meta-learner uses out-of-fold preds | Stage 15 |

**Result:** No information from validation/test leaks into training.

---

## Data Flow (16 Stages)

### Stage 1: Ingestion
**Input:** `data/raw/{symbol}_1m.parquet`
**Output:** Raw OHLCV DataFrame in memory

**Operations:**
- Schema validation (OHLCV columns, data types)
- Duplicate removal (keep last)

**Time:** ~1 second

### Stage 2: Cleaning
**Input:** Raw OHLCV DataFrame
**Output:** `data/processed/{symbol}_1m_clean.parquet`

**Operations:**
- Resample to 1-minute if needed
- Gap detection (preserved, not filled)
- Data validation and quality checks

**Time:** ~2 seconds

### Stage 3: Sessions
**Input:** Cleaned 1-min OHLCV
**Output:** Session-filtered DataFrame

**Operations:**
- Trading hours filtering (RTH vs ETH)
- Session boundary detection
- Optional extended hours removal

**Time:** ~1 second

### Stage 4: MTF Upscaling
**Input:** `data/processed/{symbol}_1m_clean.parquet`
**Output:** `data/processed/{symbol}_{timeframe}.parquet` (9 files)

**Timeframes:** 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h

**Operations:**
- Resample to higher timeframes (OHLCV aggregation)
- Align to base index (forward-fill)
- Apply shift(1) to prevent lookahead

**Time:** ~4 seconds

### Stage 5: Features
**Input:** Base OHLCV + MTF views
**Output:** `data/features/{symbol}_features.parquet` (162 features)

**Feature Families (12 total, 162 features):**
- Momentum (23): RSI, MACD, Stochastic, Williams %R, ROC, CCI, MFI
- Moving Averages (16): SMA, EMA at multiple periods + crossovers
- Volatility (25): ATR, Bollinger, Keltner, HV, Parkinson, GARCH
- Volume (15): OBV, VWAP, TWAP, Dollar Volume, Volume Ratio
- Trend (6): ADX, +DI/-DI, Supertrend
- Price (12): Returns, ratios, autocorrelation, CLV
- Microstructure (15): Amihud, Roll, Kyle, spreads, imbalances
- Entropy (12): Shannon, Lempel-Ziv, ApEn, SampEn, Hurst
- Wavelets (15): DWT coefficients, energy, entropy
- Temporal (9): Hour/DOW sin/cos, session progress
- Regime (9): Volatility regime, trend regime, composite

**Time:** ~16 seconds

### Stage 6: Regime Detection
**Input:** Feature DataFrame
**Output:** Regime-labeled DataFrame

**Operations:**
- Volatility regime (low/normal/high via ATR percentile)
- Trend regime (downtrend/sideways/uptrend via ADX)
- Composite regime (9 combinations)

**Time:** ~2 seconds

---

### Stage 7: OPTUNA Label Optimization
**Input:** Feature DataFrame
**Output:** Optimized triple-barrier parameters

**Trials:** 100

**Search Space:**
| Parameter | Range |
|-----------|-------|
| `upper_mult` | 1.0 - 4.0 |
| `lower_mult` | 1.0 - 4.0 |
| `horizon` | 5 - 60 |
| `atr_period` | 7 - 28 |

**Objective:** Maximize label quality via quick CV with LightGBM

**Output Files:**
- `optimal_barrier_params.json`
- `label_optimization_history.csv`

**Time:** ~10 minutes

### Stage 8: OPTUNA Feature Selection
**Input:** Labeled DataFrame with 162 features
**Output:** Feature selection mask

**Trials:** 100

**Search Space:** Binary include/exclude for each of 162 features

**Objective:** Maximize F1 score with minimal feature subset

**Output Files:**
- `optimal_feature_mask.json`
- `selected_features.txt` (~60-100 features)

**Time:** ~15 minutes

### Stage 9: OPTUNA Feature Pruning
**Input:** Selected features from Stage 8
**Output:** Final pruned feature set

**Trials:** 50

**Search Space:**
| Parameter | Range |
|-----------|-------|
| `importance_threshold` | 0.001 - 0.1 (log) |
| `top_k_features` | 20 - 100 |
| `importance_method` | gain, split, shap |

**Objective:** Remove low-importance features while maintaining performance

**Output Files:**
- `pruned_feature_mask.json`
- `feature_importance_ranking.csv` (~30-60 features)

**Time:** ~8 minutes

---

### Stage 10: Splits
**Input:** Optimized feature DataFrame
**Output:** Train/val/test DataFrames

**Configuration:**
- Train: 70%
- Validation: 15%
- Test: 15%
- Purge: 60 bars
- Embargo: 1440 bars

**Time:** ~1 second

### Stage 11: Scaling
**Input:** Train/val/test DataFrames
**Output:** Scaled arrays

**Operations:**
- Fit RobustScaler on train only
- Transform all splits with same scaler
- Save scaler for inference bundle

**Time:** ~1 second

### Stage 12: Adaptation
**Input:** Scaled arrays
**Output:** Model-specific tensors

**Adapters:**
- **TabularAdapter:** 2D arrays `(N, ~60)`
- **SequenceAdapter:** 3D windows `(N, seq_len, ~60)`
- **MultiStreamAdapter:** 4D tensors `(N, 9, T, 4)`

**Time:** ~2 seconds

---

### Stage 13: OPTUNA Hyperparameter Optimization
**Input:** Adapted tensors
**Output:** Optimal hyperparameters per model

**Trials:** 100 per model (2,300 total for 23 models)

**Model-Specific Search Spaces:**
- Boosting: max_depth, learning_rate, n_estimators, subsample, etc.
- Neural: hidden_size, num_layers, dropout, batch_size, seq_len, etc.
- Transformer: d_model, n_heads, patch_len, stride, etc.
- Classical: n_estimators (RF), C (SVM/LR), kernel, penalty

**Output Files (per model):**
- `{model}_best_params.json`
- `{model}_optimization_history.csv`

**Time:** ~20-60 minutes per model

### Stage 14: Training
**Input:** Adapted tensors + optimal hyperparameters
**Output:** Trained models in `experiments/runs/{run_id}/models/`

**Operations:**
- Instantiate model from registry with optimal hyperparameters
- 5-fold PurgedKFold cross-validation
- Generate out-of-fold (OOF) predictions
- Train with early stopping, sample weighting
- Evaluate on validation set
- Save model + performance report

**Models (23 total):**
- Tabular (6): XGBoost, LightGBM, CatBoost, RF, Logistic, SVM
- Neural (10): LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D
- Ensemble (3): Voting, Stacking, Blending
- Meta-Learners (4): Ridge, MLP, XGBoost, Calibrated

**Time:**
- Boosting: 10-20 seconds
- Neural: 2-5 minutes (with GPU)
- Classical: 5-60 seconds

### Stage 15: Stacking
**Input:** OOF predictions from Stage 14
**Output:** Stacked ensemble model

**Operations:**
- Align OOF predictions from heterogeneous base models
- Validate OOF alignment (same indices, no NaN)
- Train meta-learner on stacked OOF predictions
- Available meta-learners: Ridge, MLP, XGBoost, Calibrated

**Time:** 1-5 minutes

---

### Stage 16: Bundling
**Input:** Trained models, scaler, optimization artifacts
**Output:** `experiments/runs/{run_id}/bundles/{model}_bundle.pkl`

**Bundle Contents (ModelBundle V1.1.0):**
- Trained model weights (.pkl or .pt)
- Fitted RobustScaler
- Feature mask (from Stages 8-9)
- Barrier parameters (from Stage 7)
- Optimal hyperparameters (from Stage 13)
- PreprocessingGraph (feature lineage for raw inference)
- BundleMetadata (version, creation time, etc.)

**Time:** ~5 seconds per model

---

## Model Families

### Tabular Models (2D Input)

**Boosting (3 models):**
- XGBoost, LightGBM, CatBoost
- **Input:** `(N, ~60)` - optimized features after Stages 8-9
- **Strengths:** Fast, interpretable, feature interactions
- **Training Time:** 10-20 seconds

**Classical (3 models):**
- Random Forest, Logistic Regression, SVM
- **Input:** `(N, ~60)` - same as boosting
- **Strengths:** Robust baselines, simple, interpretable
- **Training Time:** 5-60 seconds

### Sequence Models (3D Input)

**Neural Networks (10 models):**
- LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D
- **Input:** `(N, seq_len, ~60)` - lookback windows (seq_len = 30-60)
- **Strengths:** Temporal dependencies, sequential patterns, multi-scale detection
- **Training Time:** 2-5 minutes (GPU)
- **Status:** All 10 implemented

**Model Details:**

**CNN (2 models):**
- InceptionTime, ResNet1D
- **Input:** `(N, seq_len, ~60)` or multi-resolution `(N, 9, T, 4)`
- **Strengths:** Multi-scale pattern detection

**Advanced Transformers (3 models):**
- PatchTST, iTransformer, TFT
- **Input:** `(N, 9, T, 4)` - raw multi-resolution OHLCV
- **Strengths:** SOTA long-term forecasting, interpretable attention

**MLP (1 model):**
- N-BEATS
- **Input:** `(N, seq_len, ~60)` or `(N, 9, T, 4)`
- **Strengths:** Interpretable decomposition, M4 competition winner

---

## Multi-Timeframe Strategies

### Configurable Primary Timeframe

**Implementation:**
- User specifies primary training timeframe per experiment (5m, 10m, 15m, 1h, etc.)
- All features computed on selected primary timeframe
- MTF enrichment is optional (not required)

**Current State:**
- 9 intraday timeframes available: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
- ~180 indicator-derived features (150 base + 30 MTF from 9 timeframes)

**Data format:**
- Tabular models: 2D arrays `(N, 180)`
- Sequence models: 3D windows `(N, seq_len, 180)`

**Status:** ✅ Complete (9 intraday timeframes, configurable per-model)

### Strategy 1: Single-TF (Baseline)

**Purpose:** Train on chosen timeframe without MTF enrichment
**Data:** Features from one timeframe only (e.g., only 5-minute)
**Models:** All families
**Status:** ✅ Complete

### Strategy 2: MTF Indicators (Optional Enrichment)

**Purpose:** Add indicator features from other timeframes
**Data:** Indicator-derived features from multiple timeframes
**Models:** Tabular models (Boosting, Classical)
**Status:** ✅ Complete (9 intraday timeframes)

### Strategy 3: MTF Ingestion (Optional for Sequence Models)

**Purpose:** Multi-stream raw OHLCV for sequence models
**Data:** Raw OHLCV bars from multiple timeframes as multi-stream input
**Shape:** `(N, T_primary, F)` + optional multi-TF streams
**Models:** Sequence models (Neural, CNN, Transformer, MLP)
**Status:** ✅ Complete

**Note:** Models can mix-and-match strategies in same experiment

---

## Meta-Learner Stacking Architecture

### Heterogeneous Base Models

**NEW APPROACH:** No same-family constraint. Train 3-4 base models from different families.

**Recommended Configuration:**
```python
# 3-4 heterogeneous base models (1 per family)
base_models = {
    "tabular": "catboost",           # OR "lightgbm"
    "cnn": "tcn",                     # Local patterns
    "transformer": "patchtst",        # OR "tft", long context
    "optional_4th": "nbeats"          # OR "ridge", different bias
}
```

**No Input Shape Restriction:** Models can have different input shapes (2D, 3D, 4D)

### Meta-Learner Training

| Method | Description | Input | Leakage Prevention |
|--------|-------------|-------|-------------------|
| **Logistic Stacking** | Linear combination of OOF preds | Base OOF outputs | Out-of-fold CV |
| **Ridge Stacking** | Regularized linear combination | Base OOF outputs | Out-of-fold CV |
| **MLP Blending** | Neural network meta-learner | Base OOF outputs | Out-of-fold CV |
| **Calibrated Blender** | Soft voting + calibration | Base OOF outputs | Holdout split |

**Recommended:**
- **Logistic/Ridge:** Fast, interpretable, prevents overfitting
- **MLP:** Learns complex base model interactions
- **Calibrated Blender:** Combines voting with probability calibration

---

## Configuration System

### Pipeline Configuration
**File:** `config/pipeline.yaml`

```yaml
phase1:
  ingestion:
    data_dir: "data/raw"
    output_dir: "data/processed"

phase2:
  mtf:
    base_timeframe: "5min"
    timeframes: ["15min", "30min", "1h", "4h", "1d"]

phase3:
  features:
    momentum: {rsi_periods: [14, 21]}
    trend: {macd: {fast: 12, slow: 26, signal: 9}}
    wavelets: {types: ["db4", "haar"], levels: 3}

phase4:
  barriers:
    MES: {profit_threshold: 0.015, loss_threshold: 0.010}
  splits: {train_pct: 0.70, val_pct: 0.15, test_pct: 0.15}
  purge_bars: 60
  embargo_bars: 1440
```

### Model Configurations
**Files:** `config/models/{model_name}.yaml`

**Example: XGBoost**
```yaml
model_params:
  objective: "multi:softprob"
  num_class: 3
  max_depth: 6
  learning_rate: 0.1

training:
  num_boost_round: 1000
  early_stopping_rounds: 50
```

**Example: LSTM**
```yaml
model_params:
  hidden_size: 128
  num_layers: 2
  dropout: 0.2

training:
  max_epochs: 200
  batch_size: 256
  lr: 0.001
  patience: 20
  seq_len: 30
```

---

## Directory Structure

```
Research/
├── data/
│   ├── raw/                       # Raw OHLCV data
│   │   ├── MES_1m.parquet
│   │   └── MGC_1m.parquet
│   ├── processed/                 # Clean OHLCV + MTF views
│   │   ├── MES_1m_clean.parquet
│   │   ├── MES_5m.parquet
│   │   ├── MES_15m.parquet
│   │   └── ...
│   ├── features/                  # Engineered features
│   │   └── MES_features.parquet  (~180 features)
│   └── splits/
│       └── scaled/                # Train/val/test splits (canonical)
│           ├── MES_train.parquet
│           ├── MES_val.parquet
│           └── MES_test.parquet
│
├── experiments/
│   └── runs/
│       └── {run_id}/
│           ├── models/            # Trained models
│           │   ├── xgboost_MES_h20.pkl
│           │   ├── lstm_MES_h20.pt
│           │   └── voting_MES_h20.pkl
│           ├── reports/           # Performance reports
│           │   └── xgboost_report.json
│           └── artifacts/         # Logs, configs, etc.
│
├── src/
│   ├── phase1/                    # Data pipeline (Phases 1-5)
│   │   └── stages/
│   │       ├── ingest/            # Phase 1: Data loading
│   │       ├── clean/             # Phase 1: Cleaning, resampling
│   │       ├── mtf/               # Phase 2: MTF upscaling
│   │       ├── features/          # Phase 3: Feature engineering
│   │       ├── labeling/          # Phase 4: Triple-barrier labeling
│   │       ├── splits/            # Phase 4: Train/val/test splits
│   │       ├── scaling/           # Phase 4: Robust scaling
│   │       └── datasets/          # Phase 5: Adapters
│   │
│   ├── models/                    # Model implementations (Phase 6-8)
│   │   ├── base.py                # BaseModel interface
│   │   ├── registry.py            # Model registry
│   │   ├── trainer.py             # Unified trainer
│   │   ├── boosting/              # XGBoost, LightGBM, CatBoost
│   │   ├── neural/                # LSTM, GRU, TCN, Transformer
│   │   ├── classical/             # Random Forest, Logistic, SVM
│   │   ├── ensemble/              # Voting, Stacking, Blending
│   │   └── meta_learners/         # Regime-aware, adaptive (planned)
│   │
│   └── cross_validation/          # Phase 3: CV and tuning
│       ├── purged_kfold.py
│       ├── cv_runner.py
│       └── oof_generator.py
│
├── config/                        # Configuration files
│   ├── pipeline.yaml              # Pipeline config
│   ├── features.yaml              # Feature config
│   ├── labeling.yaml              # Labeling config
│   ├── ensembles.yaml             # Ensemble config
│   └── models/                    # Per-model configs
│       ├── xgboost.yaml
│       ├── lstm.yaml
│       └── ...
│
├── scripts/                       # CLI scripts
│   ├── train_model.py             # Train single/ensemble models
│   ├── run_cv.py                  # Cross-validation
│   └── run_walk_forward.py        # Walk-forward validation
│
└── docs/                          # Documentation
    ├── README.md                  # Entry point
    ├── ARCHITECTURE.md            # This file
    ├── QUICK_REFERENCE.md         # Command cheatsheet
    ├── implementation/            # Implementation phases & roadmaps
    │   ├── PHASE_1_INGESTION.md
    │   ├── PHASE_2_MTF_UPSCALING.md
    │   ├── PHASE_3_FEATURES.md
    │   ├── PHASE_4_LABELING.md
    │   ├── PHASE_5_ADAPTERS.md
    │   ├── PHASE_6_TRAINING.md
    │   ├── PHASE_7_ENSEMBLES.md
    │   ├── PHASE_8_META_LEARNERS.md
    │   ├── MTF_IMPLEMENTATION_ROADMAP.md
    │   └── ADVANCED_MODELS_ROADMAP.md
    ├── guides/                    # How-to guides
    │   ├── MODEL_INTEGRATION.md
    │   ├── FEATURE_ENGINEERING.md
    │   ├── HYPERPARAMETER_TUNING.md
    │   ├── ENSEMBLE_CONFIGURATION.md
    │   └── NOTEBOOK_SETUP.md
    ├── reference/                 # Technical reference
    │   ├── MODELS.md
    │   ├── FEATURES.md
    │   ├── PIPELINE_STAGES.md
    │   ├── SLIPPAGE.md
    │   └── INFRASTRUCTURE.md
    └── archive/                   # Historical docs
```

---

## Quick Commands

### Run Full Pipeline (Data Only)
```bash
# Process single contract
./pipeline run --symbols MES

# Output: data/splits/scaled/{MES_train,MES_val,MES_test}.parquet
```

### Train Single Models
```bash
# Train XGBoost
python scripts/train_model.py --model xgboost --horizon 20 --symbol MES

# Train LSTM (specify seq_len)
python scripts/train_model.py --model lstm --horizon 20 --seq-len 30

# Train all models
python scripts/train_model.py --model all --horizon 20
```

### Train Ensembles
```bash
# Voting ensemble (tabular models)
python scripts/train_model.py \
  --model voting \
  --base-models xgboost,lightgbm,catboost \
  --horizon 20

# Stacking ensemble (sequence models)
python scripts/train_model.py \
  --model stacking \
  --base-models lstm,gru,tcn \
  --horizon 20 \
  --seq-len 30 \
  --meta-learner logistic
```

### Cross-Validation
```bash
# CV for single model
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# CV with Optuna tuning
python scripts/run_cv.py --models xgboost --horizons 20 --tune --n-trials 50
```

### List Available Models
```bash
python scripts/train_model.py --list-models
# Output: 23 models (19 base + 4 meta-learners)
# Families: Tabular (6), Neural (10), Ensemble (3), Meta-Learners (4)
```

---

## Performance Benchmarks

**Hardware:** NVIDIA RTX 4090, 64GB RAM, AMD Ryzen 9 7950X
**Dataset:** MES 1-year (~105K 5-min bars, ~73K after splits)

| Stage | Operation | Time | Memory |
|-------|-----------|------|--------|
| **1-3** | Ingestion + Cleaning + Sessions | ~4s | 50 MB |
| **4** | MTF Upscaling (9 TFs) | ~4s | 80 MB |
| **5-6** | Features + Regime | ~18s | 150 MB |
| **7** | OPTUNA Label Optimization (100 trials) | ~10min | 200 MB |
| **8** | OPTUNA Feature Selection (100 trials) | ~15min | 200 MB |
| **9** | OPTUNA Feature Pruning (50 trials) | ~8min | 200 MB |
| **10-12** | Splits + Scaling + Adaptation | ~4s | 100 MB |
| **13** | OPTUNA Hyperparams (per model) | ~20-60min | 500 MB |
| **14** | Training (per model, XGBoost) | ~15s | 500 MB |
| **14** | Training (per model, LSTM GPU) | ~3min | 2 GB |
| **15** | Stacking (5-fold OOF) | ~5min | 100 MB |
| **16** | Bundling | ~5s | 50 MB |

**Optuna Trials Summary:**
| Stage | Trials | Time |
|-------|--------|------|
| 7 | 100 | ~10 min |
| 8 | 100 | ~15 min |
| 9 | 50 | ~8 min |
| 13 | 100 x 23 models | ~8-23 hours |
| **Total** | **2,550** | **~10-25 hours** |

**Total Pipeline (1 model, no Optuna):** ~5 minutes
**Total Pipeline (1 model, with Optuna):** ~1-2 hours
**Total Pipeline (23 models, with Optuna):** ~10-25 hours (parallelizable)

---

## Extension Points

### Adding a New Model

**Step 1:** Create model file in `src/models/{family}/`

```python
from src.models import register, BaseModel

@register(name="my_model", family="boosting")
class MyModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        # Training logic
        return TrainingMetrics(...)

    def predict(self, X):
        # Prediction logic
        return PredictionOutput(...)

    def save(self, path):
        # Save model
        pass

    @classmethod
    def load(cls, path):
        # Load model
        pass
```

**Step 2:** Create config file `config/models/my_model.yaml`

**Step 3:** Train model
```bash
python scripts/train_model.py --model my_model --horizon 20
```

**That's it.** Model automatically discovered and available.

### Adding a New Feature

**Step 1:** Add feature calculation in `src/phase1/stages/features/indicators/`

```python
class MyIndicator:
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        # Calculate feature
        return pd.Series(...)
```

**Step 2:** Register in `src/phase1/stages/features/feature_engineer.py`

```python
my_indicator = MyIndicator()
df["my_feature"] = my_indicator.calculate(df)
```

**Step 3:** Update `config/features.yaml` (optional)

**Feature automatically available to all models.**

### Adding a New Timeframe (MTF Extension)

**Step 1:** Update `config/pipeline.yaml`
```yaml
phase2:
  mtf:
    timeframes: ["15min", "30min", "1h", "4h", "1d", "10min", "20min", "25min", "45min"]
```

**Step 2:** Update `src/phase1/stages/mtf/mtf_scaler.py` (if needed)

**Timeframes automatically upscaled and aligned.**

---

## Design Decisions

### Why Single-Contract Pipeline?

**Decision:** Process one contract per run, no cross-symbol features.

**Rationale:**
- Simpler: No alignment issues across symbols
- Faster: Smaller datasets, faster iteration
- Isolated: Prevents leakage between contracts
- Sufficient: Most trading strategies are single-contract

**Trade-off:** Can't model cross-symbol correlation (e.g., ES vs NQ spread)

### Why Adapters Instead of Separate Pipelines?

**Decision:** One canonical dataset + adapters for model-specific formats.

**Rationale:**
- **Single source of truth:** Canonical data in `data/splits/scaled/`
- **Reproducibility:** All models train on identical features/labels
- **Storage efficiency:** Store data once, not per model family
- **Deterministic:** Adapters are pure transformations (no stochasticity)

**Trade-off:** Adapters add slight overhead (~2 seconds), but ensures consistency.

### Why Train-Only Scaling?

**Decision:** Fit scaler on train split only, transform all splits.

**Rationale:**
- **Prevents leakage:** Scaler never sees validation/test statistics
- **Realistic:** Mimics production (scaler fit on historical data)
- **Standard practice:** Industry standard for time-series ML

**Trade-off:** Validation/test may have values outside train range (handled via robust scaler).

### Why Purge + Embargo?

**Decision:** Remove 60 bars (purge) + 1440 bars (embargo) between splits.

**Rationale:**
- **Purge:** Labels look forward `horizon` bars; purge 3x to ensure no overlap
- **Embargo:** Financial data has serial correlation; 5 days (~1440 bars) prevents temporal leakage
- **Evidence:** Proven effective in "Advances in Financial Machine Learning" (de Prado)

**Trade-off:** Lose ~10% of data, but prevents overfitting.

---

## Future Roadmap

### Completed (16-Stage Pipeline)
- 9-timeframe MTF ladder (Stage 4)
- 162 features across 12 families (Stage 5)
- All MTF strategies (single-TF, MTF indicators, MTF ingestion)
- 23 models: Tabular (6), Neural (10), Ensemble (3), Meta-Learners (4)
- Heterogeneous stacking with meta-learners (Stage 15)
- 4 Optuna optimization stages (Stages 7, 8, 9, 13)

### Short-Term
1. Advanced meta-learners (regime-aware, adaptive weighting)
2. Multi-horizon meta-learners (train across 5, 10, 15, 20 horizons)
3. Distributed Optuna with SQLite/PostgreSQL storage

### Medium-Term
1. Real-time inference pipeline with streaming predictions
2. Online learning (update models in production)
3. Optuna pruning optimization (MedianPruner, Hyperband)

### Long-Term
1. Contextual bandits for ensemble selection
2. Multi-contract correlation models (if needed)
3. AutoML wrapper for full pipeline optimization

---

## 16-Stage Quick Reference

| # | Stage | Trials | Time |
|---|-------|--------|------|
| 1 | Ingestion | - | ~1s |
| 2 | Cleaning | - | ~2s |
| 3 | Sessions | - | ~1s |
| 4 | MTF Upscaling | - | ~4s |
| 5 | Features (162) | - | ~16s |
| 6 | Regime | - | ~2s |
| **7** | **OPTUNA: Labels** | 100 | ~10min |
| **8** | **OPTUNA: Feature Selection** | 100 | ~15min |
| **9** | **OPTUNA: Feature Pruning** | 50 | ~8min |
| 10 | Splits | - | ~1s |
| 11 | Scaling | - | ~1s |
| 12 | Adaptation | - | ~2s |
| **13** | **OPTUNA: Hyperparameters** | 100/model | ~20-60min/model |
| 14 | Training | - | ~15s-3min/model |
| 15 | Stacking | - | ~5min |
| 16 | Bundling | - | ~5s |

**Total Optuna Trials:** 100 + 100 + 50 + (100 x 23) = **2,550**

---

## References

**Documentation:**
- `docs/README.md` - Entry point
- `docs/QUICK_REFERENCE.md` - Command cheatsheet
- `docs/implementation/UNIFIED_PIPELINE_ARCHITECTURE.md` - 16-stage pipeline details
- `docs/implementation/` - Detailed stage implementation guides
- `docs/guides/` - How-to guides
- `docs/reference/` - Technical reference documentation

**Key Papers:**
- "Advances in Financial Machine Learning" (de Prado) - Purge/embargo, triple-barrier labeling
- "The Elements of Statistical Learning" (Hastie et al.) - Ensemble methods
- "Attention Is All You Need" (Vaswani et al.) - Transformer architecture
- "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting" (Oreshkin et al.)
- "PatchTST: A Time Series is Worth 64 Words" (Nie et al.)
- "Optuna: A Next-generation Hyperparameter Optimization Framework" (Akiba et al.)

**Codebase:**
- `src/pipeline/` - Unified pipeline (Stages 1-16)
- `src/features/` - Feature engineering (Stage 5)
- `src/optimization/` - Optuna optimization (Stages 7-9, 13)
- `src/models/` - Model implementations (Stages 14-15)
- `src/bundling/` - Inference bundles (Stage 16)
- `scripts/` - CLI tools

---

**Last Updated:** 2026-01-18
**Architecture Version:** 4.0 (16-stage pipeline with Optuna optimization)
