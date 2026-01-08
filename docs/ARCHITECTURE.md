# ML Model Factory Architecture

**Last Updated:** 2026-01-08

---

## Overview

This is a **single-pipeline ML model factory** for training, evaluating, and deploying machine learning models on OHLCV time series data. The factory processes one futures contract at a time through a unified pipeline, then uses **model-family adapters** to serve data in the appropriate format (2D tabular, 3D sequences, 4D multi-resolution tensors) for any model type.

**Key Principle:** One canonical dataset → Deterministic adapters → Model-specific training

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW DATA INGESTION                          │
│                                                                     │
│  data/raw/{SYMBOL}_1m.parquet  →  [Validate]  →  [Clean]          │
│  (One contract: MES or MGC)                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-TIMEFRAME UPSCALING (COMPLETE)             │
│                                                                     │
│  1-min OHLCV (canonical)  →  [Upscale to 9 Intraday Timeframes]    │
│                                                                     │
│  ✅ Complete: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h              │
│     (Full 9-TF intraday ladder implemented)                         │
│                                                                     │
│  Then models choose:                                                │
│  • Primary training TF (configurable per-model)                     │
│  • MTF strategy (single-TF / MTF indicators / MTF ingestion)        │
│  • Which TFs to use for enrichment/multi-stream                     │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                            │
│                                                                     │
│  Base Indicators (~150):  RSI, MACD, ATR, Bollinger, ADX, etc.    │
│  Wavelets (~30):          Db4/Haar decomposition (3 levels)        │
│  Microstructure (~20):    Spread proxies, order flow, liquidity    │
│  MTF Indicators (~30):    Indicators from 5 timeframes             │
│  ────────────────────────────────────────────────────────────────  │
│  Total: ~180 features                                              │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    TRIPLE-BARRIER LABELING                          │
│                                                                     │
│  [Optuna Optimization]  →  Profit/Loss/Time Barriers               │
│  [Quality Weighting]    →  0.5x-1.5x based on barrier touches      │
│  [Splits]               →  70/15/15 train/val/test + purge/embargo │
│  [Robust Scaling]       →  Train-only fit, transform all splits    │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL-FAMILY ADAPTERS                           │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│  │   Tabular    │   │   Sequence   │   │  Multi-Res   │           │
│  │   Adapter    │   │   Adapter    │   │   Adapter    │           │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘           │
│         ↓                  ↓                  ↓                     │
│    2D Arrays          3D Windows         4D Tensors                │
│    (N, 180)           (N, T, 180)        (N, 9, T, 4)              │
│         ↓                  ↓                  ↓                     │
│   ┌─────────┐        ┌─────────┐        ┌─────────┐               │
│   │Boosting │        │ Neural  │        │Advanced │               │
│   │Classical│        │  TCN    │        │PatchTST │               │
│   └─────────┘        │Transform│        │iTransf. │               │
│                      └─────────┘        │TFT/NBEATS              │
│                                         └─────────┘               │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING (PHASE 6)                       │
│                                                                     │
│  Base Models (19 implemented):                                      │
│  ├─ Tabular (6):      XGBoost, LightGBM, CatBoost,                 │
│  │                    Random Forest, Logistic, SVM                 │
│  ├─ Neural (10):      LSTM, GRU, TCN, Transformer,                 │
│  │                    PatchTST, iTransformer, TFT, N-BEATS,        │
│  │                    InceptionTime, ResNet1D                       │
│  └─ Ensemble (3):     Voting, Stacking, Blending                   │
│                                                                     │
│  Meta-Learners (4 implemented):                                     │
│  └─ Inference:        Ridge Meta, MLP Meta, Calibrated, XGBoost    │
│                                                                     │
│  Total: 23 models (19 base + 4 meta-learners)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│       HETEROGENEOUS ENSEMBLE STACKING (PHASE 7 - COMPLETE)         │
│                                                                     │
│  ✅ Status: Fully implemented in trainer.py                         │
│  Features: Heterogeneous base model stacking with dual data loading│
│  Usage: --base-models xgboost,lstm,patchtst --meta-learner ridge   │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│          META-LEARNER STACKING (Planned Workflow)                   │
│                                                                     │
│  3-4 Heterogeneous Base Models (1 per family):                     │
│  ├→ Tabular: CatBoost OR LightGBM (engineered features)           │
│  ├→ CNN/TCN: TCN (local patterns)                                  │
│  ├→ Transformer: PatchTST OR TFT (long context)                   │
│  └→ Optional 4th: N-BEATS OR Ridge (different inductive bias)     │
│       ↓                                                             │
│  Generate Out-of-Fold (OOF) Predictions                           │
│       ↓                                                             │
│  Meta-Learner (Inference Family):                                  │
│  ├→ Logistic Regression (stacking)                                 │
│  ├→ Ridge Regression (stacking)                                    │
│  ├→ Small MLP (learned blending)                                   │
│  └→ Calibrated Blender (voting + calibration)                     │
│       ↓                                                             │
│  Final Ensemble Predictions                                        │
│  (Retrain bases on full train, evaluate on test)                  │
└─────────────────────────────────────────────────────────────────────┘
```

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
| **MTF shift(1)** | Prevent lookahead in multi-timeframe features | Phase 2 |
| **Purge (60 bars)** | Remove overlapping labels between splits | Phase 4 |
| **Embargo (1440 bars)** | Prevent serial correlation leakage | Phase 4 |
| **Train-only scaling** | Fit scaler on train only, transform all splits | Phase 4 |
| **OOF predictions** | Stacking meta-learner uses out-of-fold preds | Phase 7 |

**Result:** No information from validation/test leaks into training.

---

## Data Flow

### Phase 1: Canonical OHLCV Ingestion
**Input:** `data/raw/{symbol}_1m.parquet`
**Output:** `data/processed/{symbol}_1m_clean.parquet`

**Operations:**
- Schema validation (OHLCV columns, data types)
- Duplicate removal (keep last)
- Gap detection (preserved, not filled)
- Session filtering (regular vs extended hours)

**Time:** ~3 seconds (1-year data)

### Phase 2: Multi-Timeframe Upscaling
**Input:** `data/processed/{symbol}_1m_clean.parquet`
**Output:** `data/processed/{symbol}_{timeframe}.parquet` (9 files: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)

**Operations:**
- Resample to higher timeframes (OHLCV aggregation)
- Align to base index (forward-fill)
- Apply shift(1) to prevent lookahead

**Status:** ✅ Complete (9 intraday timeframes)

**Time:** ~4 seconds

### Phase 3: Feature Engineering
**Input:** Base OHLCV + MTF views
**Output:** Model-specific features based on per-model feature selection

**Per-Model Feature Selection:**
Different model families get different features tailored to their inductive biases:

**Tabular Models (Boosting + Classical):**
- Base indicators on primary TF: RSI, MACD, ATR, Bollinger, ADX (~60 features)
- Wavelets on primary TF: Db4/Haar decomposition (~24 features)
- Microstructure on primary TF: Spread proxies, order flow (~10 features)
- MTF indicators from other TFs: RSI_1m, MACD_5m, ATR_1h, etc. (~50 features)
- Price/volume features: Returns, log volume, OHLC ratios (~8 features)
- Time features: Hour, day_of_week (~2 features)
- **Total: ~200 engineered features**

**Sequence Models (Neural + CNN):**
- Base indicators on primary TF: RSI, MACD, ATR, Bollinger (~60 features)
- Wavelets on primary TF: Db4 decomposition (~24 features)
- Microstructure on primary TF: Roll spread, volume imbalance (~10 features)
- Price/volume raw features: Returns, log volume (~8 features)
- Time features: Hour, day_of_week (~2 features)
- **Total: ~150 base features (no MTF indicators - model learns from sequence)**

**Advanced Transformers (Planned):**
- Raw OHLCV bars from multiple timeframes as multi-stream input
- No pre-engineered indicators (attention learns from raw data)
- **Input: Multi-stream 1m+5m+15m raw OHLCV (3 streams × 4 OHLC)**

**Why Different Features:**
- Tabular models excel with rich engineered features and MTF indicators
- Sequence models have inherent temporal memory (no need for MTF features)
- Transformers learn multi-scale patterns from raw data via attention

**Time:** ~16 seconds

### Phase 4: Triple-Barrier Labeling
**Input:** `data/features/{symbol}_features.parquet`
**Output:** `data/splits/scaled/{symbol}_{split}.parquet` (train/val/test)

**Operations:**
- Optuna barrier optimization (100 trials, ~2 minutes)
- Triple-barrier labeling (profit/loss/time)
- Quality weighting (0.5x-1.5x)
- Time-series splits (70/15/15) with purge (60) + embargo (1440)
- Robust scaling (train-only fit)

**Time:** ~2.5 minutes

### Phase 5: Model-Family Adapters
**Input:** `data/splits/scaled/` (canonical splits)
**Output:** `TimeSeriesDataContainer` (in-memory, model-specific shape and features)

**Operations (Per-Model Feature Selection + Shape Adaptation):**
- **Tabular Adapter:**
  - Select ~200 engineered features (base indicators + MTF indicators)
  - Output: 2D arrays `(N, ~200)`
- **Sequence Adapter:**
  - Select ~150 base features (indicators + wavelets, no MTF)
  - Create 3D windows from selected features
  - Output: 3D windows `(N, seq_len, ~150)`
- **Multi-Res Adapter (Planned):**
  - Extract raw OHLCV bars from multiple timeframes
  - Build multi-stream 4D tensors
  - Output: 4D tensors `(N, 9, T, 4)` (no engineered features)

**Why Adapters Do Both:**
- **Feature Selection:** Each model gets features tailored to its inductive biases
- **Shape Transformation:** Features reshaped to model-appropriate format (2D/3D/4D)

**Time:** <2 seconds

### Phase 6: Model Training
**Input:** `TimeSeriesDataContainer`
**Output:** Trained models in `experiments/runs/{run_id}/models/`

**Operations:**
- Instantiate model from registry
- Train with early stopping, sample weighting
- Evaluate on validation set
- Save model + performance report

**Time:**
- Boosting: 10-20 seconds
- Neural: 2-5 minutes (with GPU)
- Classical: 5-60 seconds

### Phase 7: Ensemble Training
**Input:** Trained base models (same family)
**Output:** Ensemble models

**Operations:**
- **Voting:** Weighted average of predictions
- **Stacking:** Train meta-learner on OOF predictions
- **Blending:** Train meta-learner on holdout predictions

**Time:** 1-5 minutes (stacking takes longest)

### Phase 8: Meta-Learners (Planned)
**Input:** Ensemble models
**Output:** Adaptive meta-learner

**Operations:**
- Regime-aware weighting (weight by market regime)
- Confidence-based selection (weight by prediction confidence)
- Adaptive performance tracking (weight by recent accuracy)

**Time:** ~5 minutes (regime clustering + weight optimization)

---

## Model Families

### Tabular Models (2D Input)

**Boosting (3 models):**
- XGBoost, LightGBM, CatBoost
- **Input:** `(N, 180)` - all features as tabular rows
- **Strengths:** Fast, interpretable, feature interactions
- **Training Time:** 10-20 seconds

**Classical (3 models):**
- Random Forest, Logistic Regression, SVM
- **Input:** `(N, 180)` - same as boosting
- **Strengths:** Robust baselines, simple, interpretable
- **Training Time:** 5-60 seconds

### Sequence Models (3D Input)

**Neural Networks (10 models):**
- LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D
- **Input:** `(N, seq_len, 180)` - lookback windows (seq_len = 30-60)
- **Strengths:** Temporal dependencies, sequential patterns, multi-scale detection
- **Training Time:** 2-5 minutes (GPU)
- **Status:** ✅ All 10 implemented

**Model Details:**

**CNN (2 models):**
- InceptionTime, ResNet1D
- **Input:** `(N, seq_len, 180)` or multi-resolution `(N, 9, T, 4)`
- **Strengths:** Multi-scale pattern detection
- **Status:** ✅ Complete

**Advanced Transformers (3 models):**
- PatchTST, iTransformer, TFT
- **Input:** `(N, 9, T, 4)` - raw multi-resolution OHLCV
- **Strengths:** SOTA long-term forecasting, interpretable attention
- **Status:** ✅ Complete

**MLP (1 model):**
- N-BEATS
- **Input:** `(N, seq_len, 180)` or `(N, 9, T, 4)`
- **Strengths:** Interpretable decomposition, M4 competition winner
- **Status:** ✅ Complete

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

| Phase | Operation | Time | Memory |
|-------|-----------|------|--------|
| **Phase 1** | Ingestion + Cleaning | ~3s | 50 MB |
| **Phase 2** | MTF Upscaling (9 TFs) | ~4s | 80 MB |
| **Phase 3** | Feature Engineering | ~16s | 150 MB |
| **Phase 4** | Labeling + Splits + Scaling | ~2.5min | 200 MB |
| **Phase 5** | Adapters (in-memory) | <2s | +50-150 MB |
| **Phase 6** | XGBoost Training | ~15s | 500 MB |
| **Phase 6** | LSTM Training (GPU) | ~3min | 2 GB |
| **Phase 7** | Stacking (5-fold) | ~5min | +100 MB |

**Total Pipeline (Data + Train XGBoost):** ~3 minutes
**Total Pipeline (Data + Train All 19 Base Models):** ~35 minutes (with GPU)

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

### Completed (All 7 Phases)
- ✅ 9-timeframe MTF ladder (Phase 2 complete)
- ✅ Strategy 1, 2, 3 (all MTF strategies complete)
- ✅ CNN models: InceptionTime, ResNet1D
- ✅ Advanced transformers: PatchTST, iTransformer, TFT
- ✅ N-BEATS (MLP-based forecasting)
- ✅ Heterogeneous stacking with meta-learners (Phase 7 complete)

### Short-Term (Phase 8)
1. Advanced meta-learners (regime-aware, adaptive weighting)
2. Multi-horizon meta-learners (train across 5, 10, 15, 20 horizons)

### Medium-Term (Phase 9)
1. Real-time inference pipeline with streaming predictions
2. Online learning (update models in production)

### Long-Term
1. Contextual bandits for ensemble selection
2. Multi-contract correlation models (if needed)

---

## References

**Documentation:**
- `docs/README.md` - Entry point
- `docs/QUICK_REFERENCE.md` - Command cheatsheet
- `docs/implementation/` - Detailed phase implementation guides and roadmaps
- `docs/guides/` - How-to guides
- `docs/reference/` - Technical reference documentation

**Key Papers:**
- "Advances in Financial Machine Learning" (de Prado) - Purge/embargo, triple-barrier labeling
- "The Elements of Statistical Learning" (Hastie et al.) - Ensemble methods
- "Attention Is All You Need" (Vaswani et al.) - Transformer architecture
- "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting" (Oreshkin et al.)
- "PatchTST: A Time Series is Worth 64 Words" (Nie et al.)

**Codebase:**
- `src/phase1/` - Data pipeline (Phases 1-5)
- `src/models/` - Model implementations (Phases 6-8)
- `src/cross_validation/` - CV and hyperparameter tuning
- `scripts/` - CLI tools

---

**Last Updated:** 2026-01-08
**Architecture Version:** 3.0 (all 7 phases complete)
