# ML Model Factory Architecture

**Last Updated:** 2026-01-01

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
│                    MULTI-TIMEFRAME UPSCALING                        │
│                                                                     │
│  1-min OHLCV  →  [Resample]  →  5min, 15min, 30min, 1h, 4h, daily │
│  (Proper alignment with shift(1) to prevent lookahead)            │
│  Status: ⚠️ 5 of 9 timeframes (intended: 9-TF ladder)             │
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
│  Single Models (13 implemented):                                   │
│  ├─ Boosting (3):     XGBoost, LightGBM, CatBoost                 │
│  ├─ Neural (4):       LSTM, GRU, TCN, Transformer                 │
│  ├─ Classical (3):    Random Forest, Logistic, SVM                │
│  └─ Ensemble (3):     Voting, Stacking, Blending                  │
│                                                                     │
│  Advanced Models (6 planned):                                      │
│  ├─ CNN (2):          InceptionTime, 1D ResNet                     │
│  ├─ Transformers (3): PatchTST, iTransformer, TFT                 │
│  └─ MLP (1):          N-BEATS                                      │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   ENSEMBLE MODELS (PHASE 7)                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  Base Models  →  [Voting/Stacking/Blending]  →  Ensemble │     │
│  │  (Same family: all tabular OR all sequence)              │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Recommended Configurations:                                       │
│  ├─ Boosting Trio:    XGBoost + LightGBM + CatBoost               │
│  ├─ Mixed Tabular:    XGBoost + LightGBM + Random Forest          │
│  └─ All Neural:       LSTM + GRU + TCN + Transformer              │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                META-LEARNERS (PHASE 8 - Planned)                    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  Ensembles  →  [Regime-Aware/Adaptive]  →  Final Preds  │     │
│  │  (Dynamic weighting based on context)                    │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Strategies:                                                        │
│  ├─ Regime-Aware:     Weight by market regime (trend/range/vol)   │
│  ├─ Confidence-Based: Weight by prediction confidence             │
│  └─ Adaptive:         Weight by recent performance                │
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

### 2. Canonical Dataset with Adapters

**One unified pipeline:**
- Single data processing flow from raw OHLCV to features/labels
- Canonical dataset stored once in `data/splits/scaled/`
- Adapters transform canonical data to model-specific formats **on-the-fly**

**Why:**
- Single source of truth (no duplicate datasets)
- Reproducibility (same data for all models)
- Storage efficiency (canonical data stored once, adapters are deterministic)

**NOT separate pipelines:**
- There is ONE pipeline, not multiple "model-specific pipelines"
- Adapters are lightweight transformations, not separate data processing flows

### 3. Model-Family Adapters

**Three adapter types:**

| Adapter | Output Shape | Model Families | Status |
|---------|--------------|----------------|--------|
| **Tabular** | 2D `(N, F)` | Boosting, Classical | ✅ Complete |
| **Sequence** | 3D `(N, T, F)` | Neural | ✅ Complete |
| **Multi-Resolution** | 4D `(N, TF, T, 4)` | Advanced (PatchTST, etc.) | ❌ Planned |

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
**Output:** `data/processed/{symbol}_{timeframe}.parquet` (5 files: 15m, 30m, 1h, 4h, 1d)

**Operations:**
- Resample to higher timeframes (OHLCV aggregation)
- Align to 5-minute base index (forward-fill)
- Apply shift(1) to prevent lookahead

**Status:** ⚠️ 5 of 9 timeframes (intended: 9-TF ladder)

**Time:** ~4 seconds

### Phase 3: Feature Engineering
**Input:** Base OHLCV + MTF views
**Output:** `data/features/{symbol}_features.parquet` (~180 features)

**Operations:**
- Base indicators: RSI, MACD, ATR, Bollinger, ADX (~70 features)
- Wavelets: Db4/Haar decomposition, 3 levels (~30 features)
- Microstructure: Spread proxies, order flow (~20 features)
- Statistical: Skewness, kurtosis, autocorr (~15 features)
- MTF indicators: Indicators from 5 timeframes (~30 features)

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
**Output:** `TimeSeriesDataContainer` (in-memory, model-specific shape)

**Operations:**
- **Tabular:** Extract 2D arrays `(N, 180)`
- **Sequence:** Create 3D windows `(N, seq_len, 180)`
- **Multi-Res:** Build 4D tensors `(N, 9, T, 4)` (future)

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

**Neural Networks (4 models):**
- LSTM, GRU, TCN, Transformer
- **Input:** `(N, seq_len, 180)` - lookback windows (seq_len = 30-60)
- **Strengths:** Temporal dependencies, sequential patterns
- **Training Time:** 2-5 minutes (GPU)

### Advanced Models (4D Input - Planned)

**CNN (2 models):**
- InceptionTime, 1D ResNet
- **Input:** `(N, seq_len, 180)` or multi-resolution `(N, 9, T, 4)`
- **Strengths:** Multi-scale pattern detection
- **Status:** ❌ Not implemented

**Advanced Transformers (3 models):**
- PatchTST, iTransformer, TFT
- **Input:** `(N, 9, T, 4)` - raw multi-resolution OHLCV
- **Strengths:** SOTA long-term forecasting, interpretable attention
- **Status:** ❌ Not implemented (requires Phase 2 Strategy 3)

**MLP (1 model):**
- N-BEATS
- **Input:** `(N, seq_len, 180)` or `(N, 9, T, 4)`
- **Strengths:** Interpretable decomposition, M4 competition winner
- **Status:** ❌ Not implemented

---

## Multi-Timeframe Strategies

### Current: Strategy 2 (MTF Indicators)

**Implementation:**
- ~180 indicator-derived features
- 150 base indicators (5-minute data)
- 30 MTF indicators (from 5 timeframes: 15m, 30m, 1h, 4h, 1d)

**Data format:**
- Tabular models: 2D arrays `(N, 180)`
- Sequence models: 3D windows `(N, seq_len, 180)`

**Status:** ⚠️ Partial (5 of 9 timeframes)

### Planned: Strategy 1 (Single-TF Baselines)

**Purpose:** Ablation study to measure MTF value
**Data:** One timeframe only (e.g., 5-minute)
**Models:** All families
**Status:** ❌ Not implemented (simple config flag)

### Planned: Strategy 3 (MTF Raw Ingestion)

**Purpose:** Multi-resolution temporal learning
**Data:** Raw OHLCV bars from 9 timeframes as 4D tensors
**Shape:** `(N, 9, T, 4)` where:
- N: samples
- 9: timeframes (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- T: lookback window (varies by timeframe)
- 4: OHLC features

**Models:** PatchTST, iTransformer, TFT, N-BEATS
**Status:** ❌ Not implemented (requires Phase 2 extension + Phase 5 multi-res adapter)

**Roadmap:** See `docs/archive/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md`

---

## Ensemble Architecture

### Compatibility Rules

**CRITICAL:** All base models in an ensemble must have the **same input shape**.

**Valid Ensembles:**
```python
# Valid: All tabular (2D)
["xgboost", "lightgbm", "catboost"]
["xgboost", "random_forest", "logistic"]

# Valid: All sequence (3D, same seq_len)
["lstm", "gru", "tcn", "transformer"]  # seq_len=30
```

**Invalid Ensembles:**
```python
# INVALID: Mixing tabular (2D) and sequence (3D)
["xgboost", "lstm"]  # ❌ EnsembleCompatibilityError
```

### Ensemble Methods

| Method | Description | Meta-Learner | Leakage Prevention |
|--------|-------------|--------------|-------------------|
| **Voting** | Weighted avg of predictions | None | N/A (no training) |
| **Stacking** | Train on OOF predictions | Logistic/XGBoost | Out-of-fold CV |
| **Blending** | Train on holdout predictions | Logistic/XGBoost | Holdout split |

**Recommended:**
- **Voting:** Fast baseline, no overfitting risk
- **Stacking:** Best performance, prevents leakage via OOF
- **Blending:** Simpler than stacking, less data for base models

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
    ├── phases/                    # Implementation phases
    │   ├── PHASE_1_INGESTION.md
    │   ├── PHASE_2_MTF_UPSCALING.md
    │   ├── PHASE_3_FEATURES.md
    │   ├── PHASE_4_LABELING.md
    │   ├── PHASE_5_ADAPTERS.md
    │   ├── PHASE_6_TRAINING.md
    │   ├── PHASE_7_ENSEMBLES.md
    │   └── PHASE_8_META_LEARNERS.md
    ├── guides/                    # How-to guides
    │   ├── MODEL_INTEGRATION_GUIDE.md
    │   ├── FEATURE_ENGINEERING_GUIDE.md
    │   └── HYPERPARAMETER_OPTIMIZATION_GUIDE.md
    └── archive/                   # Historical docs
        └── roadmaps/
            ├── MTF_IMPLEMENTATION_ROADMAP.md
            └── ADVANCED_MODELS_ROADMAP.md
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
# Output: 13 models
```

---

## Performance Benchmarks

**Hardware:** NVIDIA RTX 4090, 64GB RAM, AMD Ryzen 9 7950X
**Dataset:** MES 1-year (~105K 5-min bars, ~73K after splits)

| Phase | Operation | Time | Memory |
|-------|-----------|------|--------|
| **Phase 1** | Ingestion + Cleaning | ~3s | 50 MB |
| **Phase 2** | MTF Upscaling (5 TFs) | ~4s | 80 MB |
| **Phase 3** | Feature Engineering | ~16s | 150 MB |
| **Phase 4** | Labeling + Splits + Scaling | ~2.5min | 200 MB |
| **Phase 5** | Adapters (in-memory) | <2s | +50-150 MB |
| **Phase 6** | XGBoost Training | ~15s | 500 MB |
| **Phase 6** | LSTM Training (GPU) | ~3min | 2 GB |
| **Phase 7** | Stacking (5-fold) | ~5min | +100 MB |

**Total Pipeline (Data + Train XGBoost):** ~3 minutes
**Total Pipeline (Data + Train All 10 Single Models):** ~20 minutes (with GPU)

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

### Short-Term (1-2 weeks)
1. Complete 9-timeframe MTF ladder (Phase 2 extension)
2. Implement Strategy 1 (single-TF baselines for ablation)
3. Add CNN models (InceptionTime, 1D ResNet)

### Medium-Term (1-2 months)
1. Implement Strategy 3 (multi-resolution raw OHLCV tensors)
2. Add advanced transformer models (PatchTST, iTransformer, TFT)
3. Add N-BEATS (MLP-based forecasting)
4. Implement meta-learners (regime-aware, adaptive)

### Long-Term (3-6 months)
1. Online learning (update models in production)
2. Multi-horizon meta-learners (train across 5, 10, 15, 20 horizons)
3. Contextual bandits for ensemble selection
4. Real-time inference pipeline
5. Multi-contract correlation models (if needed)

---

## References

**Documentation:**
- `docs/README.md` - Entry point
- `docs/QUICK_REFERENCE.md` - Command cheatsheet
- `docs/phases/` - Detailed phase implementation guides
- `docs/guides/` - How-to guides
- `docs/archive/roadmaps/` - Long-term implementation plans

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

**Last Updated:** 2026-01-01
**Architecture Version:** 2.0 (post-cleanup)
