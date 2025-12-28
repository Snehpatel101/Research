# ML Pipeline Notebook - Complete Reference

**File:** `notebooks/ML_Pipeline.ipynb`
**Purpose:** End-to-end ML training pipeline for OHLCV time series prediction
**Models:** 13 (Boosting, Neural, Classical, Ensemble)
**Features:** 150+ technical indicators

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Notebook Structure](#notebook-structure)
4. [Configuration Reference](#configuration-reference)
5. [Models (13 Total)](#models-13-total)
6. [Feature Engineering (150+)](#feature-engineering-150-features)
7. [Output Files](#output-files)
8. [Validation & Error Messages](#validation--error-messages)
9. [Workflows](#workflows)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

```
1. Open notebooks/ML_Pipeline.ipynb in Colab or Jupyter
2. Configure Section 1 (symbol, models, horizons)
3. Run All Cells (Ctrl+F9 / Cmd+F9)
4. Export trained models from Section 7
```

**Minimum Configuration:**
```python
SYMBOL = "SI"           # Your contract symbol
TRAIN_XGBOOST = True    # Enable at least one model
TRAINING_HORIZON = 20   # Prediction horizon (bars forward)
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ML PIPELINE DATA FLOW                               │
└─────────────────────────────────────────────────────────────────────────────────┘

 ╔═══════════════╗
 ║   RAW DATA    ║    data/raw/{symbol}_1m.csv
 ║   (1-min)     ║    Columns: datetime, open, high, low, close, volume
 ╚═══════╤═══════╝
         │
         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                         PHASE 1: DATA PIPELINE                                 │
 │                                                                                │
 │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │
 │   │ CLEAN   │──▶│FEATURES │──▶│ LABELS  │──▶│ SPLITS  │──▶│ SCALE   │        │
 │   │ 1m→5m   │   │  150+   │   │ Triple  │   │ 70/15/15│   │ Robust  │        │
 │   │ resample│   │indicators│   │ Barrier │   │ purge/  │   │ (train  │        │
 │   │ gaps    │   │ wavelets│   │ Optuna  │   │ embargo │   │  only)  │        │
 │   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘        │
 │                                                                                │
 └───────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
 ╔═══════════════════════════════════════════════════════════════════════════════╗
 ║                        data/splits/scaled/                                     ║
 ║   train_h{horizon}.parquet  │  val_h{horizon}.parquet  │  test_h{horizon}.parquet ║
 ║   ~70% of data              │  ~15% of data            │  ~15% of data (held out) ║
 ╚═══════════════════════════════════════════════════════════════════════════════╝
         │
         ├──────────────────────────────────────────────────────────────┐
         │                                                              │
         ▼                                                              ▼
 ┌───────────────────────────────────────┐     ┌───────────────────────────────────┐
 │       PHASE 2: MODEL TRAINING         │     │    PHASE 3: CROSS-VALIDATION      │
 │                                       │     │         (Optional)                │
 │   ┌─────────────────────────────┐    │     │   ┌─────────────────────────────┐ │
 │   │      BOOSTING (3)           │    │     │   │    PurgedKFold (5 splits)   │ │
 │   │  XGBoost │ LightGBM │CatBoost│    │     │   │    Optuna Tuning (20+trials)│ │
 │   └─────────────────────────────┘    │     │   │    Out-of-Fold Predictions  │ │
 │   ┌─────────────────────────────┐    │     │   └─────────────────────────────┘ │
 │   │       NEURAL (4)            │    │     │                                   │
 │   │ LSTM │ GRU │ TCN │Transformer│    │     │   CV_RESULTS = {                  │
 │   └─────────────────────────────┘    │     │     "xgboost": {"mean_f1": 0.52}, │
 │   ┌─────────────────────────────┐    │     │     "lightgbm": {"mean_f1": 0.51} │
 │   │      CLASSICAL (3)          │    │     │   }                               │
 │   │ RandomForest│Logistic│ SVM  │    │     │                                   │
 │   └─────────────────────────────┘    │     │   TUNING_RESULTS = {              │
 │                                       │     │     "xgboost": {"best_params":..} │
 │   TRAINING_RESULTS = {               │     │   }                               │
 │     "xgboost": {                     │     └───────────────────────────────────┘
 │       "model": <XGBoostModel>,       │
 │       "metrics": {...},              │
 │       "run_id": "20251227_143052"    │
 │     }                                │
 │   }                                  │
 └───────────────────────────────────────┘
         │
         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                       PHASE 4: ENSEMBLE (Optional)                             │
 │                                                                                │
 │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                  │
 │   │    VOTING     │   │   STACKING    │   │   BLENDING    │                  │
 │   │  Soft/Hard    │   │  Meta-Learner │   │   Holdout     │                  │
 │   │  Weighted Avg │   │  on OOF preds │   │   Meta-Learn  │                  │
 │   └───────────────┘   └───────────────┘   └───────────────┘                  │
 │                                                                                │
 │   ENSEMBLE_RESULTS = {                                                         │
 │     "voting": {"model": <VotingEnsemble>, "metrics": {...}},                  │
 │     "stacking": {"model": <StackingEnsemble>, "metrics": {...}}               │
 │   }                                                                            │
 └───────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                         PHASE 5: EXPORT                                        │
 │                                                                                │
 │   experiments/exports/{timestamp}_{symbol}_H{horizon}/                        │
 │   ├── models/                                                                  │
 │   │   └── {model}/                                                            │
 │   │       ├── model.pkl         (Pickle - sklearn compatible)                 │
 │   │       ├── model.onnx        (ONNX - optional cross-platform)              │
 │   │       └── config.json       (Training configuration)                      │
 │   ├── predictions/                                                             │
 │   │   └── {model}/                                                            │
 │   │       ├── val_predictions.csv                                             │
 │   │       └── test_predictions.csv                                            │
 │   ├── metrics/                                                                 │
 │   │   ├── training_metrics.json                                               │
 │   │   └── test_metrics.json                                                   │
 │   ├── visualizations/                                                          │
 │   │   └── {model}/*.png                                                       │
 │   ├── model_cards/                                                             │
 │   │   └── {model}_card.md                                                     │
 │   ├── manifest.json                                                            │
 │   └── README.md                                                                │
 └───────────────────────────────────────────────────────────────────────────────┘
```

---

## Notebook Structure

### Section 1: Master Configuration (Cell 2)

**Purpose:** Single configuration panel for ALL settings

| Category | Parameters |
|----------|------------|
| Data | SYMBOL, DATE_RANGE, DRIVE_DATA_PATH, CUSTOM_DATA_FILE |
| Pipeline | HORIZONS, TRAIN/VAL/TEST ratios, PURGE_BARS, EMBARGO_BARS |
| Models | 13 boolean toggles (TRAIN_XGBOOST, TRAIN_LSTM, etc.) |
| Neural | SEQUENCE_LENGTH, BATCH_SIZE, MAX_EPOCHS, EARLY_STOPPING |
| Transformer | N_HEADS, N_LAYERS, D_MODEL, SEQUENCE_LENGTH |
| Boosting | N_ESTIMATORS, EARLY_STOPPING |
| Ensemble | Base models, weights, meta-learners |
| CV | N_SPLITS, TUNE_HYPERPARAMS, N_TRIALS |
| Execution | RUN_DATA_PIPELINE, RUN_MODEL_TRAINING, SAFE_MODE |
| Reproducibility | RANDOM_SEED |
| Class Balance | USE_CLASS_WEIGHTS, USE_SAMPLE_WEIGHTS |

### Section 2: Environment Setup (Cells 4-8)

| Cell | Purpose | Outputs |
|------|---------|---------|
| 2.1 | Environment Detection | `IS_COLAB`, `PROJECT_ROOT`, paths |
| 2.2 | Install Dependencies | All packages installed |
| 2.3 | GPU Detection | `GPU_AVAILABLE`, `GPU_NAME`, `GPU_MEMORY` |
| 2.4 | Reproducibility Setup | Random seeds set (Python, NumPy, PyTorch) |
| 2.5 | Memory Utilities | `clear_memory()`, `print_memory_status()` |

### Section 3: Phase 1 - Data Pipeline (Cells 10-12)

| Cell | Purpose | Outputs |
|------|---------|---------|
| 3.1 | Verify Raw Data | `RAW_DATA_FILE`, date range detected |
| 3.2 | Run Data Pipeline | Scaled parquet files in `data/splits/scaled/` |
| 3.3 | Verify Processed Data | `FEATURE_COLS`, `LABEL_COLS`, `DATA_READY` |

### Section 4: Phase 2 - Model Training (Cells 14-18)

| Cell | Purpose | Outputs |
|------|---------|---------|
| 4.1 | Train Models | `TRAINING_RESULTS` dict |
| 4.2 | Compare Models | Comparison table, bar charts |
| 4.3 | Visualize Results | Confusion matrices, feature importance, learning curves |
| 4.4 | Transformer Attention | Attention heatmaps (transformer only) |
| 4.5 | Test Set Performance | `TEST_RESULTS`, generalization gap |

### Section 5: Phase 3 - Cross-Validation (Cells 20-21)

| Cell | Purpose | Outputs |
|------|---------|---------|
| 5.1 | Run Cross-Validation | `CV_RESULTS`, `TUNING_RESULTS` |
| 5.2 | Tuning Results | Best params, retrain recommendations |

### Section 6: Phase 4 - Ensemble (Cells 23-24)

| Cell | Purpose | Outputs |
|------|---------|---------|
| 6.1 | Train Ensemble | `ENSEMBLE_RESULTS` |
| 6.2 | Ensemble Analysis | Diversity metrics, contribution charts |

### Section 7: Results & Export (Cells 26-27)

| Cell | Purpose | Outputs |
|------|---------|---------|
| 7.1 | Final Summary | Best model, overall stats |
| 7.2 | Export Package | Complete export directory |

---

## Configuration Reference

### Data Configuration

| Parameter | Type | Default | Valid Values | Description |
|-----------|------|---------|--------------|-------------|
| `SYMBOL` | str | `"SI"` | SI, MES, MGC, ES, GC, NQ, CL, HG, ZB, ZN | Futures contract symbol |
| `DATE_RANGE` | str | `"2019-2024"` | 2019-2024, 2020-2024, 2021-2024, etc. | Date range filter |
| `DRIVE_DATA_PATH` | str | `"research/data/raw"` | Any path | Google Drive path (Colab) |
| `CUSTOM_DATA_FILE` | str | `""` | Any filename | Override auto-detection |

### Pipeline Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `HORIZONS` | str | `"5,10,15,20"` | Comma-separated ints | Prediction horizons (bars forward) |
| `TRAIN_RATIO` | float | `0.70` | 0.0-1.0 | Training set proportion |
| `VAL_RATIO` | float | `0.15` | 0.0-1.0 | Validation set proportion |
| `TEST_RATIO` | float | `0.15` | 0.0-1.0 | Test set proportion |
| `PURGE_BARS` | int | `60` | ≥0 | Bars to remove at split boundaries |
| `EMBARGO_BARS` | int | `1440` | ≥0 | Embargo period (~5 days at 5-min) |
| `TRAINING_HORIZON` | int | `20` | 5, 10, 15, 20 | Which horizon to train on |

### Model Selection (Boolean Toggles)

| Parameter | Default | Model | GPU Required | Training Time |
|-----------|---------|-------|--------------|---------------|
| `TRAIN_XGBOOST` | `True` | XGBoost | No | ~1 min |
| `TRAIN_LIGHTGBM` | `True` | LightGBM | No | ~30 sec |
| `TRAIN_CATBOOST` | `True` | CatBoost | Optional | ~2 min |
| `TRAIN_RANDOM_FOREST` | `False` | Random Forest | No | ~1 min |
| `TRAIN_LOGISTIC` | `False` | Logistic Regression | No | ~10 sec |
| `TRAIN_SVM` | `False` | Support Vector Machine | No | ~5 min |
| `TRAIN_LSTM` | `False` | LSTM | Yes | ~10 min |
| `TRAIN_GRU` | `False` | GRU | Yes | ~8 min |
| `TRAIN_TCN` | `False` | TCN | Yes | ~12 min |
| `TRAIN_TRANSFORMER` | `False` | Transformer | Yes | ~15 min |
| `TRAIN_VOTING` | `False` | Voting Ensemble | No | ~1 min |
| `TRAIN_STACKING` | `False` | Stacking Ensemble | No | ~5 min |
| `TRAIN_BLENDING` | `False` | Blending Ensemble | No | ~3 min |

### Neural Network Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `SEQUENCE_LENGTH` | int | `60` | 30-120 | Input sequence length (RNN/TCN) |
| `BATCH_SIZE` | int | `256` | 64-1024 | Training batch size |
| `MAX_EPOCHS` | int | `50` | ≥1 | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | int | `10` | ≥1 | Epochs without improvement |

### Transformer Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `TRANSFORMER_SEQUENCE_LENGTH` | int | `128` | ≥32 | Transformer input length |
| `TRANSFORMER_N_HEADS` | int | `8` | 4, 8, 16 | Attention heads |
| `TRANSFORMER_N_LAYERS` | int | `3` | 2-6 | Encoder layers |
| `TRANSFORMER_D_MODEL` | int | `256` | 128, 256, 512 | Model dimension |

### Boosting Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `N_ESTIMATORS` | int | `500` | ≥1 | Number of trees/rounds |
| `BOOSTING_EARLY_STOPPING` | int | `50` | ≥1 | Early stopping rounds |

### Ensemble Configuration

**Voting:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `VOTING_BASE_MODELS` | str | `"xgboost,lightgbm,catboost"` | Comma-separated model names |
| `VOTING_WEIGHTS` | str | `""` | Comma-separated weights (empty=equal) |

**Stacking:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `STACKING_BASE_MODELS` | str | `"xgboost,lightgbm,lstm"` | Base model names |
| `STACKING_META_LEARNER` | str | `"logistic"` | logistic, xgboost, random_forest |
| `STACKING_N_FOLDS` | int | `5` | CV folds for OOF predictions |

**Blending:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `BLENDING_BASE_MODELS` | str | `"xgboost,lightgbm,random_forest"` | Base model names |
| `BLENDING_META_LEARNER` | str | `"logistic"` | Meta-learner type |
| `BLENDING_HOLDOUT_RATIO` | float | `0.2` | Holdout set proportion |

### Class Balancing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_CLASS_WEIGHTS` | bool | `True` | Balance Long/Neutral/Short classes |
| `USE_SAMPLE_WEIGHTS` | bool | `True` | Use pipeline quality weights |

### Cross-Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RUN_CROSS_VALIDATION` | bool | `False` | Enable CV phase |
| `CV_N_SPLITS` | int | `5` | Number of folds |
| `CV_TUNE_HYPERPARAMS` | bool | `False` | Enable Optuna tuning |
| `CV_N_TRIALS` | int | `20` | Optuna trial count |
| `CV_USE_PRESCALED` | bool | `True` | Use pre-scaled data (faster) |

### Execution & Reproducibility

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RUN_DATA_PIPELINE` | bool | `True` | Run Phase 1 |
| `RUN_MODEL_TRAINING` | bool | `True` | Run Phase 2 |
| `SAFE_MODE` | bool | `False` | Low-memory mode |
| `RANDOM_SEED` | int | `42` | Seed for reproducibility (0=random) |

---

## Models (13 Total)

### Boosting Models (3)

#### XGBoost
```python
# Default parameters
n_estimators = 500
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
early_stopping_rounds = 50
objective = "multi:softprob"
eval_metric = "mlogloss"
```

#### LightGBM
```python
# Default parameters
n_estimators = 500
max_depth = -1  # unlimited
num_leaves = 31
learning_rate = 0.1
feature_fraction = 0.8
bagging_fraction = 0.8
early_stopping_rounds = 50
```

#### CatBoost
```python
# Default parameters
iterations = 500
depth = 6
learning_rate = 0.1
l2_leaf_reg = 3.0
task_type = "CPU"  # or "GPU"
early_stopping_rounds = 50
```

### Neural Models (4)

#### LSTM / GRU
```python
# Default parameters
hidden_size = 128
num_layers = 2
dropout = 0.2
bidirectional = False
batch_size = 256
learning_rate = 0.001
max_epochs = 50
early_stopping_patience = 10
```

#### TCN
```python
# Default parameters
num_channels = [64, 128, 256]
kernel_size = 3
dropout = 0.2
batch_size = 256
learning_rate = 0.001
```

#### Transformer
```python
# Default parameters
d_model = 256
n_heads = 8
n_layers = 3
d_ff = 1024
dropout = 0.1
max_seq_len = 128
batch_size = 256
learning_rate = 0.0001
```

### Classical Models (3)

#### Random Forest
```python
n_estimators = 100
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
class_weight = "balanced"
```

#### Logistic Regression
```python
C = 1.0
solver = "lbfgs"
max_iter = 1000
class_weight = "balanced"
```

#### SVM
```python
C = 1.0
kernel = "rbf"
gamma = "scale"
probability = True
class_weight = "balanced"
```

### Ensemble Models (3)

#### Voting
- Combines predictions via weighted average (soft) or majority vote (hard)
- No training required - uses pre-trained base models
- Weights: Equal by default, or specify custom weights

#### Stacking
- Trains meta-learner on out-of-fold predictions
- Uses k-fold CV to generate OOF predictions from base models
- Meta-learner: Logistic regression by default

#### Blending
- Trains meta-learner on holdout predictions
- Splits training data: (1-holdout) for base models, holdout for meta-learner
- Simpler than stacking, less prone to overfitting

---

## Feature Engineering (150+ Features)

### Technical Indicators (via `ta` library)

#### Trend (15+ features)
- SMA: 5, 10, 20, 50 periods
- EMA: 5, 10, 20, 50 periods
- MACD: line, signal, histogram
- ADX: trend strength
- Ichimoku: tenkan, kijun, senkou A/B, chikou

#### Momentum (12+ features)
- RSI: 14-period
- Stochastic: %K, %D
- Williams %R
- ROC: Rate of Change
- CCI: Commodity Channel Index
- MFI: Money Flow Index

#### Volatility (15+ features)
- Bollinger Bands: upper, middle, lower, %B, width
- ATR: Average True Range
- Keltner Channels: upper, middle, lower
- Donchian Channels: high, low, middle

#### Volume (8+ features)
- OBV: On-Balance Volume
- VWAP: Volume-Weighted Average Price
- CMF: Chaikin Money Flow
- ADI: Accumulation/Distribution Index
- Volume SMA, Volume ratio

### Custom Features

#### Price-Based (10+ features)
```
log_return, pct_change, momentum_5, momentum_10, momentum_20
range_hl (high - low)
gap (open - prev_close)
body_ratio ((close - open) / (high - low))
upper_shadow, lower_shadow
```

#### Volatility (8+ features)
```
rolling_std_5, rolling_std_10, rolling_std_20
parkinson_vol (intraday range-based)
garman_klass_vol (OHLC-based)
rogers_satchell_vol
yang_zhang_vol
realized_vol_ratio
```

#### Microstructure (9+ features)
```
amihud_illiquidity
roll_spread
kyle_lambda
corwin_schultz_spread
relative_spread
volume_imbalance
trade_intensity
price_efficiency
order_flow_imbalance
```

#### Wavelet Features (12+ features)
```
wavelet_approx (low-frequency trend)
wavelet_detail_1, _2, _3, _4 (high-frequency components)
wavelet_energy_1, _2, _3, _4
wavelet_trend_strength
wavelet_volatility
```

#### Multi-Timeframe (30+ features)
```
mtf_15min_* (all indicators recomputed)
mtf_60min_* (all indicators recomputed)
mtf_daily_* (if enabled)
cross_tf_momentum
higher_tf_trend_direction
```

#### Temporal (6+ features)
```
hour_sin, hour_cos (cyclical encoding)
day_of_week_sin, day_of_week_cos
minute_of_day
session (pre-market, regular, after-hours)
```

#### Regime Detection (8+ features)
```
volatility_regime (low/medium/high)
trend_regime (up/down/sideways)
volume_regime
combined_regime
regime_persistence
```

---

## Output Files

### Data Pipeline Outputs

| Path | Format | Description |
|------|--------|-------------|
| `data/splits/scaled/train_h{horizon}.parquet` | Parquet | Scaled training data |
| `data/splits/scaled/val_h{horizon}.parquet` | Parquet | Scaled validation data |
| `data/splits/scaled/test_h{horizon}.parquet` | Parquet | Scaled test data |
| `data/splits/unscaled/` | Parquet | Unscaled data (for per-fold CV) |

### Training Outputs

| Path | Format | Description |
|------|--------|-------------|
| `experiments/runs/{run_id}/model.pkl` | Pickle | Trained model |
| `experiments/runs/{run_id}/config.json` | JSON | Training configuration |
| `experiments/runs/{run_id}/metrics.json` | JSON | Performance metrics |
| `experiments/runs/{run_id}/predictions.json` | JSON | Validation predictions |
| `experiments/runs/{run_id}/feature_importance.json` | JSON | Feature importance (boosting/RF) |
| `experiments/runs/{run_id}/training_history.json` | JSON | Learning curves (neural) |
| `experiments/runs/training_results.json` | JSON | Summary of all training |
| `experiments/runs/tuning_results.json` | JSON | Hyperparameter tuning results |

### Export Package

```
experiments/exports/{timestamp}_{symbol}_H{horizon}/
├── models/{model}/
│   ├── model.pkl              # Sklearn-compatible
│   ├── model.onnx             # Cross-platform (optional)
│   └── config.json            # Training config
├── predictions/{model}/
│   ├── val_predictions.csv    # Validation predictions
│   └── test_predictions.csv   # Test predictions
├── metrics/
│   ├── training_metrics.json
│   ├── test_metrics.json
│   └── cv_results.json
├── visualizations/{model}/*.png
├── model_cards/{model}_card.md
├── data/
│   ├── feature_names.txt
│   ├── label_mapping.json
│   └── data_stats.json
├── manifest.json
└── README.md
```

---

## Validation & Error Messages

### Data Validation Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `[ERROR] No data file found for {SYMBOL}!` | File not found | Set `CUSTOM_DATA_FILE` or check `data/raw/` |
| `[ERROR] Missing columns: {cols}` | OHLCV columns missing | Ensure file has open, high, low, close, volume |
| `[WARNING] Data range differs from config` | Date mismatch | Adjust `DATE_RANGE` or use full dataset |

### Training Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `[Error] Data not ready` | Pipeline not run | Execute Section 3 cells first |
| `[Error] No models selected` | All toggles False | Enable at least one model |
| `[ERROR] TRAINING_HORIZON not in processed horizons` | Invalid horizon | Set to 5, 10, 15, or 20 |
| `[ERROR] {model} training failed` | Model error | Check traceback, verify data |

### GPU Warnings

| Warning | Cause | Resolution |
|---------|-------|------------|
| `[WARNING] Neural models selected but no GPU` | No CUDA | Use Colab with GPU runtime |
| `GPU: Not available (using CPU)` | No GPU detected | Change runtime or skip neural models |

### Ensemble Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `[X] Need at least 2 valid base models` | <2 trained | Train more base models first |
| `[!] Invalid weights format` | Malformed weights | Use comma-separated floats |
| `[!] Skipped (not trained/failed)` | Base model missing | Check TRAINING_RESULTS |

### Validation Checks (Built-in)

| Check | Location | Action |
|-------|----------|--------|
| OHLCV integrity | Cell 3.1 | Validates high ≥ low, etc. |
| Date range | Cell 3.1 | Warns if data differs from config |
| Label distribution | Cell 3.3 | Shows Long/Neutral/Short percentages |
| Feature correlation | Pipeline | Removes highly correlated features |
| Train-only scaling | Pipeline | Prevents data leakage |
| Purge/embargo | Pipeline | Ensures no lookahead bias |
| Class balance | Cell 4.1 | Computes and applies class weights |

---

## Workflows

### Workflow 1: Quick Model Comparison
```
1. Set SYMBOL, enable boosting models (XGBoost, LightGBM, CatBoost)
2. Run All
3. Check Section 4.2 for comparison table
4. Best model highlighted automatically
```

### Workflow 2: Neural Network Training
```
1. Enable GPU runtime (Colab: Runtime > Change runtime type > GPU)
2. Enable TRAIN_LSTM or TRAIN_TRANSFORMER
3. Adjust SEQUENCE_LENGTH, BATCH_SIZE for your GPU memory
4. Run All
5. Check Section 4.3 for learning curves
6. Check Section 4.4 for attention visualization (transformer)
```

### Workflow 3: Hyperparameter Tuning
```
1. Train base models first (Workflow 1)
2. Set RUN_CROSS_VALIDATION = True
3. Set CV_TUNE_HYPERPARAMS = True
4. Set CV_N_TRIALS = 50+ for thorough search
5. Run Section 5 cells
6. Check Section 5.2 for best params
7. Retrain models with optimized params
```

### Workflow 4: Production Ensemble
```
1. Train diverse base models (boosting + neural)
2. Enable TRAIN_STACKING = True
3. Configure STACKING_BASE_MODELS with trained models
4. Run Section 6 cells
5. Check diversity analysis (Section 6.2)
6. Export best ensemble (Section 7.2)
```

### Workflow 5: Full Pipeline (All Phases)
```
1. Configure all settings in Section 1
2. Enable: TRAIN_XGBOOST, TRAIN_LIGHTGBM, TRAIN_LSTM
3. Enable: RUN_CROSS_VALIDATION, CV_TUNE_HYPERPARAMS
4. Enable: TRAIN_STACKING
5. Run All Cells
6. ~30-60 minutes total
7. Export complete package from Section 7.2
```

---

## API Reference

### TimeSeriesDataContainer

```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer

# Load processed data
container = TimeSeriesDataContainer.from_parquet_dir(
    path="data/splits/scaled",
    horizon=20
)

# Get sklearn-compatible arrays
X_train, y_train, w_train = container.get_sklearn_arrays("train")
X_val, y_val, w_val = container.get_sklearn_arrays("val")
X_test, y_test, w_test = container.get_sklearn_arrays("test")

# Properties
container.feature_names  # List of feature columns
container.n_features     # Number of features
container.splits         # Available splits
```

### ModelRegistry

```python
from src.models import ModelRegistry

# List all models
ModelRegistry.list_all()  # ['catboost', 'gru', 'lightgbm', 'logistic', ...]
ModelRegistry.count()     # 13

# List by family
ModelRegistry.list_models()
# {'boosting': ['xgboost', 'lightgbm', 'catboost'],
#  'neural': ['lstm', 'gru', 'tcn', 'transformer'],
#  'classical': ['random_forest', 'logistic', 'svm'],
#  'ensemble': ['voting', 'stacking', 'blending']}

# Create model
model = ModelRegistry.create("xgboost", config={"max_depth": 8})

# Train
metrics = model.fit(X_train, y_train, X_val, y_val, sample_weights=w_train)

# Predict
output = model.predict(X_test)
output.class_predictions      # Shape (n_samples,) - values: -1, 0, 1
output.class_probabilities    # Shape (n_samples, 3)
output.confidence             # Shape (n_samples,)

# Save/Load
model.save("path/to/model")
model.load("path/to/model")
```

### Cross-Validation

```python
from src.cross_validation import PurgedKFold, PurgedKFoldConfig

config = PurgedKFoldConfig(
    n_splits=5,
    purge_bars=60,
    embargo_bars=1440
)
cv = PurgedKFold(config)

# Generate folds
for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "No data found" | Symbol mismatch | Check `data/raw/` for exact filename |
| GPU not detected | Wrong runtime | Colab: Runtime > Change runtime type > GPU |
| Out of memory | Large batch/sequence | Reduce BATCH_SIZE, SEQUENCE_LENGTH |
| Poor accuracy | Class imbalance | Enable USE_CLASS_WEIGHTS |
| Slow training | CPU for neural | Use GPU or stick to boosting models |
| Overfitting | No regularization | Increase dropout, reduce max_depth |
| NaN in training | Bad data | Check for inf/NaN in features |
| Kernel crash | Memory leak | Enable SAFE_MODE, reduce batch size |

---

## Key Safeguards

| Safeguard | Implementation | Prevents |
|-----------|----------------|----------|
| Train-only scaling | Scaler fit only on train split | Data leakage from val/test |
| Purge (60 bars) | Gap at split boundaries | Label leakage |
| Embargo (1440 bars) | Post-split buffer | Serial correlation leakage |
| Random seeds | Python, NumPy, PyTorch seeded | Non-reproducible results |
| Class weights | Balanced weighting | Majority class bias |
| Sample weights | Quality-based weighting | Low-quality sample influence |
| Early stopping | Monitor val loss | Overfitting |
| PurgedKFold | Time-aware CV splits | Lookahead bias in CV |

---

## Performance Expectations

| Horizon | Expected Sharpe | Win Rate | Max Drawdown |
|---------|-----------------|----------|--------------|
| H5 | 0.3 - 0.8 | 45-50% | 10-25% |
| H10 | 0.4 - 0.9 | 46-52% | 9-20% |
| H15 | 0.4 - 1.0 | 47-53% | 8-18% |
| H20 | 0.5 - 1.2 | 48-55% | 8-18% |

**Note:** Performance varies by symbol, market conditions, and model selection. These are baseline expectations from Phase 1 analysis.

---

**Document Version:** 2.0
**Last Updated:** 2025-12-27
**Generated by:** 7-agent review process
