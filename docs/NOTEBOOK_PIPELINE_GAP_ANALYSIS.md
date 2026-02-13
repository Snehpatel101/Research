# Notebook vs Pipeline Configuration Gap Analysis

**Purpose:** Identify configuration capabilities in the pipeline that are not exposed in the notebook Cell 2 interface.

**Generated:** 2026-02-05

---

## Executive Summary

| Category | Notebook Exposes | Pipeline Supports | Gap |
|----------|------------------|-------------------|-----|
| **Total Parameters** | 61 | 200+ | **139+ hidden** |
| **Model Selection** | 15 toggles | 23 models | 8 models hidden |
| **Feature Families** | 8 toggles | 15 families | 7 families hidden |
| **Feature Parameters** | 0 | 50+ parameters | **All hidden** |
| **Adapter Options** | 0 | 25+ parameters | **All hidden** |
| **CV Configuration** | 7 parameters | 15+ parameters | 8+ hidden |
| **Scaling Options** | 0 | 6 parameters | **All hidden** |
| **Labeling Options** | 5 parameters | 20+ parameters | 15+ hidden |

**Critical Finding:** The notebook exposes ~30% of available configuration. Advanced users cannot access many powerful features without modifying code.

---

## 1. Model Selection Gaps

### Exposed in Notebook (15 models)
```
USE_XGBOOST, USE_LIGHTGBM, USE_CATBOOST
USE_RANDOM_FOREST, USE_LOGISTIC, USE_SVM
USE_LSTM, USE_GRU, USE_TCN, USE_NBEATS
USE_INCEPTION_TIME, USE_RESNET_1D
USE_PATCHTST, USE_ITRANSFORMER, USE_TFT
```

### Hidden in Pipeline (8 additional models)
| Model | Type | Why Hidden |
|-------|------|------------|
| `transformer` | Vanilla Transformer | 3D hybrid, alternative to TFT |
| `voting_ensemble` | Ensemble | Requires explicit base model config |
| `stacking_ensemble` | Ensemble | Requires explicit base model config |
| `blending_ensemble` | Ensemble | Requires explicit base model config |
| `ridge_meta` | Meta-learner | Only used internally by ensembles |
| `mlp_meta` | Meta-learner | Only used internally by ensembles |
| `calibrated_meta` | Meta-learner | Only used internally by ensembles |
| `xgboost_meta` | Meta-learner | Only used internally by ensembles |

### Recommendation
- Add `USE_VANILLA_TRANSFORMER` toggle
- Expose ensemble method selection beyond just `ENSEMBLE_METHOD`
- Allow meta-learner selection: `META_LEARNER_TYPE`

---

## 2. Feature Family Gaps

### Exposed in Notebook (8 families)
```
USE_PRICE_FEATURES, USE_MOMENTUM_FEATURES, USE_VOLATILITY_FEATURES
USE_VOLUME_FEATURES, USE_TREND_FEATURES, USE_REGIME_FEATURES
USE_MICROSTRUCTURE_FEATURES, USE_WAVELET_FEATURES
```

### Hidden in Pipeline (7 additional families)
| Family | Features | Why Important |
|--------|----------|---------------|
| `moving_average` | 16 | SMA/EMA crossovers, price ratios |
| `entropy` | 12 | Information-theoretic signals |
| `order_flow` | 12 | Buy/sell pressure indicators |
| `liquidity` | 12 | Market depth signals |
| `mean_reversion` | 10 | Z-scores, variance ratios, OU estimates |
| `temporal` | 9 | Hour/day encoding (already computed) |
| `raw` | 5 | OHLCV passthrough for transformers |

### Recommendation
Add toggles:
```python
USE_MOVING_AVERAGE_FEATURES = True  # Already computed, just expose
USE_ENTROPY_FEATURES = False
USE_ORDER_FLOW_FEATURES = False
USE_LIQUIDITY_FEATURES = False
USE_MEAN_REVERSION_FEATURES = False
USE_TEMPORAL_FEATURES = True  # Already computed, just expose
USE_RAW_FEATURES = False  # For transformer raw mode
```

---

## 3. Feature Parameter Gaps (CRITICAL)

### Exposed in Notebook
```
FEATURE_SELECTION_ENABLED = True
FEATURE_SELECTION_METHOD = "mda"
FEATURE_SELECTION_N = 50
FEATURE_SET = None  # Presets only
```

### Hidden in Pipeline (50+ parameters)

#### Indicator Parameters (All Hardcoded)
| Indicator | Parameter | Hardcoded Value | Could Be |
|-----------|-----------|-----------------|----------|
| RSI | periods | 7, 14, 21 | Configurable list |
| ATR | periods | 7, 14, 21 | Configurable list |
| MACD | fast, slow, signal | 12, 26, 9 | Configurable tuple |
| Stochastic | k_period, d_period | 14, 3 | Configurable tuple |
| Bollinger | window, std_dev | 20, 2.0 | Configurable tuple |
| ADX | period | 14 | Configurable int |
| Supertrend | period, multiplier | 10, 3.0 | Configurable tuple |
| CCI | period | 20 | Configurable int |
| MFI | period | 14 | Configurable int |
| Z-Score | windows | 10, 20, 60 | Configurable list |
| Variance Ratio | lags | 2, 4, 8, 16 | Configurable list |

#### Feature Selection Parameters (Partially Exposed)
| Parameter | Pipeline Default | Exposed? |
|-----------|------------------|----------|
| `n_features` | 50 | Yes (`FEATURE_SELECTION_N`) |
| `method` | "mda" | Yes (`FEATURE_SELECTION_METHOD`) |
| `min_feature_frequency` | 0.6 | **No** |
| `n_estimators` | 100 | **No** |
| `use_clustered_importance` | False | **No** |
| `max_clusters` | 20 | **No** |
| `random_state` | 42 | **No** |

### Recommendation
Add advanced feature config section:
```python
# Advanced Feature Configuration (optional)
FEATURE_SELECTION_MIN_FREQUENCY = 0.6
FEATURE_SELECTION_N_ESTIMATORS = 100
FEATURE_SELECTION_USE_CLUSTERING = False

# Indicator Parameters (optional overrides)
INDICATOR_RSI_PERIODS = [7, 14, 21]
INDICATOR_ATR_PERIODS = [7, 14, 21]
INDICATOR_MACD_PARAMS = (12, 26, 9)
INDICATOR_BB_PARAMS = (20, 2.0)
```

---

## 4. Adapter Configuration Gaps (CRITICAL)

### Exposed in Notebook
```
(Nothing - adapters are auto-selected based on model)
```

### Hidden in Pipeline (25+ parameters)

#### Sequence Adapter (3D Models)
| Parameter | Default | Impact |
|-----------|---------|--------|
| `sequence_length` | 60 | Temporal receptive field |
| `stride` | 1 | Training set size (stride=1 = sliding window) |
| `symbol_column` | "symbol" | Per-symbol isolation |
| `lazy_load` | False | Memory optimization for >1GB files |
| `chunk_size` | 100,000 | Lazy loading chunk size |

#### Multi-Stream Adapter (4D Models)
| Parameter | Default | Impact |
|-----------|---------|--------|
| `timeframes` | ["1min", "5min", "15min", "60min"] | MTF tensor dimension |
| `sequence_length` | 60 | Per-timeframe sequence |
| `stride` | 1 | Overlap between sequences |
| `base_path` | "data/canonical" | MTF data store location |

#### Scaling Options
| Parameter | Default | Impact |
|-----------|---------|--------|
| `method` | "robust" | Scaling algorithm |
| `clip_value` | 5.0 | Outlier clipping threshold |
| `with_centering` | True | RobustScaler median subtraction |
| `quantile_range` | (25.0, 75.0) | RobustScaler IQR range |

### Recommendation
Add adapter configuration section:
```python
# Adapter Configuration (optional)
SEQUENCE_LENGTH = 60  # For RNN/CNN/Transformer
SEQUENCE_STRIDE = 1  # 1=sliding window, N=skip N bars

# Scaling Configuration
SCALER_METHOD = "robust"  # robust, standard, minmax
SCALER_CLIP_VALUE = 5.0  # 0 = no clipping
```

---

## 5. Cross-Validation Gaps

### Exposed in Notebook
```
CV_METHOD = "purged_kfold"
CV_N_SPLITS = 5
RUN_CPCV = False
RUN_PBO = False
RUN_WALK_FORWARD = False
PURGE_BARS = 60
EMBARGO_BARS = 10
```

### Hidden in Pipeline
| Parameter | Pipeline Default | What It Does |
|-----------|------------------|--------------|
| `min_train_size` | 0.3 | Minimum training fraction |
| `PURGE_MULTIPLIER` | 3 | Purge = max(horizons) × 3 |
| `auto_scale_purge_embargo` | True | Auto-calculate from horizons |
| Walk-forward `gap` | horizon-based | Gap between train and test |
| CPCV `n_groups` | 6 | Number of CPCV partitions |
| CPCV `n_test_groups` | 2 | Test groups per combination |
| PBO `n_partitions` | 16 | CSCV partitions for PBO |

### Critical Issue
**EMBARGO_BARS = 10 in notebook is DANGEROUSLY LOW!**
- Pipeline default: 1440 bars (5 days at 5min)
- Notebook default: 10 bars (~50 minutes)
- This could cause significant data leakage!

### Recommendation
```python
# Cross-Validation (with safe defaults)
CV_N_SPLITS = 5
PURGE_BARS = 180  # max(horizons) * 3
EMBARGO_BARS = 1440  # 5 days at 5min = 1440 bars
MIN_TRAIN_SIZE = 0.3
AUTO_SCALE_PURGE_EMBARGO = True  # Let pipeline calculate
```

---

## 6. Labeling Configuration Gaps

### Exposed in Notebook
```
HORIZONS = [20]
LABELING_METHOD = "triple_barrier"
BARRIER_PROFIT_TAKE = 2.0
BARRIER_STOP_LOSS = 2.0
BARRIER_MAX_HOLDING = 50
```

### Hidden in Pipeline (15+ parameters)

#### Alternative Labeling Strategies
| Strategy | Parameters | Use Case |
|----------|------------|----------|
| `adaptive_triple_barrier` | volatility_regime_col, trend_regime_col | Regime-aware barriers |
| `directional` | threshold, use_log_returns | Simple direction prediction |
| `threshold` | pct_up, pct_down, max_bars | Fixed percentage thresholds |
| `regression` | winsorize_pct, scale_factor | Continuous return prediction |
| `meta` | primary_signal_column, bet_size_method | Meta-labeling for sizing |

#### Label Balance Constraints
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_long_pct` | 0.05 | Minimum long class share |
| `min_short_pct` | 0.05 | Minimum short class share |
| `min_neutral_pct` | 0.10 | Minimum neutral share (HARD) |
| `target_neutral_low` | 0.20 | Target neutral lower bound |
| `target_neutral_high` | 0.30 | Target neutral upper bound |

### Recommendation
```python
# Advanced Labeling (optional)
LABELING_METHOD = "triple_barrier"  # Also: adaptive, directional, threshold, regression, meta

# Label Balance (optional)
LABEL_MIN_LONG_PCT = 0.05
LABEL_MIN_SHORT_PCT = 0.05
LABEL_MIN_NEUTRAL_PCT = 0.10

# Adaptive Triple Barrier (if using adaptive)
ADAPTIVE_VOLATILITY_REGIME_COL = "volatility_regime"
ADAPTIVE_TREND_REGIME_COL = "trend_regime"
```

---

## 7. Training Configuration Gaps

### Exposed in Notebook
```
OPTUNA_ENABLED = True
OPTUNA_TRIALS = 50
OPTIMIZE_FOR = "sharpe_ratio"
EXPERIMENT_NAME = "my_experiment"
RANDOM_SEED = 42
VERBOSE = True
```

### Hidden in Pipeline (30+ parameters)

#### Neural Network Training
| Parameter | Default | Impact |
|-----------|---------|--------|
| `batch_size` | 256 | Memory/speed tradeoff |
| `max_epochs` | 100 | Training duration |
| `early_stopping_patience` | 15 | Convergence detection |
| `learning_rate` | 0.001 | Optimization speed |
| `gradient_clip` | 1.0 | Training stability |
| `mixed_precision` | True | Speed/memory optimization |
| `num_workers` | 4 | Data loading parallelism |

#### OOM Recovery
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `oom_recovery_enabled` | True | Auto-recover from OOM |
| `oom_max_retries` | 3 | Retry attempts |
| `oom_batch_reduction_factor` | 0.5 | Batch size reduction |
| `oom_min_batch_size` | 8 | Minimum batch size |

#### Checkpointing
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `checkpoint_interval` | 10 | Save every N epochs |
| `keep_n_checkpoints` | 3 | Recent checkpoints to keep |
| `save_best_only` | True | Only save improvements |

#### Optuna Advanced
| Parameter | Default | Options |
|-----------|---------|---------|
| `sampler` | "tpe" | tpe, random, cmaes, grid |
| `pruner` | "median" | median, hyperband, percentile, none |
| `timeout` | 0 | Seconds (0 = unlimited) |
| `direction` | "maximize" | maximize, minimize |

### Recommendation
```python
# Neural Network Training (optional)
BATCH_SIZE = 256
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 0.001
GRADIENT_CLIP = 1.0
MIXED_PRECISION = True

# OOM Recovery (optional)
OOM_RECOVERY_ENABLED = True
OOM_MAX_RETRIES = 3

# Checkpointing (optional)
CHECKPOINT_ENABLED = True
CHECKPOINT_INTERVAL = 10

# Optuna Advanced (optional)
OPTUNA_SAMPLER = "tpe"  # tpe, random, cmaes
OPTUNA_PRUNER = "median"  # median, hyperband, none
OPTUNA_TIMEOUT = 0  # 0 = unlimited
```

---

## 8. Inference & Backtesting Gaps

### Exposed in Notebook
```
RUN_BACKTEST = True
POSITION_SIZING = "fixed"
COMMISSION_PER_TRADE = 2.50
SLIPPAGE_TICKS = 1
TICK_VALUE = 1.25
GENERATE_FINANCIAL_REPORT = True
```

### Hidden in Pipeline
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `initial_capital` | 100000.0 | Starting capital |
| `max_positions` | 1 | Max concurrent positions |
| `allow_short` | True | Allow short selling |
| `max_leverage` | 1.0 | Maximum leverage |
| `stop_loss` | None | Stop loss percentage |
| `take_profit` | None | Take profit percentage |
| `trailing_stop` | None | Trailing stop percentage |
| `fill_at` | "close" | Execution timing: close, open, vwap |
| `delay_bars` | 0 | Execution delay |
| `kelly_fraction` | 0.25 | Kelly criterion fraction |
| `volatility_target` | 0.15 | Target volatility for sizing |
| `confidence_threshold` | 0.6 | Min confidence for trading |

### Recommendation
```python
# Backtesting (extended)
INITIAL_CAPITAL = 100000.0
MAX_POSITIONS = 1
ALLOW_SHORT = True
MAX_LEVERAGE = 1.0

# Risk Management (optional)
STOP_LOSS = None  # e.g., 0.02 for 2%
TAKE_PROFIT = None  # e.g., 0.04 for 4%
TRAILING_STOP = None

# Position Sizing Advanced
POSITION_SIZING = "fixed"  # fixed, kelly, volatility, confidence
KELLY_FRACTION = 0.25  # For kelly sizing
VOLATILITY_TARGET = 0.15  # For volatility sizing
CONFIDENCE_THRESHOLD = 0.6  # For confidence sizing
```

---

## 9. MTF (Multi-Timeframe) Gaps

### Exposed in Notebook
```
MTF_ENABLED = True
MTF_TIMEFRAMES = ["15min", "30min", "1h"]
```

### Hidden in Pipeline
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `mtf_mode` | "bars" | Mode: bars, indicators, both |
| `mtf_features` | 30 per TF | Which features to compute |
| `mtf_ffill_limit` | None | Forward fill limit |
| `mtf_min_periods_ratio` | 0.5 | Minimum valid data ratio |
| `mtf_shift` | 1 | Anti-lookahead shift (MANDATORY) |

### Recommendation
```python
# MTF Advanced (optional)
MTF_MODE = "indicators"  # bars, indicators, both, multi_stream
MTF_FEATURES_PER_TF = 30  # Number of features per timeframe
MTF_FFILL_LIMIT = None  # Forward fill limit
```

---

## 10. Gap Summary Table

### High-Priority Gaps (Should Expose)

| Gap | Risk | Priority |
|-----|------|----------|
| **EMBARGO_BARS default (10 vs 1440)** | Data leakage | CRITICAL |
| Sequence length/stride | Model performance | High |
| Scaler configuration | Model convergence | High |
| Feature selection frequency | Feature stability | High |
| Neural training params | Training efficiency | High |
| Missing feature families | Feature coverage | Medium |

### Medium-Priority Gaps (Nice to Have)

| Gap | Benefit | Priority |
|-----|---------|----------|
| Indicator parameters | Domain customization | Medium |
| OOM recovery settings | Reliability | Medium |
| Checkpointing | Resumability | Medium |
| Optuna advanced | Tuning efficiency | Medium |
| Label balance constraints | Label quality | Medium |

### Low-Priority Gaps (Advanced Users Only)

| Gap | Benefit | Priority |
|-----|---------|----------|
| Lazy loading chunk size | Memory optimization | Low |
| Adapter symbol isolation | Multi-asset correctness | Low |
| Inference server config | Production deployment | Low |
| Alert configuration | Monitoring | Low |

---

## 11. Recommended Notebook Cell 2 Updates

### Minimal Update (Fix Critical Issues)
```python
# FIX: Change EMBARGO_BARS default from 10 to 1440
EMBARGO_BARS = 1440  # 5 days at 5min (was 10)

# ADD: Basic adapter config
SEQUENCE_LENGTH = 60
```

### Moderate Update (Expose Key Options)
```python
# === ADAPTER CONFIGURATION ===
SEQUENCE_LENGTH = 60  # For RNN/CNN/Transformer
SCALER_METHOD = "robust"  # robust, standard, minmax

# === TRAINING CONFIGURATION ===
BATCH_SIZE = 256
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
MIXED_PRECISION = True

# === FEATURE SELECTION ADVANCED ===
FEATURE_SELECTION_MIN_FREQUENCY = 0.6
FEATURE_SELECTION_USE_CLUSTERING = False
```

### Full Update (Expose All Major Options)
See recommended configurations in each section above.

---

## 12. Architecture Recommendation

### Current Architecture
```
Notebook Cell 2 (61 params)
    ↓
Hard-coded assembly logic
    ↓
Pipeline Config Classes (200+ params)
    ↓
Training/Inference
```

### Recommended Architecture
```
Notebook Cell 2 (100+ params with safe defaults)
    ↓
ExperimentConfig.from_notebook(cell2_vars)
    ↓
Pipeline Config Classes (validation + defaults)
    ↓
Training/Inference
```

**Key Changes:**
1. Create `ExperimentConfig.from_notebook()` that maps Cell 2 vars to pipeline config
2. Add validation layer that catches dangerous defaults (e.g., low EMBARGO_BARS)
3. Expose more params with documented safe defaults
4. Keep advanced params optional with fallback to pipeline defaults

---

*Generated from analysis of `/Users/sneh/research/notebooks/ml_factory_colab.ipynb` and `/Users/sneh/research/src/` pipeline code.*
