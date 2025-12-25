# Phase 1: Data Preparation Pipeline

## Overview

Phase 1 is the **complete** data preparation pipeline that transforms raw OHLCV trading data into model-ready datasets. It produces standardized outputs that any machine learning model in Phase 2+ can consume via the `TimeSeriesDataContainer` interface.

**Status:** COMPLETE and production-ready.

---

## What Phase 1 Does (Plain English)

Phase 1 takes **raw 1-minute OHLCV bars** and transforms them through 12 sequential stages into **clean, labeled, feature-rich datasets** ready for ML model training.

```
Raw 1-min OHLCV
    |
    v
[Ingest] --> [Clean/Resample] --> [Feature Engineering (150+)]
    |
    v
[MTF Features] --> [Triple-Barrier Labels] --> [Optuna Optimization]
    |
    v
[Final Labels] --> [Train/Val/Test Splits] --> [RobustScaler]
    |
    v
[TimeSeriesDataContainer] --> Ready for Phase 2 Models
```

---

## Pipeline Stages (12 Sequential Steps)

| Stage | Module | Description | Output |
|-------|--------|-------------|--------|
| 1 | `ingest/` | Load and validate raw OHLCV data | Validated 1-min bars |
| 2 | `clean/` | Resample 1min to 5min, handle gaps | Clean 5-min bars |
| 3 | `features/` | Engineer 150+ technical indicators | Feature-enriched data |
| 4 | `mtf/` | Multi-timeframe features (15m, 30m, 1h) | MTF features added |
| 5 | `labeling/` | Initial triple-barrier labels | Labeled data |
| 6 | `ga_optimize/` | Optuna TPE barrier optimization | Optimized parameters |
| 7 | `final_labels/` | Apply optimized parameters | Final labels |
| 8 | `splits/` | Train/val/test with purge/embargo | Split datasets |
| 9 | `scaling/` | RobustScaler (train-only fit) | Scaled features |
| 10 | `datasets/` | Build TimeSeriesDataContainer | Container object |
| 11 | `validation/` | Feature correlation and quality checks | Validation report |
| 12 | `reporting/` | Generate completion reports | Final reports |

---

## Symbol Isolation Principle

**CRITICAL:** Each symbol (MES, MGC) is processed **independently**. There is NO cross-symbol correlation or feature sharing. This ensures:

1. No data leakage between symbols
2. Symbol-specific barrier optimization
3. Clean separation for multi-symbol model training
4. Proper backtesting isolation

---

## Feature Engineering (~150-200 Features)

Phase 1 generates comprehensive features across multiple categories. All features include anti-lookahead protection via `shift(1)` to ensure features at bar[t] only use data up to bar[t-1].

### Feature Categories

#### 1. Price Features (~10 features)
Fundamental price-based calculations:

| Feature | Description | Formula |
|---------|-------------|---------|
| `return_{1,5,10,20,60}` | Simple returns | `(close[t-1] - close[t-1-n]) / close[t-1-n]` |
| `log_return_{1,5,10,20,60}` | Log returns | `log(close[t-1] / close[t-1-n])` |
| `hl_ratio` | High-to-low ratio | `high[t-1] / low[t-1]` |
| `co_ratio` | Close-to-open ratio | `close[t-1] / open[t-1]` |
| `range_pct` | Range as % of close | `(high - low)[t-1] / close[t-1]` |
| `return_autocorr_lag{1,5,10}` | Return autocorrelation | Rolling correlation with lagged self |
| `clv` | Close Location Value | `(2*close - high - low) / (high - low)` |

#### 2. Moving Averages (~15 features)
SMA and EMA at multiple periods:

| Feature | Periods | Description |
|---------|---------|-------------|
| `sma_{10,20,50,100,200}` | 5 periods | Simple moving averages |
| `ema_{9,21,50}` | 3 periods | Exponential moving averages |
| `close_to_sma_{10,20,50,100,200}` | 5 periods | Price relative to SMA |
| `close_to_ema_{9,21,50}` | 3 periods | Price relative to EMA |

#### 3. Momentum Indicators (~25 features)
Oscillators and momentum measures:

| Feature | Parameters | Description |
|---------|------------|-------------|
| `rsi_14` | 14-period | Relative Strength Index |
| `rsi_overbought` | >70 | RSI overbought flag |
| `rsi_oversold` | <30 | RSI oversold flag |
| `macd_line` | (12, 26) | MACD line |
| `macd_signal` | 9-period | MACD signal line |
| `macd_hist` | | MACD histogram |
| `macd_cross_up/down` | | MACD crossover signals |
| `stoch_k` | 14-period | Stochastic %K |
| `stoch_d` | 3-period | Stochastic %D |
| `stoch_overbought/oversold` | | Stochastic flags |
| `williams_r` | 14-period | Williams %R |
| `roc_{5,10,20}` | 3 periods | Rate of Change |
| `cci_20` | 20-period | Commodity Channel Index |
| `mfi_14` | 14-period | Money Flow Index |

#### 4. Volatility Indicators (~25 features)
Risk and volatility measures:

| Feature | Parameters | Description |
|---------|------------|-------------|
| `atr_{7,14,21}` | 3 periods | Average True Range |
| `atr_pct_{7,14,21}` | 3 periods | ATR as % of price |
| `bb_middle/upper/lower` | 20-period | Bollinger Bands |
| `bb_width` | | Band width (normalized by std) |
| `bb_position` | | Price position in bands [0,1] |
| `close_bb_zscore` | | Close z-score vs BB middle |
| `kc_middle/upper/lower` | 20-period | Keltner Channels |
| `kc_position` | | Price position in channels |
| `close_kc_atr_dev` | | Close deviation in ATR units |
| `hvol_{10,20,60}` | 3 periods | Historical volatility (annualized) |
| `parkinson_vol` | 20-period | Parkinson volatility |
| `gk_vol` | 20-period | Garman-Klass volatility |
| `return_skew_{20,60}` | 2 periods | Return skewness |
| `return_kurt_{20,60}` | 2 periods | Return kurtosis (excess) |

#### 5. Volume Indicators (~15 features)
Volume analysis and order flow proxies:

| Feature | Parameters | Description |
|---------|------------|-------------|
| `obv` | | On Balance Volume |
| `obv_sma_20` | 20-period | OBV moving average |
| `volume_sma_20` | 20-period | Volume moving average |
| `volume_ratio` | | Volume vs 20-period SMA |
| `volume_zscore` | 20-period | Volume z-score |
| `vwap` | Session | Volume Weighted Average Price |
| `price_to_vwap` | | Price deviation from VWAP |
| `dollar_volume` | | Price x Volume |
| `dollar_volume_sma_{10,20}` | | Dollar volume averages |
| `dollar_volume_ratio` | | Dollar volume vs average |

#### 6. Trend Indicators (~8 features)
Trend direction and strength:

| Feature | Parameters | Description |
|---------|------------|-------------|
| `adx_14` | 14-period | Average Directional Index |
| `plus_di_14` | 14-period | Positive Directional Index |
| `minus_di_14` | 14-period | Negative Directional Index |
| `adx_strong_trend` | >25 | Strong trend flag |
| `supertrend` | (10, 3.0) | Supertrend value |
| `supertrend_direction` | | Trend direction (1=up, -1=down) |

#### 7. Temporal Features (~9 features)
Time-based patterns:

| Feature | Description |
|---------|-------------|
| `hour_sin/cos` | Hour cyclical encoding (24h cycle) |
| `minute_sin/cos` | Minute cyclical encoding (60m cycle) |
| `dayofweek_sin/cos` | Day of week encoding (7d cycle) |
| `session_asia` | Asia session flag (00:00-08:00 UTC) |
| `session_london` | London session flag (08:00-16:00 UTC) |
| `session_ny` | New York session flag (16:00-24:00 UTC) |

#### 8. Regime Features (~2 features)
Market regime classification:

| Feature | Description |
|---------|-------------|
| `volatility_regime` | 1=high volatility, 0=low volatility |
| `trend_regime` | 1=uptrend, -1=downtrend, 0=sideways |

#### 9. Microstructure Proxies (~20 features)
Market microstructure estimates from OHLCV:

| Feature | Description | Reference |
|---------|-------------|-----------|
| `micro_amihud` | Amihud illiquidity ratio | Amihud (2002) |
| `micro_amihud_{10,20}` | Smoothed Amihud | |
| `micro_roll_spread` | Roll spread estimate | Roll (1984) |
| `micro_roll_spread_pct` | Roll spread as % | |
| `micro_kyle_lambda` | Price impact coefficient | Kyle (1985) |
| `micro_cs_spread` | Corwin-Schultz spread | Corwin & Schultz (2012) |
| `micro_rel_spread` | High-low range / close | |
| `micro_rel_spread_{10,20}` | Smoothed relative spread | |
| `micro_volume_imbalance` | Order flow proxy | |
| `micro_cum_imbalance_20` | Cumulative imbalance | |
| `micro_trade_intensity_{20,50}` | Volume vs average | |
| `micro_efficiency_{10,20}` | Price efficiency ratio | |
| `micro_vol_ratio` | Short/long volatility | |

#### 10. Wavelet Features (~24 features)
Multi-scale signal decomposition using Daubechies-4 wavelets:

| Feature | Description |
|---------|-------------|
| `wavelet_close_approx` | Low-frequency trend (normalized) |
| `wavelet_close_d{1,2,3}` | Detail coefficients (high, mid, low freq) |
| `wavelet_volume_approx` | Volume trend |
| `wavelet_volume_d{1,2,3}` | Volume detail coefficients |
| `wavelet_close_energy_approx` | Trend energy (log-transformed) |
| `wavelet_close_energy_d{1,2,3}` | Frequency band energy |
| `wavelet_volume_energy_*` | Volume energy features |
| `wavelet_close_energy_ratio` | Trend vs total energy |
| `wavelet_close_volatility` | MAD-based volatility |
| `wavelet_close_trend_strength` | Trend strength (normalized slope) |
| `wavelet_close_trend_direction` | Trend direction (-1/0/+1) |

#### 11. Multi-Timeframe (MTF) Features (~30 features)
Higher timeframe context:

| Timeframe | Features Added |
|-----------|---------------|
| 15-minute | OHLCV, RSI, MACD, ATR, Bollinger |
| 30-minute | OHLCV, RSI, MACD, ATR, Bollinger |
| 1-hour | OHLCV, RSI, MACD, ATR, Bollinger |

MTF features have suffixes like `_15m`, `_30m`, `_1h`.

---

## Labeling System

### Triple-Barrier Method

For each bar, we look forward and determine which barrier is hit first:

1. **Profit Target (upper barrier):** Price rises by `k_up * ATR` -> Label: LONG (+1)
2. **Stop Loss (lower barrier):** Price falls by `k_down * ATR` -> Label: SHORT (-1)
3. **Timeout (vertical barrier):** Neither hit within `max_bars` -> Label: NEUTRAL (0)

### Symbol-Specific Barrier Configuration

**MES (S&P 500 Micro Futures) - ASYMMETRIC:**
MES has a long-term bullish drift, so we use asymmetric barriers (k_up > k_down) to counteract this bias.

| Horizon | k_up | k_down | max_bars | Rationale |
|---------|------|--------|----------|-----------|
| H5 | 1.50 | 1.00 | 15 | Counter equity drift |
| H10 | 2.00 | 1.40 | 30 | Counter equity drift |
| H15 | 2.50 | 1.75 | 45 | Counter equity drift |
| H20 | 3.00 | 2.10 | 60 | Counter equity drift |

**MGC (Gold Micro Futures) - SYMMETRIC:**
Gold is more mean-reverting, so symmetric barriers work better.

| Horizon | k_up | k_down | max_bars | Rationale |
|---------|------|--------|----------|-----------|
| H5 | 1.20 | 1.20 | 15 | Mean-reverting asset |
| H10 | 1.60 | 1.60 | 30 | Mean-reverting asset |
| H15 | 2.00 | 2.00 | 45 | Mean-reverting asset |
| H20 | 2.50 | 2.50 | 60 | Mean-reverting asset |

### Label Optimization with Optuna

**Note:** The stage is named `ga_optimize` for historical reasons, but uses **Optuna TPE** internally (Tree-structured Parzen Estimator), not a genetic algorithm.

- **Objective:** Maximize Sharpe ratio with transaction cost penalties
- **Trials:** 100-150 per symbol/horizon combination
- **Symbol-specific constraints:** MES rewards k_up > k_down; MGC rewards k_up = k_down

### Quality Scores and Sample Weights

Each sample receives a quality score (0.5x to 1.5x) based on:

| Weight | Criteria |
|--------|----------|
| **1.5x** | Hit target quickly with minimal drawdown |
| **1.0x** | Standard signal quality |
| **0.5x** | Marginal signals, high uncertainty, near-timeout |

---

## Train/Val/Test Splits

### Split Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Train | 70% | Sufficient data for model training |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation (untouched until Phase 5) |

### Leakage Prevention

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Purge** | 60 bars | 3x max horizon (20*3=60) - prevents label leakage |
| **Embargo** | 1440 bars | 5 trading days (~5 days at 5-min) - serial correlation buffer |

```
[======== Training 70% ========][PURGE][== Val 15% ==][EMBARGO][== Test 15% ==]
                                  60                    1440
```

---

## Feature Selection Configuration

### Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Correlation threshold | 0.80 | Remove highly correlated features |
| Variance threshold | 0.01 | Remove near-constant features |

### Sample-to-Feature Ratio Guidelines

| Sample Size | Optimal Features | Maximum Features | Ratio |
|-------------|------------------|------------------|-------|
| 50K samples | 50-80 | 100 | 20:1 minimum |
| 100K samples | 80-150 | 200 | 20:1 preferred |
| 500K samples | 150-300 | 500 | 20:1 conservative |

**WARNING:** The pipeline produces 150+ features which may cause overfitting with smaller datasets (<100K samples). Use model-specific feature sets to mitigate.

---

## Model-Family Feature Sets

Phase 1 provides pre-defined feature sets optimized for different model families:

### `boosting_optimal` (XGBoost, LightGBM, CatBoost)

**Target:** 50-100 features for gradient boosting models.

| Include | Exclude | Scaler |
|---------|---------|--------|
| Returns, oscillators, trend, volatility, volume, temporal | Raw MA values | None (boosting handles raw) |

**Aliases:** `boosting`, `xgboost`, `lightgbm`, `catboost`

### `neural_optimal` (LSTM, GRU, MLP)

**Target:** Normalized features for sequential models.

| Include | Exclude | Scaler |
|---------|---------|--------|
| Returns, bounded oscillators, ratios, cyclical time | Raw prices, unbounded features | RobustScaler |

**Aliases:** `neural`, `lstm`, `gru`

### `transformer_raw` (Foundation Models)

**Target:** Minimal features - let the model learn patterns.

| Include | Exclude | Scaler |
|---------|---------|--------|
| Returns, volume ratio, cyclical time | Most indicators | StandardScaler |

**Aliases:** `transformer`, `foundation`

### `ensemble_base` (Stacking, Blending)

**Target:** Diverse features for ensemble meta-learners.

| Include | Exclude | Scaler |
|---------|---------|--------|
| Multi-category features for diversity | None specific | RobustScaler |

**Aliases:** `ensemble`, `stacking`, `blending`

---

## Output Structure

```
data/splits/
  scaled/
    train_scaled.parquet      # 70% - train your model
    val_scaled.parquet        # 15% - tune hyperparameters
    test_scaled.parquet       # 15% - final evaluation (touch once!)
    feature_scaler.pkl        # RobustScaler fitted on train
    feature_scaler.json       # Scaler parameters (JSON)
    scaling_metadata.json     # Scaling configuration
  datasets/
    core_full/                # Feature set outputs
      h5/train.parquet, val.parquet, test.parquet
      h10/train.parquet, val.parquet, test.parquet
      h15/train.parquet, val.parquet, test.parquet
      h20/train.parquet, val.parquet, test.parquet
  split_config.json           # Split configuration
```

Each parquet file contains:
- **150+ features** (technical indicators, wavelets, microstructure, MTF)
- **Labels** for 4 horizons (label_h5, label_h10, label_h15, label_h20)
- **Quality weights** (sample_weight_h5, ..., sample_weight_h20)
- **Metadata** (symbol, datetime)

---

## TimeSeriesDataContainer Interface

The container provides data in formats for all model frameworks:

```python
from src.phase1.stages.datasets import TimeSeriesDataContainer

# Load from Phase 1 outputs
container = TimeSeriesDataContainer.from_parquet_dir(
    path="data/splits/scaled",
    horizon=20  # Which horizon to use
)

# For XGBoost/LightGBM (tabular)
X_train, y_train, weights = container.get_sklearn_arrays("train")

# For LSTM/Transformer (sequences)
train_dataset = container.get_pytorch_sequences("train", seq_len=60)

# For NeuralForecast (N-HiTS, TFT)
nf_df = container.get_neuralforecast_df("train")

# Get container summary
print(container.describe())
```

---

## Quick Start

```bash
# Run the full pipeline
./pipeline run --symbols MES,MGC

# Check outputs
ls -lh data/splits/scaled/

# Use in Python
python -c "
from src.phase1.stages.datasets import TimeSeriesDataContainer
container = TimeSeriesDataContainer.from_parquet_dir('data/splits/scaled', horizon=20)
X, y, w = container.get_sklearn_arrays('train')
print(f'Ready to train: {X.shape[0]} samples, {X.shape[1]} features')
"
```

---

## Key Parameters Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Base Timeframe | 5-minute | Balance noise vs signal |
| Horizons | [5, 10, 15, 20] bars | Multiple prediction timeframes |
| Split Ratios | 70/15/15 | Standard ML split |
| Purge | 60 bars | Prevent label leakage |
| Embargo | 1440 bars | Serial correlation buffer |
| Correlation Threshold | 0.80 | Feature correlation limit |
| Variance Threshold | 0.01 | Minimum feature variance |

---

## Anti-Lookahead Protection

All features are computed with `shift(1)` to ensure:

- Feature at bar[t] uses only data available up to bar[t-1]
- No future information leaks into training
- Proper backtesting assumptions

Example:
```python
# RSI calculation with anti-lookahead
df['rsi_14'] = calculate_rsi_numba(df['close'].values, 14).shift(1)
#                                                          ^^^^^^^^
#                           This shift ensures RSI[t] uses close[t-1] and earlier
```

---

## Success Criteria (All Met)

| Criterion | Target | Status |
|-----------|--------|--------|
| Clean OHLCV data | No corruption | PASS |
| Feature count | 150+ | PASS |
| Label horizons | [5, 10, 15, 20] | PASS |
| Split ratios | 70/15/15 | PASS |
| Purge/embargo | 60/1440 bars | PASS |
| No lookahead | Validated | PASS |
| Real data only | No synthetic fallback | PASS |
| Symbol isolation | No cross-symbol leakage | PASS |

---

## Next Step

Phase 1 outputs feed directly into **Phase 2 (Model Factory)**, where any registered model can consume the standardized datasets via the `TimeSeriesDataContainer` interface.
