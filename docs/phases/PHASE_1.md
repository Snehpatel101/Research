# Phase 1: Data Preparation and Labeling

## Current Status: COMPLETE

Phase 1 is fully implemented and production-ready. The pipeline transforms raw OHLCV data into model-ready datasets with proper labeling, feature engineering, and train/val/test splits.

---

## Purpose

Phase 1 is the **data foundation** for the ML Model Factory. It ensures that:

1. All models receive **identical, standardized datasets** (fair comparison)
2. No **lookahead bias** exists in features or labels
3. **Symbol isolation** is maintained (no cross-symbol leakage)
4. **Train/val/test splits** have proper purge and embargo gaps
5. **Feature scaling** is fit only on training data

---

## Pipeline Architecture

### 12 Sequential Stages

```
Stage 1: Ingest          Raw OHLCV validation and loading
    |
Stage 2: Clean           Resample 1min -> 5min, gap handling
    |
Stage 3: Features        150+ technical indicators
    |
Stage 4: MTF             Multi-timeframe features (15m, 30m, 1h)
    |
Stage 5: Labeling        Initial triple-barrier labels
    |
Stage 6: GA Optimize     Optuna TPE barrier optimization
    |
Stage 7: Final Labels    Apply optimized parameters
    |
Stage 8: Splits          Train/val/test with purge/embargo
    |
Stage 9: Scaling         RobustScaler (train-only fit)
    |
Stage 10: Datasets       Build TimeSeriesDataContainer
    |
Stage 11: Validation     Feature correlation and quality checks
    |
Stage 12: Reporting      Generate completion reports
```

### Stage Details

| Stage | Input | Output | Key Operations |
|-------|-------|--------|----------------|
| **Ingest** | Raw parquet files | Validated DataFrame | Schema validation, timezone normalization |
| **Clean** | 1-min bars | 5-min bars | Resampling, gap filling, roll handling |
| **Features** | Clean OHLCV | 150+ columns | All indicator categories |
| **MTF** | Base features | +30 MTF features | 15m, 30m, 1h timeframes |
| **Labeling** | Features | Initial labels | Triple-barrier with default params |
| **GA Optimize** | Labels | Optimized params | Optuna TPE (100-150 trials) |
| **Final Labels** | Features + params | Final labels | Symbol-specific barriers |
| **Splits** | Labeled data | Train/val/test | 70/15/15 with purge/embargo |
| **Scaling** | Split data | Scaled data | RobustScaler fit on train |
| **Datasets** | Scaled data | Container | TimeSeriesDataContainer |
| **Validation** | Container | Report | Correlation, quality checks |
| **Reporting** | All artifacts | Reports | Summary documentation |

---

## Detailed Feature Documentation

### Feature Engineering Philosophy

Phase 1 generates features with these principles:

1. **Anti-lookahead:** All features use `shift(1)` to ensure feature[t] uses data[t-1]
2. **Safe division:** All ratios handle division by zero gracefully
3. **Stationarity:** Prefer normalized/ratio features over raw values
4. **Scale-invariance:** Features work across different price levels
5. **Symbol isolation:** No cross-symbol features (each symbol processed independently)

### Complete Feature Catalog

#### Category 1: Price Returns (~10 features)

| Feature Name | Formula | Anti-lookahead |
|--------------|---------|----------------|
| `return_1` | `(close[t-1] - close[t-2]) / close[t-2]` | shift(1) |
| `return_5` | `(close[t-1] - close[t-6]) / close[t-6]` | shift(1) |
| `return_10` | `(close[t-1] - close[t-11]) / close[t-11]` | shift(1) |
| `return_20` | `(close[t-1] - close[t-21]) / close[t-21]` | shift(1) |
| `return_60` | `(close[t-1] - close[t-61]) / close[t-61]` | shift(1) |
| `log_return_1` | `log(close[t-1] / close[t-2])` | shift(1) |
| `log_return_5` | `log(close[t-1] / close[t-6])` | shift(1) |
| `log_return_10` | `log(close[t-1] / close[t-11])` | shift(1) |
| `log_return_20` | `log(close[t-1] / close[t-21])` | shift(1) |
| `log_return_60` | `log(close[t-1] / close[t-61])` | shift(1) |

#### Category 2: Price Ratios (~6 features)

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `hl_ratio` | `high[t-1] / low[t-1]` | Intrabar range indicator |
| `co_ratio` | `close[t-1] / open[t-1]` | Bar direction indicator |
| `range_pct` | `(high - low)[t-1] / close[t-1]` | Normalized range |
| `clv` | `(2*close - high - low) / (high - low)` | Close Location Value [-1, +1] |
| `return_autocorr_lag1` | Rolling autocorr lag 1 | Trend persistence |
| `return_autocorr_lag5` | Rolling autocorr lag 5 | Medium-term persistence |
| `return_autocorr_lag10` | Rolling autocorr lag 10 | Longer-term persistence |

#### Category 3: Moving Averages (~15 features)

| Feature Name | Period | Type |
|--------------|--------|------|
| `sma_10` | 10 | Simple Moving Average |
| `sma_20` | 20 | Simple Moving Average |
| `sma_50` | 50 | Simple Moving Average |
| `sma_100` | 100 | Simple Moving Average |
| `sma_200` | 200 | Simple Moving Average |
| `ema_9` | 9 | Exponential Moving Average |
| `ema_21` | 21 | Exponential Moving Average |
| `ema_50` | 50 | Exponential Moving Average |
| `close_to_sma_10` | 10 | Price relative to SMA |
| `close_to_sma_20` | 20 | Price relative to SMA |
| `close_to_sma_50` | 50 | Price relative to SMA |
| `close_to_sma_100` | 100 | Price relative to SMA |
| `close_to_sma_200` | 200 | Price relative to SMA |
| `close_to_ema_9` | 9 | Price relative to EMA |
| `close_to_ema_21` | 21 | Price relative to EMA |
| `close_to_ema_50` | 50 | Price relative to EMA |

#### Category 4: Momentum Oscillators (~20 features)

| Feature Name | Parameters | Range | Description |
|--------------|------------|-------|-------------|
| `rsi_14` | 14-period | 0-100 | Relative Strength Index |
| `rsi_overbought` | >70 | 0/1 | Overbought flag |
| `rsi_oversold` | <30 | 0/1 | Oversold flag |
| `macd_line` | (12, 26) | unbounded | MACD line |
| `macd_signal` | 9-period | unbounded | Signal line |
| `macd_hist` | | unbounded | MACD histogram |
| `macd_cross_up` | | 0/1 | Bullish crossover |
| `macd_cross_down` | | 0/1 | Bearish crossover |
| `stoch_k` | (14, 3) | 0-100 | Stochastic %K |
| `stoch_d` | (14, 3) | 0-100 | Stochastic %D |
| `stoch_overbought` | >80 | 0/1 | Overbought flag |
| `stoch_oversold` | <20 | 0/1 | Oversold flag |
| `williams_r` | 14-period | -100-0 | Williams %R |
| `roc_5` | 5-period | % | Rate of Change 5 |
| `roc_10` | 10-period | % | Rate of Change 10 |
| `roc_20` | 20-period | % | Rate of Change 20 |
| `cci_20` | 20-period | unbounded | Commodity Channel Index |
| `mfi_14` | 14-period | 0-100 | Money Flow Index |

#### Category 5: Volatility Measures (~25 features)

| Feature Name | Parameters | Description |
|--------------|------------|-------------|
| `atr_7` | 7-period | Average True Range |
| `atr_14` | 14-period | Average True Range |
| `atr_21` | 21-period | Average True Range |
| `atr_pct_7` | 7-period | ATR as % of price |
| `atr_pct_14` | 14-period | ATR as % of price |
| `atr_pct_21` | 21-period | ATR as % of price |
| `bb_middle` | 20-period | Bollinger middle band |
| `bb_upper` | 20-period, 2 std | Upper band |
| `bb_lower` | 20-period, 2 std | Lower band |
| `bb_width` | | Band width normalized |
| `bb_position` | | Price position [0-1] |
| `close_bb_zscore` | | Z-score vs middle |
| `kc_middle` | 20-period | Keltner middle |
| `kc_upper` | 2x ATR | Upper channel |
| `kc_lower` | 2x ATR | Lower channel |
| `kc_position` | | Price position [0-1] |
| `close_kc_atr_dev` | | Deviation in ATR units |
| `hvol_10` | 10-period | Annualized volatility |
| `hvol_20` | 20-period | Annualized volatility |
| `hvol_60` | 60-period | Annualized volatility |
| `parkinson_vol` | 20-period | High-low volatility |
| `gk_vol` | 20-period | Garman-Klass volatility |
| `return_skew_20` | 20-period | Return skewness |
| `return_skew_60` | 60-period | Return skewness |
| `return_kurt_20` | 20-period | Return kurtosis |
| `return_kurt_60` | 60-period | Return kurtosis |

#### Category 6: Volume Analysis (~15 features)

| Feature Name | Parameters | Description |
|--------------|------------|-------------|
| `obv` | | On Balance Volume |
| `obv_sma_20` | 20-period | OBV moving average |
| `volume_sma_20` | 20-period | Volume moving average |
| `volume_ratio` | | Volume / SMA ratio |
| `volume_zscore` | 20-period | Volume z-score |
| `vwap` | Session | Volume Weighted Avg Price |
| `price_to_vwap` | | Price deviation from VWAP |
| `dollar_volume` | | Close x Volume |
| `dollar_volume_sma_10` | 10-period | Dollar vol average |
| `dollar_volume_sma_20` | 20-period | Dollar vol average |
| `dollar_volume_ratio` | | Dollar vol / SMA ratio |

#### Category 7: Trend Indicators (~8 features)

| Feature Name | Parameters | Description |
|--------------|------------|-------------|
| `adx_14` | 14-period | Average Directional Index |
| `plus_di_14` | 14-period | +DI (positive directional) |
| `minus_di_14` | 14-period | -DI (negative directional) |
| `adx_strong_trend` | >25 | Strong trend flag |
| `supertrend` | (10, 3.0) | Supertrend value |
| `supertrend_direction` | | Direction (1=up, -1=down) |

#### Category 8: Temporal Features (~9 features)

| Feature Name | Encoding | Description |
|--------------|----------|-------------|
| `hour_sin` | sin(2*pi*hour/24) | Hour cyclical (sine) |
| `hour_cos` | cos(2*pi*hour/24) | Hour cyclical (cosine) |
| `minute_sin` | sin(2*pi*min/60) | Minute cyclical (sine) |
| `minute_cos` | cos(2*pi*min/60) | Minute cyclical (cosine) |
| `dayofweek_sin` | sin(2*pi*dow/7) | Day cyclical (sine) |
| `dayofweek_cos` | cos(2*pi*dow/7) | Day cyclical (cosine) |
| `session_asia` | 0/1 | Asia session (00:00-08:00 UTC) |
| `session_london` | 0/1 | London session (08:00-16:00 UTC) |
| `session_ny` | 0/1 | NY session (16:00-24:00 UTC) |

#### Category 9: Regime Features (~2 features)

| Feature Name | Values | Description |
|--------------|--------|-------------|
| `volatility_regime` | 0/1 | High (1) vs low (0) volatility |
| `trend_regime` | -1/0/1 | Down (-1), sideways (0), up (1) |

#### Category 10: Microstructure Proxies (~20 features)

From OHLCV data, we estimate market microstructure characteristics:

| Feature Name | Academic Reference | Description |
|--------------|-------------------|-------------|
| `micro_amihud` | Amihud (2002) | Illiquidity ratio |ret|/vol |
| `micro_amihud_10` | | 10-period smoothed |
| `micro_amihud_20` | | 20-period smoothed |
| `micro_roll_spread` | Roll (1984) | Spread from serial covariance |
| `micro_roll_spread_pct` | | Spread as % of price |
| `micro_kyle_lambda` | Kyle (1985) | Price impact coefficient |
| `micro_cs_spread` | Corwin-Schultz (2012) | HL spread estimate |
| `micro_rel_spread` | | (H-L)/C simple spread |
| `micro_rel_spread_10` | | 10-period smoothed |
| `micro_rel_spread_20` | | 20-period smoothed |
| `micro_volume_imbalance` | | Order flow proxy (C-O)/(H-L) |
| `micro_cum_imbalance_20` | | 20-period cumulative |
| `micro_trade_intensity_20` | | Volume / 20-period avg |
| `micro_trade_intensity_50` | | Volume / 50-period avg |
| `micro_efficiency_10` | | Price efficiency |net|/sum(|changes|) |
| `micro_efficiency_20` | | 20-period efficiency |
| `micro_vol_ratio` | | Short/long volatility ratio |

#### Category 11: Wavelet Features (~24 features)

DWT decomposition using Daubechies-4 wavelets (level=3):

| Feature Name | Description |
|--------------|-------------|
| `wavelet_close_approx` | Low-frequency trend component (normalized) |
| `wavelet_close_d1` | Detail level 1 - highest frequency |
| `wavelet_close_d2` | Detail level 2 - medium frequency |
| `wavelet_close_d3` | Detail level 3 - lower frequency |
| `wavelet_volume_approx` | Volume trend component |
| `wavelet_volume_d1` | Volume detail level 1 |
| `wavelet_volume_d2` | Volume detail level 2 |
| `wavelet_volume_d3` | Volume detail level 3 |
| `wavelet_close_energy_approx` | log1p(sum of squared approx coeffs) |
| `wavelet_close_energy_d1` | Energy at detail level 1 |
| `wavelet_close_energy_d2` | Energy at detail level 2 |
| `wavelet_close_energy_d3` | Energy at detail level 3 |
| `wavelet_volume_energy_*` | Volume energy features (4 features) |
| `wavelet_close_energy_ratio` | Approx energy / total energy |
| `wavelet_close_volatility` | MAD-based robust volatility |
| `wavelet_close_trend_strength` | Normalized slope of approx |
| `wavelet_close_trend_direction` | Sign of trend (-1/0/+1) |

#### Category 12: Multi-Timeframe Features (~30 features)

Features resampled from higher timeframes (suffixes: `_15m`, `_30m`, `_1h`):

| Base Feature | Timeframes | Total Features |
|--------------|------------|----------------|
| `close` | 15m, 30m, 1h | 3 |
| `high` | 15m, 30m, 1h | 3 |
| `low` | 15m, 30m, 1h | 3 |
| `volume` | 15m, 30m, 1h | 3 |
| `rsi_14` | 15m, 30m, 1h | 3 |
| `macd_line` | 15m, 30m, 1h | 3 |
| `macd_hist` | 15m, 30m, 1h | 3 |
| `atr_14` | 15m, 30m, 1h | 3 |
| `bb_position` | 15m, 30m, 1h | 3 |
| `volume_ratio` | 15m, 30m, 1h | 3 |

---

## Labeling System Details

### Triple-Barrier Implementation

```
Price at t=0
    |
    |----[Upper Barrier: k_up * ATR]---- Hit = LONG (+1)
    |
    |
    |----[Lower Barrier: k_down * ATR]--- Hit = SHORT (-1)
    |
    |----[Vertical Barrier: max_bars]---- Timeout = NEUTRAL (0)
```

### Symbol-Specific Parameters

**Why different barriers for different symbols?**

- **MES (S&P 500):** Has long-term bullish drift (~7% annual). Without asymmetric barriers (k_up > k_down), the model would over-predict LONGs.

- **MGC (Gold):** More mean-reverting, less directional drift. Symmetric barriers work better.

### Barrier Parameters Table

| Symbol | Horizon | k_up | k_down | max_bars | Asymmetry |
|--------|---------|------|--------|----------|-----------|
| MES | H5 | 1.50 | 1.00 | 15 | 1.50x |
| MES | H10 | 2.00 | 1.40 | 30 | 1.43x |
| MES | H15 | 2.50 | 1.75 | 45 | 1.43x |
| MES | H20 | 3.00 | 2.10 | 60 | 1.43x |
| MGC | H5 | 1.20 | 1.20 | 15 | 1.00x |
| MGC | H10 | 1.60 | 1.60 | 30 | 1.00x |
| MGC | H15 | 2.00 | 2.00 | 45 | 1.00x |
| MGC | H20 | 2.50 | 2.50 | 60 | 1.00x |

### Optuna TPE Optimization

**Note:** Stage is named `ga_optimize` for historical reasons but uses Optuna TPE (Tree-structured Parzen Estimator), not genetic algorithm.

**Objective function:**
```
maximize: sharpe_ratio - penalty * transaction_costs
```

**Search space:**
| Parameter | Range | Type |
|-----------|-------|------|
| k_up | [0.5, 4.0] | float |
| k_down | [0.5, 4.0] | float |
| max_bars | [horizon, horizon*5] | int |

**Symbol-specific constraints:**
- MES: Reward k_up > k_down
- MGC: Penalize |k_up - k_down|

### Quality Scoring

Each label receives a quality score based on:

| Factor | Weight Range | Criteria |
|--------|--------------|----------|
| Time to hit | 0.5-1.5 | Faster hits = higher quality |
| Max Drawdown | 0.5-1.5 | Lower drawdown = higher quality |
| Touch type | 0.5-1.5 | Clean hits = higher quality |

**Sample Weight Formula:**
```python
sample_weight = base_weight * time_factor * drawdown_factor * touch_factor
# Bounded to [0.5, 1.5]
```

---

## Train/Val/Test Split Details

### Split Ratios

```
|<-------- Training (70%) -------->|<-- Val (15%) -->|<-- Test (15%) -->|
|----------------------------------|-----------------|------------------|
^                                  ^                 ^
|                                  |                 |
Start                          PURGE gap         EMBARGO gap
                               (60 bars)         (1440 bars)
```

### Purge and Embargo Explained

**Purge (60 bars):**
- Placed between training and validation
- Prevents label leakage (training on data whose labels depend on val prices)
- Size: 3 x max_horizon = 3 x 20 = 60 bars

**Embargo (1440 bars):**
- Placed between validation and test
- Breaks serial correlation in predictions
- Size: 5 trading days at 5-min = 288 bars/day x 5 = 1440 bars

### Why These Values?

| Parameter | Value | Calculation |
|-----------|-------|-------------|
| Purge | 60 bars | 3 x max_horizon (20) = 60 |
| Embargo | 1440 bars | 5 trading days x 288 bars/day |

---

## Model-Family Feature Sets

Phase 1 defines feature sets optimized for different model types:

### `boosting_optimal`

**For:** XGBoost, LightGBM, CatBoost

**Philosophy:** Tree-based models handle correlated features and different scales internally. Include all useful features, no scaling needed.

**Included prefixes:**
```python
include_prefixes = [
    "return_", "log_return_", "roc_",           # Price action
    "rsi_", "macd_", "stoch_", "williams_",     # Momentum
    "cci_", "mfi_",
    "adx_", "supertrend",                       # Trend
    "atr_", "hvol_", "parkinson_", "garman_",   # Volatility
    "bb_width", "bb_position", "kc_position",
    "volume_", "obv",                           # Volume
    "hour_", "dayofweek_", "session_",          # Temporal
]
```

**Recommended scaler:** None (boosting handles raw values)

### `neural_optimal`

**For:** LSTM, GRU, MLP

**Philosophy:** Neural networks prefer normalized, bounded features. Exclude raw prices and unbounded features.

**Included prefixes:**
```python
include_prefixes = [
    "return_", "log_return_",                    # Returns (already normalized)
    "rsi_", "stoch_", "williams_", "cci_", "mfi_",  # Bounded oscillators
    "hvol_", "atr_ratio", "bb_position", "kc_position",  # Normalized volatility
    "volume_ratio", "volume_zscore",             # Volume ratios
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",  # Cyclical
]
```

**Excluded prefixes:**
```python
exclude_prefixes = [
    "sma_", "ema_", "bb_upper", "bb_lower", "vwap",  # Raw prices
    "open_", "high_", "low_", "close_",
]
```

**Recommended scaler:** RobustScaler

### `transformer_raw`

**For:** Foundation models, transformers

**Philosophy:** Minimal features - let the model learn patterns from raw data.

**Included:**
```python
include_prefixes = ["return_", "log_return_", "volume_ratio"]
include_columns = ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "is_rth"]
```

**Recommended scaler:** StandardScaler
**Sequence length:** 128 (longer for transformers)

### `ensemble_base`

**For:** Stacking, blending, ensemble meta-learners

**Philosophy:** Diverse features from different categories to ensure base models have uncorrelated inputs.

**Feature groups (for diversity):**
- Group 1: Price momentum (return_, roc_)
- Group 2: Mean reversion (rsi_, bb_position, kc_position)
- Group 3: Trend following (adx_, macd_, supertrend)
- Group 4: Volatility (atr_, hvol_)
- Group 5: Volume (volume_, obv)
- Group 6: Temporal (hour_, dayofweek_)

**Includes MTF:** Yes (adds diversity)
**Recommended scaler:** RobustScaler

---

## Sample-to-Feature Ratio Guidelines

### Critical for Preventing Overfitting

| Sample Size | Min Ratio | Optimal | Max Features |
|-------------|-----------|---------|--------------|
| 50,000 | 10:1 | 20:1 | 100 |
| 100,000 | 10:1 | 20:1 | 200 |
| 200,000 | 10:1 | 20:1 | 400 |
| 500,000 | 10:1 | 20:1 | 500 |

### Mitigation Strategies

1. **Use model-specific feature sets** (boosting_optimal, neural_optimal)
2. **Apply correlation pruning** (CORRELATION_THRESHOLD = 0.80)
3. **Use regularization** (L1/L2 in models, dropout in neural nets)
4. **Collect more data** when possible

---

## Feature Selection Process

### Step 1: Remove Low Variance Features

```python
variance_threshold = 0.01
# Remove features with normalized variance < threshold
```

### Step 2: Remove Highly Correlated Features

```python
correlation_threshold = 0.80
# For each correlated pair, keep the higher-priority feature
```

### Feature Priority Ranking

When removing correlated features, we keep the more interpretable one:

| Priority | Feature Examples |
|----------|------------------|
| 100 | log_return (most fundamental) |
| 90-95 | simple_return, high_low_range |
| 85-90 | rsi, macd_hist, bb_position |
| 75-85 | stoch_k, williams_r, atr_pct |
| 60-70 | raw values (sma_, ema_, vwap) |

---

## Output Structure

### Directory Layout

```
data/splits/
|
+-- scaled/
|   +-- train_scaled.parquet      # Training data (70%)
|   +-- val_scaled.parquet        # Validation data (15%)
|   +-- test_scaled.parquet       # Test data (15%)
|   +-- feature_scaler.pkl        # RobustScaler object
|   +-- feature_scaler.json       # Scaler params (JSON)
|   +-- scaling_metadata.json     # Scaling config
|
+-- datasets/
|   +-- core_full/                # Feature set outputs
|       +-- h5/
|       |   +-- train.parquet
|       |   +-- val.parquet
|       |   +-- test.parquet
|       +-- h10/
|       |   +-- train.parquet
|       |   +-- val.parquet
|       |   +-- test.parquet
|       +-- h15/
|       |   +-- (same structure)
|       +-- h20/
|           +-- (same structure)
|
+-- split_config.json             # Split configuration
```

### Column Schema

Each parquet file contains:

| Column Type | Examples |
|-------------|----------|
| Metadata | datetime, symbol |
| Features (150+) | return_1, rsi_14, atr_pct_14, ... |
| Labels (4) | label_h5, label_h10, label_h15, label_h20 |
| Weights (4) | sample_weight_h5, ..., sample_weight_h20 |
| Quality (4) | quality_h5, ..., quality_h20 |

---

## TimeSeriesDataContainer Interface

### Loading Data

```python
from src.phase1.stages.datasets import TimeSeriesDataContainer

container = TimeSeriesDataContainer.from_parquet_dir(
    path="data/splits/scaled",
    horizon=20  # Which label horizon to use
)

print(container.describe())
# {'horizon': 20, 'n_features': 157, 'splits': {'train': {...}, 'val': {...}, 'test': {...}}}
```

### For Tabular Models (XGBoost, LightGBM)

```python
X_train, y_train, weights = container.get_sklearn_arrays("train")
X_val, y_val, _ = container.get_sklearn_arrays("val")

# X_train: numpy array (n_samples, n_features)
# y_train: numpy array (n_samples,)
# weights: numpy array (n_samples,)
```

### For Sequential Models (LSTM, GRU)

```python
train_dataset = container.get_pytorch_sequences(
    "train",
    seq_len=60,       # Sequence length
    stride=1,         # Step between sequences
    symbol_isolated=True  # Don't cross symbol boundaries
)

# Returns PyTorch Dataset yielding (X, y, weight) tuples
# X shape: (seq_len, n_features)
```

### For NeuralForecast (N-HiTS, TFT)

```python
nf_df = container.get_neuralforecast_df("train")

# Returns DataFrame with columns:
# unique_id: symbol
# ds: datetime
# y: target (label)
# sample_weight: quality weight
# [all features...]
```

---

## Validation and Quality Checks

### Stage 11: Validation

The validation stage checks:

1. **Feature correlation matrix** - Ensure correlation threshold respected
2. **Feature variance** - No near-constant features
3. **Label distribution** - Balanced classes per horizon
4. **Sample weights** - Within expected range [0.5, 1.5]
5. **No NaN values** - Clean data

### Success Criteria

| Check | Requirement | Status |
|-------|-------------|--------|
| No lookahead | shift(1) on all features | PASS |
| No NaN features | All numeric, no missing | PASS |
| Label balance | Each class > 10% | PASS |
| Feature count | 150+ | PASS |
| Correlation | < 0.80 threshold | PASS |
| Variance | > 0.01 threshold | PASS |
| Symbol isolation | No cross-symbol features | PASS |

---

## Usage Examples

### Run Complete Pipeline

```bash
# Full pipeline with default symbols
./pipeline run --symbols MES,MGC

# Check status
./pipeline status

# Validate outputs
./pipeline validate
```

### Use in Python

```python
from src.phase1.stages.datasets import TimeSeriesDataContainer

# Load container
container = TimeSeriesDataContainer.from_parquet_dir(
    path="data/splits/scaled",
    horizon=20
)

# Get data for XGBoost training
X_train, y_train, weights = container.get_sklearn_arrays("train")
X_val, y_val, _ = container.get_sklearn_arrays("val")

print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Label distribution: {pd.Series(y_train).value_counts().to_dict()}")
```

---

## Expected Performance Baselines

Based on the labeling configuration and feature engineering:

| Horizon | Expected Sharpe | Expected Win Rate | Expected Max DD |
|---------|-----------------|-------------------|-----------------|
| H5 | 0.3 - 0.8 | 45% - 50% | 10% - 25% |
| H10 | 0.4 - 0.9 | 46% - 52% | 9% - 20% |
| H15 | 0.4 - 1.0 | 47% - 53% | 8% - 18% |
| H20 | 0.5 - 1.2 | 48% - 55% | 8% - 18% |

**Note:** These are baselines before model optimization. Actual performance depends on model choice and hyperparameters.

---

## Next Steps (Phase 2)

Phase 1 outputs feed into Phase 2 (Model Factory):

1. **Model Registry** - Register model types (XGBoost, LSTM, etc.)
2. **Training Pipeline** - Train models using TimeSeriesDataContainer
3. **Evaluation Framework** - Compare models with identical metrics
4. **Ensemble Support** - Stack/blend multiple models

```python
# Phase 2 will use:
from src.models import ModelRegistry, train_model

@ModelRegistry.register("xgboost")
class XGBoostModel(BaseModel):
    def train(self, container, config):
        X, y, w = container.get_sklearn_arrays("train")
        # ... training logic
```

---

## Changelog

### 2025-12-24 (Current)
- Expanded feature documentation with complete catalog
- Added model-family feature sets documentation
- Added sample-to-feature ratio guidelines
- Added microstructure and wavelet feature details
- Improved barrier parameter documentation

### Previous Updates
- Removed synthetic data generation
- Added wavelet decomposition features
- Added microstructure proxy features
- Increased embargo to 1440 bars
- Added DataIngestor validation
