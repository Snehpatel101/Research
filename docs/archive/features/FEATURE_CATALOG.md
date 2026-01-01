# Feature Catalog - Complete Reference

**ML Model Factory Feature Engineering Documentation**

This catalog documents all 150+ features engineered for OHLCV time series modeling, including exact formulas, parameters, and implementation details.

---

## Table of Contents

- [Feature Statistics](#feature-statistics)
- [Price Features (20 features)](#price-features)
- [Moving Average Features (25 features)](#moving-average-features)
- [Momentum Indicators (23 features)](#momentum-indicators)
- [Volatility Indicators (34 features)](#volatility-indicators)
- [Volume Features (13 features)](#volume-features)
- [Trend Indicators (6 features)](#trend-indicators)
- [Temporal Features (9 features)](#temporal-features)
- [Regime Features (2 features)](#regime-features)
- [Microstructure Features (19 features)](#microstructure-features)
- [Wavelet Features (24 features)](#wavelet-features)
- [Multi-Timeframe Features (Variable)](#multi-timeframe-features)

---

## Feature Statistics

| Category | Count | Default Parameters |
|----------|-------|-------------------|
| Price Features | 20 | periods=[1,5,10,20,60], lags=[1,2,5,10,20] |
| Moving Averages | 25 | SMA=[10,20,50,100,200], EMA=[9,12,21,26,50] |
| Momentum | 23 | RSI=14, MACD=(12,26,9), Stochastic=(14,3) |
| Volatility | 34 | ATR=[7,14,21], Bollinger=20, Hvol=[10,20,60] |
| Volume | 13 | period=20, periods=[10,20] |
| Trend | 6 | ADX=14, Supertrend=(10,3.0) |
| Temporal | 9 | hour/minute/dayofweek sin/cos encoding |
| Regime | 2 | hvol_lookback=100, SMA=(50,200) |
| Microstructure | 19 | Amihud=[10,20], Roll=20, Kyle=20 |
| Wavelets | 24 | wavelet=db4, level=3, window=64 |
| **Total Base** | **~175** | All features lagged by 1 bar (no lookahead) |

**Note:** MTF features add ~40-60 additional features per MTF timeframe enabled (default: 15min, 60min).

---

## Price Features

**Category:** Basic price transformations and statistical properties

**Count:** 20 features (10 returns + 3 ratios + 5 autocorrelations + 2 additional)

### Returns (10 features)

Simple and log returns over multiple periods.

**Implementation:** `src/phase1/stages/features/price_features.py::add_returns()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `return_1` | `(close[t-1] - close[t-2]) / close[t-2]` | period=1 | 1 bar |
| `return_5` | `(close[t-1] - close[t-6]) / close[t-6]` | period=5 | 1 bar |
| `return_10` | `(close[t-1] - close[t-11]) / close[t-11]` | period=10 | 1 bar |
| `return_20` | `(close[t-1] - close[t-21]) / close[t-21]` | period=20 | 1 bar |
| `return_60` | `(close[t-1] - close[t-61]) / close[t-61]` | period=60 | 1 bar |
| `log_return_1` | `log(close[t-1] / close[t-2])` | period=1 | 1 bar |
| `log_return_5` | `log(close[t-1] / close[t-6])` | period=5 | 1 bar |
| `log_return_10` | `log(close[t-1] / close[t-11])` | period=10 | 1 bar |
| `log_return_20` | `log(close[t-1] / close[t-21])` | period=20 | 1 bar |
| `log_return_60` | `log(close[t-1] / close[t-61])` | period=60 | 1 bar |

**Stationarity:** Both simple and log returns are stationary (differenced prices).

### Price Ratios (3 features)

Intrabar price relationships.

**Implementation:** `src/phase1/stages/features/price_features.py::add_price_ratios()`

| Feature | Formula | Range | Lag |
|---------|---------|-------|-----|
| `hl_ratio` | `high[t-1] / low[t-1]` | [1.0, ∞) | 1 bar |
| `co_ratio` | `close[t-1] / open[t-1]` | (0, ∞) | 1 bar |
| `range_pct` | `(high[t-1] - low[t-1]) / close[t-1]` | [0, ∞) | 1 bar |

**Use Case:** Captures intrabar volatility and price action patterns.

### Autocorrelation (5 features)

Rolling autocorrelation of returns at different lags.

**Implementation:** `src/phase1/stages/features/price_features.py::add_autocorrelation()`

| Feature | Formula | Parameters | Interpretation |
|---------|---------|------------|----------------|
| `return_autocorr_lag1` | `rolling_autocorr(returns, lag=1, window=20)` | window=20, lag=1 | Short-term momentum/mean-reversion |
| `return_autocorr_lag2` | `rolling_autocorr(returns, lag=2, window=20)` | window=20, lag=2 | Intraday patterns |
| `return_autocorr_lag5` | `rolling_autocorr(returns, lag=5, window=20)` | window=20, lag=5 | 25-min patterns (5min bars) |
| `return_autocorr_lag10` | `rolling_autocorr(returns, lag=10, window=20)` | window=20, lag=10 | Hour-scale patterns |
| `return_autocorr_lag20` | `rolling_autocorr(returns, lag=20, window=20)` | window=20, lag=20 | Multi-hour patterns |

**Stationarity:** Autocorrelation coefficients are bounded in [-1, 1] and stationary.

### Additional Price Features (2 features)

**Implementation:** `src/phase1/stages/features/price_features.py::add_clv()`

| Feature | Formula | Range | Lag |
|---------|---------|-------|-----|
| `clv` | `(2*close - high - low) / (high - low)` | [-1, 1] | 1 bar |

**clv (Close Location Value):**
- `clv = +1`: close at high (bullish)
- `clv = -1`: close at low (bearish)
- `clv = 0`: close at midpoint

---

## Moving Average Features

**Category:** Trend-following indicators

**Count:** 25 features (10 SMA + 10 EMA + 5 crossovers)

### Simple Moving Averages (10 features)

**Implementation:** `src/phase1/stages/features/moving_averages.py::add_sma()`

**Default periods:** [10, 20, 50, 100, 200] (for 5min base timeframe)

| Feature | Formula | Period (5min) | Lookback (min) | Lag |
|---------|---------|---------------|----------------|-----|
| `sma_10` | `mean(close[t-11:t-1])` | 10 bars | 50 min | 1 bar |
| `sma_20` | `mean(close[t-21:t-1])` | 20 bars | 100 min | 1 bar |
| `sma_50` | `mean(close[t-51:t-1])` | 50 bars | 250 min (~4h) | 1 bar |
| `sma_100` | `mean(close[t-101:t-1])` | 100 bars | 500 min (~8h) | 1 bar |
| `sma_200` | `mean(close[t-201:t-1])` | 200 bars | 1000 min (~17h) | 1 bar |
| `price_to_sma_10` | `(close[t-1] / sma_10[t-1]) - 1` | - | - | 1 bar |
| `price_to_sma_20` | `(close[t-1] / sma_20[t-1]) - 1` | - | - | 1 bar |
| `price_to_sma_50` | `(close[t-1] / sma_50[t-1]) - 1` | - | - | 1 bar |
| `price_to_sma_100` | `(close[t-1] / sma_100[t-1]) - 1` | - | - | 1 bar |
| `price_to_sma_200` | `(close[t-1] / sma_200[t-1]) - 1` | - | - | 1 bar |

**Period Scaling:** Periods automatically scale with timeframe to maintain lookback duration.
- Example: SMA-20 on 15min bars → SMA-7 (both ≈100 min lookback)

### Exponential Moving Averages (10 features)

**Implementation:** `src/phase1/stages/features/moving_averages.py::add_ema()`

**Default periods:** [9, 12, 21, 26, 50] (for 5min base timeframe)

| Feature | Formula | Period (5min) | Smoothing Factor α | Lag |
|---------|---------|---------------|-------------------|-----|
| `ema_9` | `α*close + (1-α)*ema_prev` | 9 bars | 0.2 | 1 bar |
| `ema_12` | `α*close + (1-α)*ema_prev` | 12 bars | 0.1538 | 1 bar |
| `ema_21` | `α*close + (1-α)*ema_prev` | 21 bars | 0.0909 | 1 bar |
| `ema_26` | `α*close + (1-α)*ema_prev` | 26 bars | 0.0741 | 1 bar |
| `ema_50` | `α*close + (1-α)*ema_prev` | 50 bars | 0.0392 | 1 bar |
| `price_to_ema_9` | `(close[t-1] / ema_9[t-1]) - 1` | - | - | 1 bar |
| `price_to_ema_12` | `(close[t-1] / ema_12[t-1]) - 1` | - | - | 1 bar |
| `price_to_ema_21` | `(close[t-1] / ema_21[t-1]) - 1` | - | - | 1 bar |
| `price_to_ema_26` | `(close[t-1] / ema_26[t-1]) - 1` | - | - | 1 bar |
| `price_to_ema_50` | `(close[t-1] / ema_50[t-1]) - 1` | - | - | 1 bar |

**Smoothing factor:** `α = 2 / (period + 1)`

**MA Crossovers (5 features)**

Generated via MACD (see Momentum section).

---

## Momentum Indicators

**Category:** Oscillators measuring price momentum and overbought/oversold conditions

**Count:** 23 features

### RSI (Relative Strength Index) - 3 features

**Implementation:** `src/phase1/stages/features/momentum.py::add_rsi()`

| Feature | Formula | Parameters | Range | Lag |
|---------|---------|------------|-------|-----|
| `rsi_14` | `100 - (100 / (1 + RS))` where `RS = avg_gain / avg_loss` | period=14 | [0, 100] | 1 bar |
| `rsi_overbought` | `rsi_14 > 70` | threshold=70 | {0, 1} | 1 bar |
| `rsi_oversold` | `rsi_14 < 30` | threshold=30 | {0, 1} | 1 bar |

**Calculation:** Uses EMA smoothing of gains/losses over 14 periods.

**Interpretation:**
- RSI > 70: Overbought
- RSI < 30: Oversold
- RSI = 50: Neutral

### MACD (Moving Average Convergence Divergence) - 5 features

**Implementation:** `src/phase1/stages/features/momentum.py::add_macd()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `macd_line` | `EMA_12 - EMA_26` | fast=12, slow=26 | 1 bar |
| `macd_signal` | `EMA_9(macd_line)` | signal=9 | 1 bar |
| `macd_hist` | `macd_line - macd_signal` | - | 1 bar |
| `macd_cross_up` | `(macd_line > macd_signal) & (macd_line[t-1] <= macd_signal[t-1])` | - | 1 bar |
| `macd_cross_down` | `(macd_line < macd_signal) & (macd_line[t-1] >= macd_signal[t-1])` | - | 1 bar |

**Trading Signals:**
- Crossover up: Bullish signal
- Crossover down: Bearish signal
- Histogram > 0: Momentum increasing
- Histogram < 0: Momentum decreasing

### Stochastic Oscillator - 4 features

**Implementation:** `src/phase1/stages/features/momentum.py::add_stochastic()`

| Feature | Formula | Parameters | Range | Lag |
|---------|---------|------------|-------|-----|
| `stoch_k` | `100 * (close - low_14) / (high_14 - low_14)` | k_period=14 | [0, 100] | 1 bar |
| `stoch_d` | `SMA_3(stoch_k)` | d_period=3 | [0, 100] | 1 bar |
| `stoch_overbought` | `stoch_k > 80` | threshold=80 | {0, 1} | 1 bar |
| `stoch_oversold` | `stoch_k < 20` | threshold=20 | {0, 1} | 1 bar |

**Signals:**
- %K > 80: Overbought
- %K < 20: Oversold
- %K crosses %D: Momentum shift

### Williams %R - 1 feature

**Implementation:** `src/phase1/stages/features/momentum.py::add_williams_r()`

| Feature | Formula | Parameters | Range | Lag |
|---------|---------|------------|-------|-----|
| `williams_r` | `-100 * (high_14 - close) / (high_14 - low_14)` | period=14 | [-100, 0] | 1 bar |

**Interpretation:**
- Williams %R > -20: Overbought
- Williams %R < -80: Oversold

### Rate of Change (ROC) - 3 features

**Implementation:** `src/phase1/stages/features/momentum.py::add_roc()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `roc_5` | `100 * (close[t-1] - close[t-6]) / close[t-6]` | period=5 | 1 bar |
| `roc_10` | `100 * (close[t-1] - close[t-11]) / close[t-11]` | period=10 | 1 bar |
| `roc_20` | `100 * (close[t-1] - close[t-21]) / close[t-21]` | period=20 | 1 bar |

### Commodity Channel Index (CCI) - 1 feature

**Implementation:** `src/phase1/stages/features/momentum.py::add_cci()`

| Feature | Formula | Parameters | Range | Lag |
|---------|---------|------------|-------|-----|
| `cci_20` | `(TP - SMA(TP)) / (0.015 * MAD)` where `TP = (H+L+C)/3` | period=20 | unbounded | 1 bar |

**Typical values:** CCI typically ranges from -100 to +100, but can exceed these bounds during strong trends.

### Money Flow Index (MFI) - 1 feature

**Implementation:** `src/phase1/stages/features/momentum.py::add_mfi()`

| Feature | Formula | Parameters | Range | Lag |
|---------|---------|------------|-------|-----|
| `mfi_14` | `100 - (100 / (1 + MFR))` where `MFR = pos_flow / neg_flow` | period=14 | [0, 100] | 1 bar |

**Calculation:** Money flow = typical_price * volume, then sum positive/negative flows.

**Note:** Only computed if volume data is available.

---

## Volatility Indicators

**Category:** Measures of price dispersion and uncertainty

**Count:** 34 features

### Average True Range (ATR) - 6 features

**Implementation:** `src/phase1/stages/features/volatility.py::add_atr()`

**Default periods:** [7, 14, 21]

| Feature | Formula | Period | Lag |
|---------|---------|--------|-----|
| `atr_7` | `EMA_7(true_range)` where `TR = max(H-L, |H-C_prev|, |L-C_prev|)` | 7 bars | 1 bar |
| `atr_14` | `EMA_14(true_range)` | 14 bars | 1 bar |
| `atr_21` | `EMA_21(true_range)` | 21 bars | 1 bar |
| `atr_pct_7` | `atr_7 / close * 100` | 7 bars | 1 bar |
| `atr_pct_14` | `atr_14 / close * 100` | 14 bars | 1 bar |
| `atr_pct_21` | `atr_21 / close * 100` | 21 bars | 1 bar |

**Use:** Position sizing, stop-loss placement, volatility normalization.

### Bollinger Bands - 6 features

**Implementation:** `src/phase1/stages/features/volatility.py::add_bollinger_bands()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `bb_middle` | `SMA_20(close)` | period=20 | 1 bar |
| `bb_upper` | `bb_middle + 2*std_20` | period=20, std_mult=2.0 | 1 bar |
| `bb_lower` | `bb_middle - 2*std_20` | period=20, std_mult=2.0 | 1 bar |
| `bb_width` | `(bb_upper - bb_lower) / std_20` | normalized | 1 bar |
| `bb_position` | `(close - bb_lower) / (bb_upper - bb_lower)` | range=[0, 1] | 1 bar |
| `close_bb_zscore` | `(close - bb_middle) / std_20` | z-score | 1 bar |

**Stationarity:** All features except raw bands are stationary (normalized or z-scored).

### Keltner Channels - 5 features

**Implementation:** `src/phase1/stages/features/volatility.py::add_keltner_channels()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `kc_middle` | `EMA_20(close)` | period=20 | 1 bar |
| `kc_upper` | `kc_middle + 2*ATR_20` | period=20, atr_mult=2.0 | 1 bar |
| `kc_lower` | `kc_middle - 2*ATR_20` | period=20, atr_mult=2.0 | 1 bar |
| `kc_position` | `(close - kc_lower) / (kc_upper - kc_lower)` | range=[0, 1] | 1 bar |
| `close_kc_atr_dev` | `(close - kc_middle) / ATR_20` | ATR units | 1 bar |

**Difference from Bollinger:** Uses ATR instead of standard deviation for channel width.

### Historical Volatility - 3 features

**Implementation:** `src/phase1/stages/features/volatility.py::add_historical_volatility()`

**Default periods:** [10, 20, 60]

| Feature | Formula | Period | Annualization | Lag |
|---------|---------|--------|---------------|-----|
| `hvol_10` | `std(log_returns_10) * sqrt(252*78)` | 10 bars | 140.07 | 1 bar |
| `hvol_20` | `std(log_returns_20) * sqrt(252*78)` | 20 bars | 140.07 | 1 bar |
| `hvol_60` | `std(log_returns_60) * sqrt(252*78)` | 60 bars | 140.07 | 1 bar |

**Annualization factor:** `sqrt(bars_per_day * trading_days_per_year)` = `sqrt(78 * 252)` = 140.07 for 5min bars.

**Note:** Annualization factors are timeframe-aware (see `constants.py::get_annualization_factor()`).

### Advanced Volatility Estimators - 8 features

High-efficiency volatility estimators using OHLC data.

**Parkinson Volatility (1 feature):**

**Implementation:** `src/phase1/stages/features/volatility.py::add_parkinson_volatility()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `parkinson_vol` | `sqrt((1/(4*ln(2))) * mean((ln(H/L))^2)) * ann_factor` | period=20 | 1 bar |

**Efficiency:** More efficient than close-to-close for normally distributed returns.

**Garman-Klass Volatility (1 feature):**

**Implementation:** `src/phase1/stages/features/volatility.py::add_garman_klass_volatility()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `gk_vol` | `sqrt(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2) * ann_factor` | period=20 | 1 bar |

**Efficiency:** More efficient than Parkinson, uses OHLC.

**Rogers-Satchell Volatility (1 feature):**

**Implementation:** `src/phase1/stages/features/volatility.py::add_rogers_satchell_volatility()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `rs_vol` | `sqrt(mean((ln(H/C))(ln(H/O)) + (ln(L/C))(ln(L/O)))) * ann_factor` | period=20 | 1 bar |

**Use:** Handles non-zero drift (trending markets) better than GK.

**Yang-Zhang Volatility (1 feature):**

**Implementation:** `src/phase1/stages/features/volatility.py::add_yang_zhang_volatility()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `yz_vol` | Combined overnight + open-close + RS components | period=20 | 1 bar |

**Formula:**
```
YZ = sqrt(vol_overnight^2 + k*vol_open_close^2 + (1-k)*vol_RS^2)
k = 0.34 / (1.34 + (n+1)/(n-1))
```

**Use:** Handles both drift and opening gaps, considered most efficient OHLC estimator.

**Higher Moments (4 features):**

**Implementation:** `src/phase1/stages/features/volatility.py::add_higher_moments()`

**Default periods:** [20, 60]

| Feature | Formula | Period | Interpretation | Lag |
|---------|---------|--------|----------------|-----|
| `return_skew_20` | `rolling_skew(returns, 20)` | 20 bars | Asymmetry (crash risk if negative) | 1 bar |
| `return_skew_60` | `rolling_skew(returns, 60)` | 60 bars | Asymmetry (longer window) | 1 bar |
| `return_kurt_20` | `rolling_kurtosis(returns, 20)` | 20 bars | Tail fatness (extreme moves) | 1 bar |
| `return_kurt_60` | `rolling_kurtosis(returns, 60)` | 60 bars | Tail fatness (longer window) | 1 bar |

**Interpretation:**
- **Skewness:**
  - Negative: Fat left tail (crash risk)
  - Positive: Fat right tail (rally risk)
  - Zero: Symmetric distribution
- **Kurtosis (excess):**
  - High: Fat tails, more extreme moves
  - Low: Thin tails, fewer extreme moves
  - Zero: Normal distribution

---

## Volume Features

**Category:** Trading activity and liquidity indicators

**Count:** 13 features

**Note:** All volume features are skipped if volume data is unavailable or zero.

### On-Balance Volume (OBV) - 2 features

**Implementation:** `src/phase1/stages/features/volume.py::add_volume_features()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `obv` | `cumsum(sign(close_diff) * volume)` | - | 1 bar |
| `obv_sma_20` | `SMA_20(obv)` | period=20 | 1 bar |

**Interpretation:** Rising OBV confirms uptrend; falling OBV confirms downtrend.

### Volume Analysis - 3 features

**Implementation:** `src/phase1/stages/features/volume.py::add_volume_features()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `volume_sma_20` | `SMA_20(volume)` | period=20 | 1 bar |
| `volume_ratio` | `volume / volume_sma_20` | - | 1 bar |
| `volume_zscore` | `(volume - mean_20) / std_20` | period=20 | 1 bar |

**Use:** Detect volume spikes (breakout confirmation) and volume drying up (trend exhaustion).

### VWAP (Volume Weighted Average Price) - 2 features

**Implementation:** `src/phase1/stages/features/volume.py::add_vwap()`

| Feature | Formula | Session | Lag |
|---------|---------|---------|-----|
| `vwap` | `cumsum(typical_price * volume) / cumsum(volume)` | Daily reset | 1 bar |
| `price_to_vwap` | `(close - vwap) / vwap` | - | 1 bar |

**Session reset:** VWAP resets at start of each trading day.

**Use:** Institutional benchmark, mean-reversion signal when price deviates significantly.

### Dollar Volume - 3 features

**Implementation:** `src/phase1/stages/features/volume.py::add_dollar_volume()`

**Default periods:** [10, 20]

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `dollar_volume` | `close * volume` | - | 1 bar |
| `dollar_volume_sma_10` | `SMA_10(dollar_volume)` | period=10 | 1 bar |
| `dollar_volume_sma_20` | `SMA_20(dollar_volume)` | period=20 | 1 bar |
| `dollar_volume_ratio` | `dollar_volume / dollar_volume_sma_20` | - | 1 bar |

**Use:** Better liquidity proxy than raw volume (accounts for price differences).

---

## Trend Indicators

**Category:** Directional movement and trend strength

**Count:** 6 features

### ADX (Average Directional Index) - 4 features

**Implementation:** `src/phase1/stages/features/trend.py::add_adx()`

| Feature | Formula | Period | Range | Lag |
|---------|---------|--------|-------|-----|
| `adx_14` | Smoothed difference between +DI and -DI | 14 | [0, 100] | 1 bar |
| `plus_di_14` | `100 * EMA(+DM) / ATR` | 14 | [0, 100] | 1 bar |
| `minus_di_14` | `100 * EMA(-DM) / ATR` | 14 | [0, 100] | 1 bar |
| `adx_strong_trend` | `adx_14 > 25` | threshold=25 | {0, 1} | 1 bar |

**Calculation:**
- `+DM` (Positive Directional Movement): `max(high - high_prev, 0)` if high-high_prev > low_prev-low
- `-DM` (Negative Directional Movement): `max(low_prev - low, 0)` if low_prev-low > high-high_prev
- `+DI`: `100 * EMA(+DM) / ATR`
- `-DI`: `100 * EMA(-DM) / ATR`
- `DX`: `100 * |+DI - -DI| / (+DI + -DI)`
- `ADX`: `EMA(DX)`

**Interpretation:**
- ADX < 20: Weak trend (range-bound)
- ADX 20-25: Developing trend
- ADX > 25: Strong trend
- ADX > 50: Very strong trend

### Supertrend - 2 features

**Implementation:** `src/phase1/stages/features/trend.py::add_supertrend()`

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `supertrend` | Dynamic support/resistance based on ATR | period=10, mult=3.0 | 1 bar |
| `supertrend_direction` | 1=uptrend, -1=downtrend | - | {-1, 1} | 1 bar |

**Algorithm:**
```python
basic_upper = (high + low) / 2 + mult * ATR
basic_lower = (high + low) / 2 - mult * ATR

# Upper band tightens in downtrend
upper_band = min(basic_upper, prev_upper) if close_prev > prev_upper else basic_upper

# Lower band tightens in uptrend
lower_band = max(basic_lower, prev_lower) if close_prev < prev_lower else basic_lower

# Trend switches on band break
if close > upper_band: direction = 1, supertrend = lower_band
if close < lower_band: direction = -1, supertrend = upper_band
```

**Use:** Trailing stop-loss, trend-following entry/exit.

---

## Temporal Features

**Category:** Time-based cyclical patterns

**Count:** 9 features

**Implementation:** `src/phase1/stages/features/temporal.py::add_temporal_features()`

### Cyclical Time Encoding (6 features)

Sin/cos encoding preserves cyclical nature of time (e.g., 23:59 is close to 00:00).

| Feature | Formula | Cycle | Range |
|---------|---------|-------|-------|
| `hour_sin` | `sin(2π * hour / 24)` | 24-hour | [-1, 1] |
| `hour_cos` | `cos(2π * hour / 24)` | 24-hour | [-1, 1] |
| `minute_sin` | `sin(2π * minute / 60)` | 60-minute | [-1, 1] |
| `minute_cos` | `cos(2π * minute / 60)` | 60-minute | [-1, 1] |
| `dayofweek_sin` | `sin(2π * dayofweek / 7)` | 7-day | [-1, 1] |
| `dayofweek_cos` | `cos(2π * dayofweek / 7)` | 7-day | [-1, 1] |

**Why sin/cos?** Prevents discontinuity (e.g., hour=23 and hour=0 are encoded as similar values).

### Trading Sessions (3 features)

**Implementation:** `src/phase1/stages/features/temporal.py::add_temporal_features()`

One-hot encoded trading sessions (UTC-based, 8-hour blocks):

| Feature | Formula | Hours (UTC) | Markets Active |
|---------|---------|-------------|----------------|
| `session_asia` | `hour in [0, 8)` | 00:00-08:00 | Tokyo, Hong Kong, Singapore |
| `session_london` | `hour in [8, 16)` | 08:00-16:00 | London, Frankfurt |
| `session_ny` | `hour in [16, 24)` | 16:00-24:00 | New York, Chicago |

**Note:** Temporal features have NO lag (they describe the current bar's timestamp, not price action).

---

## Regime Features

**Category:** Market state classification

**Count:** 2 features (basic) or 3+ features (advanced)

**Implementation:** `src/phase1/stages/features/regime.py::add_regime_features()`

### Basic Regime Features (2 features)

| Feature | Formula | Parameters | Values | Lag |
|---------|---------|------------|--------|-----|
| `volatility_regime` | `hvol_20 > rolling_median(hvol_20, 100)` | lookback=100 | {0, 1} | 1 bar |
| `trend_regime` | SMA-based trend classification | SMA=(50, 200) | {-1, 0, 1} | 1 bar |

**Volatility Regime:**
- 0: Low volatility (below median)
- 1: High volatility (above median)

**Trend Regime:**
- 1: Uptrend (close > SMA50 > SMA200)
- -1: Downtrend (close < SMA50 < SMA200)
- 0: Sideways (other configurations)

### Advanced Regime Features (3+ features)

**Implementation:** `src/phase1/stages/regime/` (separate package)

Adds Hurst exponent-based market structure classification:

| Feature | Formula | Parameters | Values |
|---------|---------|------------|--------|
| `structure_regime` | Hurst exponent classification | lookback=100 | {mean_reverting, random, trending} |
| `hurst_exponent` | R/S analysis | lookback=100 | [0, 1] |
| `composite_regime` | Combined vol + trend + structure | - | Multi-dimensional |

**Hurst Exponent Interpretation:**
- H < 0.5: Mean-reverting (anti-persistent)
- H ≈ 0.5: Random walk
- H > 0.5: Trending (persistent)

**Use Cases:**
1. **Model selection:** Train separate models per regime
2. **Parameter adaptation:** Adjust triple-barrier params per regime
3. **Feature weighting:** Weight features differently in different regimes

---

## Microstructure Features

**Category:** Liquidity, spread, and order flow proxies derived from OHLCV

**Count:** 19 features

**Implementation:** `src/phase1/stages/features/microstructure.py::add_microstructure_features()`

**Note:** These features estimate microstructure properties without tick-level data.

### Amihud Illiquidity - 3 features

Price impact per unit of volume traded.

| Feature | Formula | Parameters | Interpretation | Lag |
|---------|---------|------------|----------------|-----|
| `micro_amihud` | `|return| / volume` | - | Higher = less liquid | 1 bar |
| `micro_amihud_10` | `SMA_10(micro_amihud)` | period=10 | Smoothed illiquidity | 1 bar |
| `micro_amihud_20` | `SMA_20(micro_amihud)` | period=20 | Smoothed illiquidity | 1 bar |

**Reference:** Amihud (2002), "Illiquidity and Stock Returns"

### Roll Spread - 2 features

Bid-ask spread estimator from serial covariance.

| Feature | Formula | Parameters | Interpretation | Lag |
|---------|---------|------------|----------------|-----|
| `micro_roll_spread` | `2 * sqrt(max(-cov(Δp, Δp_lag), 0))` | period=20 | Spread in price units | 1 bar |
| `micro_roll_spread_pct` | `micro_roll_spread / close * 100` | period=20 | Spread as % of price | 1 bar |

**Reference:** Roll (1984), "A Simple Implicit Measure of the Effective Bid-Ask Spread"

**Intuition:** Negative serial covariance in price changes indicates bid-ask bounce.

### Kyle's Lambda - 1 feature

Price impact coefficient (lambda in Kyle's model).

| Feature | Formula | Parameters | Interpretation | Lag |
|---------|---------|------------|----------------|-----|
| `micro_kyle_lambda` | `|return| / avg_volume_20` | volume_period=20 | Price impact per unit volume | 1 bar |

**Reference:** Kyle (1985), "Continuous Auctions and Insider Trading"

### Corwin-Schultz Spread - 1 feature

High-low spread estimator.

| Feature | Formula | Parameters | Range | Lag |
|---------|---------|------------|-------|-----|
| `micro_cs_spread` | Derived from 2-day high-low ratios | - | [0, 1] | 1 bar |

**Reference:** Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads"

**Formula (simplified):**
```python
beta = log(high/low)^2 + log(high_prev/low_prev)^2
gamma = log(max(high, high_prev) / min(low, low_prev))^2
alpha = (sqrt(2*beta) - sqrt(beta)) / denom - sqrt(gamma / denom)
spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
```

### Relative Spread - 3 features

High-low range normalized by close.

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `micro_rel_spread` | `(high - low) / close` | - | 1 bar |
| `micro_rel_spread_10` | `SMA_10(micro_rel_spread)` | period=10 | 1 bar |
| `micro_rel_spread_20` | `SMA_20(micro_rel_spread)` | period=20 | 1 bar |

### Volume Imbalance - 2 features

Order flow proxy from close location within range.

| Feature | Formula | Parameters | Range | Lag |
|---------|---------|------------|-------|-----|
| `micro_volume_imbalance` | `(close - open) / (high - low)` | - | [-1, 1] | 1 bar |
| `micro_cum_imbalance_20` | `sum_20(micro_volume_imbalance)` | period=20 | unbounded | 1 bar |

**Interpretation:**
- Positive: Buying pressure (close near high)
- Negative: Selling pressure (close near low)

### Trade Intensity - 2 features

Volume relative to recent average.

| Feature | Formula | Parameters | Interpretation | Lag |
|---------|---------|------------|----------------|-----|
| `micro_trade_intensity_20` | `volume / avg_volume_20` | period=20 | Relative activity | 1 bar |
| `micro_trade_intensity_50` | `volume / avg_volume_50` | period=50 | Relative activity | 1 bar |

**Use:** High intensity may indicate informed trading or market events.

### Price Efficiency - 2 features

Trending vs choppy market measure.

| Feature | Formula | Parameters | Range | Lag |
|---------|---------|------------|-------|-----|
| `micro_efficiency_10` | `|net_change_10| / sum_abs_changes_10` | period=10 | [0, 1] | 1 bar |
| `micro_efficiency_20` | `|net_change_20| / sum_abs_changes_20` | period=20 | [0, 1] | 1 bar |

**Interpretation:**
- Efficiency ≈ 1: Strong trend (straight line)
- Efficiency ≈ 0: Choppy/mean-reverting

### Realized Volatility Ratio - 1 feature

Short-term vs long-term volatility (volatility clustering).

| Feature | Formula | Parameters | Interpretation | Lag |
|---------|---------|------------|----------------|-----|
| `micro_vol_ratio` | `std_5(returns) / std_20(returns)` | short=5, long=20 | Ratio > 1 = vol expansion | 1 bar |

---

## Wavelet Features

**Category:** Multi-scale signal decomposition (frequency domain analysis)

**Count:** 24 features (default config: level=3, price+volume, all components)

**Implementation:** `src/phase1/stages/features/wavelets.py::add_wavelet_features()`

**Note:** Requires PyWavelets library. Features are skipped if unavailable or insufficient data.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wavelet` | `db4` | Wavelet family (db4, db8, sym5, coif3, haar) |
| `level` | `3` | Decomposition levels (3 = 4 frequency bands) |
| `window` | `64` | Rolling window size for causal computation |
| `normalize` | `True` | Z-score normalization of coefficients |

**Supported Wavelets:**
- `db4`: Daubechies 4 (good general-purpose)
- `db8`: Daubechies 8 (smoother)
- `sym5`: Symlet 5 (nearly symmetric, less phase distortion)
- `coif3`: Coiflet 3 (symmetric with vanishing moments)
- `haar`: Haar (simplest, good for abrupt changes)

### Price Wavelet Coefficients (4 features)

Decompose close price into frequency components.

| Feature | Formula | Frequency Band | Lag |
|---------|---------|----------------|-----|
| `wavelet_close_approx` | Approximation coefficient (trend) | Low-frequency | 1 bar |
| `wavelet_close_d1` | Detail coefficient level 1 | High-frequency (noise) | 1 bar |
| `wavelet_close_d2` | Detail coefficient level 2 | Mid-frequency | 1 bar |
| `wavelet_close_d3` | Detail coefficient level 3 | Low-frequency (cycles) | 1 bar |

**Interpretation:**
- `approx`: Smoothed trend (low-pass filter)
- `d1`: High-frequency noise and microstructure
- `d2`: Medium-term oscillations
- `d3`: Longer-term cycles

### Volume Wavelet Coefficients (4 features)

Same decomposition for volume (if enabled and volume available).

| Feature | Formula | Frequency Band | Lag |
|---------|---------|----------------|-----|
| `wavelet_volume_approx` | Volume trend | Low-frequency | 1 bar |
| `wavelet_volume_d1` | Volume high-frequency | High-frequency | 1 bar |
| `wavelet_volume_d2` | Volume mid-frequency | Mid-frequency | 1 bar |
| `wavelet_volume_d3` | Volume low-frequency cycles | Low-frequency | 1 bar |

### Wavelet Energy Features (10 features)

Energy (power) at each frequency band.

**Price Energy (5 features):**

| Feature | Formula | Interpretation | Lag |
|---------|---------|----------------|-----|
| `wavelet_close_energy_approx` | `log1p(sum(approx^2))` | Trend strength | 1 bar |
| `wavelet_close_energy_d1` | `log1p(sum(d1^2))` | High-frequency power | 1 bar |
| `wavelet_close_energy_d2` | `log1p(sum(d2^2))` | Mid-frequency power | 1 bar |
| `wavelet_close_energy_d3` | `log1p(sum(d3^2))` | Low-frequency power | 1 bar |
| `wavelet_close_energy_ratio` | `approx_energy / total_energy` | Trend dominance [0, 1] | 1 bar |

**Volume Energy (5 features):**

Same as price energy but for volume signal.

### Wavelet Volatility (1 feature)

Robust volatility estimate using MAD of detail coefficients.

| Feature | Formula | Parameters | Lag |
|---------|---------|------------|-----|
| `wavelet_close_volatility` | `MAD(detail_coeffs) / 0.6745` | window=64 | 1 bar |

**Advantage:** More robust to trends than standard deviation.

### Wavelet Trend Strength (2 features)

Trend strength and direction from approximation coefficient slope.

| Feature | Formula | Parameters | Range | Lag |
|---------|---------|------------|-------|-----|
| `wavelet_close_trend_strength` | `|slope(approx)| / std(close)` | window=64, level=3 | [0, ∞) | 1 bar |
| `wavelet_close_trend_direction` | `sign(slope(approx))` | window=64, level=3 | {-1, 1} | 1 bar |

---

## Multi-Timeframe Features

**Category:** Indicators computed on higher timeframes

**Count:** Variable (depends on MTF timeframes enabled)

**Implementation:** `src/phase1/stages/mtf/generator.py::add_mtf_features()`

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_timeframe` | `5min` | Base data timeframe |
| `mtf_timeframes` | `[15min, 60min]` | Higher timeframes to resample |
| `include_ohlcv` | `True` | Include resampled OHLCV bars |
| `include_indicators` | `True` | Include indicators computed on MTF |

**Supported MTF timeframes:**
- Short-term: 10min, 15min, 30min, 45min
- Hourly: 60min (1h)
- Multi-hour: 4h (240min)
- Daily: daily (1d)

**Constraint:** MTF timeframes must be > base timeframe.

### MTF OHLCV Features (5 features per timeframe)

Resampled OHLC and volume bars from higher timeframe.

**Example for 15min MTF (suffix `_15m`):**

| Feature | Resampling Rule | Lag |
|---------|-----------------|-----|
| `open_15m` | First open in 15min window | 1 bar |
| `high_15m` | Max high in 15min window | 1 bar |
| `low_15m` | Min low in 15min window | 1 bar |
| `close_15m` | Last close in 15min window | 1 bar |
| `volume_15m` | Sum volume in 15min window | 1 bar |

**Example for 60min MTF (suffix `_1h`):**

| Feature | Resampling Rule | Lag |
|---------|-----------------|-----|
| `open_1h` | First open in 60min window | 1 bar |
| `high_1h` | Max high in 60min window | 1 bar |
| `low_1h` | Min low in 60min window | 1 bar |
| `close_1h` | Last close in 60min window | 1 bar |
| `volume_1h` | Sum volume in 60min window | 1 bar |

### MTF Indicator Features (Variable per timeframe)

Subset of base indicators recomputed on MTF data.

**Default MTF indicators (per timeframe):**
- Returns: `return_1`, `return_5`, `return_10`
- Moving averages: `sma_20`, `sma_50`, `ema_9`, `ema_21`
- Momentum: `rsi_14`, `macd_line`, `macd_signal`
- Volatility: `atr_14`, `bb_width`, `hvol_20`
- Volume: `obv`, `volume_ratio` (if volume available)

**Example for 15min MTF:**

| Feature | Formula (on 15min bars) | Lag |
|---------|-------------------------|-----|
| `return_1_15m` | `pct_change(close_15m, 1)` | 1 bar |
| `sma_20_15m` | `SMA_20(close_15m)` | 1 bar |
| `rsi_14_15m` | `RSI_14(close_15m)` | 1 bar |
| `atr_14_15m` | `ATR_14(H_15m, L_15m, C_15m)` | 1 bar |
| `hvol_20_15m` | `std_20(log_returns_15m) * ann_factor` | 1 bar |

**Feature Count Estimate:**

For default config (MTF=[15min, 60min], both OHLCV + indicators):
- Per timeframe: ~5 OHLCV + ~15-20 indicators = ~20-25 features
- Total MTF features: ~40-50 features

**MTF Feature Naming Convention:**
```
{indicator_name}_{mtf_suffix}

where mtf_suffix:
  15min → _15m
  30min → _30m
  60min or 1h → _1h
  4h or 240min → _4h
  daily or 1d → _1d
```

---

## Feature Engineering Parameters Summary

### Period Scaling

**All indicator periods are timeframe-aware and scale automatically to maintain consistent lookback duration.**

**Scaling formula:**
```python
scaled_period = round((period * source_timeframe_minutes) / target_timeframe_minutes)
minimum_period = 2  # enforced minimum
```

**Example: RSI-14 scaling**

| Target Timeframe | Scaled Period | Lookback Duration |
|------------------|---------------|-------------------|
| 1min | 70 | 70 min |
| 5min (base) | 14 | 70 min |
| 15min | 5 | 75 min |
| 60min | 2 | 120 min (min enforced) |

**Implementation:** `src/phase1/stages/features/scaling.py::PeriodScaler`

### Annualization Factors

**Volatility annualization factors are timeframe-aware.**

**Formula:**
```python
annualization_factor = sqrt(bars_per_day * trading_days_per_year)
bars_per_day = trading_hours_per_day * 60 / timeframe_minutes
```

**Annualization factors (6.5h regular session):**

| Timeframe | Bars/Day | Annualization Factor |
|-----------|----------|---------------------|
| 1min | 390 | 313.21 |
| 5min | 78 | 140.07 |
| 15min | 26 | 80.92 |
| 60min | 6.5 | 40.45 |

**Implementation:** `src/phase1/stages/features/constants.py::get_annualization_factor()`

### Anti-Lookahead Enforcement

**ALL features are shifted by 1 bar to prevent lookahead bias.**

**Pattern:**
```python
# WRONG (lookahead bias):
df['sma_20'] = df['close'].rolling(20).mean()

# CORRECT (no lookahead):
df['sma_20'] = df['close'].rolling(20).mean().shift(1)
```

**Result:** Feature at bar `t` uses data only up to bar `t-1`.

**Exception:** Temporal features (hour, minute, dayofweek) have no lag since they describe the current bar's timestamp, not future price action.

---

## Feature Quality and Validation

### NaN Handling

**Column-level filtering:**
- Columns with NaN rate > `nan_threshold` (default 0.9) are dropped before row-wise NaN removal
- Prevents all-NaN columns from wiping the entire dataset

**Row-level filtering:**
- After column filtering, rows with any NaN are dropped
- Final dataset contains no NaN values

**Implementation:** `src/phase1/stages/features/nan_handling.py::clean_nan_columns()`

### Feature Metadata

Every feature includes metadata documenting:
- Formula/calculation method
- Parameters used
- Lag/shift applied
- Expected range/distribution

**Storage:** `{symbol}_feature_metadata.json`

**Example:**
```json
{
  "rsi_14": "14-period Relative Strength Index (lagged)",
  "sma_50": "50-period Simple Moving Average (lagged)",
  "wavelet_close_approx": "Wavelet approx db4 L3 normalized (lagged)"
}
```

### Stationarity

**Stationary features (safe for ML):**
- All returns and log returns
- All ratios and percentages
- Normalized features (z-scores, BB position, etc.)
- Oscillators (RSI, Stochastic, Williams %R, etc.)
- Temporal features (sin/cos encoding)

**Non-stationary features (use with caution or transform):**
- Raw prices (open, high, low, close)
- Raw moving averages (SMA, EMA values)
- Cumulative indicators (OBV, MACD line)
- Wavelet approximation coefficients (raw)

**Recommended practice:**
- Use price-to-MA ratios instead of raw MA values
- Use normalized/z-scored versions when available
- Apply differencing or log-transform if needed

---

## Feature Engineering Pipeline Flow

```
Raw OHLCV Data (1min or 5min resampled)
    ↓
[ Validation & Cleaning ]
    ↓
[ Base Feature Engineering ] (order matters for dependencies)
    1. Returns & Price Ratios
    2. Moving Averages (SMA, EMA)
    3. Momentum (RSI, MACD, Stochastic, etc.)
    4. Volatility (ATR, Bollinger, Keltner, Hvol, etc.)
    5. Volume (OBV, VWAP, Dollar Volume)
    6. Trend (ADX, Supertrend)
    7. Temporal (Hour, Minute, Day, Session)
    8. Regime (Volatility, Trend regimes)
    9. Autocorrelation & Higher Moments
    10. Microstructure (Amihud, Roll, Kyle, etc.)
    ↓
[ Wavelet Features ] (if enabled and sufficient data)
    - Coefficients (approx + details)
    - Energy features
    - Wavelet volatility
    - Trend strength
    ↓
[ Multi-Timeframe Features ] (if enabled and sufficient data)
    - Resample to higher timeframes
    - Compute MTF OHLCV
    - Compute MTF indicators
    ↓
[ NaN Cleaning ]
    - Drop high-NaN columns (> nan_threshold)
    - Drop rows with any NaN
    ↓
[ Feature Metadata Recording ]
    - Document all features with formulas/params
    ↓
Engineered Feature DataFrame
    ↓
[ Save to Parquet ]
    - {symbol}_features.parquet
    - {symbol}_feature_metadata.json
```

---

## References

**Academic Papers:**
- Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects"
- Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices"
- Garman, M. B., & Klass, M. J. (1980). "On the Estimation of Security Price Volatilities from Historical Data"
- Kyle, A. S. (1985). "Continuous Auctions and Insider Trading"
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning" (Chapter 8: Feature Importance)
- Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the Rate of Return"
- Rogers, L. C. G., & Satchell, S. E. (1991). "Estimating Variance from High, Low and Closing Prices"
- Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market"
- Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices"

**Implementation:**
- Technical Analysis Library: TA-Lib
- Wavelet Analysis: PyWavelets
- Time Series: pandas resampling
- Numba optimization: `src/phase1/stages/features/numba_functions.py`

---

## See Also

- [Feature Selection Configurations](./FEATURE_SELECTION_CONFIGS.md)
- [Multi-Timeframe Feature Configurations](./MTF_FEATURE_CONFIGS.md)
- [Model-Specific Feature Lists](./MODEL_FEATURE_REQUIREMENTS.md)
- [Feature Engineering Implementation Guide](/docs/guides/FEATURE_ENGINEERING_GUIDE.md)
