# Best OHLCV Features for ML Trading Models

A comprehensive research-backed catalog of features for OHLCV-based machine learning trading systems.

**Research Date:** 2024-12-24
**Based On:** Academic literature, industry practice, and codebase analysis

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Feature Categories](#feature-categories)
3. [Volatility Estimators](#1-volatility-estimators)
4. [Microstructure Features](#2-microstructure-features-from-ohlcv)
5. [Momentum Features](#3-momentum-features)
6. [Cross-Temporal Features](#4-cross-temporal-features)
7. [Market Regime Features](#5-market-regime-features)
8. [Wavelet Features](#6-wavelet-features)
9. [Features by Model Type](#7-features-by-model-type)
10. [MTF Feature Recommendations](#8-multi-timeframe-mtf-recommendations)
11. [Missing Features to Add](#9-missing-features-to-add)
12. [Implementation Status](#10-implementation-status)

---

## Executive Summary

### Key Findings from Academic Research

1. **Statistical Significance**: Studies show incorporating technical indicators in ML algorithms reduces prediction errors by up to 60% ([arXiv 2412.15448](https://arxiv.org/html/2412.15448v1/))

2. **Momentum Dominance**: Multi-model ML frameworks indicate momentum-based indicators are the most influential predictors ([MDPI 2024](https://www.mdpi.com/2504-2289/9/10/248))

3. **Model-Specific Features**: XGBoost achieves 71% accuracy vs LSTM's 62-67% on same features, suggesting feature requirements differ by model ([Research Comparison](https://netanel.io/posts/xgb_vs_lstm/))

4. **Overfitting Warning**: Strong in-sample R-squared (0.749-0.812) but negative out-of-sample R-squared (-0.020 to -0.016) highlights critical need for proper validation

### Feature Priority Matrix

| Priority | Category | Reason |
|----------|----------|--------|
| Critical | Volatility (Yang-Zhang, Parkinson) | 5x more efficient than close-to-close |
| Critical | Momentum (RSI, MACD, ROC) | Strongest predictive power in academic studies |
| High | Microstructure (Amihud, Roll) | Liquidity predicts returns |
| High | Regime (HMM, Hurst) | Improves Sharpe by filtering bad trades |
| Medium | Wavelets | Captures non-stationary patterns |
| Medium | Cross-temporal (autocorr, spectral) | Long-memory effects |

---

## Feature Categories

### Currently Implemented (Phase 1)

| Category | Count | Module |
|----------|-------|--------|
| Volatility | 15+ | `volatility.py` |
| Momentum | 20+ | `momentum.py` |
| Microstructure | 20+ | `microstructure.py` |
| Wavelets | 24+ | `wavelets.py` |
| Regime | 5+ | `regime.py` |
| Moving Averages | 10+ | `moving_averages.py` |
| Trend | 8+ | `trend.py` |
| Volume | 10+ | `volume.py` |
| Temporal | 8+ | `temporal.py` |
| Price | 15+ | `price_features.py` |

**Total: ~150+ features**

---

## 1. Volatility Estimators

### 1.1 Range-Based Volatility (IMPLEMENTED)

Range-based estimators are 5x more efficient than close-to-close volatility ([Portfolio Optimizer](https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/)).

#### Parkinson (1980)
```
Parkinson_Vol = sqrt(1/(4*ln(2)) * (ln(H/L))^2) * AnnFactor
```
- Uses only high-low range
- 5x more efficient than close-to-close
- **Limitation**: Assumes zero drift, overestimates during trends

**Status**: IMPLEMENTED in `volatility.py`

#### Garman-Klass (1980)
```
GK_Vol = sqrt(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2) * AnnFactor
```
- Uses OHLC data
- More efficient than Parkinson
- Best estimator when accounting for noise ([Molnar 2012](https://onlinelibrary.wiley.com/doi/10.1002/fut.20197))
- **Limitation**: Assumes zero drift

**Status**: IMPLEMENTED in `volatility.py`

#### Rogers-Satchell (1991)
```
RS_Vol = sqrt(ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O))
```
- Allows non-zero drift
- Better for trending markets
- **Limitation**: Doesn't handle opening jumps

**Status**: MISSING - ADD TO CODEBASE

#### Yang-Zhang (2000)
```
YZ_Vol = sqrt(Vol_overnight + k*Vol_close_to_close + (1-k)*Vol_Rogers_Satchell)
```
Where k minimizes variance.

- Handles both drift and opening jumps
- Minimum variance estimator
- **Best overall choice** for most situations

**Status**: MISSING - ADD TO CODEBASE (HIGH PRIORITY)

### 1.2 Realized Volatility Components (PARTIALLY IMPLEMENTED)

#### Bipower Variation (Barndorff-Nielsen & Shephard 2004)
```
BV = (pi/2) * sum(|r_t| * |r_{t-1}|)
```
- Robust to jumps
- Separates continuous volatility from jump component
- **Key insight**: Difference between RV and BV estimates jump variation

**Reference**: [Oxford Academic](https://academic.oup.com/jfec/article-abstract/2/1/1/960705)

**Status**: MISSING - ADD TO CODEBASE (MEDIUM PRIORITY)

#### Jump Detection
```
Jump_Component = RV - BV (when RV > BV)
Jump_Test = (RV - BV) / sqrt(variance_estimator)
```
- Identifies discontinuous price movements
- Critical for risk management

**Status**: MISSING - ADD TO CODEBASE

### 1.3 Volatility of Volatility (PARTIALLY IMPLEMENTED)

```python
# Rolling volatility of volatility
vol_of_vol = hvol.rolling(20).std()

# Volatility ratio (short/long)
vol_ratio = hvol_5 / hvol_20
```

**Status**: `micro_vol_ratio` implemented, full vol-of-vol MISSING

---

## 2. Microstructure Features (from OHLCV)

These features estimate liquidity and informed trading without order book data.

### 2.1 Amihud Illiquidity (2002) - IMPLEMENTED
```
Amihud = |Return| / Dollar_Volume
```
- Measures price impact per unit volume
- Higher values = less liquid
- Good proxy for Kyle's lambda

**Reference**: [Amihud 2002](https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf)

**Status**: IMPLEMENTED in `microstructure.py`

### 2.2 Roll Spread (1984) - IMPLEMENTED
```
Spread = 2 * sqrt(max(-Cov(dp_t, dp_{t-1}), 0))
```
- Estimates bid-ask spread from serial covariance
- Negative covariance implies bid-ask bounce
- Works with daily OHLCV data

**Status**: IMPLEMENTED in `microstructure.py`

### 2.3 Kyle's Lambda (1985) - IMPLEMENTED
```
Lambda = |Return| / Average_Volume
```
- Price impact coefficient
- Proxy for market depth

**Reference**: [Kyle Lambda Documentation](https://frds.io/measures/kyle_lambda/)

**Status**: IMPLEMENTED in `microstructure.py`

### 2.4 Corwin-Schultz Spread (2012) - IMPLEMENTED
```
alpha = (sqrt(2*beta) - sqrt(beta))/(3 - 2*sqrt(2)) - sqrt(gamma/(3 - 2*sqrt(2)))
CS_Spread = 2*(exp(alpha) - 1)/(1 + exp(alpha))
```
Where:
- beta = sum of squared log(H/L) over 2 consecutive bars
- gamma = squared log of 2-bar high to 2-bar low

**Key Insight**: Highs are more likely at ask, lows at bid

**Reference**: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1106193)

**Status**: IMPLEMENTED in `microstructure.py`

### 2.5 VPIN (Volume-Synchronized Probability of Informed Trading) - MISSING

VPIN uses volume buckets instead of time windows to detect informed trading.

```python
# Pseudo-code for VPIN
def calculate_vpin(trades, bucket_size, n_buckets):
    # Classify each trade as buy/sell using tick rule or bulk classification
    # Aggregate into volume buckets of fixed size
    # Calculate order imbalance in each bucket
    # VPIN = mean absolute imbalance over n trailing buckets
    pass
```

**Key Properties**:
- Leading indicator (signaled Flash Crash hours before)
- Volume-synchronized timing more stable than clock-time
- Requires trade classification (can proxy from OHLCV using close position in range)

**Reference**: [QuantResearch VPIN](https://www.quantresearch.org/VPIN.pdf)

**Status**: MISSING - ADD TO CODEBASE (HIGH PRIORITY)

### 2.6 High-Low Amihud Variant - MISSING

Uses range instead of absolute return to capture intraday volatility:
```
HL_Amihud = (High - Low) / Volume
```

**Status**: MISSING - simple addition to existing Amihud code

---

## 3. Momentum Features

### 3.1 RSI (Relative Strength Index) - IMPLEMENTED
```
RSI = 100 - (100 / (1 + RS))
RS = Avg_Gain / Avg_Loss
```
- Traditional: 70 overbought, 30 oversold
- Academic research confirms predictive power

**Status**: IMPLEMENTED in `momentum.py`

### 3.2 MACD - IMPLEMENTED
```
MACD_Line = EMA(12) - EMA(26)
Signal = EMA(MACD_Line, 9)
Histogram = MACD_Line - Signal
```

**Status**: IMPLEMENTED in `momentum.py`

### 3.3 Rate of Change (ROC) - IMPLEMENTED
```
ROC = (Close - Close_n) / Close_n * 100
```

**Status**: IMPLEMENTED in `momentum.py`

### 3.4 Moving Hurst Indicator - MISSING

Novel indicator based on Hurst exponent:
```python
def moving_hurst(prices, window=100):
    """
    Calculate rolling Hurst exponent.
    H > 0.5: trending
    H < 0.5: mean-reverting
    H = 0.5: random walk
    """
    # Use R/S analysis or DFA method
    pass
```

**Reference**: [Hurst Trading Signals](https://www.scitepress.org/papers/2018/66670/66670.pdf)

**Status**: MISSING - ADD TO CODEBASE (HIGH PRIORITY)

### 3.5 Momentum Quality Score - MISSING

Composite momentum score that weights different timeframes:
```python
momentum_quality = w1 * roc_5 + w2 * roc_20 + w3 * roc_60
# With consistency check
momentum_consistency = sign(roc_5) == sign(roc_20) == sign(roc_60)
```

**Status**: MISSING - simple composite

---

## 4. Cross-Temporal Features

### 4.1 Autocorrelation Features - MISSING

```python
# Lagged autocorrelation
acf_1 = returns.autocorr(lag=1)
acf_5 = returns.autocorr(lag=5)
acf_20 = returns.autocorr(lag=20)

# Partial autocorrelation
pacf = sm.tsa.pacf(returns, nlags=20)
```

**Use Case**: Detect mean-reversion vs trending behavior

**Status**: MISSING - ADD TO CODEBASE

### 4.2 Hurst Exponent - PARTIAL

```python
def hurst_exponent(prices, max_lag=100):
    """
    H > 0.5: Persistent (trending)
    H < 0.5: Anti-persistent (mean-reverting)
    H = 0.5: Random walk (no memory)
    """
    # R/S Analysis or DFA method
    pass
```

**Academic Significance**: Incompatibility with EMH when H != 0.5

**Reference**: [Springer Article](https://link.springer.com/article/10.1186/s40854-022-00394-x)

**Status**: In `stages/regime` but not exposed as features

### 4.3 Spectral Features - MISSING

```python
from scipy import signal

def spectral_features(prices, window):
    freqs, psd = signal.welch(prices[-window:])

    # Dominant frequency
    dominant_freq = freqs[np.argmax(psd)]

    # Spectral entropy (disorder)
    psd_norm = psd / psd.sum()
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    # Low frequency power ratio
    low_freq_power = psd[freqs < 0.1].sum() / psd.sum()

    return dominant_freq, spectral_entropy, low_freq_power
```

**Status**: MISSING - ADD TO CODEBASE

### 4.4 Fractal Dimension - MISSING

Measures price path complexity:
```python
def fractal_dimension(prices, window):
    """
    Higuchi or box-counting method.
    Higher FD = more complex/choppy
    Lower FD = smoother trends
    """
    pass
```

**Status**: MISSING - ADD TO CODEBASE

---

## 5. Market Regime Features

### 5.1 Hidden Markov Model States - MISSING

HMMs can identify latent market regimes (bull/bear/neutral):

```python
from hmmlearn import hmm

def fit_hmm_regimes(returns, n_states=3):
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")
    model.fit(returns.reshape(-1, 1))

    # Get regime probabilities
    states = model.predict(returns.reshape(-1, 1))
    state_probs = model.predict_proba(returns.reshape(-1, 1))

    return states, state_probs
```

**Academic Support**: Kritzman et al. 2012 showed HMM regime detection improves portfolio returns

**Reference**: [QuantStart HMM](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)

**Status**: MISSING - ADD TO CODEBASE (HIGH PRIORITY)

### 5.2 Volatility Regime - IMPLEMENTED

Simple high/low classification based on rolling volatility percentile.

**Status**: IMPLEMENTED in `regime.py`

### 5.3 Trend Regime - IMPLEMENTED

Based on price vs moving average relationships.

**Status**: IMPLEMENTED in `regime.py`

### 5.4 Structure Regime (Hurst-based) - PARTIAL

Classifies market as trending/mean-reverting/random.

**Status**: In `stages/regime` but not fully exposed

### 5.5 Composite Regime Score - MISSING

```python
# Combine multiple regime indicators
composite_regime = (
    0.4 * volatility_regime +
    0.3 * trend_regime +
    0.3 * structure_regime
)
```

**Status**: MISSING - simple composite

---

## 6. Wavelet Features

### 6.1 Wavelet Decomposition - IMPLEMENTED

Multi-scale decomposition using DWT:
- **Approximation coefficients**: Low-frequency trend
- **Detail coefficients**: High-frequency noise at multiple levels

Recommended wavelets for financial data:
- `db4` (Daubechies 4): Good general-purpose
- `db8`: Smoother approximation
- `haar`: Captures abrupt changes
- `sym5`: Nearly symmetric, less phase distortion

**Reference**: [Springer Wavelet Learning](https://link.springer.com/article/10.1007/s10489-021-02218-4)

**Status**: IMPLEMENTED in `wavelets.py`

### 6.2 Wavelet Energy Features - IMPLEMENTED

Energy at each decomposition level indicates signal power in frequency bands.

**Status**: IMPLEMENTED in `wavelets.py`

### 6.3 Wavelet Volatility - IMPLEMENTED

Uses MAD of detail coefficients - more robust to trends.

**Status**: IMPLEMENTED in `wavelets.py`

### 6.4 Wavelet Trend Strength - IMPLEMENTED

Slope of approximation coefficients normalized by volatility.

**Status**: IMPLEMENTED in `wavelets.py`

### 6.5 Wavelet Coherence (Cross-asset) - MISSING

For multi-asset strategies:
```python
import pywt

def wavelet_coherence(series1, series2, wavelet='cmor1.5-1.0'):
    """
    Measures time-frequency correlation between two series.
    Useful for pairs trading and correlation regime detection.
    """
    pass
```

**Status**: MISSING - useful for multi-asset

---

## 7. Features by Model Type

### 7.1 XGBoost / Gradient Boosting

XGBoost achieves higher accuracy (71%) vs LSTM (62-67%) on same data.

**Best Features for XGBoost**:
| Feature Type | Reason |
|--------------|--------|
| Momentum indicators | Strong split-based importance |
| Lagged returns | Multiple lags capture patterns |
| Volatility ratios | Stationary, bounded |
| Binary regime flags | Easy decision boundaries |
| Ranked percentiles | Robust to outliers |

**Preprocessing**:
- No need for strict stationarity
- Can handle missing values
- Feature importance for selection

### 7.2 LSTM / RNN

LSTMs excel at sequential dependencies and smaller datasets.

**Best Features for LSTM**:
| Feature Type | Reason |
|--------------|--------|
| Raw OHLCV sequences | Learns patterns directly |
| Differenced prices | Stationarity helps |
| Normalized indicators | Bounded inputs important |
| Time embeddings | Cyclical patterns |
| Wavelet coefficients | Multi-scale patterns |

**Preprocessing**:
- Standardization critical (mean=0, std=1)
- Sequence padding/truncation
- Consider input normalization layer

### 7.3 Transformers

Transformers capture long-range dependencies via attention.

**Best Features for Transformers**:
| Feature Type | Reason |
|--------------|--------|
| Positional embeddings | Temporal structure |
| Multi-head compatible | Separate attention per feature type |
| Longer sequences | Better long-term dependency |
| Raw + engineered mix | Attention learns importance |
| Cross-asset features | Multi-head cross-attention |

**Reference**: [Transformer Stock Prediction](https://ojs.bonviewpress.com/index.php/FSI/article/download/5703/1501/33635)

### 7.4 Hybrid Approaches (Recommended)

XGBoost for feature selection + LSTM for prediction:
```python
# Stage 1: XGBoost feature importance
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
important_features = get_top_features(xgb_model, n=50)

# Stage 2: LSTM with selected features
lstm_input = X_train[important_features]
lstm_model.fit(lstm_input, y_train)
```

---

## 8. Multi-Timeframe (MTF) Recommendations

### 8.1 Optimal Timeframe Combinations

Based on practitioner experience and academic research:

| Trading Style | Timeframes | Ratio |
|---------------|------------|-------|
| Scalping | 1m, 5m, 15m | 1:5:15 |
| Day Trading | 5m, 15m, 1h | 1:3:12 |
| Swing | 15m, 1h, 4h | 1:4:16 |
| Position | 1h, 4h, 1d | 1:4:24 |

**Reference**: [TradeCiety MTF Analysis](https://tradeciety.com/how-to-perform-a-multiple-time-frame-analysis)

### 8.2 MTF Feature Aggregation

For each base timeframe, aggregate from higher timeframes:

```python
MTF_WINDOWS = {
    '5min': {
        '15min': 3,    # 3 x 5min bars
        '30min': 6,
        '1h': 12,
        '4h': 48,
        '1d': 288      # ~288 5min bars per day
    }
}
```

### 8.3 Features to Aggregate by Timeframe

| Timeframe | Key Features | Purpose |
|-----------|--------------|---------|
| 1m/5m | Microstructure, volume | Entry timing |
| 15m/30m | Momentum, volatility | Setup confirmation |
| 1h/4h | Trend, regime | Direction bias |
| 1d | Support/resistance, range | Major levels |

### 8.4 MTF Alignment Score

```python
def mtf_alignment_score(signals_by_tf):
    """
    Measures agreement across timeframes.
    58% win rate when aligned vs 39% when not.
    """
    signs = [np.sign(s) for s in signals_by_tf.values()]
    agreement = len(set(signs)) == 1
    strength = np.mean([abs(s) for s in signals_by_tf.values()])
    return agreement, strength
```

**Reference**: [Real Trading](https://realtrading.com/trading-blog/rule-of-three-multi-timeframe-analysis/)

---

## 9. Missing Features to Add

### High Priority (Academic Support + High Impact)

| Feature | Category | Implementation Effort |
|---------|----------|----------------------|
| Yang-Zhang Volatility | Volatility | Low |
| Rogers-Satchell Volatility | Volatility | Low |
| Bipower Variation | Volatility | Medium |
| Jump Detection | Volatility | Medium |
| VPIN | Microstructure | Medium |
| Rolling Hurst Exponent | Cross-temporal | Medium |
| HMM Regimes | Regime | High |
| Autocorrelation | Cross-temporal | Low |

### Medium Priority

| Feature | Category | Implementation Effort |
|---------|----------|----------------------|
| Spectral Features | Cross-temporal | Medium |
| Fractal Dimension | Cross-temporal | Medium |
| Vol-of-Vol | Volatility | Low |
| High-Low Amihud | Microstructure | Low |
| Composite Momentum | Momentum | Low |

### Low Priority (Nice to Have)

| Feature | Category | Implementation Effort |
|---------|----------|----------------------|
| Wavelet Coherence | Wavelets | Medium |
| Partial Autocorrelation | Cross-temporal | Low |
| MTF Alignment Score | MTF | Low |

---

## 10. Implementation Status

### Current Coverage

```
IMPLEMENTED (150+ features):
[x] Parkinson volatility
[x] Garman-Klass volatility
[x] Historical volatility (multiple periods)
[x] ATR
[x] Bollinger Bands
[x] Keltner Channels
[x] Higher moments (skewness, kurtosis)
[x] Amihud illiquidity
[x] Roll spread
[x] Kyle's lambda
[x] Corwin-Schultz spread
[x] Volume imbalance
[x] Trade intensity
[x] Price efficiency
[x] RSI
[x] MACD
[x] Stochastic
[x] Williams %R
[x] ROC
[x] CCI
[x] MFI
[x] Wavelet decomposition (all levels)
[x] Wavelet energy
[x] Wavelet volatility
[x] Wavelet trend strength
[x] Basic regime detection
[x] Moving averages (SMA, EMA)
[x] Trend indicators (ADX, etc.)
[x] Volume indicators (OBV, VWAP, etc.)
[x] Temporal features (hour, day, etc.)
[x] Price features (returns, gaps, etc.)

MISSING (Priority additions):
[ ] Yang-Zhang volatility      <- HIGH PRIORITY
[ ] Rogers-Satchell volatility <- HIGH PRIORITY
[ ] Bipower variation          <- MEDIUM PRIORITY
[ ] Jump detection             <- MEDIUM PRIORITY
[ ] VPIN                       <- HIGH PRIORITY
[ ] Rolling Hurst exponent     <- HIGH PRIORITY (expose from regime)
[ ] HMM regime states          <- HIGH PRIORITY
[ ] Autocorrelation features   <- MEDIUM PRIORITY
[ ] Spectral features          <- MEDIUM PRIORITY
[ ] Fractal dimension          <- LOW PRIORITY
```

### Recommended Next Steps

1. **Add Yang-Zhang and Rogers-Satchell volatility** - Simple formulas, immediate value
2. **Expose Hurst exponent as feature** - Already implemented in regime module
3. **Add autocorrelation features** - Simple rolling calculation
4. **Implement VPIN** - Requires volume bucket logic but high value
5. **Add HMM regime states** - Requires hmmlearn dependency

---

## References

### Academic Papers

1. Amihud, Y. (2002). "Illiquidity and Stock Returns." Journal of Financial Markets.
2. Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread."
3. Corwin, S.A. & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads."
4. Kyle, A.S. (1985). "Continuous Auctions and Insider Trading."
5. Barndorff-Nielsen, O.E. & Shephard, N. (2004). "Power and Bipower Variation."
6. Yang, D. & Zhang, Q. (2000). "Drift Independent Volatility Estimation."
7. Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the Rate of Return."
8. Garman, M.B. & Klass, M.J. (1980). "On the Estimation of Security Price Volatilities from Historical Data."
9. Rogers, L.C.G. & Satchell, S.E. (1991). "Estimating Variance from High, Low and Closing Prices."

### Online Resources

- [arXiv: Technical Indicators in ML Trading](https://arxiv.org/html/2412.15448v1/)
- [Portfolio Optimizer: Range-Based Volatility](https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/)
- [QuantStart: HMM Market Regime Detection](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Macrosynergy: Hurst Exponent](https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/)
- [MLFinLab: Volatility Estimators](https://www.mlfinlab.com/en/latest/feature_engineering/volatility_estimators.html)

---

*Document generated as part of Phase 1 feature engineering research for the ML Model Factory project.*
