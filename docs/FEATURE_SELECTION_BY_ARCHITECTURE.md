# Feature Selection Guide by Model Architecture

## Executive Summary

This document provides statistically-grounded feature set definitions for each model architecture in the ML factory. Feature selection is critical for model performance - too few features causes underfitting, too many causes overfitting and computational overhead.

**Key Insight**: Different architectures have fundamentally different inductive biases. A feature set that works well for XGBoost may harm LSTM performance. This guide provides architecture-specific recommendations based on:

1. Statistical properties of the features
2. Model architecture constraints
3. Bias-variance tradeoff for each model type
4. Correlation structure and redundancy analysis

---

## Available Feature Categories (~180 total)

Based on the codebase analysis, features are generated in these categories:

### 1. Price & Returns (12 features)
```
return_1, return_5, return_10, return_20
log_return_1, log_return_5, log_return_10, log_return_20
range_pct, hl_ratio, co_ratio, clv
```
**Properties**: Stationary, bounded variance, no scaling needed

### 2. Momentum Oscillators (18 features)
```
rsi_14, rsi_overbought, rsi_oversold
stoch_k, stoch_d, stoch_overbought, stoch_oversold
williams_r
roc_5, roc_10, roc_20
cci_20
mfi_14
macd_line, macd_signal, macd_hist, macd_cross_up, macd_cross_down
```
**Properties**: Most bounded [0,100] or [-100,100], mean-reverting signals

### 3. Volatility (22 features)
```
atr_7, atr_14, atr_21, atr_pct_7, atr_pct_14, atr_pct_21
hvol_10, hvol_20, hvol_60
parkinson_vol, gk_vol, rs_vol, yz_vol
bb_middle, bb_upper, bb_lower, bb_width, bb_position, close_bb_zscore
kc_middle, kc_upper, kc_lower, kc_position, close_kc_atr_dev
return_skew_20, return_skew_60, return_kurt_20, return_kurt_60
```
**Properties**: Some bounded (positions), some unbounded (raw volatility)

### 4. Trend (8 features)
```
adx_14, adx_di_plus, adx_di_minus
supertrend, supertrend_direction
sma_10, sma_20, sma_50 (positions relative to price)
ema_9, ema_21, ema_50 (positions relative to price)
```
**Properties**: ADX bounded [0,100], trend signals binary/continuous

### 5. Volume (10 features)
```
obv, obv_zscore
vwap, price_to_vwap
volume_ratio_20, volume_zscore_20
dollar_volume, dollar_volume_zscore
volume_sma_20, volume_ema_9
```
**Properties**: OBV unbounded, ratios/zscores bounded

### 6. Temporal (12 features)
```
hour_sin, hour_cos
minute_sin, minute_cos
dayofweek_sin, dayofweek_cos
session_id, is_rth
trend_regime, volatility_regime
quarter_hour_sin, quarter_hour_cos
```
**Properties**: Sin/cos bounded [-1,1], regimes categorical

### 7. Wavelets (24 features with default settings)
```
wavelet_close_approx, wavelet_close_d1, wavelet_close_d2, wavelet_close_d3
wavelet_volume_approx, wavelet_volume_d1, wavelet_volume_d2, wavelet_volume_d3
wavelet_close_energy_approx, wavelet_close_energy_d1, wavelet_close_energy_d2, wavelet_close_energy_d3
wavelet_volume_energy_approx, wavelet_volume_energy_d1, wavelet_volume_energy_d2, wavelet_volume_energy_d3
wavelet_close_energy_ratio
wavelet_volume_energy_ratio
wavelet_close_volatility
wavelet_close_trend_strength, wavelet_close_trend_direction
```
**Properties**: Normalized coefficients, energy ratios bounded [0,1]

### 8. Microstructure (22 features)
```
micro_amihud, micro_amihud_10, micro_amihud_20
micro_roll_spread, micro_roll_spread_pct
micro_kyle_lambda
micro_cs_spread
micro_rel_spread, micro_rel_spread_10, micro_rel_spread_20
micro_volume_imbalance, micro_cum_imbalance_20
micro_trade_intensity_20, micro_trade_intensity_50
micro_efficiency_10, micro_efficiency_20
micro_vol_ratio
```
**Properties**: Ratios and proxies, mostly bounded

### 9. MTF Features (~30-50 depending on config)
```
{indicator}_{timeframe} format, e.g.:
rsi_14_15m, rsi_14_1h
macd_hist_15m, macd_hist_1h
atr_14_15m, atr_14_1h
close_15m, high_15m, low_15m (if include_ohlcv=True)
```
**Properties**: Same as base indicators but from higher timeframes

---

## Correlation Analysis & Redundancy

### High Correlation Groups (>0.80)

Based on statistical analysis of typical financial data:

**Group 1: RSI-like oscillators**
- `rsi_14`, `stoch_k`, `williams_r` (corr > 0.85)
- **Keep**: `rsi_14` (most interpretable)
- **Alternative**: `stoch_k` (faster signals)

**Group 2: Volatility measures**
- `hvol_20`, `parkinson_vol`, `gk_vol`, `rs_vol`, `yz_vol` (corr > 0.90)
- **Keep**: `yz_vol` (most comprehensive), `hvol_20` (simplest)

**Group 3: ATR variants**
- `atr_7`, `atr_14`, `atr_21` (corr > 0.95)
- **Keep**: `atr_14` (standard), `atr_pct_14` (normalized)

**Group 4: Moving averages**
- `sma_20`, `ema_21`, `bb_middle`, `kc_middle` (corr > 0.99)
- **Keep**: `bb_position`, `kc_position` (normalized forms)

**Group 5: Return lags**
- `return_1`, `log_return_1` (corr > 0.99)
- **Keep**: `log_return_*` (better statistical properties)

**Group 6: Volume measures**
- `obv`, `dollar_volume` (corr varies by regime)
- **Keep both**: Different information content

### Deduplication Strategy

```python
# Features to EXCLUDE (redundant with others in same group)
REDUNDANT_FEATURES = [
    # RSI group - keep rsi_14 only
    "stoch_k", "stoch_d", "williams_r",
    # Volatility group - keep yz_vol, hvol_20, atr_14
    "parkinson_vol", "gk_vol", "rs_vol",
    "hvol_10", "hvol_60",  # keep hvol_20
    "atr_7", "atr_21",  # keep atr_14
    # MA group - keep normalized versions
    "sma_10", "sma_20", "sma_50",
    "ema_9", "ema_21", "ema_50",
    "bb_middle", "bb_upper", "bb_lower",
    "kc_middle", "kc_upper", "kc_lower",
    # Returns - keep log versions
    "return_1", "return_5", "return_10", "return_20",
]
```

---

## Feature Selection Methods by Model Type

### 1. Tree-Based Models (XGBoost, LightGBM, CatBoost)

**Recommended Method**: MDI (Mean Decrease in Impurity) + SHAP

**Why**:
- Trees naturally handle feature interactions
- MDI provides fast, stable importance scores
- SHAP adds interpretability for feature selection validation

**Selection Process**:
1. Train with all ~150 features
2. Compute MDI importance
3. Remove features with importance < 0.001 (noise)
4. Validate top features with SHAP dependence plots

**Note**: Tree models are robust to correlated features - no need for aggressive deduplication.

### 2. RNNs (LSTM, GRU)

**Recommended Method**: MDA (Mean Decrease in Accuracy) + Correlation Filtering

**Why**:
- RNNs suffer from vanishing gradients with high-dimensional inputs
- Correlated features create redundant hidden state updates
- MDA respects temporal structure better than MDI

**Selection Process**:
1. Remove highly correlated features (threshold 0.80)
2. Train baseline LSTM with all remaining features
3. Compute MDA via walk-forward permutation importance
4. Select top 40-50 features by stability across folds

### 3. CNNs (TCN, InceptionTime, ResNet1D)

**Recommended Method**: Mutual Information + Scale Consistency Check

**Why**:
- CNNs learn spatial patterns - features should be scale-consistent
- Mutual information captures non-linear dependencies
- Filters work best on similarly-scaled features

**Selection Process**:
1. Normalize all features to similar scale
2. Remove near-constant features (var < 0.01)
3. Compute mutual information with target
4. Select features with MI > threshold AND consistent scale

### 4. Transformers (PatchTST)

**Recommended Method**: Minimal Features + Let Model Learn

**Why**:
- Transformers learn representations from raw data
- Pre-engineered features may conflict with learned attention
- Attention dilution with too many features

**Selection Process**:
1. Use minimal feature set (returns + temporal encoding)
2. Let transformer learn patterns from raw sequences
3. Only add features that provide non-redundant signal (e.g., volume ratio)

### 5. N-BEATS

**Recommended Method**: Univariate / Near-Univariate

**Why**:
- N-BEATS designed for automatic decomposition
- Adding features defeats its purpose
- Trend/seasonality blocks learn these internally

**Selection Process**:
1. Primary: Use close price only (univariate)
2. Alternative: Close + volume (bivariate)
3. No engineered features

---

## Optimal Feature Counts by Architecture

Based on bias-variance tradeoff and empirical results:

| Model Type | Optimal Count | Range | Rationale |
|------------|---------------|-------|-----------|
| **XGBoost/LightGBM/CatBoost** | 80-120 | 60-150 | High capacity, regularization handles many features |
| **LSTM/GRU** | 35-45 | 30-50 | Limited hidden state capacity, curse of dimensionality |
| **TCN** | 45-55 | 40-60 | Dilated convolutions need consistent feature space |
| **InceptionTime** | 50-60 | 45-70 | Multiple filter sizes, moderate feature tolerance |
| **ResNet1D** | 45-55 | 40-60 | Skip connections help, but still limited |
| **PatchTST** | 15-25 | 10-30 | Attention dilution with many features |
| **N-BEATS** | 1-2 | 1-5 | Designed for univariate decomposition |

### Sample Size Considerations

Minimum samples-to-features ratios:

| Ratio | Description | Application |
|-------|-------------|-------------|
| 10:1 | Absolute minimum | Only for tree models with strong regularization |
| 20:1 | Recommended | Neural networks with dropout |
| 50:1 | Conservative | High-stakes production models |

Example: With 50,000 training samples:
- Tree models: max ~500 features (10:1), recommended ~150 (20:1)
- Neural models: max ~250 features (20:1), recommended ~100 (50:1)
- Transformers: max ~100 features, recommended ~25

---

## Concrete Feature Set Definitions

### Boosting Models (XGBoost, LightGBM, CatBoost) - 78 features

```python
BOOSTING_FEATURES = [
    # Returns (8)
    "log_return_1", "log_return_5", "log_return_10", "log_return_20",
    "roc_5", "roc_10", "roc_20", "clv",

    # Momentum (10)
    "rsi_14", "rsi_overbought", "rsi_oversold",
    "macd_line", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d",
    "cci_20", "mfi_14",

    # Volatility (12)
    "atr_14", "atr_pct_14",
    "hvol_20", "yz_vol",
    "bb_width", "bb_position", "close_bb_zscore",
    "kc_position", "close_kc_atr_dev",
    "return_skew_20", "return_kurt_20",
    "micro_vol_ratio",

    # Trend (6)
    "adx_14", "adx_di_plus", "adx_di_minus",
    "supertrend", "supertrend_direction",
    "wavelet_close_trend_strength",

    # Volume (8)
    "volume_ratio_20", "volume_zscore_20",
    "obv_zscore", "price_to_vwap",
    "micro_amihud_20", "micro_kyle_lambda",
    "micro_trade_intensity_20", "micro_volume_imbalance",

    # Temporal (8)
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "is_rth", "session_id",
    "trend_regime", "volatility_regime",

    # Wavelets (10)
    "wavelet_close_approx", "wavelet_close_d1", "wavelet_close_d2",
    "wavelet_close_energy_ratio",
    "wavelet_volume_approx", "wavelet_volume_d1",
    "wavelet_close_volatility",
    "wavelet_close_trend_direction",
    "wavelet_volume_energy_ratio",
    "wavelet_volume_energy_d1",

    # Microstructure (8)
    "micro_roll_spread_pct", "micro_cs_spread",
    "micro_rel_spread_20", "micro_efficiency_10",
    "micro_efficiency_20", "micro_cum_imbalance_20",
    "micro_amihud", "micro_trade_intensity_50",

    # MTF (8) - if enabled
    "rsi_14_15m", "rsi_14_1h",
    "macd_hist_15m", "macd_hist_1h",
    "atr_pct_14_15m", "atr_pct_14_1h",
    "bb_position_15m", "bb_position_1h",
]
```

### LSTM/GRU - 43 features

```python
LSTM_GRU_FEATURES = [
    # Returns (4) - stationary, good for sequences
    "log_return_1", "log_return_5", "log_return_10", "log_return_20",

    # Bounded Oscillators (8) - normalized [0,100]
    "rsi_14",
    "stoch_k", "stoch_d",
    "mfi_14", "cci_20",
    "bb_position", "kc_position",
    "williams_r",

    # Volatility Ratios (6) - normalized
    "atr_pct_14",
    "hvol_20",
    "close_bb_zscore", "close_kc_atr_dev",
    "micro_vol_ratio",
    "return_skew_20",

    # Volume Ratios (4)
    "volume_ratio_20", "volume_zscore_20",
    "price_to_vwap", "obv_zscore",

    # Trend Signals (4)
    "adx_14",
    "macd_hist",  # normalized version
    "supertrend_direction",
    "wavelet_close_trend_direction",

    # Temporal (6) - cyclical encoding
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "trend_regime", "volatility_regime",

    # Wavelets (7) - multi-scale patterns
    "wavelet_close_approx",
    "wavelet_close_d1", "wavelet_close_d2", "wavelet_close_d3",
    "wavelet_close_energy_ratio",
    "wavelet_close_volatility",
    "wavelet_close_trend_strength",

    # Microstructure (4)
    "micro_volume_imbalance",
    "micro_efficiency_20",
    "micro_rel_spread_20",
    "micro_trade_intensity_20",
]
```

### TCN - 50 features

```python
TCN_FEATURES = [
    # Returns (5) - core for pattern recognition
    "log_return_1", "log_return_5", "log_return_10", "log_return_20",
    "clv",

    # Momentum (9)
    "rsi_14",
    "stoch_k", "stoch_d",
    "williams_r", "cci_20", "mfi_14",
    "macd_hist",
    "roc_5", "roc_10",

    # Volatility (10)
    "atr_pct_14",
    "hvol_20",
    "bb_position", "close_bb_zscore",
    "kc_position", "close_kc_atr_dev",
    "return_skew_20", "return_kurt_20",
    "micro_vol_ratio",
    "yz_vol",

    # Autocorrelation (3) - captures serial dependence
    "return_autocorr_1", "return_autocorr_5", "return_autocorr_10",

    # Volume (5)
    "volume_ratio_20", "volume_zscore_20",
    "price_to_vwap", "obv_zscore",
    "micro_volume_imbalance",

    # Wavelets (8) - multi-scale for dilated convolutions
    "wavelet_close_approx",
    "wavelet_close_d1", "wavelet_close_d2", "wavelet_close_d3",
    "wavelet_close_energy_ratio",
    "wavelet_volume_energy_ratio",
    "wavelet_close_volatility",
    "wavelet_close_trend_strength",

    # Temporal (6)
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "trend_regime", "volatility_regime",

    # Microstructure (4)
    "micro_efficiency_10", "micro_efficiency_20",
    "micro_rel_spread_20",
    "micro_trade_intensity_20",
]
```

### PatchTST - 23 features

```python
PATCHTST_FEATURES = [
    # Minimal returns (4) - let transformer learn patterns
    "log_return_1", "log_return_5", "log_return_10", "log_return_20",

    # Core oscillator (1)
    "rsi_14",

    # Key positions (3) - normalized
    "bb_position", "kc_position", "price_to_vwap",

    # Volume signal (2)
    "volume_ratio_20", "micro_volume_imbalance",

    # Temporal encoding (6) - critical for attention
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "is_rth",
    "volatility_regime",

    # Volatility context (3)
    "atr_pct_14", "micro_vol_ratio", "hvol_20",

    # Trend context (2)
    "supertrend_direction",
    "wavelet_close_trend_direction",

    # Efficiency (2)
    "micro_efficiency_10", "micro_efficiency_20",
]
```

### N-BEATS - 1-2 features (Univariate focus)

```python
NBEATS_FEATURES = [
    # Primary: univariate close price
    "close",  # or log_return_1 for stationary version

    # Optional: volume for bivariate
    # "volume_ratio_20",
]
```

**Important**: N-BEATS performs automatic trend-seasonality decomposition. Adding engineered features typically degrades performance by:
1. Introducing redundant information
2. Conflicting with internal decomposition blocks
3. Increasing model complexity without benefit

### InceptionTime - 55 features

```python
INCEPTIONTIME_FEATURES = [
    # Returns (5)
    "log_return_1", "log_return_5", "log_return_10", "log_return_20",
    "clv",

    # Momentum (10)
    "rsi_14",
    "stoch_k", "stoch_d",
    "williams_r", "cci_20", "mfi_14",
    "macd_line", "macd_signal", "macd_hist",
    "roc_10",

    # Volatility (12)
    "atr_14", "atr_pct_14",
    "hvol_20", "yz_vol",
    "bb_width", "bb_position", "close_bb_zscore",
    "kc_position", "close_kc_atr_dev",
    "return_skew_20", "return_kurt_20",
    "micro_vol_ratio",

    # Volume (6)
    "volume_ratio_20", "volume_zscore_20",
    "obv_zscore", "price_to_vwap",
    "micro_volume_imbalance",
    "micro_trade_intensity_20",

    # Wavelets (10) - multiple scales for inception modules
    "wavelet_close_approx",
    "wavelet_close_d1", "wavelet_close_d2", "wavelet_close_d3",
    "wavelet_close_energy_approx", "wavelet_close_energy_d1",
    "wavelet_close_energy_ratio",
    "wavelet_volume_approx",
    "wavelet_close_volatility",
    "wavelet_close_trend_strength",

    # Temporal (6)
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "trend_regime", "volatility_regime",

    # Microstructure (6)
    "micro_amihud_20",
    "micro_roll_spread_pct",
    "micro_efficiency_10", "micro_efficiency_20",
    "micro_rel_spread_20",
    "micro_cs_spread",
]
```

### ResNet1D - 48 features

```python
RESNET1D_FEATURES = [
    # Returns (4)
    "log_return_1", "log_return_5", "log_return_10", "log_return_20",

    # Momentum (8)
    "rsi_14",
    "stoch_k", "stoch_d",
    "cci_20", "mfi_14",
    "macd_hist",
    "roc_5", "roc_10",

    # Volatility (10)
    "atr_pct_14",
    "hvol_20", "yz_vol",
    "bb_position", "close_bb_zscore",
    "kc_position", "close_kc_atr_dev",
    "return_skew_20", "return_kurt_20",
    "micro_vol_ratio",

    # Volume (5)
    "volume_ratio_20", "volume_zscore_20",
    "price_to_vwap", "obv_zscore",
    "micro_volume_imbalance",

    # Wavelets (8) - residual learning benefits from multi-scale
    "wavelet_close_approx",
    "wavelet_close_d1", "wavelet_close_d2", "wavelet_close_d3",
    "wavelet_close_energy_ratio",
    "wavelet_close_volatility",
    "wavelet_close_trend_strength",
    "wavelet_close_trend_direction",

    # Temporal (6)
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "trend_regime", "volatility_regime",

    # Microstructure (7)
    "micro_efficiency_10", "micro_efficiency_20",
    "micro_rel_spread_20",
    "micro_trade_intensity_20",
    "micro_amihud_20",
    "micro_roll_spread_pct",
    "micro_cs_spread",
]
```

---

## Implementation in Existing Codebase

The feature sets defined here align with the existing `feature_sets.py` configuration. To use:

### Option 1: Use Existing Feature Set Definitions

```python
from src.phase1.config.feature_sets import FEATURE_SET_DEFINITIONS

# For boosting models
feature_set = FEATURE_SET_DEFINITIONS["boosting_optimal"]

# For neural models
feature_set = FEATURE_SET_DEFINITIONS["neural_optimal"]

# For TCN
feature_set = FEATURE_SET_DEFINITIONS["tcn_optimal"]

# For transformers
feature_set = FEATURE_SET_DEFINITIONS["patchtst_optimal"]
```

### Option 2: Resolve Feature Set for DataFrame

```python
from src.phase1.utils.feature_sets import resolve_feature_set
from src.phase1.config.feature_sets import FEATURE_SET_DEFINITIONS

# Get actual column list for a DataFrame
df = pd.read_parquet("data/splits/scaled/train.parquet")
feature_columns = resolve_feature_set(df, FEATURE_SET_DEFINITIONS["tcn_optimal"])
```

### Option 3: CLI Usage

```bash
# Train with model-specific feature set
python scripts/train_model.py --model tcn --horizon 20 --feature-set tcn_optimal
python scripts/train_model.py --model xgboost --horizon 20 --feature-set boosting_optimal
python scripts/train_model.py --model patchtst --horizon 20 --feature-set patchtst_optimal
```

---

## Walk-Forward Feature Selection

For dynamic feature selection that adapts to data, use the existing `WalkForwardFeatureSelector`:

```python
from src.cross_validation.feature_selector import WalkForwardFeatureSelector

# Initialize selector
selector = WalkForwardFeatureSelector(
    n_features_to_select=50,  # Target feature count
    selection_method="mda",   # MDA more reliable than MDI
    min_feature_frequency=0.6,  # Feature must appear in 60% of folds
)

# Run walk-forward selection
cv_splits = list(purged_kfold.split(X, y))
result = selector.select_features_walkforward(X, y, cv_splits)

# Get stable features
stable_features = result.stable_features
print(f"Selected {len(stable_features)} stable features")
```

---

## Summary Recommendations

| Model | Feature Count | Key Feature Types | Avoid |
|-------|--------------|-------------------|-------|
| **XGBoost/LightGBM/CatBoost** | 78 | All types, MTF included | None (robust to all) |
| **LSTM/GRU** | 43 | Bounded oscillators, ratios | Raw prices, unbounded |
| **TCN** | 50 | Multi-scale wavelets, autocorr | Highly correlated groups |
| **PatchTST** | 23 | Minimal returns + temporal | Engineered indicators |
| **N-BEATS** | 1-2 | Close only (univariate) | ALL engineered features |
| **InceptionTime** | 55 | Multi-scale, diverse types | Single-scale redundancy |
| **ResNet1D** | 48 | Wavelets, momentum | Feature redundancy |

### Key Principles

1. **Trees are robust**: XGBoost/LightGBM can handle 100+ features with internal feature selection
2. **RNNs need bounded features**: Avoid raw prices, use normalized ratios
3. **CNNs need scale consistency**: All features should be similarly scaled
4. **Transformers want minimal preprocessing**: Let attention learn patterns
5. **N-BEATS is special**: Designed for univariate, adding features hurts
6. **Always deduplicate**: Remove highly correlated features for neural models
7. **Walk-forward for production**: Use `WalkForwardFeatureSelector` for adaptive selection
