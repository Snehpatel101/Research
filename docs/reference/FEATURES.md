# Feature Catalog

## Overview
The pipeline generates 50+ technical indicators across multiple categories. Features are computed on resampled 5-minute bars with proper forward-fill handling.

## Feature Categories

### 1. Price-Based Features
- **Returns**: Simple returns, log returns (1, 5, 15, 60 bar lookbacks)
- **Price Ratios**: close/open, high/low, (high-low)/close
- **Price Momentum**: Rate of change over multiple windows

### 2. Volatility Features
- **ATR (Average True Range)**: 14-period ATR and ATR ratio
- **Parkinson**: High-low range-based volatility estimator
- **Garman-Klass**: OHLC-based volatility (more efficient than close-to-close)
- **Rolling Std**: Returns volatility over 20/60 periods

### 3. Volume Features
- **Volume Ratios**: Current volume vs. rolling mean (20, 60 periods)
- **Volume-Price Correlation**: 20-period rolling correlation
- **Volume Momentum**: Rate of change in volume

### 4. Moving Averages
- **SMA**: 20, 50, 200-period simple moving averages
- **EMA**: 12, 26-period exponential moving averages
- **Price-MA Distance**: (close - MA) / MA for mean reversion signals

### 5. Momentum Indicators
- **RSI**: 14-period Relative Strength Index
- **MACD**: 12/26/9 MACD, signal line, histogram
- **Stochastic**: %K and %D oscillators
- **ROC**: Rate of change over multiple periods

### 6. Trend Indicators
- **ADX**: Average Directional Index (14-period)
- **Bollinger Bands**: 20-period bands, width, %B position
- **Donchian Channels**: 20-period high/low channels

### 7. Pattern Features
- **Candle Patterns**: Doji, hammer, engulfing (via heuristics)
- **Support/Resistance**: Distance to recent highs/lows
- **Pivot Points**: Standard, Fibonacci, Woodie pivots

### 8. Statistical Features
- **Skewness**: Returns distribution skew (20/60 periods)
- **Kurtosis**: Returns tail heaviness (20/60 periods)
- **Autocorrelation**: Returns persistence at lag 1, 5

## Feature Selection

### Correlation Filtering
- **Threshold**: 0.70 (aggressive pruning)
- **Method**: Keep most interpretable feature from correlated groups
- **Rationale**: Reduce multicollinearity, improve stability

### Variance Filtering
- **Threshold**: 0.01
- **Purpose**: Remove near-constant features with no discriminative power

### Genetic Algorithm Optimization
- **Objective**: Maximize Sharpe ratio - transaction cost penalty
- **Penalty**: 0.5 bps per trade for MES, 1.0 bps for MGC
- **Generations**: 50
- **Population**: 100
- **Selection**: Tournament selection (k=3)

## Feature Engineering Pipeline

1. **Stage 3**: Compute all raw features from OHLCV
2. **Stage 4**: Apply GA optimization for feature subset selection
3. **Stage 7.5**: Scale features using RobustScaler (IQR normalization)
4. **Validation**: Check for look-ahead bias, NaN leakage, stationarity

## Configuration

Features are configured in:
- `/home/jake/Desktop/Research/src/config/features.py` - Feature selection thresholds
- `/home/jake/Desktop/Research/src/config/feature_sets.py` - Feature set definitions
- `/home/jake/Desktop/Research/src/stages/features/engineer.py` - Feature computation logic

## Cross-Asset Features (Disabled)

Cross-asset features (MES-MGC correlations, spreads) were removed in Dec 2025 due to:
- Length validation complexity when symbols have different bar counts
- Minimal predictive value (empirical testing)
- Increased maintenance burden

May be re-enabled in Phase 2 with proper sequence alignment.

## References

- Feature correlation analysis: `/home/jake/Desktop/Research/results/feature_selection_report.json`
- GA optimization results: `/home/jake/Desktop/Research/config/ga_results/optimization_summary.json`
- Feature importance (post-training): Will be available after Phase 2 model training
