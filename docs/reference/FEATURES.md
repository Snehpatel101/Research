# Feature Catalog

## Overview
Features are generated during Stage 3 (feature_engineering). The pipeline produces 150+ features depending on:
- Feature set selection (`feature_set` in `PipelineConfig`)
- Multi-timeframe (MTF) mode and timeframes
- Cross-asset features (disabled by default)

Use the run artifacts to see the exact feature list:
- `runs/<run_id>/artifacts/feature_set_manifest.json`
- `runs/<run_id>/artifacts/dataset_manifest.json`

## Core Feature Groups

### Price and Returns
- Simple and log returns
- OHLC ranges (high-low, close-open)
- Body/wick ratios

### Moving Averages
- SMA/EMA families
- Price-to-MA distances and ratios

### Momentum
- RSI, MACD, ROC, stochastic
- Directional momentum deltas

### Volatility
- ATR and ATR-derived metrics
- Bollinger-band position/width
- Rolling volatility measures

### Volume
- Volume z-scores and rolling ratios
- Volume/price interaction metrics

### Trend and Regime
- Trend flags from MA alignment
- Volatility regime features
- Optional structure regime (advanced detectors)

### Temporal
- Hour/day-of-week cyclic encodings
- Session boundary flags

### Wavelet Features
Multi-scale decomposition for cycle and trend detection:
- Wavelet coefficients at multiple levels (approximation + detail)
- Energy distribution across scales
- Trend extraction via low-frequency components
- Noise filtering via high-frequency removal

Configuration: `src/phase1/stages/features/wavelets.py`

### Microstructure Features
Market microstructure proxies from OHLCV data:
- **Bid-ask spread proxy:** Estimated from high-low range
- **Order flow imbalance:** Inferred from close position within range
- **Volume-weighted price deviation:** VWAP-based metrics
- **Trade intensity:** Volume per price movement
- **Amihud illiquidity:** Price impact of volume

Configuration: `src/phase1/stages/features/microstructure.py`

## Multi-Timeframe (MTF) Features
MTF features are optional and configurable (bars, indicators, or both). When enabled, higher timeframe features are shifted to prevent lookahead bias.

Supported timeframes: 5min, 15min, 30min, 1h, 4h, daily

## Cross-Asset Features (disabled by default)
Cross-asset features (MES/MGC correlations, spread, beta) exist but are disabled by default in `src/phase1/config/features.py`. Enable only when both symbols are present and aligned.

## Feature Selection and Validation
Feature selection is part of Stage 8 validation:
- Correlation filtering
- Variance thresholding
- Optional stationarity checks

Thresholds are configurable; see `src/phase1/stages/validation/run.py` and `src/phase1/config/features.py`.

## Configuration References

- Feature sets: `src/phase1/config/feature_sets.py`
- Feature config: `src/phase1/config/features.py`
- Feature engineering: `src/phase1/stages/features/`
- Wavelet features: `src/phase1/stages/features/wavelets.py`
- Microstructure features: `src/phase1/stages/features/microstructure.py`
- Validation: `src/phase1/stages/validation/`
