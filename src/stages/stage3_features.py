"""
Stage 3: Feature Engineering

This module has been refactored into src/stages/features/
This file remains for backward compatibility.

The feature engineering functionality is now modular:
- features/constants.py: Annualization constants
- features/numba_functions.py: High-performance Numba calculations
- features/price_features.py: Returns and price ratios
- features/moving_averages.py: SMA, EMA features
- features/momentum.py: RSI, MACD, Stochastic, etc.
- features/volatility.py: ATR, Bollinger Bands, Keltner, etc.
- features/volume.py: OBV, VWAP, volume features
- features/trend.py: ADX, Supertrend
- features/temporal.py: Time encoding features
- features/regime.py: Market regime indicators
- features/cross_asset.py: Cross-asset features (MES-MGC)
- features/engineer.py: Main FeatureEngineer class

Usage:
    from stages.features import FeatureEngineer
    engineer = FeatureEngineer(input_dir='data/clean', output_dir='data/features')
    results = engineer.process_directory()
"""

# Re-export everything from the features package for backward compatibility
from src.stages.features import (
    # Main class
    FeatureEngineer,
    main,
    # Constants
    ANNUALIZATION_FACTOR,
    BARS_PER_DAY,
    TRADING_DAYS_PER_YEAR,
    # Numba functions
    calculate_sma_numba,
    calculate_ema_numba,
    calculate_rsi_numba,
    calculate_atr_numba,
    calculate_stochastic_numba,
    calculate_rolling_correlation_numba,
    calculate_rolling_beta_numba,
    calculate_adx_numba,
    # Price features
    add_returns,
    add_price_ratios,
    # Moving averages
    add_sma,
    add_ema,
    # Momentum
    add_rsi,
    add_macd,
    add_stochastic,
    add_williams_r,
    add_roc,
    add_cci,
    add_mfi,
    # Volatility
    add_atr,
    add_bollinger_bands,
    add_keltner_channels,
    add_historical_volatility,
    add_parkinson_volatility,
    add_garman_klass_volatility,
    # Volume
    add_volume_features,
    add_vwap,
    add_obv,
    # Trend
    add_adx,
    add_supertrend,
    # Temporal
    add_temporal_features,
    add_session_features,
    # Regime
    add_regime_features,
    add_volatility_regime,
    add_trend_regime,
    # Cross Asset
    add_cross_asset_features,
)

__all__ = [
    'FeatureEngineer',
    'main',
    'ANNUALIZATION_FACTOR',
    'BARS_PER_DAY',
    'TRADING_DAYS_PER_YEAR',
    'calculate_sma_numba',
    'calculate_ema_numba',
    'calculate_rsi_numba',
    'calculate_atr_numba',
    'calculate_stochastic_numba',
    'calculate_rolling_correlation_numba',
    'calculate_rolling_beta_numba',
    'calculate_adx_numba',
    'add_returns',
    'add_price_ratios',
    'add_sma',
    'add_ema',
    'add_rsi',
    'add_macd',
    'add_stochastic',
    'add_williams_r',
    'add_roc',
    'add_cci',
    'add_mfi',
    'add_atr',
    'add_bollinger_bands',
    'add_keltner_channels',
    'add_historical_volatility',
    'add_parkinson_volatility',
    'add_garman_klass_volatility',
    'add_volume_features',
    'add_vwap',
    'add_obv',
    'add_adx',
    'add_supertrend',
    'add_temporal_features',
    'add_session_features',
    'add_regime_features',
    'add_volatility_regime',
    'add_trend_regime',
    'add_cross_asset_features',
]

if __name__ == '__main__':
    main()
