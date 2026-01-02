"""
Feature Engineering Package.

This package provides comprehensive feature engineering for financial time series,
generating 50+ technical indicators across multiple categories.

Main Components:
    FeatureEngineer: Main class for orchestrating feature engineering
    constants: Annualization factors and time constants
    numba_functions: High-performance Numba-optimized calculations

Feature Categories:
    price_features: Returns and price ratios
    moving_averages: SMA, EMA, and price-to-MA ratios
    momentum: RSI, MACD, Stochastic, Williams %R, ROC, CCI, MFI
    volatility: ATR, Bollinger Bands, Keltner Channels, historical volatility
    volume: OBV, VWAP, volume ratios
    trend: ADX, Supertrend
    temporal: Time encoding, trading sessions
    regime: Volatility and trend regime classification
    wavelets: Multi-scale wavelet decomposition (DWT coefficients, energy, trend)

Note: Each symbol is processed independently (no cross-symbol correlation).

Example:
    >>> from stages.features import FeatureEngineer
    >>> engineer = FeatureEngineer(input_dir='data/clean', output_dir='data/features')
    >>> results = engineer.process_directory()
"""

# Main class
# MTF Features
from ..mtf import (
    MTF_TIMEFRAMES,
    MTFFeatureGenerator,
    add_mtf_features,
    validate_mtf_alignment,
)
from .cli import main

# Constants
from .constants import (
    ANNUALIZATION_FACTOR,
    ANNUALIZATION_FACTOR_MAP,
    BARS_PER_DAY,
    BARS_PER_DAY_MAP,
    TIMEFRAME_MINUTES,
    TRADING_DAYS_PER_YEAR,
    TRADING_HOURS_EXTENDED,
    TRADING_HOURS_REGULAR,
    get_annualization_factor,
    get_bars_per_day,
)
from .engineer import FeatureEngineer

# Feature functions - Microstructure
from .microstructure import (
    add_amihud_illiquidity,
    add_corwin_schultz_spread,
    add_kyle_lambda,
    add_microstructure_features,
    add_price_efficiency,
    add_realized_volatility_ratio,
    add_relative_spread,
    add_roll_spread,
    add_trade_intensity,
    add_volume_imbalance,
)

# Feature functions - Momentum
from .momentum import (
    add_cci,
    add_macd,
    add_mfi,
    add_roc,
    add_rsi,
    add_stochastic,
    add_williams_r,
)

# Feature functions - Moving Averages
from .moving_averages import (
    add_ema,
    add_sma,
)

# Numba functions
from .numba_functions import (
    calculate_adx_numba,
    calculate_atr_numba,
    calculate_ema_numba,
    calculate_rsi_numba,
    calculate_sma_numba,
    calculate_stochastic_numba,
)

# Feature functions - Price
from .price_features import (
    add_price_ratios,
    add_returns,
)

# Feature functions - Regime
from .regime import (
    add_regime_features,
    add_trend_regime,
    add_volatility_regime,
)

# Period scaling
from .scaling import (
    BASE_TIMEFRAME_MINUTES,
    DEFAULT_BASE_PERIODS,
    PeriodScaler,
    create_period_config,
    get_scaled_periods,
    get_timeframe_minutes,
    get_unique_scaled_periods,
    scale_period,
)

# Feature functions - Temporal
from .temporal import (
    add_session_features,
    add_temporal_features,
)

# Feature functions - Trend
from .trend import (
    add_adx,
    add_supertrend,
)

# Feature functions - Volatility
from .volatility import (
    add_atr,
    add_bollinger_bands,
    add_garman_klass_volatility,
    add_historical_volatility,
    add_keltner_channels,
    add_parkinson_volatility,
)

# Feature functions - Volume
from .volume import (
    add_obv,
    add_volume_features,
    add_vwap,
)

# Feature functions - Wavelets
from .wavelets import (
    DEFAULT_LEVEL,
    DEFAULT_WAVELET,
    DEFAULT_WINDOW,
    PYWT_AVAILABLE,
    SUPPORTED_WAVELETS,
    add_wavelet_coefficients,
    add_wavelet_energy,
    add_wavelet_features,
    add_wavelet_trend_strength,
    add_wavelet_volatility,
)

__all__ = [
    # Main class
    "FeatureEngineer",
    "main",
    # Constants
    "ANNUALIZATION_FACTOR",
    "ANNUALIZATION_FACTOR_MAP",
    "BARS_PER_DAY",
    "BARS_PER_DAY_MAP",
    "TIMEFRAME_MINUTES",
    "TRADING_DAYS_PER_YEAR",
    "TRADING_HOURS_EXTENDED",
    "TRADING_HOURS_REGULAR",
    "get_annualization_factor",
    "get_bars_per_day",
    # Numba functions
    "calculate_sma_numba",
    "calculate_ema_numba",
    "calculate_rsi_numba",
    "calculate_atr_numba",
    "calculate_stochastic_numba",
    "calculate_adx_numba",
    # Price features
    "add_returns",
    "add_price_ratios",
    # Moving averages
    "add_sma",
    "add_ema",
    # Momentum
    "add_rsi",
    "add_macd",
    "add_stochastic",
    "add_williams_r",
    "add_roc",
    "add_cci",
    "add_mfi",
    # Volatility
    "add_atr",
    "add_bollinger_bands",
    "add_keltner_channels",
    "add_historical_volatility",
    "add_parkinson_volatility",
    "add_garman_klass_volatility",
    # Volume
    "add_volume_features",
    "add_vwap",
    "add_obv",
    # Trend
    "add_adx",
    "add_supertrend",
    # Temporal
    "add_temporal_features",
    "add_session_features",
    # Regime
    "add_regime_features",
    "add_volatility_regime",
    "add_trend_regime",
    # Microstructure
    "add_microstructure_features",
    "add_amihud_illiquidity",
    "add_roll_spread",
    "add_kyle_lambda",
    "add_corwin_schultz_spread",
    "add_relative_spread",
    "add_volume_imbalance",
    "add_trade_intensity",
    "add_price_efficiency",
    "add_realized_volatility_ratio",
    # Wavelets
    "add_wavelet_features",
    "add_wavelet_coefficients",
    "add_wavelet_energy",
    "add_wavelet_volatility",
    "add_wavelet_trend_strength",
    "SUPPORTED_WAVELETS",
    "DEFAULT_WAVELET",
    "DEFAULT_LEVEL",
    "DEFAULT_WINDOW",
    "PYWT_AVAILABLE",
    # Period scaling
    "scale_period",
    "get_scaled_periods",
    "get_unique_scaled_periods",
    "create_period_config",
    "get_timeframe_minutes",
    "PeriodScaler",
    "DEFAULT_BASE_PERIODS",
    "BASE_TIMEFRAME_MINUTES",
    # MTF Features
    "MTFFeatureGenerator",
    "add_mtf_features",
    "validate_mtf_alignment",
    "MTF_TIMEFRAMES",
]
