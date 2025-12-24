"""
Feature configuration for the ensemble trading pipeline.

This module contains configuration for:
- Feature selection thresholds (correlation, variance)
- Cross-asset feature configuration
- Multi-timeframe (MTF) feature configuration
"""

# =============================================================================
# FEATURE SELECTION CONFIGURATION
# =============================================================================
# CORRELATION_THRESHOLD: Maximum allowed correlation between features.
# Features with correlation above this threshold will be removed (keeping the
# most interpretable feature from each correlated group).
#
# CRITICAL: The previous default of 0.95 was too lenient. Features with 0.70+
# correlation cause multicollinearity issues during ML training, leading to:
# - Unstable coefficient estimates
# - Inflated variance in predictions
# - Reduced model interpretability
#
# Lowering to 0.70 provides aggressive pruning which is desirable for:
# - Reducing overfitting in ensemble models
# - Improving feature importance reliability
# - Faster training with fewer redundant features
#
# Lower values = more aggressive pruning = fewer features = less multicollinearity
# Higher values = less pruning = more features = potential multicollinearity
CORRELATION_THRESHOLD = 0.70

# VARIANCE_THRESHOLD: Minimum variance for a feature to be retained.
# Features with variance below this threshold are considered near-constant
# and provide no discriminative power.
VARIANCE_THRESHOLD = 0.01


# =============================================================================
# CROSS-ASSET FEATURE CONFIGURATION
# =============================================================================
# Cross-asset features capture relationships between MES (S&P 500 futures) and
# MGC (Gold futures), which often exhibit interesting correlation dynamics:
# - Risk-on/risk-off regimes: MES up, MGC down (and vice versa)
# - Flight to safety: MES down, MGC up during market stress
# - Inflation hedging: Both assets may move together during inflation concerns
#
# These features are computed when both symbols are present in the data.

CROSS_ASSET_FEATURES = {
    # CRITICAL: Cross-asset features require BOTH MES and MGC symbols
    # Since we trade one market at a time, these features are disabled by default
    # Set to True only when running multi-symbol analysis with both MES and MGC
    'enabled': False,
    'symbols': ['MES', 'MGC'],  # Symbol pair for cross-asset features
    'min_symbols': 2,  # Require at least 2 symbols for cross-asset features
    'features': {
        'mes_mgc_correlation_20': {
            'description': '20-bar rolling correlation between MES and MGC returns',
            'lookback': 20
        },
        'mes_mgc_spread_zscore': {
            'description': 'Z-score of spread between normalized MES and MGC prices',
            'lookback': 20
        },
        'mes_mgc_beta': {
            'description': 'Rolling beta of MES returns vs MGC returns',
            'lookback': 20
        },
        'relative_strength': {
            'description': 'MES return minus MGC return (momentum divergence)',
            'lookback': 20
        }
    }
}


# =============================================================================
# MULTI-TIMEFRAME (MTF) FEATURE CONFIGURATION
# =============================================================================
# MTF features capture market structure from higher timeframes and align them
# to the base timeframe (5min) without lookahead bias.
#
# Two types of MTF features are supported:
# 1. MTF OHLCV: Raw OHLCV values from higher TFs (e.g., close_15m, high_1h)
# 2. MTF Indicators: Technical indicators computed on higher TFs (e.g., rsi_14_15m)
#
# ANTI-LOOKAHEAD DESIGN:
# All MTF features use shift(1) on higher TF data before alignment, ensuring
# we only use COMPLETED higher TF bars. This prevents using information from
# the current (incomplete) higher TF bar which would cause lookahead bias.
#
# Example: At 10:05 (5min bar), the 15min bar for 10:00-10:15 is incomplete.
# We use the PREVIOUS completed 15min bar (09:45-10:00) for features.

MTF_CONFIG = {
    # Master enable/disable for MTF features
    'enabled': True,

    # Base timeframe of the input data
    'base_timeframe': '5min',

    # Higher timeframes to compute features for
    # Must be integer multiples of base_timeframe
    # Supported: '15min', '30min', '60min' (or '1h')
    'mtf_timeframes': ['15min', '60min'],

    # Include raw OHLCV from higher TFs
    # Features: open_15m, high_15m, low_15m, close_15m, volume_15m, etc.
    'include_ohlcv': True,

    # Include indicators computed on higher TFs
    # Features: sma_20_15m, rsi_14_15m, atr_14_1h, etc.
    'include_indicators': True,

    # Minimum data requirements
    # Need sufficient data for indicator warmup at higher TF
    'min_base_bars': 1000,  # Minimum bars at base TF
    'min_mtf_bars': 50,     # Minimum bars at each higher TF after resampling

    # Indicator configuration for higher TFs
    'indicators': {
        'sma_periods': [20, 50],
        'ema_periods': [9, 21],
        'rsi_period': 14,
        'atr_period': 14,
        'bb_period': 20,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
    },

    # Feature groups to include
    # Set to False to exclude specific feature groups
    'feature_groups': {
        'moving_averages': True,   # SMA, EMA
        'momentum': True,          # RSI, MACD
        'volatility': True,        # ATR, BB position
        'price_ratios': True,      # close/SMA ratios
    }
}


def get_mtf_config() -> dict:
    """
    Get the MTF configuration dictionary.

    Returns
    -------
    dict
        Copy of MTF_CONFIG

    Notes
    -----
    Returns a copy to prevent accidental modification of the global config.
    """
    import copy
    return copy.deepcopy(MTF_CONFIG)


def validate_mtf_config(config: dict = None) -> list[str]:
    """
    Validate MTF configuration values.

    Parameters
    ----------
    config : dict, optional
        MTF config dict to validate. Uses MTF_CONFIG if not provided.

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    if config is None:
        config = MTF_CONFIG

    errors = []

    # Validate base timeframe
    valid_base_tfs = ['1min', '5min']
    if config.get('base_timeframe') not in valid_base_tfs:
        errors.append(
            f"base_timeframe must be one of {valid_base_tfs}, "
            f"got '{config.get('base_timeframe')}'"
        )

    # Validate MTF timeframes
    valid_mtf_tfs = ['15min', '30min', '60min', '1h']
    for tf in config.get('mtf_timeframes', []):
        if tf not in valid_mtf_tfs:
            errors.append(
                f"MTF timeframe '{tf}' not in valid list {valid_mtf_tfs}"
            )

    # Validate minimum bars
    if config.get('min_base_bars', 0) < 100:
        errors.append(
            f"min_base_bars must be >= 100, got {config.get('min_base_bars')}"
        )

    if config.get('min_mtf_bars', 0) < 30:
        errors.append(
            f"min_mtf_bars must be >= 30, got {config.get('min_mtf_bars')}"
        )

    return errors


def get_cross_asset_config() -> dict:
    """
    Get the cross-asset feature configuration dictionary.

    Returns
    -------
    dict
        Copy of CROSS_ASSET_FEATURES
    """
    import copy
    return copy.deepcopy(CROSS_ASSET_FEATURES)


def validate_feature_thresholds() -> list[str]:
    """
    Validate feature selection threshold values.

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    errors = []

    if not (0 < CORRELATION_THRESHOLD <= 1.0):
        errors.append(f"CORRELATION_THRESHOLD must be in (0, 1.0], got {CORRELATION_THRESHOLD}")

    if VARIANCE_THRESHOLD < 0:
        errors.append(f"VARIANCE_THRESHOLD must be non-negative, got {VARIANCE_THRESHOLD}")

    return errors


# =============================================================================
# STATIONARITY TEST CONFIGURATION
# =============================================================================
STATIONARITY_TESTS = {
    'enabled': False,
    'max_features': 5,
}


def get_stationarity_config() -> dict:
    """Get a copy of the stationarity test configuration."""
    import copy
    return copy.deepcopy(STATIONARITY_TESTS)


def validate_stationarity_config() -> list[str]:
    """Validate stationarity test configuration."""
    errors = []
    if STATIONARITY_TESTS.get('max_features', 0) < 1:
        errors.append(
            f"STATIONARITY_TESTS.max_features must be >= 1, got {STATIONARITY_TESTS.get('max_features')}"
        )
    return errors


# =============================================================================
# DRIFT CONFIGURATION
# =============================================================================
DRIFT_CONFIG = {
    'enabled': True,
    'psi_threshold': 0.2,
    'bins': 10,
    'max_features': 200,
}


def get_drift_config() -> dict:
    """Get a copy of the drift configuration."""
    import copy
    return copy.deepcopy(DRIFT_CONFIG)


def validate_drift_config() -> list[str]:
    """Validate drift configuration."""
    errors = []
    if DRIFT_CONFIG.get('bins', 0) < 2:
        errors.append(
            f"DRIFT_CONFIG.bins must be >= 2, got {DRIFT_CONFIG.get('bins')}"
        )
    if DRIFT_CONFIG.get('max_features', 0) < 1:
        errors.append(
            f"DRIFT_CONFIG.max_features must be >= 1, got {DRIFT_CONFIG.get('max_features')}"
        )
    return errors


def get_cross_asset_feature_names() -> list[str]:
    """
    Get list of cross-asset feature names.

    Returns
    -------
    list[str]
        List of cross-asset feature column names
    """
    return list(CROSS_ASSET_FEATURES['features'].keys())


def is_cross_asset_feature(feature_name: str) -> bool:
    """
    Check if a feature name is a cross-asset feature.

    Parameters
    ----------
    feature_name : str
        Name of the feature to check

    Returns
    -------
    bool
        True if the feature is a cross-asset feature, False otherwise
    """
    return feature_name in CROSS_ASSET_FEATURES['features']
