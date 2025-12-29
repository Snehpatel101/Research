"""
Feature configuration for the ensemble trading pipeline.

This module contains configuration for:
- Supported timeframes for resampling
- Feature selection thresholds (correlation, variance)
- Cross-asset feature configuration
- Multi-timeframe (MTF) feature configuration
"""


# =============================================================================
# TIMEFRAME CONFIGURATION
# =============================================================================
# Supported timeframes for resampling pipeline.
# Input data (1min bars) can be resampled to any of these target timeframes.
# The base timeframe for ML features is typically 5min.
SUPPORTED_TIMEFRAMES = [
    '1min', '5min', '10min', '15min', '20min', '30min', '45min', '60min'
]

# Mapping from timeframe string to pandas frequency for resampling
TIMEFRAME_TO_FREQ = {
    '1min': '1min',
    '5min': '5min',
    '10min': '10min',
    '15min': '15min',
    '20min': '20min',
    '30min': '30min',
    '45min': '45min',
    '60min': '60min',
    '1h': '1h',
}


def validate_timeframe(timeframe: str) -> None:
    """
    Validate that a timeframe string is supported.

    Parameters
    ----------
    timeframe : str
        Timeframe string to validate (e.g., '5min', '15min')

    Raises
    ------
    ValueError
        If the timeframe is not in SUPPORTED_TIMEFRAMES
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: '{timeframe}'. "
            f"Supported timeframes: {SUPPORTED_TIMEFRAMES}"
        )


def parse_timeframe_to_minutes(timeframe: str) -> int:
    """
    Parse a timeframe string to minutes.

    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., '5min', '1h')

    Returns
    -------
    int
        Number of minutes

    Raises
    ------
    ValueError
        If the timeframe format is not recognized
    """
    timeframe = timeframe.lower().strip()

    if timeframe.endswith('min'):
        try:
            return int(timeframe[:-3])
        except ValueError:
            raise ValueError(f"Invalid timeframe format: '{timeframe}'")
    elif timeframe.endswith('h'):
        try:
            return int(timeframe[:-1]) * 60
        except ValueError:
            raise ValueError(f"Invalid timeframe format: '{timeframe}'")
    else:
        raise ValueError(
            f"Unrecognized timeframe format: '{timeframe}'. "
            f"Expected format like '5min' or '1h'"
        )


def auto_scale_purge_embargo(
    horizons: list,
    purge_multiplier: float | None = None,
    embargo_multiplier: float | None = None,
) -> tuple:
    """
    Auto-scale purge and embargo bars based on label horizons.

    This delegates to the shared horizon_config implementation so
    all code paths use the same defaults and validation.
    """
    from src.common.horizon_config import auto_scale_purge_embargo as _auto_scale

    return _auto_scale(
        horizons,
        purge_multiplier=purge_multiplier,
        embargo_multiplier=embargo_multiplier,
    )


def validate_horizons(horizons: list) -> list:
    """
    Validate label horizons.

    Parameters
    ----------
    horizons : list
        List of horizon values

    Returns
    -------
    list
        List of validation error messages (empty if valid)
    """
    errors = []

    if not horizons:
        errors.append("At least one horizon must be specified")
        return errors

    for h in horizons:
        if not isinstance(h, int):
            errors.append(f"Horizon must be an integer, got {type(h).__name__}: {h}")
        elif h < 1:
            errors.append(f"Horizon must be >= 1, got {h}")
        elif h > 100:
            errors.append(f"Horizon {h} is very large (> 100 bars). Consider smaller values.")

    return errors


# Import canonical horizon definitions from the centralized module
# Do NOT define horizons locally - always import from src.common.horizon_config

# =============================================================================
# FEATURE SELECTION CONFIGURATION
# =============================================================================
# CORRELATION_THRESHOLD: Maximum allowed correlation between features.
# Features with correlation above this threshold will be removed (keeping the
# most interpretable feature from each correlated group).
#
# Industry Standard: 0.80 is widely accepted in ML practice as the threshold
# where multicollinearity begins to cause significant issues.
#
# Rationale for 0.80:
# - Below 0.80: Multicollinearity is generally acceptable for most model types
# - 0.80-0.90: Moderate correlation; some models (e.g., linear) may suffer
# - Above 0.90: High correlation; likely redundant information
#
# Note: 0.70 was previously used but is too aggressive for this codebase:
# - With 150+ features, aggressive pruning can remove useful signal
# - Tree-based models (XGBoost, LightGBM) handle correlated features well
# - Neural networks with dropout are robust to moderate correlation
#
# Model-family considerations:
# - Boosting (XGBoost, LightGBM): Tolerant of correlation, 0.80-0.90 acceptable
# - Neural (LSTM, Transformer): Moderate tolerance, 0.80 preferred
# - Linear models: Most sensitive, may need 0.70 for stability
#
# Lower values = more aggressive pruning = fewer features = less multicollinearity
# Higher values = less pruning = more features = potential multicollinearity
CORRELATION_THRESHOLD = 0.80

# VARIANCE_THRESHOLD: Minimum variance for a feature to be retained.
# Features with variance below this threshold are considered near-constant
# and provide no discriminative power.
VARIANCE_THRESHOLD = 0.01


# =============================================================================
# SYMBOL ISOLATION POLICY
# =============================================================================
# Each symbol is processed in complete isolation. There are NO cross-symbol or
# cross-asset features (no correlation, beta, spread, or relative strength
# features between symbols). This ensures:
# 1. No data leakage between symbols
# 2. Each model can be trained on a single symbol's data
# 3. No dependencies on other symbols' data availability
# 4. Clean separation for production deployment


def is_cross_asset_feature(feature_name: str) -> bool:
    """
    Check if a feature name represents a cross-asset feature.

    Cross-asset features are NOT generated in this pipeline. Each symbol is
    processed in complete isolation with no cross-symbol operations.

    This function always returns False since cross-asset features have been
    removed from the pipeline. It is kept for backward compatibility with
    validation code that may still call it.

    Parameters
    ----------
    feature_name : str
        The feature column name to check

    Returns
    -------
    bool
        Always returns False since cross-asset features are not supported
    """
    # Cross-asset features are not generated - always return False
    # Legacy patterns that would have been cross-asset (now removed):
    # - mes_mgc_*, relative_strength, beta_*, spread_*, correlation_*
    return False


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
    'base_timeframe': '1min',

    # Higher timeframes to compute features for
    # All TFs must be >= base timeframe (1min) since we resample UP
    # Higher TFs capture broader market structure
    'mtf_timeframes': ['5min', '10min', '15min', '30min', '45min', '60min'],

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
    valid_mtf_tfs = ['1min', '5min', '10min', '15min', '30min', '45min', '60min', '1h']
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


