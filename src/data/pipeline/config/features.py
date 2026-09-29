"""
Feature configuration for the ensemble trading pipeline.

This module contains configuration for:
- Supported timeframes for resampling (imported from common module)
- Feature selection thresholds (correlation, variance)
- Symbol isolation policy (no cross-asset features)
- Multi-timeframe (MTF) feature configuration

NOTE: Timeframe definitions are imported from src.core.common.timeframes.
Do not redefine timeframe constants here - use the common module instead.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# TIMEFRAME CONFIGURATION
# =============================================================================
# Import canonical timeframe definitions from the single source of truth
from src.core.common.timeframes import (
    is_valid_timeframe,
)


def auto_scale_purge_embargo(
    horizons: list[int],
    purge_multiplier: float | None = None,
    embargo_multiplier: float | None = None,
) -> tuple[int, int]:
    """
    Auto-scale purge and embargo bars based on label horizons.

    This delegates to the shared horizon_config implementation so
    all code paths use the same defaults and validation.
    """
    from src.core.common.horizon_config import auto_scale_purge_embargo as _auto_scale

    return _auto_scale(
        horizons,
        purge_multiplier=purge_multiplier,
        embargo_multiplier=embargo_multiplier,
    )


def validate_horizons(horizons: list[int], data_length: int | None = None) -> list[str]:
    """
    Validate label horizons and optionally check against data length.

    Parameters
    ----------
    horizons : list
        List of horizon values
    data_length : int, optional
        Length of the dataset. If provided, validates that max(horizons) < data_length / 10
        to ensure sufficient samples for meaningful model training.

    Returns
    -------
    list
        List of validation error messages (empty if valid)

    Notes
    -----
    When data_length is provided, this function checks that horizons are not too large
    relative to the data size. The rule max(horizons) < data_length / 10 ensures at
    least 10 samples per horizon, which is a minimum requirement for meaningful statistics.

    Examples
    --------
    >>> validate_horizons([5, 20])  # Basic validation
    []
    >>> validate_horizons([5, 20], data_length=100)  # 20 >= 10, too large
    ['Horizon 20 may be too large for data length 100 (max recommended: 10)...']
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

    # Validate horizons against data length (LBL-005)
    if data_length is not None and data_length > 0:
        max_recommended_horizon = data_length // 10
        for h in horizons:
            if isinstance(h, int) and h >= max_recommended_horizon:
                errors.append(
                    f"Horizon {h} may be too large for data length {data_length} "
                    f"(max recommended: {max_recommended_horizon}). "
                    f"Horizons should be < 10% of data length to ensure sufficient training samples."
                )

    return errors


# Import canonical horizon definitions from the centralized module
# Do NOT define horizons locally - always import from src.core.common.horizon_config

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

MTF_CONFIG: dict[str, Any] = {
    # Master enable/disable for MTF features
    "enabled": True,
    # Base timeframe of the input data
    "base_timeframe": "1min",
    # Higher timeframes to compute features for
    # All TFs must be >= base timeframe (1min) since we resample UP
    # Higher TFs capture broader market structure
    "mtf_timeframes": ["5min", "10min", "15min", "30min", "45min", "60min"],
    # Include raw OHLCV from higher TFs
    # Features: open_15m, high_15m, low_15m, close_15m, volume_15m, etc.
    "include_ohlcv": True,
    # Include indicators computed on higher TFs
    # Features: sma_20_15m, rsi_14_15m, atr_14_1h, etc.
    "include_indicators": True,
    # Minimum data requirements
    # Need sufficient data for indicator warmup at higher TF
    "min_base_bars": 1000,  # Minimum bars at base TF
    "min_mtf_bars": 50,  # Minimum bars at each higher TF after resampling
    # Indicator configuration for higher TFs
    "indicators": {
        "sma_periods": [20, 50],
        "ema_periods": [9, 21],
        "rsi_period": 14,
        "atr_period": 14,
        "bb_period": 20,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
    },
    # Feature groups to include
    # Set to False to exclude specific feature groups
    "feature_groups": {
        "moving_averages": True,  # SMA, EMA
        "momentum": True,  # RSI, MACD
        "volatility": True,  # ATR, BB position
        "price_ratios": True,  # close/SMA ratios
    },
}


def validate_mtf_config(config: dict[str, Any] | None = None) -> list[str]:
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
    valid_base_tfs = ["1min", "5min"]
    if config.get("base_timeframe") not in valid_base_tfs:
        errors.append(
            f"base_timeframe must be one of {valid_base_tfs}, "
            f"got '{config.get('base_timeframe')}'"
        )

    # Validate MTF timeframes using common module
    for tf in config.get("mtf_timeframes", []):
        if not is_valid_timeframe(tf, allow_extended=True):
            errors.append(
                f"MTF timeframe '{tf}' is not valid. "
                f"Use canonical forms like '5min', '15min', '60min', etc."
            )

    # Validate minimum bars
    if config.get("min_base_bars", 0) < 100:
        errors.append(f"min_base_bars must be >= 100, got {config.get('min_base_bars')}")

    if config.get("min_mtf_bars", 0) < 30:
        errors.append(f"min_mtf_bars must be >= 30, got {config.get('min_mtf_bars')}")

    return errors


# =============================================================================
# STATIONARITY TEST CONFIGURATION
# =============================================================================
STATIONARITY_TESTS = {
    "enabled": False,
    "max_features": 5,
}


# =============================================================================
# DRIFT CONFIGURATION
# =============================================================================
DRIFT_CONFIG = {
    "enabled": True,
    "psi_threshold": 0.2,
    "bins": 10,
    "max_features": 200,
}


def get_drift_config() -> dict[str, Any]:
    """Get a copy of the drift configuration."""
    import copy

    return copy.deepcopy(DRIFT_CONFIG)
