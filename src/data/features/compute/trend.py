"""
Trend feature computation - ADX, Directional Indicators, Supertrend.

PHASE_1 Unified Features: 6 TREND features.

Phase 24: Added caching for ADX/DI and Supertrend to avoid redundant computation.
- ADX/DI computed once per DataFrame, cached by id(df)
- Supertrend computed once per DataFrame, cached by id(df)
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

# =============================================================================
# MODULE-LEVEL CACHES (Phase 24: Avoid redundant computation)
# =============================================================================

# Cache for ADX/DI computation: {id(df): (plus_di, minus_di, adx)}
_di_adx_cache: dict[int, tuple[pd.Series, pd.Series, pd.Series]] = {}

# Cache for Supertrend computation: {id(df): (supertrend, direction)}
_supertrend_cache: dict[int, tuple[pd.Series, pd.Series]] = {}


def clear_trend_cache() -> None:
    """Clear all trend computation caches. Call between different DataFrames."""
    _di_adx_cache.clear()
    _supertrend_cache.clear()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _true_range(df: pd.DataFrame) -> pd.Series:
    """True Range calculation."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr


def _smoothed_ma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (equivalent to EMA with alpha=1/period)."""
    return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _compute_directional_movement(
    df: pd.DataFrame, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute +DM, -DM, and True Range for directional indicators.

    Returns:
        Tuple of (smoothed +DM, smoothed -DM, smoothed TR)
    """
    # Calculate directional movement
    high_diff = df["high"] - df["high"].shift(1)
    low_diff = df["low"].shift(1) - df["low"]

    # +DM: positive when high increases more than low decreases
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    # -DM: positive when low decreases more than high increases
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

    # True Range
    tr = _true_range(df)

    # Apply Wilder's smoothing
    smoothed_plus_dm = _smoothed_ma(plus_dm, period)
    smoothed_minus_dm = _smoothed_ma(minus_dm, period)
    smoothed_tr = _smoothed_ma(tr, period)

    return smoothed_plus_dm, smoothed_minus_dm, smoothed_tr


def _compute_di_adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute +DI, -DI, and ADX.

    Returns:
        Tuple of (+DI, -DI, ADX)
    """
    smoothed_plus_dm, smoothed_minus_dm, smoothed_tr = _compute_directional_movement(df, period)

    # Avoid division by zero
    smoothed_tr = smoothed_tr.replace(0, np.nan)

    # Directional Indicators
    plus_di = (smoothed_plus_dm / smoothed_tr) * 100
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100

    # Directional Index
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = (np.abs(plus_di - minus_di) / di_sum) * 100

    # ADX is smoothed DX
    adx = _smoothed_ma(dx, period)

    return plus_di, minus_di, adx


def _get_di_adx_cached(
    df: pd.DataFrame, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Get cached ADX/DI computation or compute if not cached.

    Phase 24: Avoids computing ADX/DI 4x for the same DataFrame.
    """
    cache_key = id(df)
    if cache_key not in _di_adx_cache:
        _di_adx_cache[cache_key] = _compute_di_adx(df, period)
    return _di_adx_cache[cache_key]


# =============================================================================
# ADX FEATURES
# =============================================================================


def compute_adx_14(df: pd.DataFrame) -> pd.Series:
    """
    14-period Average Directional Index.

    ADX measures trend strength regardless of direction.
    - > 25: Strong trend
    - < 20: Weak trend / ranging market
    """
    _, _, adx = _get_di_adx_cached(df, period=14)
    return adx


def compute_plus_di_14(df: pd.DataFrame) -> pd.Series:
    """
    14-period +DI (Plus Directional Indicator).

    Measures upward trend strength.
    """
    plus_di, _, _ = _get_di_adx_cached(df, period=14)
    return plus_di


def compute_minus_di_14(df: pd.DataFrame) -> pd.Series:
    """
    14-period -DI (Minus Directional Indicator).

    Measures downward trend strength.
    """
    _, minus_di, _ = _get_di_adx_cached(df, period=14)
    return minus_di


def compute_adx_strong_trend(df: pd.DataFrame) -> pd.Series:
    """
    ADX strong trend flag (ADX > 25).

    Binary indicator for strong trending conditions.
    """
    _, _, adx = _get_di_adx_cached(df, period=14)
    return (adx > 25).astype(float)


# =============================================================================
# SUPERTREND FEATURES
# =============================================================================


def _compute_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """
    Compute Supertrend indicator.

    Supertrend is a trend-following indicator that uses ATR to determine
    support/resistance levels.

    Returns:
        Tuple of (supertrend_line, direction)
        Direction: 1 for uptrend, -1 for downtrend
    """
    # Calculate ATR
    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Basic upper and lower bands
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # Initialize arrays
    n = len(df)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    # Convert to numpy for faster computation
    close = df["close"].values
    upper = basic_upper.values
    lower = basic_lower.values

    # Initialize first valid values
    first_valid = period - 1
    if first_valid >= n:
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)

    supertrend[first_valid] = basic_upper.iloc[first_valid]
    direction[first_valid] = 1

    # Calculate Supertrend
    for i in range(first_valid + 1, n):
        # Adjust bands based on previous close
        if lower[i] > supertrend[i - 1] if direction[i - 1] == 1 else True:
            final_lower = lower[i]
        else:
            final_lower = supertrend[i - 1] if direction[i - 1] == 1 else lower[i]

        if upper[i] < supertrend[i - 1] if direction[i - 1] == -1 else True:
            final_upper = upper[i]
        else:
            final_upper = supertrend[i - 1] if direction[i - 1] == -1 else upper[i]

        # Determine trend direction
        if direction[i - 1] == 1:
            if close[i] <= final_lower:
                direction[i] = -1
                supertrend[i] = final_upper
            else:
                direction[i] = 1
                supertrend[i] = final_lower
        else:
            if close[i] >= final_upper:
                direction[i] = 1
                supertrend[i] = final_lower
            else:
                direction[i] = -1
                supertrend[i] = final_upper

    # Set NaN for warmup period
    supertrend[:first_valid] = np.nan
    direction[:first_valid] = np.nan

    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)


def _get_supertrend_cached(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """
    Get cached Supertrend computation or compute if not cached.

    Phase 24: Avoids computing Supertrend 2x for the same DataFrame.
    """
    cache_key = id(df)
    if cache_key not in _supertrend_cache:
        _supertrend_cache[cache_key] = _compute_supertrend(df, period, multiplier)
    return _supertrend_cache[cache_key]


def compute_supertrend(df: pd.DataFrame) -> pd.Series:
    """
    Supertrend line (10, 3.0).

    The Supertrend value acts as dynamic support (in uptrend) or
    resistance (in downtrend).
    """
    supertrend, _ = _get_supertrend_cached(df, period=10, multiplier=3.0)
    return supertrend


def compute_supertrend_direction(df: pd.DataFrame) -> pd.Series:
    """
    Supertrend direction (10, 3.0).

    Returns:
        1.0 for uptrend
        -1.0 for downtrend
    """
    _, direction = _get_supertrend_cached(df, period=10, multiplier=3.0)
    return direction


# =============================================================================
# FEATURE MAP
# =============================================================================

TREND_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # ADX
    "adx_14": compute_adx_14,
    "plus_di_14": compute_plus_di_14,
    "minus_di_14": compute_minus_di_14,
    "adx_strong_trend": compute_adx_strong_trend,
    # Supertrend
    "supertrend": compute_supertrend,
    "supertrend_direction": compute_supertrend_direction,
}

# Feature family metadata
FEATURE_FAMILY = "trend"
FEATURE_COUNT = 6

__all__ = [
    "TREND_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    "compute_adx_14",
    "compute_plus_di_14",
    "compute_minus_di_14",
    "compute_adx_strong_trend",
    "compute_supertrend",
    "compute_supertrend_direction",
    "clear_trend_cache",
]
