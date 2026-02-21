"""
Volume feature computation - OBV, VWAP, TWAP, dollar volume.

PHASE_1 Unified Features: 15 VOLUME features.

Performance: Uses DataFrame-id based caching to avoid recomputing base features
when derived features are computed. Cache is cleared when DataFrame changes.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data.features.compute._helpers import rolling_std as _rolling_std
from src.data.features.compute._helpers import sma as _sma

# =============================================================================
# CACHING INFRASTRUCTURE
# =============================================================================

# Module-level cache keyed by (DataFrame id, feature name)
# Stores computed results to avoid redundant calculations
_volume_cache: dict[tuple[int, str], pd.Series] = {}
_cache_df_id: int | None = None


def _get_cached(df: pd.DataFrame, key: str) -> pd.Series | None:
    """Get cached result if DataFrame hasn't changed."""
    global _cache_df_id
    df_id = id(df)
    if df_id != _cache_df_id:
        # DataFrame changed, clear cache
        _volume_cache.clear()
        _cache_df_id = df_id
        return None
    return _volume_cache.get((df_id, key))


def _set_cached(df: pd.DataFrame, key: str, value: pd.Series) -> pd.Series:
    """Store result in cache and return it."""
    global _cache_df_id
    df_id = id(df)
    if df_id != _cache_df_id:
        _volume_cache.clear()
        _cache_df_id = df_id
    _volume_cache[(df_id, key)] = value
    return value


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


# _sma, _rolling_std imported from _helpers.py (Phase H5 consolidation)


def _get_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract datetime index from DataFrame."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    elif "datetime" in df.columns:
        return pd.to_datetime(df["datetime"])
    else:
        # Return a default index if no datetime available
        return None


# =============================================================================
# OBV FEATURES
# =============================================================================


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume.

    OBV adds volume on up days and subtracts on down days.
    Measures buying/selling pressure.
    """
    cached = _get_cached(df, "obv")
    if cached is not None:
        return cached

    close_diff = df["close"].diff()
    direction = np.sign(close_diff)

    # First row has no direction, use 0
    direction = direction.fillna(0)

    obv = (direction * df["volume"]).cumsum()
    return _set_cached(df, "obv", obv)


def compute_obv_sma_20(df: pd.DataFrame) -> pd.Series:
    """20-period SMA of OBV."""
    obv = compute_obv(df)  # Uses cache
    return _sma(obv, window=20)


# =============================================================================
# VOLUME SMA AND RATIO FEATURES
# =============================================================================


def compute_volume_sma_20(df: pd.DataFrame) -> pd.Series:
    """20-period volume SMA."""
    return _sma(df["volume"], window=20)


def compute_volume_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Volume ratio (current volume / 20-period average).

    > 1.0 indicates above-average volume
    """
    vol_sma = _sma(df["volume"], window=20)
    # Avoid division by zero
    vol_sma = vol_sma.replace(0, np.nan)
    return df["volume"] / vol_sma


def compute_volume_zscore(df: pd.DataFrame) -> pd.Series:
    """
    Volume z-score (how many standard deviations from mean).

    Useful for detecting volume anomalies.
    """
    vol_sma = _sma(df["volume"], window=20)
    vol_std = _rolling_std(df["volume"], window=20)

    # Avoid division by zero
    vol_std = vol_std.replace(0, np.nan)

    return (df["volume"] - vol_sma) / vol_std


# =============================================================================
# VWAP FEATURES
# =============================================================================


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price (session VWAP).

    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)

    Note: This computes a rolling VWAP. For true session VWAP,
    reset at session boundaries.
    """
    cached = _get_cached(df, "vwap")
    if cached is not None:
        return cached

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()

    # Avoid division by zero
    cum_vol = cum_vol.replace(0, np.nan)

    return _set_cached(df, "vwap", cum_tp_vol / cum_vol)


def compute_price_to_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Price deviation from VWAP.

    > 1.0 means price is above VWAP (bullish short-term)
    < 1.0 means price is below VWAP (bearish short-term)
    """
    vwap = compute_vwap(df)  # Uses cache
    # Avoid division by zero
    vwap = vwap.replace(0, np.nan)
    return df["close"] / vwap


# =============================================================================
# TWAP FEATURES
# =============================================================================


def compute_twap(df: pd.DataFrame) -> pd.Series:
    """
    Time Weighted Average Price (cumulative).

    Simple average of typical prices over time.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    return typical_price.expanding(min_periods=1).mean()


def compute_twap_10(df: pd.DataFrame) -> pd.Series:
    """10-period rolling TWAP."""
    cached = _get_cached(df, "twap_10")
    if cached is not None:
        return cached

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    return _set_cached(df, "twap_10", typical_price.rolling(window=10, min_periods=10).mean())


def compute_twap_20(df: pd.DataFrame) -> pd.Series:
    """20-period rolling TWAP."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    return typical_price.rolling(window=20, min_periods=20).mean()


def compute_price_to_twap_10(df: pd.DataFrame) -> pd.Series:
    """
    Price to TWAP-10 ratio.

    Similar to price_to_vwap but using time-weighted average.
    """
    twap_10 = compute_twap_10(df)  # Uses cache
    # Avoid division by zero
    twap_10 = twap_10.replace(0, np.nan)
    return df["close"] / twap_10


# =============================================================================
# DOLLAR VOLUME FEATURES
# =============================================================================


def compute_dollar_volume(df: pd.DataFrame) -> pd.Series:
    """
    Dollar volume (price x volume).

    Measures total value traded per bar.
    """
    cached = _get_cached(df, "dollar_volume")
    if cached is not None:
        return cached

    return _set_cached(df, "dollar_volume", df["close"] * df["volume"])


def compute_dollar_volume_sma_10(df: pd.DataFrame) -> pd.Series:
    """10-period SMA of dollar volume."""
    dollar_vol = compute_dollar_volume(df)  # Uses cache
    return _sma(dollar_vol, window=10)


def compute_dollar_volume_sma_20(df: pd.DataFrame) -> pd.Series:
    """20-period SMA of dollar volume."""
    dollar_vol = compute_dollar_volume(df)  # Uses cache
    return _sma(dollar_vol, window=20)


def compute_dollar_volume_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Dollar volume ratio (current / 20-period average).

    Measures relative liquidity.
    """
    dollar_vol = compute_dollar_volume(df)  # Uses cache
    dollar_vol_sma = _sma(dollar_vol, window=20)

    # Avoid division by zero
    dollar_vol_sma = dollar_vol_sma.replace(0, np.nan)

    return dollar_vol / dollar_vol_sma


# =============================================================================
# FEATURE MAP
# =============================================================================

VOLUME_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # OBV
    "obv": compute_obv,
    "obv_sma_20": compute_obv_sma_20,
    # Volume SMA/Ratio
    "volume_sma_20": compute_volume_sma_20,
    "volume_ratio": compute_volume_ratio,
    "volume_zscore": compute_volume_zscore,
    # VWAP
    "vwap": compute_vwap,
    "price_to_vwap": compute_price_to_vwap,
    # TWAP
    "twap": compute_twap,
    "twap_10": compute_twap_10,
    "twap_20": compute_twap_20,
    "price_to_twap_10": compute_price_to_twap_10,
    # Dollar Volume
    "dollar_volume": compute_dollar_volume,
    "dollar_volume_sma_10": compute_dollar_volume_sma_10,
    "dollar_volume_sma_20": compute_dollar_volume_sma_20,
    "dollar_volume_ratio": compute_dollar_volume_ratio,
}

# Feature family metadata
FEATURE_FAMILY = "volume"
FEATURE_COUNT = 15

__all__ = [
    "VOLUME_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    # OBV
    "compute_obv",
    "compute_obv_sma_20",
    # Volume SMA/Ratio
    "compute_volume_sma_20",
    "compute_volume_ratio",
    "compute_volume_zscore",
    # VWAP
    "compute_vwap",
    "compute_price_to_vwap",
    # TWAP
    "compute_twap",
    "compute_twap_10",
    "compute_twap_20",
    "compute_price_to_twap_10",
    # Dollar Volume
    "compute_dollar_volume",
    "compute_dollar_volume_sma_10",
    "compute_dollar_volume_sma_20",
    "compute_dollar_volume_ratio",
]
