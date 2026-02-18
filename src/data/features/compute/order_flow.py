"""
Order flow imbalance indicators estimated from OHLCV data.

PHASE_19A ML Pipeline Enhancement: ORDER_FLOW features.

These features estimate buy/sell pressure and order flow dynamics
from OHLCV bar structure without requiring tick-level data.

Performance: Uses DataFrame-id based caching to avoid redundant
order imbalance calculations across features.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

# =============================================================================
# CACHING INFRASTRUCTURE
# =============================================================================

# Module-level cache for order imbalance
_order_imbalance_cache: dict[int, pd.Series] = {}
_cache_df_id: int | None = None


def _clear_cache_if_df_changed(df: pd.DataFrame) -> None:
    """Clear cache if DataFrame has changed."""
    global _cache_df_id
    df_id = id(df)
    if df_id != _cache_df_id:
        _order_imbalance_cache.clear()
        _cache_df_id = df_id


def _get_order_imbalance_cached(df: pd.DataFrame) -> pd.Series:
    """Get cached order imbalance or compute and cache it."""
    _clear_cache_if_df_changed(df)
    df_id = id(df)

    if df_id in _order_imbalance_cache:
        return _order_imbalance_cache[df_id]

    # Compute order imbalance
    hl_range = df["high"] - df["low"]
    hl_range = hl_range.replace(0, np.nan)
    buy_pressure = (df["close"] - df["low"]) / hl_range
    result = buy_pressure.fillna(0.5)

    _order_imbalance_cache[df_id] = result
    return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


# =============================================================================
# ORDER IMBALANCE FEATURES
# =============================================================================


def compute_order_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Estimate buy/sell volume ratio from OHLC bar structure.

    Uses the position of close within high-low range to estimate
    what proportion of volume was buyer-initiated vs seller-initiated.

    Uses caching to avoid redundant computation when called multiple times.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns

    Returns:
        Series with order imbalance ratio (0-1, where 1 = all buying)
    """
    return _get_order_imbalance_cached(df)


def compute_net_order_flow_5(df: pd.DataFrame) -> pd.Series:
    """
    5-period cumulative order imbalance.

    Positive values indicate sustained buying pressure,
    negative values indicate sustained selling pressure.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with net order flow (-1 to 1 range)
    """
    imbalance = compute_order_imbalance(df)
    # Center around 0 (subtract 0.5)
    centered = imbalance - 0.5
    # Rolling sum normalized by window
    net_flow = centered.rolling(window=5, min_periods=5).sum() / 5
    return net_flow


def compute_net_order_flow_10(df: pd.DataFrame) -> pd.Series:
    """
    10-period cumulative order imbalance.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with net order flow (-1 to 1 range)
    """
    imbalance = compute_order_imbalance(df)
    centered = imbalance - 0.5
    net_flow = centered.rolling(window=10, min_periods=10).sum() / 10
    return net_flow


def compute_net_order_flow_20(df: pd.DataFrame) -> pd.Series:
    """
    20-period cumulative order imbalance.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with net order flow (-1 to 1 range)
    """
    imbalance = compute_order_imbalance(df)
    centered = imbalance - 0.5
    net_flow = centered.rolling(window=20, min_periods=20).sum() / 20
    return net_flow


# =============================================================================
# BUY/SELL PRESSURE FEATURES
# =============================================================================


def compute_buy_volume(df: pd.DataFrame) -> pd.Series:
    """
    Estimate buyer-initiated volume from OHLCV.

    Uses bar structure to split total volume into estimated
    buyer-initiated component.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with estimated buy volume
    """
    imbalance = compute_order_imbalance(df)
    return df["volume"] * imbalance


def compute_sell_volume(df: pd.DataFrame) -> pd.Series:
    """
    Estimate seller-initiated volume from OHLCV.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with estimated sell volume
    """
    imbalance = compute_order_imbalance(df)
    return df["volume"] * (1 - imbalance)


def compute_pressure_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Pressure ratio: positive = buying dominates, negative = selling dominates.

    Computed as (buy_volume - sell_volume) / total_volume.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with pressure ratio (-1 to 1 range)
    """
    imbalance = compute_order_imbalance(df)

    buy_vol = df["volume"] * imbalance
    sell_vol = df["volume"] * (1 - imbalance)

    total = buy_vol + sell_vol
    pressure_ratio = (buy_vol - sell_vol) / total.replace(0, np.nan)

    return pressure_ratio


# =============================================================================
# VOLUME DELTA FEATURES
# =============================================================================


def compute_volume_delta_5(df: pd.DataFrame) -> pd.Series:
    """
    5-period volume delta: difference between up-bar and down-bar volume.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with smoothed volume delta
    """
    # Up bar: close > open, down bar: close < open
    is_up = df["close"] > df["open"]

    up_volume = df["volume"].where(is_up, 0)
    down_volume = df["volume"].where(~is_up, 0)

    delta = up_volume - down_volume

    return delta.rolling(window=5, min_periods=5).mean()


def compute_volume_delta_10(df: pd.DataFrame) -> pd.Series:
    """
    10-period volume delta.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with smoothed volume delta
    """
    is_up = df["close"] > df["open"]

    up_volume = df["volume"].where(is_up, 0)
    down_volume = df["volume"].where(~is_up, 0)

    delta = up_volume - down_volume

    return delta.rolling(window=10, min_periods=10).mean()


def compute_volume_delta_20(df: pd.DataFrame) -> pd.Series:
    """
    20-period volume delta.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with smoothed volume delta
    """
    is_up = df["close"] > df["open"]

    up_volume = df["volume"].where(is_up, 0)
    down_volume = df["volume"].where(~is_up, 0)

    delta = up_volume - down_volume

    return delta.rolling(window=20, min_periods=20).mean()


# =============================================================================
# CUMULATIVE ORDER FLOW FEATURES
# =============================================================================


def compute_cum_order_flow(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative order flow (similar to OBV but weighted by bar position).

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with cumulative order flow
    """
    imbalance = compute_order_imbalance(df)
    # Center around 0 and multiply by volume
    centered = imbalance - 0.5
    flow = centered * df["volume"]
    return flow.cumsum()


def compute_cum_order_flow_normalized(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative order flow normalized by cumulative volume.

    Provides a bounded measure of aggregate buying/selling pressure.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with normalized cumulative flow (-1 to 1 range)
    """
    imbalance = compute_order_imbalance(df)
    centered = imbalance - 0.5
    flow = centered * df["volume"]
    cum_flow = flow.cumsum()
    cum_vol = df["volume"].cumsum()

    # Avoid division by zero
    cum_vol = cum_vol.replace(0, np.nan)

    return cum_flow / cum_vol


# =============================================================================
# AGGREGATE FEATURE COMPUTATION
# =============================================================================


def compute_order_flow_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Compute all order flow features for a DataFrame.

    Args:
        df: DataFrame with OHLCV columns
        windows: List of windows for rolling features (default: [5, 10, 20])

    Returns:
        DataFrame with all order flow features
    """
    if windows is None:
        windows = [5, 10, 20]

    features = pd.DataFrame(index=df.index)

    # Basic imbalance
    features["order_imbalance"] = compute_order_imbalance(df)

    # Net flow at different windows
    for w in windows:
        imbalance = compute_order_imbalance(df)
        centered = imbalance - 0.5
        features[f"net_order_flow_{w}"] = centered.rolling(window=w, min_periods=w).sum() / w

        is_up = df["close"] > df["open"]
        up_volume = df["volume"].where(is_up, 0)
        down_volume = df["volume"].where(~is_up, 0)
        delta = up_volume - down_volume
        features[f"volume_delta_{w}"] = delta.rolling(window=w, min_periods=w).mean()

    # Pressure components
    features["pressure_ratio"] = compute_pressure_ratio(df)

    return features


# =============================================================================
# FEATURE MAP
# =============================================================================

ORDER_FLOW_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Order Imbalance
    "order_imbalance": compute_order_imbalance,
    "net_order_flow_5": compute_net_order_flow_5,
    "net_order_flow_10": compute_net_order_flow_10,
    "net_order_flow_20": compute_net_order_flow_20,
    # Buy/Sell Pressure
    "buy_volume": compute_buy_volume,
    "sell_volume": compute_sell_volume,
    "pressure_ratio": compute_pressure_ratio,
    # Volume Delta
    "volume_delta_5": compute_volume_delta_5,
    "volume_delta_10": compute_volume_delta_10,
    "volume_delta_20": compute_volume_delta_20,
    # Cumulative Flow
    "cum_order_flow": compute_cum_order_flow,
    "cum_order_flow_normalized": compute_cum_order_flow_normalized,
}

# Feature family metadata
FEATURE_FAMILY = "order_flow"
FEATURE_COUNT = 12

__all__ = [
    "ORDER_FLOW_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    # Order Imbalance
    "compute_order_imbalance",
    "compute_net_order_flow_5",
    "compute_net_order_flow_10",
    "compute_net_order_flow_20",
    # Buy/Sell Pressure
    "compute_buy_volume",
    "compute_sell_volume",
    "compute_pressure_ratio",
    # Volume Delta
    "compute_volume_delta_5",
    "compute_volume_delta_10",
    "compute_volume_delta_20",
    # Cumulative Flow
    "compute_cum_order_flow",
    "compute_cum_order_flow_normalized",
    # Aggregate
    "compute_order_flow_features",
]
