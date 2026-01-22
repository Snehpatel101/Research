"""
Adaptive transaction cost model for day trading.

This module implements time-of-day and volume-adjusted transaction costs
for more realistic triple-barrier labeling.

Key Features:
- Time-of-day slippage multipliers (open/close = 2.5x, mid-session = 1.0x)
- Volume-adjusted market impact (square root law)
- Volatility regime adjustments

References:
    - Kyle (1985): "Continuous Auctions and Insider Trading"
    - Almgren & Chriss (2000): "Optimal execution of portfolio transactions"
    - Hasbrouck (2007): "Empirical Market Microstructure"
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_time_of_day_multiplier(timestamp: pd.Timestamp) -> float:
    """
    Slippage multiplier based on time of day.

    Market hours (ET):
    - 09:30-10:00: Open (volatile, wide spreads) → 2.5x
    - 10:00-14:00: Mid-session (liquid, tight spreads) → 1.0x
    - 14:00-15:00: Pre-close (moderate) → 1.5x
    - 15:00-16:00: Close (volatile, wide spreads) → 2.5x
    - Outside hours: 3.0x (very illiquid)

    Args:
        timestamp: Trading timestamp

    Returns:
        Slippage multiplier (1.0 = baseline)
    """
    hour = timestamp.hour
    minute = timestamp.minute

    if hour < 9 or hour >= 16:
        return 3.0

    if hour == 9 and minute >= 30:
        return 2.5

    if hour == 9 and minute < 30:
        return 3.0

    if 10 <= hour < 14:
        return 1.0

    if hour == 14:
        return 1.5

    if hour == 15:
        return 2.5

    return 1.0


def get_volume_impact_multiplier(
    order_size_contracts: float,
    typical_volume: float,
) -> float:
    """
    Market impact based on order size relative to typical volume.

    Uses square root law: impact ∝ sqrt(order_size / typical_volume)
    This is well-established in market microstructure literature.

    Args:
        order_size_contracts: Number of contracts to trade
        typical_volume: Typical minute volume (contracts)

    Returns:
        Impact multiplier >= 1.0

    Reference:
        Kyle (1985), Almgren & Chriss (2000)
    """
    if typical_volume <= 0:
        return 1.0

    volume_fraction = order_size_contracts / typical_volume

    impact = 1.0 + 0.5 * np.sqrt(volume_fraction)

    return float(max(1.0, impact))


def get_volatility_multiplier(
    atr_current: float,
    atr_mean: float,
) -> float:
    """
    Volatility-adjusted slippage.

    Higher volatility → wider spreads → more slippage.
    Linear scaling with volatility ratio.

    Args:
        atr_current: Current ATR value
        atr_mean: Historical mean ATR

    Returns:
        Volatility multiplier (capped at [0.5, 2.0])
    """
    if atr_mean <= 0:
        return 1.0

    vol_ratio = atr_current / atr_mean

    return float(max(0.5, min(2.0, vol_ratio)))


def compute_adaptive_slippage(
    symbol: str,
    timestamp: pd.Timestamp,
    order_size_contracts: float,
    atr_current: float,
    df_volume_stats: pd.DataFrame,
    base_slippage_ticks: float,
) -> float:
    """
    Compute adaptive slippage for a trade.

    Combines three adjustment factors:
    1. Time-of-day (open/close = 2.5x)
    2. Volume impact (sqrt law)
    3. Volatility regime (linear scaling)

    Args:
        symbol: Contract symbol (MES, MGC, etc.)
        timestamp: Trade timestamp
        order_size_contracts: Number of contracts
        atr_current: Current ATR value
        df_volume_stats: DataFrame with typical volume and ATR statistics
        base_slippage_ticks: Base slippage in ticks

    Returns:
        Adjusted slippage in ticks

    Example:
        >>> df_stats = pd.DataFrame({
        ...     'median_volume': [1000],
        ...     'mean_atr': [5.0]
        ... }, index=['MES'])
        >>> slippage = compute_adaptive_slippage(
        ...     symbol='MES',
        ...     timestamp=pd.Timestamp('2024-01-02 09:35:00'),
        ...     order_size_contracts=1,
        ...     atr_current=6.0,
        ...     df_volume_stats=df_stats,
        ...     base_slippage_ticks=0.5
        ... )
        >>> slippage  # ~1.5 ticks (2.5x time-of-day * 1.2x vol * 0.5 base)
    """
    time_mult = get_time_of_day_multiplier(timestamp)

    typical_volume = df_volume_stats.loc[symbol, "median_volume"]
    volume_mult = get_volume_impact_multiplier(order_size_contracts, typical_volume)

    mean_atr = df_volume_stats.loc[symbol, "mean_atr"]
    vol_mult = get_volatility_multiplier(atr_current, mean_atr)

    adjusted_slippage = base_slippage_ticks * time_mult * volume_mult * vol_mult

    return float(adjusted_slippage)


def compute_cost_in_atr_adaptive(
    df: pd.DataFrame,
    symbol: str,
    base_cost_ticks: float,
    tick_value: float,
    order_size_contracts: float = 1.0,
) -> pd.Series:
    """
    Compute time-varying cost_in_atr for triple-barrier labeling.

    This function computes adaptive transaction costs for each bar,
    accounting for time-of-day, volume, and volatility effects.

    For triple-barrier labeling, we typically use the MEDIAN cost
    to ensure consistent barriers across all samples. However,
    this function returns the full series for analysis.

    Args:
        df: DataFrame with timestamp index, volume, and atr_14 columns
        symbol: Contract symbol
        base_cost_ticks: Base commission + slippage (ticks)
        tick_value: Dollar value per tick
        order_size_contracts: Number of contracts (default 1 for labeling)

    Returns:
        Series of cost_in_atr values (time-varying)

    Example:
        >>> df = pd.read_parquet('data/processed/MES_5min.parquet')
        >>> costs = compute_cost_in_atr_adaptive(
        ...     df=df,
        ...     symbol='MES',
        ...     base_cost_ticks=0.5,
        ...     tick_value=1.25,
        ...     order_size_contracts=1.0
        ... )
        >>> median_cost = costs.median()  # Use for triple-barrier
        >>> print(f"Median cost_in_atr: {median_cost:.4f}")
    """
    volume_stats = pd.DataFrame(
        {
            "median_volume": [df["volume"].median()],
            "mean_atr": [df["atr_14"].mean()],
        },
        index=[symbol],
    )

    adaptive_costs = []

    for idx, row in df.iterrows():
        slippage_ticks = compute_adaptive_slippage(
            symbol=symbol,
            timestamp=idx,
            order_size_contracts=order_size_contracts,
            atr_current=row["atr_14"],
            df_volume_stats=volume_stats,
            base_slippage_ticks=base_cost_ticks,
        )

        cost_dollars = slippage_ticks * tick_value

        if row["atr_14"] > 0:
            cost_in_atr = cost_dollars / row["atr_14"]
        else:
            cost_in_atr = 0.0

        adaptive_costs.append(cost_in_atr)

    return pd.Series(adaptive_costs, index=df.index, name="cost_in_atr_adaptive")


def compute_cost_in_atr_adaptive_fast(
    df: pd.DataFrame,
    symbol: str,
    base_cost_ticks: float,
    tick_value: float,
    order_size_contracts: float = 1.0,
) -> float:
    """
    Fast version: compute median adaptive cost directly.

    This is optimized for triple-barrier labeling where we only
    need the median cost, not the full time series.

    Args:
        df: DataFrame with timestamp index, volume, and atr_14 columns
        symbol: Contract symbol
        base_cost_ticks: Base commission + slippage (ticks)
        tick_value: Dollar value per tick
        order_size_contracts: Number of contracts (default 1)

    Returns:
        Median cost_in_atr (scalar)
    """
    costs_series = compute_cost_in_atr_adaptive(
        df=df,
        symbol=symbol,
        base_cost_ticks=base_cost_ticks,
        tick_value=tick_value,
        order_size_contracts=order_size_contracts,
    )

    median_cost = costs_series.median()

    logger.info(
        f"Adaptive costs for {symbol}: "
        f"median={median_cost:.4f}, "
        f"range=[{costs_series.min():.4f}, {costs_series.max():.4f}]"
    )

    return float(median_cost)


__all__ = [
    "get_time_of_day_multiplier",
    "get_volume_impact_multiplier",
    "get_volatility_multiplier",
    "compute_adaptive_slippage",
    "compute_cost_in_atr_adaptive",
    "compute_cost_in_atr_adaptive_fast",
]
