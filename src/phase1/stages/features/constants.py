"""
Timeframe-aware constants for feature engineering.

This module provides dynamic calculation of bars_per_day and annualization
factors based on the trading timeframe, replacing hardcoded values.

Key Functions:
    get_bars_per_day: Calculate bars per trading day for a given timeframe
    get_annualization_factor: Calculate annualization factor for volatility scaling

Example:
    >>> from stages.features.constants import get_annualization_factor
    >>> factor_5min = get_annualization_factor('5min')  # ~140.07
    >>> factor_15min = get_annualization_factor('15min')  # ~80.92
"""

import math

import numpy as np

# =============================================================================
# TRADING TIME CONSTANTS
# =============================================================================

# Trading hours per day
# Regular trading hours: 6.5 hours (9:30 AM - 4:00 PM ET for equities)
# Extended hours for futures (CME ES/MES): 23 hours with 1 hour maintenance
TRADING_HOURS_REGULAR = 6.5
TRADING_HOURS_EXTENDED = 23.0

# Trading days per year (standard for US markets)
TRADING_DAYS_PER_YEAR = 252

# Timeframe to minutes mapping
TIMEFRAME_MINUTES: dict[str, int] = {
    "1min": 1,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "20min": 20,
    "30min": 30,
    "45min": 45,
    "60min": 60,
    "1h": 60,
}


# =============================================================================
# DYNAMIC CALCULATION FUNCTIONS
# =============================================================================


def get_bars_per_day(timeframe: str, extended_hours: bool = False) -> float:
    """
    Calculate bars per trading day for a given timeframe.

    Parameters
    ----------
    timeframe : str
        Bar timeframe (e.g., '5min', '15min', '1h')
    extended_hours : bool, default False
        If True, use 23-hour session (futures).
        If False, use 6.5-hour regular session (equities).

    Returns
    -------
    float
        Number of bars per trading day

    Raises
    ------
    ValueError
        If timeframe is not recognized

    Examples
    --------
    >>> get_bars_per_day('5min')
    78.0
    >>> get_bars_per_day('15min')
    26.0
    >>> get_bars_per_day('1h')
    6.5
    >>> get_bars_per_day('5min', extended_hours=True)
    276.0
    """
    minutes = TIMEFRAME_MINUTES.get(timeframe.lower())
    if minutes is None:
        raise ValueError(
            f"Unknown timeframe: '{timeframe}'. "
            f"Supported timeframes: {list(TIMEFRAME_MINUTES.keys())}"
        )

    hours = TRADING_HOURS_EXTENDED if extended_hours else TRADING_HOURS_REGULAR
    minutes_per_day = hours * 60

    return minutes_per_day / minutes


def get_annualization_factor(timeframe: str, extended_hours: bool = False) -> float:
    """
    Calculate annualization factor for volatility scaling.

    Used to annualize returns/volatility:
    - Daily: annual_vol = daily_vol * sqrt(252)
    - Intraday: annual_vol = bar_vol * sqrt(bars_per_day * 252)

    Parameters
    ----------
    timeframe : str
        Bar timeframe (e.g., '5min', '15min', '1h')
    extended_hours : bool, default False
        Whether to use extended trading hours (23h for futures)

    Returns
    -------
    float
        Annualization factor

    Raises
    ------
    ValueError
        If timeframe is not recognized

    Examples
    --------
    >>> factor = get_annualization_factor('5min')
    >>> abs(factor - 140.07) < 0.01
    True
    >>> get_annualization_factor('15min') < get_annualization_factor('5min')
    True
    """
    bars_per_day = get_bars_per_day(timeframe, extended_hours)
    return math.sqrt(bars_per_day * TRADING_DAYS_PER_YEAR)


# =============================================================================
# PRE-COMPUTED LOOKUP TABLES (Regular Hours)
# =============================================================================

# Bars per day for common timeframes (regular 6.5-hour session)
BARS_PER_DAY_MAP: dict[str, float] = {
    tf: get_bars_per_day(tf, extended_hours=False) for tf in TIMEFRAME_MINUTES.keys()
}

# Annualization factors for common timeframes (regular session)
ANNUALIZATION_FACTOR_MAP: dict[str, float] = {
    tf: get_annualization_factor(tf, extended_hours=False) for tf in TIMEFRAME_MINUTES.keys()
}


# =============================================================================
# BACKWARD COMPATIBILITY CONSTANTS
# =============================================================================

# Default values for 5-minute bars (most common timeframe)
# These maintain backward compatibility with existing code
BARS_PER_DAY = 78  # 6.5 hours * 60 / 5 = 78 bars
ANNUALIZATION_FACTOR = np.sqrt(TRADING_DAYS_PER_YEAR * BARS_PER_DAY)  # ~140.07


__all__ = [
    # Trading time constants
    "TRADING_HOURS_REGULAR",
    "TRADING_HOURS_EXTENDED",
    "TRADING_DAYS_PER_YEAR",
    "TIMEFRAME_MINUTES",
    # Dynamic functions
    "get_bars_per_day",
    "get_annualization_factor",
    # Lookup tables
    "BARS_PER_DAY_MAP",
    "ANNUALIZATION_FACTOR_MAP",
    # Backward compatibility (5min defaults)
    "BARS_PER_DAY",
    "ANNUALIZATION_FACTOR",
]
