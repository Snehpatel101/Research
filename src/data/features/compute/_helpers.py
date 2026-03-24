"""
Shared helper functions for feature computation.

Phase 29: Consolidated duplicate functions to compute once and cache.
Previously had 4 duplicate _log_returns() definitions across:
- entropy.py
- volatility.py
- regime.py
- microstructure.py

Phase H5: Consolidated duplicate _sma, _ema, _rolling_std definitions from:
- momentum.py, moving_average.py, volatility.py, volume.py, regime.py,
  order_flow.py, mean_reversion.py, liquidity.py, microstructure.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """
    Calculate log returns.

    This is the canonical implementation - all feature modules should import
    from here to avoid duplicate computation.

    Args:
        close: Series of close prices

    Returns:
        Series of log returns: log(close_t / close_{t-1})
    """
    return np.log(close / close.shift(1))


def sma(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """
    Simple moving average.

    This is the canonical implementation. All feature compute modules should
    import from here instead of defining their own ``_sma``.

    Args:
        series: Input time series.
        window: Rolling window size.
        min_periods: Minimum number of observations required. Defaults to
            *window* when ``None``.

    Returns:
        Simple moving average values.
    """
    if min_periods is None:
        min_periods = window
    return series.rolling(window=window, min_periods=min_periods).mean()


def ema(series: pd.Series, span: int, min_periods: int | None = None) -> pd.Series:
    """
    Exponential moving average.

    This is the canonical implementation. All feature compute modules should
    import from here instead of defining their own ``_ema``.

    Args:
        series: Input time series.
        span: EMA span (decay period).
        min_periods: Minimum number of observations required. Defaults to
            *span* when ``None``.

    Returns:
        Exponential moving average values.
    """
    if min_periods is None:
        min_periods = span
    return series.ewm(span=span, min_periods=min_periods, adjust=False).mean()


def session_cumsum(series: pd.Series) -> pd.Series:
    """
    Cumulative sum that resets at each trading day boundary.

    Prevents positional leakage by ensuring cumulative features do not
    encode absolute position within the dataset.  For intraday data with a
    ``DatetimeIndex``, the cumsum resets at each new calendar date.  If no
    ``DatetimeIndex`` is available, falls back to a plain cumsum.

    Args:
        series: Input time series with a DatetimeIndex.

    Returns:
        Session-aware cumulative sum (resets daily).
    """
    if isinstance(series.index, pd.DatetimeIndex):
        return series.groupby(series.index.date).cumsum()
    return series.cumsum()


def rolling_std(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """
    Rolling standard deviation.

    This is the canonical implementation. All feature compute modules should
    import from here instead of defining their own ``_rolling_std``.

    Args:
        series: Input time series.
        window: Rolling window size.
        min_periods: Minimum number of observations required. Defaults to
            *window* when ``None``.

    Returns:
        Rolling standard deviation values.
    """
    if min_periods is None:
        min_periods = window
    return series.rolling(window=window, min_periods=min_periods).std()
