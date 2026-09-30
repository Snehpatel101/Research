"""
Common math utilities used across the codebase.

This module provides safe mathematical operations that handle edge cases
like division by zero, NaN values, and moving averages.
"""

import math

import numpy as np
import pandas as pd

# An exponentially weighted average forgets its starting value geometrically;
# once that start's weight is below this fraction the average no longer depends
# on where the series began (for feature warmup purposes).
EWM_SETTLE_TOLERANCE = 1e-3


def safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    fill_value: float = 0.0,
) -> pd.Series | np.ndarray:
    """
    Safely divide, returning fill_value where denominator is zero or NaN.

    Parameters
    ----------
    numerator : pd.Series | np.ndarray
        The numerator values
    denominator : pd.Series | np.ndarray
        The denominator values
    fill_value : float, default 0.0
        Value to use where division is undefined (zero/NaN denominator)

    Returns
    -------
    pd.Series | np.ndarray
        Result of division with fill_value where undefined

    Examples
    --------
    >>> import pandas as pd
    >>> num = pd.Series([10, 20, 30])
    >>> denom = pd.Series([2, 0, 5])
    >>> safe_divide(num, denom)
    0    5.0
    1    0.0
    2    6.0
    dtype: float64
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
        else:
            result = np.where(np.isfinite(result), result, fill_value)
    return result


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple moving average.

    Parameters
    ----------
    series : pd.Series
        Input time series
    period : int
        Rolling window size

    Returns
    -------
    pd.Series
        Simple moving average values

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> sma(s, 3)
    0    1.0
    1    1.5
    2    2.0
    3    3.0
    4    4.0
    dtype: float64
    """
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential moving average.

    Parameters
    ----------
    series : pd.Series
        Input time series
    period : int
        EMA span (decay period)

    Returns
    -------
    pd.Series
        Exponential moving average values

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> ema(s, 3).round(2)
    0    1.0
    1    1.5
    2    2.25
    3    3.12
    4    4.06
    dtype: float64
    """
    return series.ewm(span=period, adjust=False).mean()


def normalize_series(
    series: pd.Series,
    method: str = "zscore",
    window: int | None = None,
) -> pd.Series:
    """
    Normalize a series using various methods.

    Parameters
    ----------
    series : pd.Series
        Input time series
    method : str, default "zscore"
        Normalization method: "zscore", "minmax", or "robust"
    window : int | None, default None
        If provided, use rolling statistics instead of full series

    Returns
    -------
    pd.Series
        Normalized values
    """
    if window is not None:
        if method == "zscore":
            mean = series.rolling(window, min_periods=1).mean()
            std = series.rolling(window, min_periods=1).std()
            return safe_divide(series - mean, std)
        elif method == "minmax":
            roll_min = series.rolling(window, min_periods=1).min()
            roll_max = series.rolling(window, min_periods=1).max()
            return safe_divide(series - roll_min, roll_max - roll_min)
        elif method == "robust":
            median = series.rolling(window, min_periods=1).median()
            q1 = series.rolling(window, min_periods=1).quantile(0.25)
            q3 = series.rolling(window, min_periods=1).quantile(0.75)
            iqr = q3 - q1
            return safe_divide(series - median, iqr)
    else:
        if method == "zscore":
            return safe_divide(series - series.mean(), series.std())
        elif method == "minmax":
            return safe_divide(series - series.min(), series.max() - series.min())
        elif method == "robust":
            median = series.median()
            iqr = series.quantile(0.75) - series.quantile(0.25)
            return safe_divide(series - median, iqr)

    raise ValueError(f"Unknown normalization method: {method}")


def ewm_settle_bars(alpha: float, tolerance: float = EWM_SETTLE_TOLERANCE) -> int:
    """Bars until an EWM's starting value weighs less than ``tolerance`` ((1 - alpha)^n)."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if alpha == 1.0:
        return 1
    return math.ceil(math.log(tolerance) / math.log1p(-alpha))


def span_alpha(span: int) -> float:
    """Smoothing factor of an EWM with ``span`` (pandas / TA convention)."""
    return 2.0 / (span + 1)


def span_settle_bars(span: int, tolerance: float = EWM_SETTLE_TOLERANCE) -> int:
    """:func:`ewm_settle_bars` of an EWM with ``span`` (alpha = 2 / (span + 1))."""
    return ewm_settle_bars(span_alpha(span), tolerance)


def wilder_settle_bars(period: int, tolerance: float = EWM_SETTLE_TOLERANCE) -> int:
    """:func:`ewm_settle_bars` of Wilder smoothing over ``period`` (alpha = 1 / period)."""
    return ewm_settle_bars(1.0 / period, tolerance)


def cascade_settle_bars(
    inner_alpha: float, outer_alpha: float, tolerance: float = EWM_SETTLE_TOLERANCE
) -> int:
    """Bars until an EWM of an EWM (e.g. a MACD signal line) forgets its start.

    The sum of both settling times: the inner EWM's start has faded after its
    own settling time, and the outer EWM's memory of the inner error after its
    own. A start error can be several times a feature's spread (an EMA starts
    at the first price of the window), so this headroom over the tightest
    geometric bound is what keeps such features within ``tolerance`` of their
    spread, not just of their start error.
    """
    return ewm_settle_bars(inner_alpha, tolerance) + ewm_settle_bars(outer_alpha, tolerance)


__all__ = [
    "EWM_SETTLE_TOLERANCE",
    "cascade_settle_bars",
    "ewm_settle_bars",
    "span_alpha",
    "span_settle_bars",
    "wilder_settle_bars",
    "safe_divide",
    "sma",
    "ema",
    "normalize_series",
]
