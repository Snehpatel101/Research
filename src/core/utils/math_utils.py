"""
Common math utilities used across the codebase.

This module provides safe mathematical operations that handle edge cases
like division by zero, NaN values, and moving averages.
"""

import numpy as np
import pandas as pd


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


__all__ = [
    "safe_divide",
    "sma",
    "ema",
    "normalize_series",
]
