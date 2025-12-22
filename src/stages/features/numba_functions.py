"""
Numba-optimized functions for feature calculations.

All functions in this module use the @jit decorator with nopython=True
for maximum performance. These are the core computational routines
used by the feature engineering modules.
"""

import numpy as np
from numba import jit
from typing import Tuple


@jit(nopython=True)
def calculate_sma_numba(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using Numba.

    Parameters
    ----------
    arr : np.ndarray
        Input price array
    period : int
        Lookback period for SMA calculation

    Returns
    -------
    np.ndarray
        SMA values with NaN for initial warmup period
    """
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        result[i] = np.mean(arr[i - period + 1:i + 1])

    return result


@jit(nopython=True)
def calculate_ema_numba(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using Numba.

    Handles input arrays with leading NaN values (e.g., from prior calculations).

    Parameters
    ----------
    arr : np.ndarray
        Input price array (may have leading NaN values)
    period : int
        Lookback period for EMA calculation

    Returns
    -------
    np.ndarray
        EMA values with NaN for initial warmup period
    """
    n = len(arr)
    result = np.full(n, np.nan)

    alpha = 2.0 / (period + 1)

    # Find first valid (non-NaN) index
    first_valid = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            first_valid = i
            break
    else:
        # All NaN, return all NaN
        return result

    # Calculate starting index for EMA (need 'period' valid values)
    start_idx = first_valid + period - 1

    if start_idx >= n:
        # Not enough data for EMA calculation
        return result

    # Start with SMA of first 'period' valid values
    sma_sum = 0.0
    for i in range(first_valid, first_valid + period):
        sma_sum += arr[i]
    result[start_idx] = sma_sum / period

    # Calculate EMA for remaining values
    for i in range(start_idx + 1, n):
        if not np.isnan(arr[i]):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        else:
            result[i] = result[i - 1]  # Carry forward if input is NaN

    return result


@jit(nopython=True)
def calculate_rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close price array
    period : int, default 14
        RSI calculation period

    Returns
    -------
    np.ndarray
        RSI values (0-100 scale) with NaN for warmup period
    """
    n = len(close)
    rsi = np.full(n, np.nan)

    # Calculate price changes
    deltas = np.diff(close)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate RSI for remaining periods
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@jit(nopython=True)
def calculate_atr_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range using Numba.

    Parameters
    ----------
    high : np.ndarray
        High price array
    low : np.ndarray
        Low price array
    close : np.ndarray
        Close price array
    period : int, default 14
        ATR calculation period

    Returns
    -------
    np.ndarray
        ATR values with NaN for warmup period
    """
    n = len(high)
    tr = np.zeros(n)
    atr = np.full(n, np.nan)

    # Calculate True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Calculate ATR
    atr[period] = np.mean(tr[1:period + 1])

    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


@jit(nopython=True)
def calculate_stochastic_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator using Numba.

    Parameters
    ----------
    high : np.ndarray
        High price array
    low : np.ndarray
        Low price array
    close : np.ndarray
        Close price array
    k_period : int, default 14
        %K calculation period
    d_period : int, default 3
        %D smoothing period

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (%K values, %D values)
    """
    n = len(close)
    k = np.full(n, np.nan)
    d = np.full(n, np.nan)

    for i in range(k_period - 1, n):
        high_max = np.max(high[i - k_period + 1:i + 1])
        low_min = np.min(low[i - k_period + 1:i + 1])

        if high_max - low_min != 0:
            k[i] = 100.0 * (close[i] - low_min) / (high_max - low_min)
        else:
            k[i] = 50.0

    # Calculate %D (SMA of %K)
    for i in range(k_period + d_period - 2, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])

    return k, d


@jit(nopython=True)
def calculate_rolling_correlation_numba(
    x: np.ndarray,
    y: np.ndarray,
    period: int
) -> np.ndarray:
    """
    Calculate rolling correlation using Numba.

    Parameters
    ----------
    x : np.ndarray
        First time series
    y : np.ndarray
        Second time series
    period : int
        Rolling window period

    Returns
    -------
    np.ndarray
        Rolling correlation values
    """
    n = len(x)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        x_window = x[i - period + 1:i + 1]
        y_window = y[i - period + 1:i + 1]

        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)

        x_centered = x_window - x_mean
        y_centered = y_window - y_mean

        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))

        if denominator > 0:
            result[i] = numerator / denominator
        else:
            result[i] = 0.0

    return result


@jit(nopython=True)
def calculate_rolling_beta_numba(
    y: np.ndarray,
    x: np.ndarray,
    period: int
) -> np.ndarray:
    """
    Calculate rolling beta (regression coefficient) of y on x using Numba.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (e.g., asset returns)
    x : np.ndarray
        Independent variable (e.g., market returns)
    period : int
        Rolling window period

    Returns
    -------
    np.ndarray
        Rolling beta values
    """
    n = len(x)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        x_window = x[i - period + 1:i + 1]
        y_window = y[i - period + 1:i + 1]

        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)

        x_centered = x_window - x_mean
        y_centered = y_window - y_mean

        denominator = np.sum(x_centered ** 2)

        if denominator > 0:
            result[i] = np.sum(x_centered * y_centered) / denominator
        else:
            result[i] = 0.0

    return result


@jit(nopython=True)
def calculate_adx_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ADX, +DI, -DI using Numba.

    Parameters
    ----------
    high : np.ndarray
        High price array
    low : np.ndarray
        Low price array
    close : np.ndarray
        Close price array
    period : int, default 14
        ADX calculation period

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (ADX, +DI, -DI) arrays
    """
    n = len(close)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    adx = np.full(n, np.nan)

    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]

        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff

    # Smooth TR and DM
    tr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)

    tr_smooth[period] = np.sum(tr[1:period + 1])
    plus_dm_smooth[period] = np.sum(plus_dm[1:period + 1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i]

    # Calculate DI
    for i in range(period, n):
        if tr_smooth[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]

    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX is EMA of DX
    adx[2 * period - 1] = np.mean(dx[period:2 * period])
    for i in range(2 * period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


__all__ = [
    'calculate_sma_numba',
    'calculate_ema_numba',
    'calculate_rsi_numba',
    'calculate_atr_numba',
    'calculate_stochastic_numba',
    'calculate_rolling_correlation_numba',
    'calculate_rolling_beta_numba',
    'calculate_adx_numba',
]
