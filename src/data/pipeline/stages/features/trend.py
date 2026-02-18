"""
Trend indicator features for feature engineering.

This module provides functions to calculate trend-based technical
indicators including ADX and Supertrend.
"""

import logging

import numpy as np
import pandas as pd
from numba import njit

from .numba_functions import calculate_adx_numba, calculate_atr_numba

logger = logging.getLogger(__name__)


def _np_shift1(arr: np.ndarray) -> np.ndarray:
    """Shift array by 1 using numpy (avoids pd.Series overhead)."""
    result = np.empty_like(arr, dtype=np.float64)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


def add_adx(df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 14) -> pd.DataFrame:
    """
    Add ADX and Directional Indicators.

    Calculates ADX, +DI, -DI, and trend strength flag.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 14
        ADX period

    Returns
    -------
    pd.DataFrame
        DataFrame with ADX features added
    """
    logger.info(f"Adding ADX features with period: {period}")

    adx, plus_di, minus_di = calculate_adx_numba(
        df["high"].values, df["low"].values, df["close"].values, period
    )

    # ANTI-LOOKAHEAD: shift(1) ensures ADX at bar[t] uses data up to bar[t-1]
    adx_col = f"adx_{period}"
    plus_di_col = f"plus_di_{period}"
    minus_di_col = f"minus_di_{period}"

    adx_shifted = _np_shift1(adx)
    plus_di_shifted = _np_shift1(plus_di)
    minus_di_shifted = _np_shift1(minus_di)

    # Batch concat to avoid fragmentation
    new_cols = {
        adx_col: adx_shifted,
        plus_di_col: plus_di_shifted,
        minus_di_col: minus_di_shifted,
        "adx_strong_trend": (adx_shifted > 25).astype(int),
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    feature_metadata[adx_col] = f"Average Directional Index ({period}, lagged)"
    feature_metadata[plus_di_col] = f"+DI ({period}, lagged)"
    feature_metadata[minus_di_col] = f"-DI ({period}, lagged)"
    feature_metadata["adx_strong_trend"] = "ADX strong trend flag (>25, lagged)"

    return df


@njit(cache=True)
def _supertrend_loop(
    close: np.ndarray,
    basic_upper: np.ndarray,
    basic_lower: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-optimized Supertrend band/direction loop."""
    n = len(close)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    upper_band[period] = basic_upper[period]
    lower_band[period] = basic_lower[period]
    supertrend[period] = basic_lower[period]
    direction[period] = 1.0

    for i in range(period + 1, n):
        if basic_upper[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
            upper_band[i] = basic_upper[i]
        else:
            upper_band[i] = upper_band[i - 1]

        if basic_lower[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
            lower_band[i] = basic_lower[i]
        else:
            lower_band[i] = lower_band[i - 1]

        if direction[i - 1] == 1.0:
            if close[i] < lower_band[i]:
                direction[i] = -1.0
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1.0
                supertrend[i] = lower_band[i]
        else:
            if close[i] > upper_band[i]:
                direction[i] = 1.0
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1.0
                supertrend[i] = upper_band[i]

    supertrend[:period] = np.nan
    direction[:period] = np.nan
    return supertrend, direction


def add_supertrend(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Add Supertrend indicator.

    Calculates Supertrend with configurable period and multiplier.

    Supertrend Algorithm:
    - Upper Band = (High + Low) / 2 + multiplier * ATR
    - Lower Band = (High + Low) / 2 - multiplier * ATR
    - In uptrend: Supertrend = Lower Band (support)
    - In downtrend: Supertrend = Upper Band (resistance)

    Band update rules:
    - Upper band can only decrease (tighten) during downtrend
    - Lower band can only increase (tighten) during uptrend

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 10
        ATR period for Supertrend
    multiplier : float, default 3.0
        ATR multiplier

    Returns
    -------
    pd.DataFrame
        DataFrame with Supertrend features added
    """
    logger.info(f"Adding Supertrend with period: {period}, multiplier: {multiplier}")

    # Extract numpy arrays for performance
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Calculate ATR
    atr = calculate_atr_numba(high, low, close, period)

    # Calculate basic bands: midpoint +/- multiplier * ATR
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Run Numba-optimized loop
    supertrend, direction = _supertrend_loop(close, basic_upper, basic_lower, period)

    # ANTI-LOOKAHEAD: shift(1) ensures Supertrend at bar[t] uses data up to bar[t-1]
    # Batch concat to avoid fragmentation
    new_cols = {
        "supertrend": _np_shift1(supertrend),
        "supertrend_direction": _np_shift1(direction),
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    feature_metadata["supertrend"] = f"Supertrend ({period},{multiplier}, lagged)"
    feature_metadata["supertrend_direction"] = "Supertrend direction (1=up, -1=down, lagged)"

    return df


__all__ = [
    "add_adx",
    "add_supertrend",
]
