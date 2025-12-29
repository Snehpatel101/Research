"""
Trend indicator features for feature engineering.

This module provides functions to calculate trend-based technical
indicators including ADX and Supertrend.
"""

import logging

import numpy as np
import pandas as pd

from .numba_functions import calculate_adx_numba, calculate_atr_numba

logger = logging.getLogger(__name__)


def add_adx(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    period: int = 14
) -> pd.DataFrame:
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
        df['high'].values,
        df['low'].values,
        df['close'].values,
        period
    )

    # ANTI-LOOKAHEAD: shift(1) ensures ADX at bar[t] uses data up to bar[t-1]
    adx_col = f'adx_{period}'
    plus_di_col = f'plus_di_{period}'
    minus_di_col = f'minus_di_{period}'

    df[adx_col] = pd.Series(adx).shift(1).values
    df[plus_di_col] = pd.Series(plus_di).shift(1).values
    df[minus_di_col] = pd.Series(minus_di).shift(1).values

    # Trend strength - already shifted via adx column
    df['adx_strong_trend'] = (df[adx_col] > 25).astype(int)

    feature_metadata[adx_col] = f"Average Directional Index ({period}, lagged)"
    feature_metadata[plus_di_col] = f"+DI ({period}, lagged)"
    feature_metadata[minus_di_col] = f"-DI ({period}, lagged)"
    feature_metadata['adx_strong_trend'] = "ADX strong trend flag (>25, lagged)"

    return df


def add_supertrend(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    period: int = 10,
    multiplier: float = 3.0
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
    n = len(df)

    # Extract numpy arrays for performance
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Calculate ATR
    atr = calculate_atr_numba(high, low, close, period)

    # Calculate basic bands: midpoint +/- multiplier * ATR
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Initialize output arrays
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = uptrend, -1 = downtrend

    # Set initial values at first valid ATR index (period, not period-1)
    # ATR is first valid at index 'period', so we start there
    # Start in uptrend by convention
    upper_band[period] = basic_upper[period]
    lower_band[period] = basic_lower[period]
    supertrend[period] = basic_lower[period]
    direction[period] = 1

    for i in range(period + 1, n):
        # Update upper band: can only decrease, or reset if price broke above
        if basic_upper[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
            upper_band[i] = basic_upper[i]
        else:
            upper_band[i] = upper_band[i - 1]

        # Update lower band: can only increase, or reset if price broke below
        if basic_lower[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
            lower_band[i] = basic_lower[i]
        else:
            lower_band[i] = lower_band[i - 1]

        # Determine trend direction based on previous direction and current price
        if direction[i - 1] == 1:  # Was in uptrend
            if close[i] < lower_band[i]:
                # Price broke below support -> switch to downtrend
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                # Stay in uptrend
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:  # Was in downtrend
            if close[i] > upper_band[i]:
                # Price broke above resistance -> switch to uptrend
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                # Stay in downtrend
                direction[i] = -1
                supertrend[i] = upper_band[i]

    # Set NaN for warmup period where ATR is not valid
    # ATR is first valid at index 'period', so indices 0 to period-1 are NaN
    supertrend[:period] = np.nan
    direction[:period] = np.nan

    # ANTI-LOOKAHEAD: shift(1) ensures Supertrend at bar[t] uses data up to bar[t-1]
    df['supertrend'] = pd.Series(supertrend).shift(1).values
    df['supertrend_direction'] = pd.Series(direction).shift(1).values

    feature_metadata['supertrend'] = f"Supertrend ({period},{multiplier}, lagged)"
    feature_metadata['supertrend_direction'] = "Supertrend direction (1=up, -1=down, lagged)"

    return df


__all__ = [
    'add_adx',
    'add_supertrend',
]
