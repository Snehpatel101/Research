"""
Trend indicator features for feature engineering.

This module provides functions to calculate trend-based technical
indicators including ADX and Supertrend.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

from .numba_functions import calculate_adx_numba, calculate_atr_numba

logger = logging.getLogger(__name__)


def add_adx(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add ADX and Directional Indicators.

    Calculates 14-period ADX, +DI, -DI, and trend strength flag.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with ADX features added
    """
    logger.info("Adding ADX features...")

    adx, plus_di, minus_di = calculate_adx_numba(
        df['high'].values,
        df['low'].values,
        df['close'].values,
        14
    )

    df['adx_14'] = adx
    df['plus_di_14'] = plus_di
    df['minus_di_14'] = minus_di

    # Trend strength
    df['adx_strong_trend'] = (df['adx_14'] > 25).astype(int)

    feature_metadata['adx_14'] = "Average Directional Index (14)"
    feature_metadata['plus_di_14'] = "+DI (14)"
    feature_metadata['minus_di_14'] = "-DI (14)"
    feature_metadata['adx_strong_trend'] = "ADX strong trend flag (>25)"

    return df


def add_supertrend(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Supertrend indicator.

    Calculates Supertrend with period=10 and multiplier=3.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with Supertrend features added
    """
    logger.info("Adding Supertrend...")

    period = 10
    multiplier = 3

    # Calculate ATR
    atr = calculate_atr_numba(df['high'].values, df['low'].values, df['close'].values, period)

    # Calculate basic bands
    hl_avg = (df['high'] + df['low']) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)

    # Initialize supertrend
    supertrend = np.full(len(df), np.nan)
    direction = np.ones(len(df))

    for i in range(1, len(df)):
        if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
            # Upper band
            if upper_band[i] < supertrend[i-1] if not np.isnan(supertrend[i-1]) else True:
                upper_band[i] = supertrend[i-1] if not np.isnan(supertrend[i-1]) else upper_band[i]

            # Lower band
            if lower_band[i] > supertrend[i-1] if not np.isnan(supertrend[i-1]) else True:
                lower_band[i] = supertrend[i-1] if not np.isnan(supertrend[i-1]) else lower_band[i]

            # Determine supertrend
            if df['close'].iloc[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1

    df['supertrend'] = supertrend
    df['supertrend_direction'] = direction

    feature_metadata['supertrend'] = "Supertrend (10,3)"
    feature_metadata['supertrend_direction'] = "Supertrend direction (1=up, -1=down)"

    return df


__all__ = [
    'add_adx',
    'add_supertrend',
]
