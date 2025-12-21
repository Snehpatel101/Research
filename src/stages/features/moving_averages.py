"""
Moving average features for feature engineering.

This module provides functions to calculate Simple Moving Averages (SMA)
and Exponential Moving Averages (EMA) along with price-to-MA ratios.
"""

import pandas as pd
import logging
from typing import Dict

from .numba_functions import calculate_sma_numba, calculate_ema_numba

logger = logging.getLogger(__name__)


def add_sma(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Simple Moving Averages.

    Calculates SMA for multiple periods and price position relative to each SMA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with SMA features added
    """
    logger.info("Adding SMA features...")

    periods = [10, 20, 50, 100, 200]

    for period in periods:
        df[f'sma_{period}'] = calculate_sma_numba(df['close'].values, period)

        # Price position relative to SMA
        df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1

        feature_metadata[f'sma_{period}'] = f"{period}-period Simple Moving Average"
        feature_metadata[f'price_to_sma_{period}'] = f"Price deviation from SMA-{period}"

    return df


def add_ema(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Exponential Moving Averages.

    Calculates EMA for multiple periods and price position relative to each EMA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with EMA features added
    """
    logger.info("Adding EMA features...")

    periods = [9, 12, 21, 26, 50]

    for period in periods:
        df[f'ema_{period}'] = calculate_ema_numba(df['close'].values, period)

        # Price position relative to EMA
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

        feature_metadata[f'ema_{period}'] = f"{period}-period Exponential Moving Average"
        feature_metadata[f'price_to_ema_{period}'] = f"Price deviation from EMA-{period}"

    return df


__all__ = [
    'add_sma',
    'add_ema',
]
