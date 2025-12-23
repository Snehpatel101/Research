"""
Moving average features for feature engineering.

This module provides functions to calculate Simple Moving Averages (SMA)
and Exponential Moving Averages (EMA) along with price-to-MA ratios.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

from .numba_functions import calculate_sma_numba, calculate_ema_numba

logger = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide, returning NaN when denominator is zero."""
    return numerator / denominator.replace(0, np.nan)


def add_sma(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    periods: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Add Simple Moving Averages.

    Calculates SMA for multiple periods and price position relative to each SMA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of SMA periods. Default: [10, 20, 50, 100, 200]

    Returns
    -------
    pd.DataFrame
        DataFrame with SMA features added
    """
    if periods is None:
        periods = [10, 20, 50, 100, 200]

    logger.info(f"Adding SMA features with periods: {periods}")

    for period in periods:
        # ANTI-LOOKAHEAD: shift(1) ensures SMA at bar[t] uses data up to bar[t-1]
        sma_raw = pd.Series(calculate_sma_numba(df['close'].values, period))
        df[f'sma_{period}'] = sma_raw.shift(1).values

        # Price position relative to SMA uses previous close vs shifted SMA
        # This compares close[t-1] to SMA[t-1], available at bar[t]
        df[f'price_to_sma_{period}'] = _safe_divide(
            df['close'].shift(1), df[f'sma_{period}']
        ) - 1

        feature_metadata[f'sma_{period}'] = f"{period}-period Simple Moving Average (lagged)"
        feature_metadata[f'price_to_sma_{period}'] = f"Price deviation from SMA-{period} (lagged)"

    return df


def add_ema(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    periods: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Add Exponential Moving Averages.

    Calculates EMA for multiple periods and price position relative to each EMA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of EMA periods. Default: [9, 12, 21, 26, 50]

    Returns
    -------
    pd.DataFrame
        DataFrame with EMA features added
    """
    if periods is None:
        periods = [9, 12, 21, 26, 50]

    logger.info(f"Adding EMA features with periods: {periods}")

    for period in periods:
        # ANTI-LOOKAHEAD: shift(1) ensures EMA at bar[t] uses data up to bar[t-1]
        ema_raw = pd.Series(calculate_ema_numba(df['close'].values, period))
        df[f'ema_{period}'] = ema_raw.shift(1).values

        # Price position relative to EMA uses previous close vs shifted EMA
        # This compares close[t-1] to EMA[t-1], available at bar[t]
        df[f'price_to_ema_{period}'] = _safe_divide(
            df['close'].shift(1), df[f'ema_{period}']
        ) - 1

        feature_metadata[f'ema_{period}'] = f"{period}-period Exponential Moving Average (lagged)"
        feature_metadata[f'price_to_ema_{period}'] = f"Price deviation from EMA-{period} (lagged)"

    return df


__all__ = [
    'add_sma',
    'add_ema',
]
