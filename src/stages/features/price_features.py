"""
Price-based features for feature engineering.

This module provides functions to calculate price-based features
including returns and price ratios.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def add_returns(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add return features.

    Calculates both simple and log returns over multiple periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with return features added
    """
    logger.info("Adding return features...")

    periods = [1, 5, 10, 20, 60]

    for period in periods:
        # Simple returns
        df[f'return_{period}'] = df['close'].pct_change(period)

        # Log returns
        df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        feature_metadata[f'return_{period}'] = f"{period}-period simple return"
        feature_metadata[f'log_return_{period}'] = f"{period}-period log return"

    return df


def add_price_ratios(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add price ratio features.

    Calculates high/low ratio, close/open ratio, and range as percentage of close.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with price ratio features added
    """
    logger.info("Adding price ratio features...")

    # High/Low ratio
    df['hl_ratio'] = df['high'] / df['low']

    # Close/Open ratio
    df['co_ratio'] = df['close'] / df['open']

    # Range as percentage of close
    df['range_pct'] = (df['high'] - df['low']) / df['close']

    feature_metadata['hl_ratio'] = "High to low ratio"
    feature_metadata['co_ratio'] = "Close to open ratio"
    feature_metadata['range_pct'] = "Range as percentage of close"

    return df


__all__ = [
    'add_returns',
    'add_price_ratios',
]
