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


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide, returning NaN when denominator is zero."""
    return numerator / denominator.replace(0, np.nan)


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
        # ANTI-LOOKAHEAD: shift(1) ensures return at bar[t] uses close[t-1] vs close[t-1-period]
        # Without shift, return at bar[t] would use close[t], which is not yet available
        df[f'return_{period}'] = df['close'].pct_change(period).shift(1)

        # Log returns
        # ANTI-LOOKAHEAD: shift(1) ensures log return at bar[t] uses close[t-1] vs close[t-1-period]
        df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period)).shift(1)

        feature_metadata[f'return_{period}'] = f"{period}-period simple return (lagged)"
        feature_metadata[f'log_return_{period}'] = f"{period}-period log return (lagged)"

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

    # ANTI-LOOKAHEAD: All price ratios use previous bar's OHLC data
    # Features at bar[t] must only use data available at bar[t-1]

    # High/Low ratio (safe: low could be 0)
    df['hl_ratio'] = _safe_divide(df['high'].shift(1), df['low'].shift(1))

    # Close/Open ratio (safe: open could be 0)
    df['co_ratio'] = _safe_divide(df['close'].shift(1), df['open'].shift(1))

    # Range as percentage of close (safe: close could be 0)
    df['range_pct'] = _safe_divide(
        df['high'].shift(1) - df['low'].shift(1),
        df['close'].shift(1)
    )

    feature_metadata['hl_ratio'] = "High to low ratio (previous bar)"
    feature_metadata['co_ratio'] = "Close to open ratio (previous bar)"
    feature_metadata['range_pct'] = "Range as percentage of close (previous bar)"

    return df


__all__ = [
    'add_returns',
    'add_price_ratios',
]
