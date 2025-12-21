"""
Regime indicator features for feature engineering.

This module provides functions to calculate market regime indicators
including volatility regime and trend regime classifications.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def add_regime_features(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add regime indicators.

    Calculates volatility regime (high/low) and trend regime (up/down/sideways).
    Requires that volatility and moving average features have already been added.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hvol_20, sma_50, sma_200, and close columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with regime features added
    """
    logger.info("Adding regime features...")

    # Volatility regime (high/low based on historical volatility)
    if 'hvol_20' in df.columns:
        hvol_median = df['hvol_20'].rolling(window=100).median()
        df['volatility_regime'] = (df['hvol_20'] > hvol_median).astype(int)

        feature_metadata['volatility_regime'] = "Volatility regime (1=high, 0=low)"

    # Trend regime based on price vs moving averages
    if 'sma_50' in df.columns and 'sma_200' in df.columns:
        # Uptrend: price > SMA50 > SMA200
        uptrend = (df['close'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])

        # Downtrend: price < SMA50 < SMA200
        downtrend = (df['close'] < df['sma_50']) & (df['sma_50'] < df['sma_200'])

        # Trend regime: 1=up, -1=down, 0=sideways
        df['trend_regime'] = np.where(uptrend, 1, np.where(downtrend, -1, 0))

        feature_metadata['trend_regime'] = "Trend regime (1=up, -1=down, 0=sideways)"

    return df


def add_volatility_regime(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add volatility regime indicator only.

    Standalone function for volatility regime classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hvol_20 column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with volatility regime added
    """
    logger.info("Adding volatility regime...")

    if 'hvol_20' not in df.columns:
        logger.warning("hvol_20 not found, skipping volatility regime")
        return df

    hvol_median = df['hvol_20'].rolling(window=100).median()
    df['volatility_regime'] = (df['hvol_20'] > hvol_median).astype(int)

    feature_metadata['volatility_regime'] = "Volatility regime (1=high, 0=low)"

    return df


def add_trend_regime(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add trend regime indicator only.

    Standalone function for trend regime classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with sma_50, sma_200, and close columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with trend regime added
    """
    logger.info("Adding trend regime...")

    if 'sma_50' not in df.columns or 'sma_200' not in df.columns:
        logger.warning("SMA features not found, skipping trend regime")
        return df

    # Uptrend: price > SMA50 > SMA200
    uptrend = (df['close'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])

    # Downtrend: price < SMA50 < SMA200
    downtrend = (df['close'] < df['sma_50']) & (df['sma_50'] < df['sma_200'])

    # Trend regime: 1=up, -1=down, 0=sideways
    df['trend_regime'] = np.where(uptrend, 1, np.where(downtrend, -1, 0))

    feature_metadata['trend_regime'] = "Trend regime (1=up, -1=down, 0=sideways)"

    return df


__all__ = [
    'add_regime_features',
    'add_volatility_regime',
    'add_trend_regime',
]
