"""
Volume-based features for feature engineering.

This module provides functions to calculate volume-based technical
indicators including OBV, VWAP, volume ratios, and volume z-scores.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def add_volume_features(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add volume-based features.

    Calculates OBV, volume SMA, volume ratio, and volume z-score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and 'volume' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with volume features added (if volume data exists)
    """
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        logger.info("Skipping volume features (no volume data)")
        return df

    logger.info("Adding volume features...")

    # OBV
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv

    # OBV SMA
    df['obv_sma_20'] = obv.rolling(window=20).mean()

    # Volume SMA and ratios
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Volume z-score
    vol_mean = df['volume'].rolling(window=20).mean()
    vol_std = df['volume'].rolling(window=20).std()
    df['volume_zscore'] = (df['volume'] - vol_mean) / vol_std

    feature_metadata['obv'] = "On Balance Volume"
    feature_metadata['obv_sma_20'] = "OBV 20-period SMA"
    feature_metadata['volume_sma_20'] = "Volume 20-period SMA"
    feature_metadata['volume_ratio'] = "Volume ratio to 20-period SMA"
    feature_metadata['volume_zscore'] = "Volume z-score (20)"

    return df


def add_vwap(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add VWAP (session-based).

    Calculates Volume Weighted Average Price with daily session resets
    and price-to-VWAP ratio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC, volume, and datetime columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with VWAP features added (if volume data exists)
    """
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        logger.info("Skipping VWAP (no volume data)")
        return df

    logger.info("Adding VWAP...")

    # Typical price
    tp = (df['high'] + df['low'] + df['close']) / 3

    # Session-based VWAP (daily reset)
    df['date'] = df['datetime'].dt.date

    vwap_list = []
    for date, group in df.groupby('date'):
        cum_tp_vol = (tp.loc[group.index] * df.loc[group.index, 'volume']).cumsum()
        cum_vol = df.loc[group.index, 'volume'].cumsum()
        vwap = cum_tp_vol / cum_vol
        vwap_list.append(vwap)

    df['vwap'] = pd.concat(vwap_list)

    # Price to VWAP ratio
    df['price_to_vwap'] = df['close'] / df['vwap'] - 1

    df = df.drop('date', axis=1)

    feature_metadata['vwap'] = "Volume Weighted Average Price (session)"
    feature_metadata['price_to_vwap'] = "Price deviation from VWAP"

    return df


def add_obv(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add On Balance Volume indicator.

    This is a standalone OBV function if you need just OBV without other volume features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and 'volume' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with OBV added (if volume data exists)
    """
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        logger.info("Skipping OBV (no volume data)")
        return df

    logger.info("Adding OBV...")

    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv

    feature_metadata['obv'] = "On Balance Volume"

    return df


__all__ = [
    'add_volume_features',
    'add_vwap',
    'add_obv',
]
