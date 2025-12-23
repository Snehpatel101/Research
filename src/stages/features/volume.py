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


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide, returning NaN when denominator is zero."""
    return numerator / denominator.replace(0, np.nan)


def add_volume_features(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    period: int = 20
) -> pd.DataFrame:
    """
    Add volume-based features.

    Calculates OBV, volume SMA, volume ratio, and volume z-score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and 'volume' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Period for volume SMA and z-score

    Returns
    -------
    pd.DataFrame
        DataFrame with volume features added (if volume data exists)
    """
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        logger.info("Skipping volume features (no volume data)")
        return df

    logger.info(f"Adding volume features with period: {period}")

    # ANTI-LOOKAHEAD: All volume features shifted by 1 bar
    # OBV - computed then shifted
    obv_raw = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv_raw.shift(1)

    # OBV SMA - computed on raw, then shifted
    obv_sma_col = f'obv_sma_{period}'
    df[obv_sma_col] = obv_raw.rolling(window=period).mean().shift(1)

    # Volume SMA and ratios - all shifted
    volume_sma_raw = df['volume'].rolling(window=period).mean()
    volume_sma_col = f'volume_sma_{period}'
    df[volume_sma_col] = volume_sma_raw.shift(1)
    df['volume_ratio'] = (df['volume'] / volume_sma_raw).shift(1)

    # Volume z-score (safe: std could be 0 when volume is constant)
    vol_mean = df['volume'].rolling(window=period).mean()
    vol_std = df['volume'].rolling(window=period).std()
    volume_zscore_raw = _safe_divide(df['volume'] - vol_mean, vol_std)
    df['volume_zscore'] = volume_zscore_raw.shift(1)

    feature_metadata['obv'] = "On Balance Volume (lagged)"
    feature_metadata[obv_sma_col] = f"OBV {period}-period SMA (lagged)"
    feature_metadata[volume_sma_col] = f"Volume {period}-period SMA (lagged)"
    feature_metadata['volume_ratio'] = f"Volume ratio to {period}-period SMA (lagged)"
    feature_metadata['volume_zscore'] = f"Volume z-score ({period}, lagged)"

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

    # Calculate VWAP per day without breaking index alignment
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['date'] = df['datetime'].dt.date

    # Cumulative volume and price*volume per session (vectorized with transform)
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['tp_vol'] = df['typical_price'] * df['volume']
    df['cum_vwap_num'] = df.groupby('date')['tp_vol'].cumsum()

    # Safe division - fallback to typical price if no volume
    vwap_raw = np.where(
        df['cum_vol'] > 0,
        df['cum_vwap_num'] / df['cum_vol'],
        df['typical_price']
    )

    # ANTI-LOOKAHEAD: shift(1) ensures VWAP at bar[t] uses data up to bar[t-1]
    df['vwap'] = pd.Series(vwap_raw, index=df.index).shift(1)

    # Price to VWAP ratio - compare previous close to previous VWAP
    df['price_to_vwap'] = df['close'].shift(1) / df['vwap'] - 1

    # Cleanup temporary columns
    df = df.drop(columns=['date', 'cum_vol', 'cum_vwap_num', 'typical_price', 'tp_vol'])

    feature_metadata['vwap'] = "Volume Weighted Average Price (session, lagged)"
    feature_metadata['price_to_vwap'] = "Price deviation from VWAP (lagged)"

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

    # ANTI-LOOKAHEAD: shift(1) ensures OBV at bar[t] uses data up to bar[t-1]
    obv_raw = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv_raw.shift(1)

    feature_metadata['obv'] = "On Balance Volume (lagged)"

    return df


__all__ = [
    'add_volume_features',
    'add_vwap',
    'add_obv',
]
