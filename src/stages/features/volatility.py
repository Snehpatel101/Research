"""
Volatility indicator features for feature engineering.

This module provides functions to calculate volatility-based technical
indicators including ATR, Bollinger Bands, Keltner Channels, and various
historical volatility measures.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

from .constants import ANNUALIZATION_FACTOR
from .numba_functions import calculate_atr_numba, calculate_ema_numba

logger = logging.getLogger(__name__)


def add_atr(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Average True Range features.

    Calculates ATR for multiple periods with both absolute and percentage values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with ATR features added
    """
    logger.info("Adding ATR features...")

    periods = [7, 14, 21]

    for period in periods:
        atr = calculate_atr_numba(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            period
        )
        df[f'atr_{period}'] = atr

        # ATR as percentage of close
        df[f'atr_pct_{period}'] = (atr / df['close']) * 100

        feature_metadata[f'atr_{period}'] = f"Average True Range ({period})"
        feature_metadata[f'atr_pct_{period}'] = f"ATR as % of price ({period})"

    return df


def add_bollinger_bands(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Bollinger Bands features.

    Calculates 20-period Bollinger Bands with 2 standard deviations,
    band width, and price position within the bands.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with Bollinger Band features added
    """
    logger.info("Adding Bollinger Bands...")

    period = 20
    std_mult = 2

    df['bb_middle'] = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()

    df['bb_upper'] = df['bb_middle'] + (std_mult * bb_std)
    df['bb_lower'] = df['bb_middle'] - (std_mult * bb_std)

    # Bollinger Band width
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Price position in bands
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    feature_metadata['bb_middle'] = "Bollinger Band middle (20,2)"
    feature_metadata['bb_upper'] = "Bollinger Band upper (20,2)"
    feature_metadata['bb_lower'] = "Bollinger Band lower (20,2)"
    feature_metadata['bb_width'] = "Bollinger Band width"
    feature_metadata['bb_position'] = "Price position in Bollinger Bands"

    return df


def add_keltner_channels(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Keltner Channels features.

    Calculates 20-period Keltner Channels with 2x ATR multiplier
    and price position within the channels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with Keltner Channel features added
    """
    logger.info("Adding Keltner Channels...")

    period = 20
    atr_mult = 2

    ema = calculate_ema_numba(df['close'].values, period)
    atr = calculate_atr_numba(df['high'].values, df['low'].values, df['close'].values, period)

    df['kc_middle'] = ema
    df['kc_upper'] = ema + (atr_mult * atr)
    df['kc_lower'] = ema - (atr_mult * atr)

    # Price position in channels
    df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])

    feature_metadata['kc_middle'] = "Keltner Channel middle (20,2)"
    feature_metadata['kc_upper'] = "Keltner Channel upper (20,2)"
    feature_metadata['kc_lower'] = "Keltner Channel lower (20,2)"
    feature_metadata['kc_position'] = "Price position in Keltner Channels"

    return df


def add_historical_volatility(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add historical volatility features.

    Calculates annualized historical volatility for multiple periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with historical volatility features added
    """
    logger.info("Adding historical volatility...")

    periods = [10, 20, 60]
    log_returns = np.log(df['close'] / df['close'].shift(1))

    for period in periods:
        df[f'hvol_{period}'] = log_returns.rolling(window=period).std() * ANNUALIZATION_FACTOR

        feature_metadata[f'hvol_{period}'] = f"Historical volatility ({period})"

    return df


def add_parkinson_volatility(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Parkinson volatility.

    Parkinson volatility uses high-low range to estimate volatility,
    which is more efficient than close-to-close volatility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'high' and 'low' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with Parkinson volatility added
    """
    logger.info("Adding Parkinson volatility...")

    period = 20
    hl_ratio = np.log(df['high'] / df['low'])
    df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) *
                                  (hl_ratio ** 2).rolling(window=period).mean()) * ANNUALIZATION_FACTOR

    feature_metadata['parkinson_vol'] = "Parkinson volatility (20)"

    return df


def add_garman_klass_volatility(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Garman-Klass volatility.

    Garman-Klass volatility uses OHLC data for more efficient volatility estimation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with Garman-Klass volatility added
    """
    logger.info("Adding Garman-Klass volatility...")

    period = 20
    hl = np.log(df['high'] / df['low'])
    co = np.log(df['close'] / df['open'])

    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    df['gk_vol'] = np.sqrt(gk.rolling(window=period).mean()) * ANNUALIZATION_FACTOR

    feature_metadata['gk_vol'] = "Garman-Klass volatility (20)"

    return df


__all__ = [
    'add_atr',
    'add_bollinger_bands',
    'add_keltner_channels',
    'add_historical_volatility',
    'add_parkinson_volatility',
    'add_garman_klass_volatility',
]
