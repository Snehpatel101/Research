"""
Volatility indicator features for feature engineering.

This module provides functions to calculate volatility-based technical
indicators including ATR, Bollinger Bands, Keltner Channels, and various
historical volatility measures.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

from .constants import ANNUALIZATION_FACTOR, get_annualization_factor
from .numba_functions import calculate_atr_numba, calculate_ema_numba

logger = logging.getLogger(__name__)


def add_atr(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    periods: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Add Average True Range features.

    Calculates ATR for multiple periods with both absolute and percentage values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of ATR periods. Default: [7, 14, 21]

    Returns
    -------
    pd.DataFrame
        DataFrame with ATR features added
    """
    if periods is None:
        periods = [7, 14, 21]

    logger.info(f"Adding ATR features with periods: {periods}")

    for period in periods:
        atr = calculate_atr_numba(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            period
        )
        df[f'atr_{period}'] = atr

        # ATR as percentage of close (safe division)
        close_safe = df['close'].replace(0, np.nan)
        df[f'atr_pct_{period}'] = (atr / close_safe) * 100

        feature_metadata[f'atr_{period}'] = f"Average True Range ({period})"
        feature_metadata[f'atr_pct_{period}'] = f"ATR as % of price ({period})"

    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    period: int = 20,
    std_mult: float = 2.0
) -> pd.DataFrame:
    """
    Add Bollinger Bands features.

    Calculates Bollinger Bands with configurable period and standard deviation,
    band width (normalized), and price position within the bands.

    All features are made stationary using z-scores and normalized values
    to avoid dependency on absolute price levels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Bollinger Band period
    std_mult : float, default 2.0
        Standard deviation multiplier

    Returns
    -------
    pd.DataFrame
        DataFrame with Bollinger Band features added
    """
    logger.info(f"Adding Bollinger Bands with period: {period}")

    df['bb_middle'] = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()

    df['bb_upper'] = df['bb_middle'] + (std_mult * bb_std)
    df['bb_lower'] = df['bb_middle'] - (std_mult * bb_std)

    # Bollinger Band width normalized by std (stationary)
    # This is equivalent to band_width / std, making it scale-invariant
    bb_std_safe = bb_std.replace(0, np.nan)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_std_safe

    # Price position in bands (already stationary - bounded [0,1])
    # Use safe division to handle band collapse
    band_range = df['bb_upper'] - df['bb_lower']
    band_range_safe = band_range.replace(0, np.nan)
    df['bb_position'] = (df['close'] - df['bb_lower']) / band_range_safe

    # Add close price z-score relative to BB middle (stationary)
    df['close_bb_zscore'] = (df['close'] - df['bb_middle']) / bb_std_safe

    feature_metadata['bb_middle'] = f"Bollinger Band middle ({period},{std_mult})"
    feature_metadata['bb_upper'] = f"Bollinger Band upper ({period},{std_mult})"
    feature_metadata['bb_lower'] = f"Bollinger Band lower ({period},{std_mult})"
    feature_metadata['bb_width'] = "Bollinger Band width (normalized by std)"
    feature_metadata['bb_position'] = "Price position in Bollinger Bands [0,1]"
    feature_metadata['close_bb_zscore'] = "Close price z-score relative to BB middle"

    return df


def add_keltner_channels(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    period: int = 20,
    atr_mult: float = 2.0
) -> pd.DataFrame:
    """
    Add Keltner Channels features.

    Calculates Keltner Channels with configurable period and ATR multiplier
    and price position within the channels.

    All features use safe division and stationary representations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Keltner Channel period
    atr_mult : float, default 2.0
        ATR multiplier

    Returns
    -------
    pd.DataFrame
        DataFrame with Keltner Channel features added
    """
    logger.info(f"Adding Keltner Channels with period: {period}")

    ema = calculate_ema_numba(df['close'].values, period)
    atr = calculate_atr_numba(df['high'].values, df['low'].values, df['close'].values, period)

    df['kc_middle'] = ema
    df['kc_upper'] = ema + (atr_mult * atr)
    df['kc_lower'] = ema - (atr_mult * atr)

    # Price position in channels (already stationary - bounded [0,1])
    # Use safe division to handle channel collapse
    channel_range = df['kc_upper'] - df['kc_lower']
    channel_range_safe = channel_range.replace(0, np.nan)
    df['kc_position'] = (df['close'] - df['kc_lower']) / channel_range_safe

    # Add close deviation from KC middle in ATR units (stationary)
    atr_safe = pd.Series(atr).replace(0, np.nan)
    df['close_kc_atr_dev'] = (df['close'] - df['kc_middle']) / atr_safe

    feature_metadata['kc_middle'] = f"Keltner Channel middle ({period},{atr_mult})"
    feature_metadata['kc_upper'] = f"Keltner Channel upper ({period},{atr_mult})"
    feature_metadata['kc_lower'] = f"Keltner Channel lower ({period},{atr_mult})"
    feature_metadata['kc_position'] = "Price position in Keltner Channels [0,1]"
    feature_metadata['close_kc_atr_dev'] = "Close deviation from KC middle in ATR units"

    return df


def add_historical_volatility(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    periods: Optional[List[int]] = None,
    timeframe: str = '5min'
) -> pd.DataFrame:
    """
    Add historical volatility features.

    Calculates annualized historical volatility for multiple periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of volatility periods. Default: [10, 20, 60]
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with historical volatility features added
    """
    if periods is None:
        periods = [10, 20, 60]

    logger.info(f"Adding historical volatility with periods: {periods}")

    log_returns = np.log(df['close'] / df['close'].shift(1))
    annualization_factor = get_annualization_factor(timeframe)

    for period in periods:
        df[f'hvol_{period}'] = log_returns.rolling(window=period).std() * annualization_factor

        feature_metadata[f'hvol_{period}'] = f"Historical volatility ({period})"

    return df


def add_parkinson_volatility(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    period: int = 20,
    timeframe: str = '5min'
) -> pd.DataFrame:
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
    period : int, default 20
        Parkinson volatility period
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Parkinson volatility added
    """
    logger.info(f"Adding Parkinson volatility with period: {period}")

    hl_ratio = np.log(df['high'] / df['low'])
    annualization_factor = get_annualization_factor(timeframe)
    df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) *
                                  (hl_ratio ** 2).rolling(window=period).mean()) * annualization_factor

    feature_metadata['parkinson_vol'] = f"Parkinson volatility ({period})"

    return df


def add_garman_klass_volatility(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    period: int = 20,
    timeframe: str = '5min'
) -> pd.DataFrame:
    """
    Add Garman-Klass volatility.

    Garman-Klass volatility uses OHLC data for more efficient volatility estimation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Garman-Klass volatility period
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Garman-Klass volatility added
    """
    logger.info(f"Adding Garman-Klass volatility with period: {period}")

    hl = np.log(df['high'] / df['low'])
    co = np.log(df['close'] / df['open'])
    annualization_factor = get_annualization_factor(timeframe)

    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    df['gk_vol'] = np.sqrt(gk.rolling(window=period).mean()) * annualization_factor

    feature_metadata['gk_vol'] = f"Garman-Klass volatility ({period})"

    return df


__all__ = [
    'add_atr',
    'add_bollinger_bands',
    'add_keltner_channels',
    'add_historical_volatility',
    'add_parkinson_volatility',
    'add_garman_klass_volatility',
]
