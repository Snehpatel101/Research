"""
Momentum indicator features for feature engineering.

This module provides functions to calculate momentum-based technical
indicators including RSI, MACD, Stochastic, Williams %R, ROC, CCI, and MFI.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

from .numba_functions import (
    calculate_rsi_numba,
    calculate_ema_numba,
    calculate_stochastic_numba,
)

logger = logging.getLogger(__name__)


def add_rsi(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add RSI features.

    Calculates 14-period RSI with overbought/oversold flags.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with RSI features added
    """
    logger.info("Adding RSI features...")

    df['rsi_14'] = calculate_rsi_numba(df['close'].values, 14)

    # Overbought/Oversold flags
    # ANTI-LOOKAHEAD: Shift flags forward 1 bar so flag at bar[t] is based on bar[t-1] RSI
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int).shift(1)
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int).shift(1)

    feature_metadata['rsi_14'] = "14-period Relative Strength Index"
    feature_metadata['rsi_overbought'] = "RSI overbought flag (>70)"
    feature_metadata['rsi_oversold'] = "RSI oversold flag (<30)"

    return df


def add_macd(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add MACD features.

    Calculates MACD line (12,26), signal line (9), histogram, and crossover signals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with MACD features added
    """
    logger.info("Adding MACD features...")

    # MACD line
    ema_12 = calculate_ema_numba(df['close'].values, 12)
    ema_26 = calculate_ema_numba(df['close'].values, 26)
    df['macd_line'] = ema_12 - ema_26

    # Signal line
    df['macd_signal'] = calculate_ema_numba(df['macd_line'].values, 9)

    # MACD histogram
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # MACD crossovers
    # ANTI-LOOKAHEAD: Shift signals forward 1 bar so crossover at bar[t] is based on
    # comparison between bar[t-1] and bar[t-2], making it available before bar[t] close
    df['macd_cross_up'] = ((df['macd_line'] > df['macd_signal']) &
                           (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))).astype(int).shift(1)
    df['macd_cross_down'] = ((df['macd_line'] < df['macd_signal']) &
                             (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))).astype(int).shift(1)

    feature_metadata['macd_line'] = "MACD line (12,26)"
    feature_metadata['macd_signal'] = "MACD signal line (9)"
    feature_metadata['macd_hist'] = "MACD histogram"
    feature_metadata['macd_cross_up'] = "MACD bullish crossover"
    feature_metadata['macd_cross_down'] = "MACD bearish crossover"

    return df


def add_stochastic(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Stochastic Oscillator features.

    Calculates %K and %D with overbought/oversold flags.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with Stochastic features added
    """
    logger.info("Adding Stochastic features...")

    k, d = calculate_stochastic_numba(
        df['high'].values,
        df['low'].values,
        df['close'].values,
        k_period=14,
        d_period=3
    )

    df['stoch_k'] = k
    df['stoch_d'] = d

    # Overbought/Oversold
    # ANTI-LOOKAHEAD: Shift flags forward 1 bar so flag at bar[t] is based on bar[t-1] stochastic
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int).shift(1)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int).shift(1)

    feature_metadata['stoch_k'] = "Stochastic %K (14,3)"
    feature_metadata['stoch_d'] = "Stochastic %D (14,3)"
    feature_metadata['stoch_overbought'] = "Stochastic overbought flag (>80)"
    feature_metadata['stoch_oversold'] = "Stochastic oversold flag (<20)"

    return df


def add_williams_r(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Williams %R indicator.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with Williams %R added
    """
    logger.info("Adding Williams %R...")

    period = 14
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()

    df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)

    feature_metadata['williams_r'] = "Williams %R (14)"

    return df


def add_roc(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Rate of Change features.

    Calculates ROC for multiple periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with ROC features added
    """
    logger.info("Adding ROC features...")

    periods = [5, 10, 20]

    for period in periods:
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                               df['close'].shift(period) * 100)

        feature_metadata[f'roc_{period}'] = f"Rate of Change ({period})"

    return df


def add_cci(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Commodity Channel Index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with CCI added
    """
    logger.info("Adding CCI...")

    period = 20
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

    df['cci_20'] = (tp - sma_tp) / (0.015 * mad)

    feature_metadata['cci_20'] = "Commodity Channel Index (20)"

    return df


def add_mfi(df: pd.DataFrame, feature_metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Add Money Flow Index (if volume available).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC and volume columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with MFI added (if volume data exists)
    """
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        logger.info("Skipping MFI (no volume data)")
        return df

    logger.info("Adding MFI...")

    period = 14
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']

    mf_pos = mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
    mf_neg = mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()

    mfr = mf_pos / mf_neg
    df['mfi_14'] = 100 - (100 / (1 + mfr))

    feature_metadata['mfi_14'] = "Money Flow Index (14)"

    return df


__all__ = [
    'add_rsi',
    'add_macd',
    'add_stochastic',
    'add_williams_r',
    'add_roc',
    'add_cci',
    'add_mfi',
]
