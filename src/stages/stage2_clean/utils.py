"""
Data Cleaning Utilities - Low-level functions for OHLC validation and resampling.

This module provides core utility functions for data cleaning:
- OHLC validation and correction
- Gap detection and filling
- Resampling operations
- ATR calculation for outlier detection

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Extracted from stage2_clean.py
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from numba import jit

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@jit(nopython=True)
def calculate_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using Numba for performance.

    Parameters:
    -----------
    high : High prices
    low : Low prices
    close : Close prices
    period : ATR period

    Returns:
    --------
    np.ndarray : ATR values
    """
    n = len(high)
    tr = np.zeros(n)
    atr = np.zeros(n)

    # Calculate True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    # Calculate ATR
    atr[period] = np.mean(tr[1:period+1])

    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    return atr


def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and fix OHLC relationships:
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - High >= Low

    Parameters:
    -----------
    df : Input DataFrame with OHLC columns

    Returns:
    --------
    pd.DataFrame : DataFrame with corrected OHLC values
    """
    logger.info("Validating OHLC relationships...")

    df = df.copy()

    # Fix high
    df['high'] = df[['high', 'open', 'close']].max(axis=1)

    # Fix low
    df['low'] = df[['low', 'open', 'close']].min(axis=1)

    # Ensure high >= low
    mask = df['high'] < df['low']
    if mask.any():
        df.loc[mask, ['high', 'low']] = df.loc[mask, ['low', 'high']].values
        logger.warning(f"Fixed {mask.sum()} bars where high < low")

    return df


def detect_gaps_simple(df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """
    Detect and report data gaps (simple version).

    Parameters:
    -----------
    df : Input DataFrame
    freq : Expected frequency ('1min' or '5min')

    Returns:
    --------
    pd.DataFrame : DataFrame with gap information
    """
    dt = df['datetime']

    # Use numpy for safe operations
    dt_values = dt.values if hasattr(dt, 'values') else dt
    time_diff = np.diff(dt_values.astype('datetime64[ns]').astype(np.int64)) / 1e9 / 60

    expected_gap = 1 if freq == '1min' else 5
    gap_mask = time_diff > expected_gap * 2

    gap_positions = np.where(gap_mask)[0]

    if len(gap_positions) > 0:
        gaps_df = pd.DataFrame({
            'gap_start': dt.iloc[gap_positions].values,
            'gap_end': dt.iloc[gap_positions + 1].values,
            'gap_minutes': time_diff[gap_positions]
        })
        logger.info(f"Found {len(gap_positions)} gaps in data")
        return gaps_df

    return pd.DataFrame()


def fill_gaps_simple(df: pd.DataFrame, max_gap_minutes: int = 60) -> pd.DataFrame:
    """
    Forward-fill small gaps in data (simple version).

    Parameters:
    -----------
    df : Input DataFrame
    max_gap_minutes : Maximum gap to fill (in minutes)

    Returns:
    --------
    pd.DataFrame : DataFrame with filled gaps
    """
    logger.info(f"Filling gaps up to {max_gap_minutes} minutes...")

    df = df.copy()
    df = df.set_index('datetime')

    # Create complete datetime index
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='1min'
    )

    # Reindex and forward fill
    df = df.reindex(full_index)

    # Only fill small gaps
    df = df.ffill(limit=max_gap_minutes)

    # Drop remaining NaNs
    df = df.dropna()

    df = df.reset_index().rename(columns={'index': 'datetime'})

    logger.info(f"After gap filling: {len(df):,} rows")
    return df


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute data to 5-minute bars.

    Parameters:
    -----------
    df : Input DataFrame with 1-minute OHLCV data

    Returns:
    --------
    pd.DataFrame : Resampled 5-minute bar data
    """
    logger.info("Resampling to 5-minute bars...")

    df = df.copy()
    df = df.set_index('datetime')

    resampled = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    resampled = resampled.reset_index()

    logger.info(f"Resampled to {len(resampled):,} 5-min bars")
    return resampled
