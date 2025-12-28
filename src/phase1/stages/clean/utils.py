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

DEFAULT_ROLL_GAP_THRESHOLD = 0.10
DEFAULT_ROLL_WINDOW_BARS = 5
SESSION_ID_OUTSIDE = 'outside'


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
    original_index = df.index

    # Create complete datetime index
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='1min'
    )

    # Reindex and forward fill
    df = df.reindex(full_index)
    df['missing_bar'] = (~df.index.isin(original_index)).astype(int)

    # Only fill small gaps
    df = df.ffill(limit=max_gap_minutes)

    # Drop remaining NaNs
    df = df.dropna()

    df = df.reset_index().rename(columns={'index': 'datetime'})

    logger.info(f"After gap filling: {len(df):,} rows")
    return df


def add_roll_flags(
    df: pd.DataFrame,
    price_column: str = 'close',
    pct_threshold: float = DEFAULT_ROLL_GAP_THRESHOLD,
    window_bars: int = DEFAULT_ROLL_WINDOW_BARS
) -> pd.DataFrame:
    """
    Add contract roll event/window flags based on large price gaps.

    The roll window marks a band of bars around the detected roll event.
    """
    if df.empty:
        return df
    if price_column not in df.columns:
        raise ValueError(f"Missing required column for roll detection: {price_column}")

    df = df.copy()
    pct_change = df[price_column].pct_change().abs()
    roll_event = pct_change >= pct_threshold
    roll_window = np.zeros(len(df), dtype=bool)

    if roll_event.any():
        indices = np.where(roll_event.fillna(False).values)[0]
        for idx in indices:
            start = max(0, idx - window_bars)
            end = min(len(df) - 1, idx + window_bars)
            roll_window[start:end + 1] = True

    df['roll_event'] = roll_event.fillna(False).astype(int)
    df['roll_window'] = roll_window.astype(int)
    return df


def add_session_id(
    df: pd.DataFrame,
    datetime_column: str = 'datetime',
    outside_label: str = SESSION_ID_OUTSIDE
) -> pd.DataFrame:
    """
    Add a session_id column using CME session definitions (UTC).
    """
    if df.empty:
        return df
    if datetime_column not in df.columns:
        raise ValueError(f"Missing required datetime column: {datetime_column}")

    from src.phase1.stages.sessions import SessionFilter, SessionName

    df = df.copy()
    session_filter = SessionFilter(datetime_column=datetime_column)
    sessions = session_filter.classify_sessions_vectorized(df[datetime_column])
    def _resolve_session(value: object) -> str:
        if value is None or pd.isna(value):
            return outside_label
        if isinstance(value, SessionName):
            return value.value
        return str(value)

    df['session_id'] = sessions.apply(_resolve_session)
    return df


def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str = '5min',
    include_metadata: bool = True
) -> pd.DataFrame:
    """
    Resample OHLCV data to a target timeframe.

    This function resamples 1-minute (or any lower resolution) OHLCV data
    to a target timeframe using proper aggregation rules:
    - Open: first value in the period
    - High: maximum value in the period
    - Low: minimum value in the period
    - Close: last value in the period
    - Volume: sum of values in the period

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with OHLCV data. Must have columns:
        'datetime', 'open', 'high', 'low', 'close', 'volume'
    target_timeframe : str
        Target timeframe for resampling. Supported values:
        '1min', '5min', '10min', '15min', '20min', '30min', '45min', '60min'
    include_metadata : bool
        If True, adds a 'timeframe' column to the output DataFrame

    Returns:
    --------
    pd.DataFrame : Resampled OHLCV data with optional timeframe metadata

    Raises:
    -------
    ValueError : If target_timeframe is not supported or required columns are missing

    Examples:
    ---------
    >>> df_5min = resample_ohlcv(df_1min, '5min')
    >>> df_15min = resample_ohlcv(df_1min, '15min')
    >>> df_30min = resample_ohlcv(df_5min, '30min')  # Can resample from 5min to 30min
    """
    # Import validation function from config
    from src.phase1.config import (
        SUPPORTED_TIMEFRAMES,
        validate_timeframe,
        parse_timeframe_to_minutes
    )

    # Validate target timeframe
    validate_timeframe(target_timeframe)

    # Validate required columns
    required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Expected columns: {required_columns}"
        )

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    # Get target minutes for resampling
    target_minutes = parse_timeframe_to_minutes(target_timeframe)

    # Build pandas frequency string
    freq = f'{target_minutes}min' if target_minutes < 60 else f'{target_minutes // 60}h'

    logger.info(f"Resampling to {target_timeframe} ({target_minutes}-minute bars)...")

    df = df.copy()

    # Handle symbol column if present - preserve it
    has_symbol = 'symbol' in df.columns
    symbol_value = df['symbol'].iloc[0] if has_symbol else None

    df = df.set_index('datetime')

    # Define OHLCV aggregation rules
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    optional_flag_cols = ['missing_bar', 'filled', 'roll_event', 'roll_window']
    for col in optional_flag_cols:
        if col in df.columns:
            agg_rules[col] = 'max'

    # Perform resampling
    # ANTI-LOOKAHEAD: Use closed='left', label='left' explicitly
    # A bar at 09:30 represents [09:30:00, 09:34:59], timestamp = period start
    resampled = df.resample(freq, closed='left', label='left').agg(agg_rules)

    # Drop rows where we couldn't compute all values (e.g., no data in period)
    resampled = resampled.dropna()

    # Reset index to get datetime as a column
    resampled = resampled.reset_index()

    # Add metadata if requested
    if include_metadata:
        resampled['timeframe'] = target_timeframe

    # Restore symbol column if it was present
    if has_symbol and symbol_value is not None:
        resampled['symbol'] = symbol_value

    logger.info(f"Resampled to {len(resampled):,} {target_timeframe} bars")

    return resampled


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute data to 5-minute bars.

    This is a convenience wrapper around resample_ohlcv() for backward compatibility.
    New code should use resample_ohlcv() directly for configurable timeframes.

    Parameters:
    -----------
    df : Input DataFrame with 1-minute OHLCV data

    Returns:
    --------
    pd.DataFrame : Resampled 5-minute bar data

    See Also:
    ---------
    resample_ohlcv : Generic resampling function with configurable timeframe
    """
    # Use the generic function, but exclude metadata for backward compatibility
    result = resample_ohlcv(df, target_timeframe='5min', include_metadata=False)
    return result


def get_resampling_info(source_timeframe: str, target_timeframe: str) -> dict:
    """
    Get information about a resampling operation.

    Parameters:
    -----------
    source_timeframe : str
        Source timeframe (e.g., '1min', '5min')
    target_timeframe : str
        Target timeframe (e.g., '15min', '30min')

    Returns:
    --------
    dict : Information including bars_per_target, scale_factor, etc.

    Raises:
    -------
    ValueError : If target_timeframe is smaller than source_timeframe
    """
    from src.phase1.config import parse_timeframe_to_minutes, validate_timeframe

    validate_timeframe(source_timeframe)
    validate_timeframe(target_timeframe)

    source_minutes = parse_timeframe_to_minutes(source_timeframe)
    target_minutes = parse_timeframe_to_minutes(target_timeframe)

    if target_minutes < source_minutes:
        raise ValueError(
            f"Cannot resample from {source_timeframe} to {target_timeframe}. "
            f"Target timeframe must be >= source timeframe."
        )

    bars_per_target = target_minutes // source_minutes

    return {
        'source_timeframe': source_timeframe,
        'target_timeframe': target_timeframe,
        'source_minutes': source_minutes,
        'target_minutes': target_minutes,
        'bars_per_target': bars_per_target,
        'scale_factor': target_minutes / source_minutes,
        'description': f'{bars_per_target} {source_timeframe} bars = 1 {target_timeframe} bar'
    }
