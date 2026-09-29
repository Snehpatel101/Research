"""
OHLCV resampling.

``resample_ohlcv`` is the single resampling function for training
(``MLFactory``, ``UnifiedDataPreparation``) and inference (``PreprocessingGraph``), so
both build identical bars. ``closed="left", label="left"`` is the anti-lookahead
convention: a bar stamped 09:30 covers [09:30:00, 09:34:59].
"""

import logging

import pandas as pd

from src.core.common.timeframes import timeframe_to_minutes as parse_timeframe_to_minutes
from src.core.common.timeframes import validate_timeframe

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def resample_ohlcv(
    df: pd.DataFrame, target_timeframe: str = "5min", include_metadata: bool = True
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
    # Validate target timeframe
    validate_timeframe(target_timeframe)

    # Validate required columns
    required_columns = ["datetime", "open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. " f"Expected columns: {required_columns}"
        )

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    # Get target minutes for resampling
    target_minutes = parse_timeframe_to_minutes(target_timeframe)

    # Build pandas frequency string
    freq = f"{target_minutes}min" if target_minutes < 60 else f"{target_minutes // 60}h"

    logger.info(f"Resampling to {target_timeframe} ({target_minutes}-minute bars)...")

    df = df.copy()

    # Handle symbol column if present - preserve it
    has_symbol = "symbol" in df.columns
    symbol_value = df["symbol"].iloc[0] if has_symbol else None

    df = df.set_index("datetime")

    # Define OHLCV aggregation rules
    agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    optional_flag_cols = ["missing_bar", "filled", "roll_event", "roll_window"]
    for col in optional_flag_cols:
        if col in df.columns:
            agg_rules[col] = "max"

    # Perform resampling
    # ANTI-LOOKAHEAD: Use closed='left', label='left' explicitly
    # A bar at 09:30 represents [09:30:00, 09:34:59], timestamp = period start
    resampled = df.resample(freq, closed="left", label="left").agg(agg_rules)

    # Drop rows where we couldn't compute all values (e.g., no data in period)
    resampled = resampled.dropna()

    # Reset index to get datetime as a column
    resampled = resampled.reset_index()

    # Add metadata if requested
    if include_metadata:
        resampled["timeframe"] = target_timeframe

    # Restore symbol column if it was present
    if has_symbol and symbol_value is not None:
        resampled["symbol"] = symbol_value

    logger.info(f"Resampled to {len(resampled):,} {target_timeframe} bars")

    return resampled
