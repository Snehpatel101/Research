"""
Simple Data Cleaning Pipeline Function

This module provides a simple convenience function for cleaning single files
without using the full DataCleaner class.

Supports Multi-Timeframe (MTF) resampling through the target_timeframe parameter.

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-22 - Added MTF support with configurable target_timeframe
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional
import logging

from .utils import (
    validate_ohlc,
    detect_gaps_simple,
    fill_gaps_simple,
    resample_ohlcv,
    resample_to_5min,  # Kept for backward compatibility
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def clean_symbol_data(
    input_path: Path,
    output_path: Path,
    symbol: str,
    target_timeframe: str = '5min',
    include_timeframe_metadata: bool = True,
    max_gap_minutes: int = 30
) -> pd.DataFrame:
    """
    Complete cleaning pipeline for a single symbol.

    This is the primary entry point for the pipeline. It performs:
    1. Data loading and datetime validation
    2. OHLC validation
    3. Gap detection and filling
    4. Resampling to target timeframe (configurable, defaults to 5min)
    5. Final OHLC validation
    6. Add timeframe metadata (optional)

    Parameters:
    -----------
    input_path : Path
        Path to input data file (parquet or CSV)
    output_path : Path
        Path to save cleaned data
    symbol : str
        Symbol name (added to output)
    target_timeframe : str
        Target timeframe for resampling. Default: '5min'
        Supported: '1min', '5min', '10min', '15min', '20min', '30min', '45min', '60min'
    include_timeframe_metadata : bool
        If True, adds 'timeframe' column to output. Default: True
    max_gap_minutes : int
        Maximum gap to fill in minutes. Default: 30

    Returns:
    --------
    pd.DataFrame : Cleaned and resampled data

    Raises:
    -------
    ValueError : If target_timeframe is not supported
    FileNotFoundError : If input_path does not exist

    Examples:
    ---------
    >>> # Standard 5-minute resampling (default)
    >>> df = clean_symbol_data(Path('raw/MES.parquet'), Path('clean/MES.parquet'), 'MES')

    >>> # 15-minute resampling
    >>> df = clean_symbol_data(
    ...     Path('raw/MES.parquet'),
    ...     Path('clean/MES_15min.parquet'),
    ...     'MES',
    ...     target_timeframe='15min'
    ... )

    >>> # 30-minute resampling without timeframe column
    >>> df = clean_symbol_data(
    ...     Path('raw/MES.parquet'),
    ...     Path('clean/MES_30min.parquet'),
    ...     'MES',
    ...     target_timeframe='30min',
    ...     include_timeframe_metadata=False
    ... )
    """
    # Validate target timeframe at the boundary
    from config import validate_timeframe
    validate_timeframe(target_timeframe)

    logger.info("=" * 60)
    logger.info(f"Cleaning data for {symbol}")
    logger.info(f"Target timeframe: {target_timeframe}")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading {input_path}")
    if str(input_path).endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)

    logger.info(f"Loaded {len(df):,} rows")

    # Ensure datetime column
    if 'datetime' not in df.columns:
        # Try common datetime column names
        for col in ['timestamp', 'date', 'time', 'DateTime', 'Timestamp']:
            if col in df.columns:
                df = df.rename(columns={col: 'datetime'})
                break

    if 'datetime' not in df.columns:
        raise ValueError(
            "No datetime column found. Expected one of: "
            "'datetime', 'timestamp', 'date', 'time', 'DateTime', 'Timestamp'"
        )

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Step 1: Validate OHLC
    df = validate_ohlc(df)

    # Step 2: Detect gaps
    gaps = detect_gaps_simple(df)
    if len(gaps) > 0:
        logger.info(f"Largest gap: {gaps['gap_minutes'].max():.0f} minutes")

    # Step 3: Fill small gaps
    df = fill_gaps_simple(df, max_gap_minutes=max_gap_minutes)

    # Step 4: Re-validate OHLC after gap filling
    df = validate_ohlc(df)

    # Step 5: Resample to target timeframe
    df = resample_ohlcv(df, target_timeframe, include_metadata=include_timeframe_metadata)

    # Step 6: Final validation
    df = validate_ohlc(df)

    # Add symbol column
    df['symbol'] = symbol

    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")
    logger.info(f"Output: {len(df):,} {target_timeframe} bars")

    return df


def clean_symbol_data_multi_timeframe(
    input_path: Path,
    output_dir: Path,
    symbol: str,
    timeframes: list[str] = None,
    include_timeframe_metadata: bool = True,
    max_gap_minutes: int = 30
) -> dict[str, pd.DataFrame]:
    """
    Clean and resample data to multiple timeframes at once.

    This is useful for creating datasets at different resolutions
    from the same source data in a single pass.

    Parameters:
    -----------
    input_path : Path
        Path to input data file (1-minute bars)
    output_dir : Path
        Directory to save cleaned data files
    symbol : str
        Symbol name
    timeframes : list[str]
        List of target timeframes. Default: ['5min', '15min', '30min']
    include_timeframe_metadata : bool
        If True, adds 'timeframe' column to outputs
    max_gap_minutes : int
        Maximum gap to fill in minutes

    Returns:
    --------
    dict[str, pd.DataFrame] : Dictionary mapping timeframe to cleaned DataFrame

    Examples:
    ---------
    >>> results = clean_symbol_data_multi_timeframe(
    ...     Path('raw/MES.parquet'),
    ...     Path('clean/'),
    ...     'MES',
    ...     timeframes=['5min', '15min', '30min', '60min']
    ... )
    >>> df_15min = results['15min']
    """
    from config import validate_timeframe

    if timeframes is None:
        timeframes = ['5min', '15min', '30min']

    # Validate all timeframes at the boundary
    for tf in timeframes:
        validate_timeframe(tf)

    logger.info("=" * 60)
    logger.info(f"Multi-timeframe cleaning for {symbol}")
    logger.info(f"Target timeframes: {timeframes}")
    logger.info("=" * 60)

    # Load and clean data once (before resampling)
    logger.info(f"Loading {input_path}")
    if str(input_path).endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)

    logger.info(f"Loaded {len(df):,} rows")

    # Ensure datetime column
    if 'datetime' not in df.columns:
        for col in ['timestamp', 'date', 'time', 'DateTime', 'Timestamp']:
            if col in df.columns:
                df = df.rename(columns={col: 'datetime'})
                break

    if 'datetime' not in df.columns:
        raise ValueError("No datetime column found")

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Clean once
    df = validate_ohlc(df)
    gaps = detect_gaps_simple(df)
    if len(gaps) > 0:
        logger.info(f"Largest gap: {gaps['gap_minutes'].max():.0f} minutes")
    df = fill_gaps_simple(df, max_gap_minutes=max_gap_minutes)
    df = validate_ohlc(df)

    # Add symbol
    df['symbol'] = symbol

    # Resample to each timeframe
    results = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tf in timeframes:
        logger.info(f"Resampling to {tf}...")
        df_resampled = resample_ohlcv(df, tf, include_metadata=include_timeframe_metadata)
        df_resampled = validate_ohlc(df_resampled)

        # Save
        output_path = output_dir / f"{symbol}_{tf}.parquet"
        df_resampled.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df_resampled):,} {tf} bars to {output_path}")

        results[tf] = df_resampled

    return results
