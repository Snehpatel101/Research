"""
Simple Data Cleaning Pipeline Function

This module provides a simple convenience function for cleaning single files
without using the full DataCleaner class.

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Extracted from stage2_clean.py
"""

import pandas as pd
from pathlib import Path
from typing import Union
import logging

from .utils import validate_ohlc, detect_gaps_simple, fill_gaps_simple, resample_to_5min

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def clean_symbol_data(input_path: Path, output_path: Path, symbol: str) -> pd.DataFrame:
    """
    Complete cleaning pipeline for a single symbol.

    This is the primary entry point for the pipeline. It performs:
    1. Data loading and datetime validation
    2. OHLC validation
    3. Gap detection and filling
    4. Resampling to 5-minute bars
    5. Final OHLC validation

    Parameters:
    -----------
    input_path : Path to input data file
    output_path : Path to save cleaned data
    symbol : Symbol name (added to output)

    Returns:
    --------
    pd.DataFrame : Cleaned data
    """
    logger.info(f"="*60)
    logger.info(f"Cleaning data for {symbol}")
    logger.info(f"="*60)

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

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Step 1: Validate OHLC
    df = validate_ohlc(df)

    # Step 2: Detect gaps
    gaps = detect_gaps_simple(df)
    if len(gaps) > 0:
        logger.info(f"Largest gap: {gaps['gap_minutes'].max():.0f} minutes")

    # Step 3: Fill small gaps
    df = fill_gaps_simple(df, max_gap_minutes=30)

    # Step 4: Re-validate OHLC after gap filling
    df = validate_ohlc(df)

    # Step 5: Resample to 5-min
    df = resample_to_5min(df)

    # Step 6: Final validation
    df = validate_ohlc(df)

    # Add symbol column
    df['symbol'] = symbol

    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")

    return df
