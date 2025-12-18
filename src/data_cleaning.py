"""
Data Cleaning Module for Ensemble Trading System
Handles: OHLC validation, gap filling, resampling to 5-min bars
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and fix OHLC relationships:
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - High >= Low
    """
    logger.info("Validating OHLC relationships...")

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


def detect_gaps(df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """Detect and report data gaps."""
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


def fill_gaps(df: pd.DataFrame, max_gap_minutes: int = 60) -> pd.DataFrame:
    """Forward-fill small gaps in data."""
    logger.info(f"Filling gaps up to {max_gap_minutes} minutes...")

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
    """Resample 1-minute data to 5-minute bars."""
    logger.info("Resampling to 5-minute bars...")

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


def clean_symbol_data(input_path: Path, output_path: Path, symbol: str) -> pd.DataFrame:
    """Complete cleaning pipeline for a single symbol."""
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
    gaps = detect_gaps(df)
    if len(gaps) > 0:
        logger.info(f"Largest gap: {gaps['gap_minutes'].max():.0f} minutes")

    # Step 3: Fill small gaps
    df = fill_gaps(df, max_gap_minutes=30)

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


def main():
    """Run cleaning for all symbols."""
    from config import RAW_DATA_DIR, CLEAN_DATA_DIR, SYMBOLS

    for symbol in SYMBOLS:
        # Try parquet first, then CSV
        input_path = RAW_DATA_DIR / f"{symbol}_1m.parquet"
        if not input_path.exists():
            input_path = RAW_DATA_DIR / f"{symbol}_1m.csv"

        output_path = CLEAN_DATA_DIR / f"{symbol}_5m_clean.parquet"

        if input_path.exists():
            clean_symbol_data(input_path, output_path, symbol)
        else:
            logger.warning(f"No data found for {symbol} at {input_path}")


if __name__ == "__main__":
    main()
