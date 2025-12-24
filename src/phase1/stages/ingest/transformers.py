"""
Data transformation functions for the ingestion pipeline.

Handles:
- Column name standardization
- Timezone conversion
"""

import logging
from typing import Dict

import pandas as pd
import pytz

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Standard column names
STANDARD_COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume']

# Common column name mappings (case-insensitive)
COLUMN_MAPPINGS: Dict[str, str] = {
    'timestamp': 'datetime',
    'time': 'datetime',
    'date': 'datetime',
    'dt': 'datetime',
    'o': 'open',
    'h': 'high',
    'l': 'low',
    'c': 'close',
    'v': 'volume',
    'vol': 'volume',
    'trade_volume': 'volume',
}

# Timezone mappings
TIMEZONE_MAP: Dict[str, str] = {
    'EST': 'America/New_York',
    'EDT': 'America/New_York',
    'CST': 'America/Chicago',
    'CDT': 'America/Chicago',
    'PST': 'America/Los_Angeles',
    'PDT': 'America/Los_Angeles',
    'GMT': 'GMT',
    'UTC': 'UTC',
}


def standardize_columns(
    df: pd.DataFrame,
    column_mappings: Dict[str, str] = None,
    copy: bool = True
) -> pd.DataFrame:
    """
    Standardize column names to expected format.

    Parameters:
    -----------
    df : Input DataFrame
    column_mappings : Custom column mappings (defaults to COLUMN_MAPPINGS)
    copy : If True, create a copy of the DataFrame. If False, modify in place.

    Returns:
    --------
    pd.DataFrame : DataFrame with standardized column names
    """
    logger.info("Standardizing column names...")

    if column_mappings is None:
        column_mappings = COLUMN_MAPPINGS

    if copy:
        df = df.copy()

    # Convert all column names to lowercase for mapping
    df.columns = df.columns.str.lower().str.strip()

    # Apply column mappings
    rename_dict = {}
    for col in df.columns:
        if col in column_mappings:
            rename_dict[col] = column_mappings[col]

    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.info(f"Renamed columns: {rename_dict}")

    # Check for required columns
    missing_cols = set(STANDARD_COLS) - set(df.columns)
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}")
        # Check if we have the columns under different names
        available_cols = df.columns.tolist()
        logger.info(f"Available columns: {available_cols}")

    return df


def handle_timezone(
    df: pd.DataFrame,
    source_timezone: str = 'UTC',
    timezone_map: Dict[str, str] = None,
    copy: bool = True
) -> pd.DataFrame:
    """
    Convert datetime to UTC timezone.

    Parameters:
    -----------
    df : Input DataFrame
    source_timezone : Source timezone of the data (default: 'UTC')
    timezone_map : Custom timezone mappings (defaults to TIMEZONE_MAP)
    copy : If True, create a copy of the DataFrame. If False, modify in place.

    Returns:
    --------
    pd.DataFrame : DataFrame with UTC datetime
    """
    logger.info(f"Converting timezone from {source_timezone} to UTC...")

    if timezone_map is None:
        timezone_map = TIMEZONE_MAP

    if copy:
        df = df.copy()

    # Ensure datetime column exists and is datetime type
    if 'datetime' not in df.columns:
        raise ValueError("No 'datetime' column found in data")

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Handle timezone conversion
    if df['datetime'].dt.tz is None:
        # Naive datetime - localize to source timezone first
        source_tz = timezone_map.get(source_timezone, source_timezone)
        try:
            tz = pytz.timezone(source_tz)
            df['datetime'] = df['datetime'].dt.tz_localize(tz)
            logger.info(f"Localized to {source_tz}")
        except Exception as e:
            logger.warning(f"Could not localize to {source_tz}: {e}. Assuming UTC.")
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')

    # Convert to UTC
    if df['datetime'].dt.tz.zone != 'UTC':
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')
        logger.info("Converted to UTC")

    # Remove timezone info (store as naive UTC)
    df['datetime'] = df['datetime'].dt.tz_localize(None)

    return df
