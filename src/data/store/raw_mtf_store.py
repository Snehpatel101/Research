"""
Raw multi-timeframe OHLCV store for 4D model training.

This module provides storage and retrieval for raw OHLCV data across
multiple timeframes, which is required for 4D models (PatchTST, iTransformer,
TFT, N-BEATS) that need access to raw price data at various time resolutions.

The standard 9 timeframes are:
- 1min, 3min, 5min, 10min, 15min, 30min, 60min, 2h, 4h

Usage
-----
>>> from src.data.store import save_raw_mtf, load_raw_mtf, load_all_timeframes
>>>
>>> # Save raw OHLCV data
>>> save_raw_mtf("MES", "5min", "train", df)
>>>
>>> # Load specific timeframe
>>> df = load_raw_mtf("MES", "5min", "train")
>>>
>>> # Load all timeframes
>>> all_tf = load_all_timeframes("MES", "train")
>>> df_5min = all_tf["5min"]
>>> df_15min = all_tf["15min"]
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.core.exceptions import (
    InvalidSplitError,
    InvalidTimeframeError,
    TimeframeNotFoundError,
)

logger = logging.getLogger(__name__)

# Standard timeframes for MTF analysis
# These represent the canonical set of timeframes used for 4D model training
TIMEFRAMES: list[str] = [
    "1min",
    "3min",
    "5min",
    "10min",
    "15min",
    "30min",
    "60min",
    "2h",
    "4h",
]

# Valid splits for train/validation/test partitioning
VALID_SPLITS: list[str] = ["train", "val", "test"]

# Default compression for parquet files
DEFAULT_COMPRESSION: str = "snappy"


def _validate_timeframe(timeframe: str) -> None:
    """
    Validate that a timeframe is in the standard set.

    Parameters
    ----------
    timeframe : str
        Timeframe string to validate

    Raises
    ------
    InvalidTimeframeError
        If timeframe is not in TIMEFRAMES
    """
    if timeframe not in TIMEFRAMES:
        raise InvalidTimeframeError(
            f"Invalid timeframe '{timeframe}'. " f"Valid timeframes are: {TIMEFRAMES}"
        )


def _validate_split(split: str) -> None:
    """
    Validate that a split is valid.

    Parameters
    ----------
    split : str
        Split string to validate

    Raises
    ------
    InvalidSplitError
        If split is not in VALID_SPLITS
    """
    if split not in VALID_SPLITS:
        raise InvalidSplitError(f"Invalid split '{split}'. " f"Valid splits are: {VALID_SPLITS}")


def get_mtf_path(
    symbol: str,
    timeframe: str,
    split: str,
    base_path: str | Path = "data/canonical",
) -> Path:
    """
    Get the path for a raw MTF parquet file.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "MES", "MGC")
    timeframe : str
        Timeframe string (e.g., "5min", "1h")
    split : str
        Data split ("train", "val", "test")
    base_path : str or Path, default "data/canonical"
        Base directory for data storage

    Returns
    -------
    Path
        Full path to the parquet file

    Examples
    --------
    >>> path = get_mtf_path("MES", "5min", "train")
    >>> print(path)
    data/canonical/raw_mtf/MES_5min_train.parquet
    """
    _validate_timeframe(timeframe)
    _validate_split(split)

    base = Path(base_path)
    mtf_dir = base / "raw_mtf"
    filename = f"{symbol}_{timeframe}_{split}.parquet"

    return mtf_dir / filename


def save_raw_mtf(
    symbol: str,
    timeframe: str,
    split: str,
    df: pd.DataFrame,
    base_path: str | Path = "data/canonical",
    compression: str = DEFAULT_COMPRESSION,
) -> Path:
    """
    Save raw OHLCV data for a specific timeframe and split.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "MES", "MGC")
    timeframe : str
        Timeframe string (e.g., "5min", "1h")
    split : str
        Data split ("train", "val", "test")
    df : pd.DataFrame
        DataFrame containing OHLCV data. Expected columns:
        - datetime/timestamp: Time index
        - open, high, low, close: Price data
        - volume: Trading volume
    base_path : str or Path, default "data/canonical"
        Base directory for data storage
    compression : str, default "snappy"
        Parquet compression codec

    Returns
    -------
    Path
        Path to the saved parquet file

    Raises
    ------
    InvalidTimeframeError
        If timeframe is not in the standard set
    InvalidSplitError
        If split is not valid
    ValueError
        If DataFrame is empty

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "datetime": pd.date_range("2024-01-01", periods=100, freq="5min"),
    ...     "open": [100.0] * 100,
    ...     "high": [101.0] * 100,
    ...     "low": [99.0] * 100,
    ...     "close": [100.5] * 100,
    ...     "volume": [1000] * 100,
    ... })
    >>> path = save_raw_mtf("MES", "5min", "train", df)
    """
    if df.empty:
        raise ValueError("Cannot save empty DataFrame")

    path = get_mtf_path(symbol, timeframe, split, base_path)

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df.to_parquet(path, compression=compression, index=True)

    logger.info(f"Saved raw MTF data: {symbol}/{timeframe}/{split} " f"({len(df)} rows) -> {path}")

    return path


def load_raw_mtf(
    symbol: str,
    timeframe: str,
    split: str,
    base_path: str | Path = "data/canonical",
) -> pd.DataFrame:
    """
    Load raw OHLCV data for a specific timeframe and split.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "MES", "MGC")
    timeframe : str
        Timeframe string (e.g., "5min", "1h")
    split : str
        Data split ("train", "val", "test")
    base_path : str or Path, default "data/canonical"
        Base directory for data storage

    Returns
    -------
    pd.DataFrame
        DataFrame containing OHLCV data

    Raises
    ------
    InvalidTimeframeError
        If timeframe is not in the standard set
    InvalidSplitError
        If split is not valid
    TimeframeNotFoundError
        If the requested file does not exist

    Examples
    --------
    >>> df = load_raw_mtf("MES", "5min", "train")
    >>> print(df.columns.tolist())
    ['datetime', 'open', 'high', 'low', 'close', 'volume']
    """
    path = get_mtf_path(symbol, timeframe, split, base_path)

    if not path.exists():
        raise TimeframeNotFoundError(
            f"Raw MTF data not found: {symbol}/{timeframe}/{split} " f"(expected at {path})"
        )

    df = pd.read_parquet(path)

    logger.debug(f"Loaded raw MTF data: {symbol}/{timeframe}/{split} ({len(df)} rows)")

    return df


def load_all_timeframes(
    symbol: str,
    split: str,
    base_path: str | Path = "data/canonical",
    timeframes: list[str] | None = None,
    missing_ok: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load raw OHLCV data for all timeframes (or a specified subset).

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "MES", "MGC")
    split : str
        Data split ("train", "val", "test")
    base_path : str or Path, default "data/canonical"
        Base directory for data storage
    timeframes : list of str, optional
        Specific timeframes to load. If None, loads all standard timeframes.
    missing_ok : bool, default False
        If True, skip missing timeframes instead of raising an error.
        Missing timeframes will not appear in the returned dictionary.

    Returns
    -------
    dict of str to pd.DataFrame
        Dictionary mapping timeframe strings to DataFrames

    Raises
    ------
    InvalidSplitError
        If split is not valid
    InvalidTimeframeError
        If any timeframe in the list is not in the standard set
    TimeframeNotFoundError
        If missing_ok is False and any requested file does not exist

    Examples
    --------
    >>> # Load all timeframes
    >>> all_tf = load_all_timeframes("MES", "train")
    >>> print(list(all_tf.keys()))
    ['1min', '3min', '5min', '10min', '15min', '30min', '60min', '2h', '4h']

    >>> # Load specific timeframes
    >>> subset = load_all_timeframes("MES", "train", timeframes=["5min", "15min", "60min"])
    >>> print(list(subset.keys()))
    ['5min', '15min', '60min']

    >>> # Load with missing tolerance
    >>> partial = load_all_timeframes("MES", "train", missing_ok=True)
    """
    _validate_split(split)

    if timeframes is None:
        timeframes_to_load = TIMEFRAMES
    else:
        # Validate all requested timeframes
        for tf in timeframes:
            _validate_timeframe(tf)
        timeframes_to_load = timeframes

    result: dict[str, pd.DataFrame] = {}

    for tf in timeframes_to_load:
        try:
            df = load_raw_mtf(symbol, tf, split, base_path)
            result[tf] = df
        except TimeframeNotFoundError:
            if missing_ok:
                logger.debug(f"Skipping missing timeframe: {symbol}/{tf}/{split}")
            else:
                raise

    logger.info(
        f"Loaded {len(result)}/{len(timeframes_to_load)} timeframes " f"for {symbol}/{split}"
    )

    return result


def list_available_timeframes(
    symbol: str,
    split: str,
    base_path: str | Path = "data/canonical",
) -> list[str]:
    """
    List which timeframes are available for a given symbol and split.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "MES", "MGC")
    split : str
        Data split ("train", "val", "test")
    base_path : str or Path, default "data/canonical"
        Base directory for data storage

    Returns
    -------
    list of str
        List of available timeframe strings

    Examples
    --------
    >>> available = list_available_timeframes("MES", "train")
    >>> print(available)
    ['5min', '15min', '60min']
    """
    _validate_split(split)

    available = []
    for tf in TIMEFRAMES:
        path = get_mtf_path(symbol, tf, split, base_path)
        if path.exists():
            available.append(tf)

    return available


def delete_raw_mtf(
    symbol: str,
    timeframe: str,
    split: str,
    base_path: str | Path = "data/canonical",
) -> bool:
    """
    Delete raw OHLCV data for a specific timeframe and split.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "MES", "MGC")
    timeframe : str
        Timeframe string (e.g., "5min", "1h")
    split : str
        Data split ("train", "val", "test")
    base_path : str or Path, default "data/canonical"
        Base directory for data storage

    Returns
    -------
    bool
        True if file was deleted, False if it didn't exist

    Examples
    --------
    >>> deleted = delete_raw_mtf("MES", "5min", "train")
    >>> print(deleted)
    True
    """
    path = get_mtf_path(symbol, timeframe, split, base_path)

    if path.exists():
        path.unlink()
        logger.info(f"Deleted raw MTF data: {path}")
        return True
    else:
        logger.debug(f"File not found for deletion: {path}")
        return False
