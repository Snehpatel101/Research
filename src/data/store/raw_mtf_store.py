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
from threading import Lock
from typing import NamedTuple

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


# =============================================================================
# MTF Cache - Memoize loaded parquet files with mtime invalidation
# =============================================================================


class _CacheEntry(NamedTuple):
    """Cache entry storing loaded DataFrame and file modification time."""

    df: pd.DataFrame
    mtime: float


class _MTFCache:
    """
    Thread-safe in-memory cache for MTF parquet files.

    Caches loaded DataFrames keyed by file path, with automatic invalidation
    when the source file's modification time changes.

    Attributes
    ----------
    maxsize : int
        Maximum number of entries to keep in cache
    """

    def __init__(self, maxsize: int = 100):
        """
        Initialize the MTF cache.

        Parameters
        ----------
        maxsize : int, default 100
            Maximum number of entries to cache
        """
        self._cache: dict[str, _CacheEntry] = {}
        self._maxsize = maxsize
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, path: Path, copy: bool = True) -> pd.DataFrame | None:
        """
        Get cached DataFrame if valid (file unchanged).

        Parameters
        ----------
        path : Path
            Path to the parquet file
        copy : bool, default True
            If True, return a copy of the cached DataFrame (safe).
            If False, return a view (faster, but caller must not modify).

        Returns
        -------
        pd.DataFrame or None
            Cached DataFrame if valid, None otherwise
        """
        key = str(path)
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check if file still exists and mtime matches
            if not path.exists():
                del self._cache[key]
                return None

            current_mtime = path.stat().st_mtime
            if current_mtime != entry.mtime:
                # File changed, invalidate cache entry
                del self._cache[key]
                logger.debug(f"Cache invalidated (mtime changed): {path.name}")
                return None

            self._hits += 1
            logger.debug(f"Cache hit: {path.name}")
            if copy:
                return entry.df.copy()
            return entry.df

    def put(self, path: Path, df: pd.DataFrame) -> None:
        """
        Store DataFrame in cache.

        Parameters
        ----------
        path : Path
            Path to the parquet file
        df : pd.DataFrame
            DataFrame to cache
        """
        key = str(path)
        mtime = path.stat().st_mtime

        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._maxsize and key not in self._cache:
                # Simple FIFO eviction - remove first item
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Cache eviction: {oldest_key}")

            self._cache[key] = _CacheEntry(df=df.copy(), mtime=mtime)
            self._misses += 1
            logger.debug(f"Cache miss, stored: {path.name}")

    def invalidate(self, path: Path) -> bool:
        """
        Invalidate a cache entry.

        Parameters
        ----------
        path : Path
            Path to invalidate

        Returns
        -------
        bool
            True if entry was removed
        """
        key = str(path)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            logger.info("MTF cache cleared")

    def stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Dictionary with hits, misses, and size
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "maxsize": self._maxsize,
            }


# Global cache instance
_mtf_cache = _MTFCache(maxsize=100)


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

    # Invalidate cache for this path since file changed
    _mtf_cache.invalidate(path)

    return path


def load_raw_mtf(
    symbol: str,
    timeframe: str,
    split: str,
    base_path: str | Path = "data/canonical",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load raw OHLCV data for a specific timeframe and split.

    Uses an in-memory cache with mtime-based invalidation to avoid
    repeated disk reads of unchanged files.

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
    use_cache : bool, default True
        Whether to use the in-memory cache. Set to False to always
        read from disk.

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

    # Try cache first if enabled
    if use_cache:
        cached_df = _mtf_cache.get(path)
        if cached_df is not None:
            return cached_df

    # Load from disk
    df = pd.read_parquet(path)

    # Store in cache
    if use_cache:
        _mtf_cache.put(path, df)

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
        # Invalidate cache entry
        _mtf_cache.invalidate(path)
        logger.info(f"Deleted raw MTF data: {path}")
        return True
    else:
        logger.debug(f"File not found for deletion: {path}")
        return False


# =============================================================================
# Cache Management Functions
# =============================================================================


def get_mtf_cache_stats() -> dict[str, int]:
    """
    Get MTF cache statistics.

    Returns
    -------
    dict
        Dictionary with keys: hits, misses, size, maxsize

    Examples
    --------
    >>> stats = get_mtf_cache_stats()
    >>> print(f"Hit rate: {stats['hits'] / (stats['hits'] + stats['misses']):.2%}")
    """
    return _mtf_cache.stats()


def clear_mtf_cache() -> None:
    """
    Clear all entries from the MTF cache.

    Use this when you want to force fresh reads from disk.

    Examples
    --------
    >>> clear_mtf_cache()
    """
    _mtf_cache.clear()
