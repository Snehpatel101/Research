"""
Parquet-based feature caching with content-addressable storage.

Implements a cache layer that stores computed features in Parquet format,
using hash-based keys for content-addressable storage. This enables
efficient cache invalidation based on input data and configuration changes.

Cache key computation:
    cache_key = hash(input_data_checksum + feature_config + symbol + version)

Cache directory structure:
    cache_dir/
        {feature_set}/
            {version}/
                {symbol}/
                    features.parquet
                    metadata.json
                    checksum.sha256
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """
    Metadata for a cached feature set.

    Stores information about when the cache was created, what data
    it was computed from, and validation checksums.
    """

    cache_key: str
    feature_set: str
    version: str
    symbol: str
    input_checksum: str  # Checksum of input data
    config_hash: str  # Hash of feature config
    row_count: int
    column_count: int
    columns: list[str]
    date_start: str
    date_end: str
    created_at: datetime = field(default_factory=datetime.now)
    parquet_checksum: str = ""  # Checksum of parquet file
    compression: str = "snappy"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "cache_key": self.cache_key,
            "feature_set": self.feature_set,
            "version": self.version,
            "symbol": self.symbol,
            "input_checksum": self.input_checksum,
            "config_hash": self.config_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "created_at": self.created_at.isoformat(),
            "parquet_checksum": self.parquet_checksum,
            "compression": self.compression,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CacheMetadata:
        """Create from dictionary."""
        return cls(
            cache_key=data["cache_key"],
            feature_set=data["feature_set"],
            version=data["version"],
            symbol=data["symbol"],
            input_checksum=data["input_checksum"],
            config_hash=data["config_hash"],
            row_count=data["row_count"],
            column_count=data["column_count"],
            columns=data["columns"],
            date_start=data["date_start"],
            date_end=data["date_end"],
            created_at=datetime.fromisoformat(data["created_at"]),
            parquet_checksum=data.get("parquet_checksum", ""),
            compression=data.get("compression", "snappy"),
            metadata=data.get("metadata", {}),
        )


class FeatureCache:
    """
    Parquet-based feature cache with content-addressable storage.

    Provides efficient caching of computed features with automatic
    invalidation based on input data changes and configuration changes.

    Parameters
    ----------
    cache_dir : Path or str
        Root directory for cache storage
    compression : str, default "snappy"
        Parquet compression codec
    validate_on_read : bool, default True
        Whether to validate checksums on cache reads

    Examples
    --------
    >>> cache = FeatureCache(cache_dir="data/feature_cache")
    >>> # Check cache
    >>> cache_key = cache.compute_cache_key(
    ...     input_checksum="abc123",
    ...     config_hash="def456",
    ...     symbol="MES",
    ...     version="1.0.0"
    ... )
    >>> if cache.has(feature_set="core", cache_key=cache_key):
    ...     df, metadata = cache.get(feature_set="core", cache_key=cache_key)
    ... else:
    ...     df = compute_features(...)
    ...     cache.put(df, feature_set="core", cache_key=cache_key, ...)
    """

    def __init__(
        self,
        cache_dir: Path | str,
        compression: str = "snappy",
        validate_on_read: bool = True,
    ) -> None:
        """
        Initialize feature cache.

        Parameters
        ----------
        cache_dir : Path or str
            Root directory for cache storage
        compression : str, default "snappy"
            Parquet compression codec
        validate_on_read : bool, default True
            Whether to validate checksums on cache reads
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.validate_on_read = validate_on_read

        logger.info(f"Initialized FeatureCache at {self.cache_dir}")

    def compute_cache_key(
        self,
        input_checksum: str,
        config_hash: str,
        symbol: str,
        version: str,
    ) -> str:
        """
        Compute cache key from input parameters.

        The cache key is a content-addressable identifier that changes
        when any of the inputs change.

        Parameters
        ----------
        input_checksum : str
            Checksum of input data
        config_hash : str
            Hash of feature configuration
        symbol : str
            Symbol name
        version : str
            Feature version

        Returns
        -------
        str
            Cache key (SHA256 hash)
        """
        key_parts = f"{input_checksum}:{config_hash}:{symbol}:{version}"
        return hashlib.sha256(key_parts.encode()).hexdigest()[:32]

    def _get_cache_path(self, feature_set: str, cache_key: str) -> Path:
        """Get path to cache directory for a feature set and key."""
        return self.cache_dir / feature_set / cache_key

    def _get_parquet_path(self, feature_set: str, cache_key: str) -> Path:
        """Get path to parquet file."""
        return self._get_cache_path(feature_set, cache_key) / "features.parquet"

    def _get_metadata_path(self, feature_set: str, cache_key: str) -> Path:
        """Get path to metadata file."""
        return self._get_cache_path(feature_set, cache_key) / "metadata.json"

    def _get_checksum_path(self, feature_set: str, cache_key: str) -> Path:
        """Get path to checksum file."""
        return self._get_cache_path(feature_set, cache_key) / "checksum.sha256"

    def has(self, feature_set: str, cache_key: str) -> bool:
        """
        Check if cache entry exists.

        Parameters
        ----------
        feature_set : str
            Feature set name
        cache_key : str
            Cache key

        Returns
        -------
        bool
            True if cache entry exists with valid files
        """
        parquet_path = self._get_parquet_path(feature_set, cache_key)
        metadata_path = self._get_metadata_path(feature_set, cache_key)
        return parquet_path.exists() and metadata_path.exists()

    def get(
        self,
        feature_set: str,
        cache_key: str,
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, CacheMetadata] | None:
        """
        Get cached features.

        Parameters
        ----------
        feature_set : str
            Feature set name
        cache_key : str
            Cache key
        columns : list[str], optional
            Specific columns to load (for efficiency)

        Returns
        -------
        tuple[DataFrame, CacheMetadata] or None
            Cached data and metadata, or None if not found or invalid

        Raises
        ------
        ValueError
            If validation fails and validate_on_read is True
        """
        if not self.has(feature_set, cache_key):
            logger.debug(f"Cache miss: {feature_set}/{cache_key}")
            return None

        parquet_path = self._get_parquet_path(feature_set, cache_key)
        metadata_path = self._get_metadata_path(feature_set, cache_key)

        # Load metadata
        try:
            with open(metadata_path) as f:
                metadata = CacheMetadata.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return None

        # Validate checksum if enabled
        if self.validate_on_read and metadata.parquet_checksum:
            actual_checksum = compute_file_checksum(parquet_path)
            if actual_checksum != metadata.parquet_checksum:
                logger.error(
                    f"Cache checksum mismatch for {feature_set}/{cache_key}: "
                    f"expected {metadata.parquet_checksum}, got {actual_checksum}"
                )
                raise ValueError(f"Cache integrity check failed for {feature_set}/{cache_key}")

        # Load parquet data
        try:
            if columns:
                df = pd.read_parquet(parquet_path, columns=columns)
            else:
                df = pd.read_parquet(parquet_path)
        except Exception as e:
            logger.error(f"Failed to load cached parquet: {e}")
            return None

        logger.info(
            f"Cache hit: {feature_set}/{cache_key} "
            f"({metadata.row_count} rows, {metadata.column_count} columns)"
        )
        return df, metadata

    def put(
        self,
        df: pd.DataFrame,
        feature_set: str,
        version: str,
        symbol: str,
        input_checksum: str,
        config_hash: str,
        date_start: str | datetime,
        date_end: str | datetime,
        cache_key: str | None = None,
        metadata: dict | None = None,
    ) -> CacheMetadata:
        """
        Store features in cache.

        Parameters
        ----------
        df : DataFrame
            Features to cache
        feature_set : str
            Feature set name
        version : str
            Feature version
        symbol : str
            Symbol name
        input_checksum : str
            Checksum of input data
        config_hash : str
            Hash of feature config
        date_start : str or datetime
            Start of data date range
        date_end : str or datetime
            End of data date range
        cache_key : str, optional
            Cache key (computed if not provided)
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        CacheMetadata
            Metadata for cached entry
        """
        # Compute cache key if not provided
        if cache_key is None:
            cache_key = self.compute_cache_key(
                input_checksum=input_checksum,
                config_hash=config_hash,
                symbol=symbol,
                version=version,
            )

        # Create cache directory
        cache_path = self._get_cache_path(feature_set, cache_key)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Write parquet file
        parquet_path = self._get_parquet_path(feature_set, cache_key)
        df.to_parquet(
            parquet_path,
            engine="pyarrow",
            compression=self.compression,
            index=False,
        )

        # Compute parquet checksum
        parquet_checksum = compute_file_checksum(parquet_path)

        # Create metadata
        cache_metadata = CacheMetadata(
            cache_key=cache_key,
            feature_set=feature_set,
            version=version,
            symbol=symbol,
            input_checksum=input_checksum,
            config_hash=config_hash,
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            date_start=date_start.isoformat() if isinstance(date_start, datetime) else date_start,
            date_end=date_end.isoformat() if isinstance(date_end, datetime) else date_end,
            parquet_checksum=parquet_checksum,
            compression=self.compression,
            metadata=metadata or {},
        )

        # Write metadata
        metadata_path = self._get_metadata_path(feature_set, cache_key)
        with open(metadata_path, "w") as f:
            json.dump(cache_metadata.to_dict(), f, indent=2, default=str)

        # Write checksum file
        checksum_path = self._get_checksum_path(feature_set, cache_key)
        with open(checksum_path, "w") as f:
            f.write(f"{parquet_checksum}  features.parquet\n")

        logger.info(
            f"Cached {feature_set}/{cache_key}: "
            f"{cache_metadata.row_count} rows, {cache_metadata.column_count} columns"
        )

        return cache_metadata

    def invalidate(self, feature_set: str, cache_key: str) -> bool:
        """
        Invalidate (remove) a cache entry.

        Parameters
        ----------
        feature_set : str
            Feature set name
        cache_key : str
            Cache key

        Returns
        -------
        bool
            True if entry was removed
        """
        cache_path = self._get_cache_path(feature_set, cache_key)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            logger.info(f"Invalidated cache: {feature_set}/{cache_key}")
            return True
        return False

    def invalidate_feature_set(self, feature_set: str) -> int:
        """
        Invalidate all cache entries for a feature set.

        Parameters
        ----------
        feature_set : str
            Feature set name

        Returns
        -------
        int
            Number of entries invalidated
        """
        feature_set_path = self.cache_dir / feature_set
        if not feature_set_path.exists():
            return 0

        count = 0
        for entry_path in feature_set_path.iterdir():
            if entry_path.is_dir():
                shutil.rmtree(entry_path)
                count += 1

        if count > 0:
            logger.info(f"Invalidated {count} cache entries for {feature_set}")

        return count

    def list_entries(self, feature_set: str | None = None) -> list[CacheMetadata]:
        """
        List all cache entries.

        Parameters
        ----------
        feature_set : str, optional
            Filter by feature set name

        Returns
        -------
        list[CacheMetadata]
            List of cache metadata entries
        """
        entries = []

        if feature_set:
            search_path = self.cache_dir / feature_set
            if not search_path.exists():
                return entries
            feature_sets = [search_path]
        else:
            feature_sets = [p for p in self.cache_dir.iterdir() if p.is_dir()]

        for fs_path in feature_sets:
            for entry_path in fs_path.iterdir():
                if not entry_path.is_dir():
                    continue
                metadata_path = entry_path / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            metadata = CacheMetadata.from_dict(json.load(f))
                        entries.append(metadata)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

        return entries

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Cache statistics including size and entry counts
        """
        total_size = 0
        total_entries = 0
        feature_sets: dict[str, dict] = {}

        for fs_path in self.cache_dir.iterdir():
            if not fs_path.is_dir():
                continue

            fs_name = fs_path.name
            fs_size = 0
            fs_entries = 0

            for entry_path in fs_path.iterdir():
                if not entry_path.is_dir():
                    continue

                parquet_path = entry_path / "features.parquet"
                if parquet_path.exists():
                    fs_size += parquet_path.stat().st_size
                    fs_entries += 1

            feature_sets[fs_name] = {
                "entries": fs_entries,
                "size_bytes": fs_size,
                "size_mb": round(fs_size / (1024 * 1024), 2),
            }

            total_size += fs_size
            total_entries += fs_entries

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "feature_sets": feature_sets,
            "cache_dir": str(self.cache_dir),
        }

    def cleanup_old_entries(
        self,
        max_age_days: int = 30,
        feature_set: str | None = None,
    ) -> int:
        """
        Remove cache entries older than specified age.

        Parameters
        ----------
        max_age_days : int, default 30
            Maximum age in days
        feature_set : str, optional
            Filter by feature set

        Returns
        -------
        int
            Number of entries removed
        """
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        removed = 0

        for entry in self.list_entries(feature_set):
            if entry.created_at.timestamp() < cutoff:
                if self.invalidate(entry.feature_set, entry.cache_key):
                    removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} cache entries older than {max_age_days} days")

        return removed


def compute_file_checksum(file_path: Path | str) -> str:
    """
    Compute SHA256 checksum of a file.

    Parameters
    ----------
    file_path : Path or str
        Path to file

    Returns
    -------
    str
        SHA256 hex digest

    Raises
    ------
    FileNotFoundError
        If file does not exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_dataframe_checksum(df: pd.DataFrame) -> str:
    """
    Compute checksum of a DataFrame's content.

    Uses a hash of the serialized DataFrame for content addressing.

    Parameters
    ----------
    df : DataFrame
        DataFrame to checksum

    Returns
    -------
    str
        SHA256 hex digest (first 32 chars)
    """
    # Hash based on shape and column hash
    # For performance, we sample instead of hashing entire content
    sha256 = hashlib.sha256()

    # Include shape
    sha256.update(f"shape:{df.shape}".encode())

    # Include column names and dtypes
    for col in df.columns:
        sha256.update(f"col:{col}:{df[col].dtype}".encode())

    # Include first/last rows and sample for content fingerprint
    if len(df) > 0:
        sha256.update(df.iloc[0].to_json().encode())
        sha256.update(df.iloc[-1].to_json().encode())

        # Sample middle rows if dataset is large
        if len(df) > 1000:
            sample_indices = [len(df) // 4, len(df) // 2, 3 * len(df) // 4]
            for idx in sample_indices:
                sha256.update(df.iloc[idx].to_json().encode())

    return sha256.hexdigest()[:32]
