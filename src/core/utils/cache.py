"""
Data caching utilities for the ML Model Factory pipeline.

Provides intelligent caching of pipeline artifacts (features, labels, scaled data)
with automatic invalidation based on source file modification times.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import pickle
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from src.core.utils.memory import (
    CacheConfig,
    CacheManager,
    check_available_memory,
    estimate_object_size,
)

logger = logging.getLogger(__name__)

# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CacheMetadata:
    """Metadata about a cached item."""

    key: str
    source_files: list[str]
    source_mtimes: dict[str, float]
    created_at: float
    size_bytes: int
    checksum: str | None = None

    def is_stale(self) -> bool:
        """
        Check if cache is stale based on source file modification times.

        Returns:
            True if any source file has been modified since caching
        """
        for source_file, cached_mtime in self.source_mtimes.items():
            path = Path(source_file)
            if not path.exists():
                logger.debug(f"Source file missing: {source_file}")
                return True
            current_mtime = path.stat().st_mtime
            if current_mtime > cached_mtime:
                logger.debug(
                    f"Source file modified: {source_file} "
                    f"(cached: {cached_mtime}, current: {current_mtime})"
                )
                return True
        return False


@dataclass
class DataCacheConfig:
    """Configuration for DataCache."""

    # Memory limits
    max_memory_bytes: int = 2 * 1024**3  # 2GB default for data cache
    enable_disk_fallback: bool = True
    disk_cache_dir: Path | None = None

    # Invalidation settings
    check_source_mtimes: bool = True

    # Size thresholds
    disk_fallback_threshold_bytes: int = 500 * 1024 * 1024  # 500MB

    def __post_init__(self) -> None:
        if self.disk_cache_dir is None:
            self.disk_cache_dir = Path.home() / ".cache" / "ml_factory"


class DataCache:
    """
    Cache for pipeline artifacts with intelligent invalidation.

    Features:
    - Caches features, labels, and scaled data
    - Automatic invalidation based on source file modification times
    - Disk-based fallback for large datasets
    - Thread-safe operations

    Example:
        cache = DataCache()
        cache.put_features("15min", X_features, source_files=["data/raw/MES_1m.parquet"])
        X_features = cache.get_features("15min")  # Returns cached data or None if stale
    """

    def __init__(self, config: DataCacheConfig | None = None):
        """
        Initialize data cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self._config = config or DataCacheConfig()
        self._memory_cache = CacheManager(
            CacheConfig(
                max_memory_bytes=self._config.max_memory_bytes,
                enable_staleness_detection=False,  # We handle staleness ourselves
            )
        )
        self._metadata: dict[str, CacheMetadata] = {}
        self._lock = threading.RLock()

        # Create disk cache directory if needed
        if self._config.enable_disk_fallback and self._config.disk_cache_dir:
            self._config.disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, category: str, identifier: str) -> str:
        """Create cache key from category and identifier."""
        return f"{category}:{identifier}"

    def _get_source_mtimes(self, source_files: list[str]) -> dict[str, float]:
        """Get modification times for source files."""
        mtimes = {}
        for source_file in source_files:
            path = Path(source_file)
            if path.exists():
                mtimes[source_file] = path.stat().st_mtime
        return mtimes

    def _compute_checksum(self, data: Any) -> str:
        """Compute checksum for data integrity verification."""
        if isinstance(data, np.ndarray):
            return hashlib.md5(data.tobytes()).hexdigest()[:16]
        elif isinstance(data, pd.DataFrame):
            # Use DataFrame's values for hashing (simpler and more portable)
            return hashlib.md5(data.values.tobytes()).hexdigest()[:16]
        else:
            # Generic approach: pickle and hash
            return hashlib.md5(pickle.dumps(data)).hexdigest()[:16]

    def _should_use_disk(self, size_bytes: int) -> bool:
        """Determine if data should be stored on disk."""
        if not self._config.enable_disk_fallback:
            return False
        if size_bytes > self._config.disk_fallback_threshold_bytes:
            return True
        # Also use disk if memory is low
        available = check_available_memory()
        return size_bytes > available * 0.5

    def _get_disk_path(self, key: str) -> Path:
        """Get disk cache path for a key."""
        # Sanitize key for filesystem
        safe_key = key.replace(":", "_").replace("/", "_")
        assert self._config.disk_cache_dir is not None
        return self._config.disk_cache_dir / f"{safe_key}.pkl"

    def _save_to_disk(self, key: str, data: Any, metadata: CacheMetadata) -> bool:
        """Save data to disk cache."""
        try:
            disk_path = self._get_disk_path(key)
            with open(disk_path, "wb") as f:
                pickle.dump({"data": data, "metadata": metadata}, f)
            logger.debug(f"Saved to disk cache: {disk_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
            return False

    def _load_from_disk(self, key: str) -> tuple[Any, CacheMetadata] | None:
        """Load data from disk cache."""
        disk_path = self._get_disk_path(key)
        if not disk_path.exists():
            return None
        try:
            from src.core.utils.safe_pickle import safe_pickle_load

            cached = safe_pickle_load(disk_path)
            return cached["data"], cached["metadata"]
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None

    def _remove_from_disk(self, key: str) -> None:
        """Remove data from disk cache."""
        disk_path = self._get_disk_path(key)
        if disk_path.exists():
            try:
                disk_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove disk cache: {e}")

    def put(
        self,
        category: str,
        identifier: str,
        data: Any,
        source_files: list[str] | None = None,
    ) -> bool:
        """
        Store data in cache.

        Args:
            category: Data category (features, labels, scaled)
            identifier: Unique identifier (e.g., timeframe, horizon)
            data: Data to cache
            source_files: List of source files for invalidation tracking

        Returns:
            True if cached successfully
        """
        key = self._make_key(category, identifier)
        size = estimate_object_size(data)
        source_files = source_files or []

        # Check size limit
        if size > self._config.max_memory_bytes * 2:
            logger.warning(
                f"Data for '{key}' exceeds cache limit "
                f"({size / 1024**3:.2f}GB > {self._config.max_memory_bytes * 2 / 1024**3:.2f}GB). "
                f"Not caching."
            )
            return False

        with self._lock:
            # Create metadata
            metadata = CacheMetadata(
                key=key,
                source_files=source_files,
                source_mtimes=(
                    self._get_source_mtimes(source_files)
                    if self._config.check_source_mtimes
                    else {}
                ),
                created_at=time.time(),
                size_bytes=size,
                checksum=self._compute_checksum(data),
            )

            # Decide where to store
            if self._should_use_disk(size):
                logger.info(
                    f"Caching '{key}' to disk ({size / 1024**2:.1f}MB exceeds memory threshold)"
                )
                success = self._save_to_disk(key, data, metadata)
                if success:
                    self._metadata[key] = metadata
                return success
            else:
                # Store in memory
                success = self._memory_cache.put(key, data)
                if success:
                    self._metadata[key] = metadata
                return success

    def get(
        self,
        category: str,
        identifier: str,
        validate_freshness: bool = True,
    ) -> Any | None:
        """
        Retrieve data from cache.

        Args:
            category: Data category
            identifier: Unique identifier
            validate_freshness: If True, check source file modification times

        Returns:
            Cached data or None if not found/stale
        """
        key = self._make_key(category, identifier)

        with self._lock:
            metadata = self._metadata.get(key)
            if metadata is None:
                return None

            # Check freshness
            if validate_freshness and self._config.check_source_mtimes and metadata.is_stale():
                logger.info(f"Cache entry '{key}' is stale, invalidating")
                self.invalidate(category, identifier)
                return None

            # Try memory cache first
            data = self._memory_cache.get(key)
            if data is not None:
                return data

            # Try disk cache
            if self._config.enable_disk_fallback:
                result = self._load_from_disk(key)
                if result is not None:
                    data, _ = result
                    return data

            return None

    def invalidate(self, category: str, identifier: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            category: Data category
            identifier: Unique identifier

        Returns:
            True if entry was invalidated
        """
        key = self._make_key(category, identifier)

        with self._lock:
            removed = False

            # Remove from memory cache
            if self._memory_cache.invalidate(key):
                removed = True

            # Remove from disk cache
            self._remove_from_disk(key)

            # Remove metadata
            if key in self._metadata:
                del self._metadata[key]
                removed = True

            return removed

    def invalidate_category(self, category: str) -> int:
        """
        Invalidate all entries in a category.

        Args:
            category: Data category to invalidate

        Returns:
            Number of entries invalidated
        """
        prefix = f"{category}:"
        count = 0

        with self._lock:
            keys_to_remove = [k for k in self._metadata if k.startswith(prefix)]
            for key in keys_to_remove:
                identifier = key.split(":", 1)[1]
                if self.invalidate(category, identifier):
                    count += 1

        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            self._metadata.clear()

            # Clear disk cache
            if self._config.enable_disk_fallback and self._config.disk_cache_dir:
                for path in self._config.disk_cache_dir.glob("*.pkl"):
                    with contextlib.suppress(Exception):
                        path.unlink()

    # Convenience methods for specific data types

    def put_features(
        self,
        timeframe: str,
        data: np.ndarray | pd.DataFrame,
        source_files: list[str] | None = None,
    ) -> bool:
        """Cache feature data for a timeframe."""
        return self.put("features", timeframe, data, source_files)

    def get_features(
        self,
        timeframe: str,
        validate_freshness: bool = True,
    ) -> np.ndarray | pd.DataFrame | None:
        """Get cached feature data for a timeframe."""
        return self.get("features", timeframe, validate_freshness)

    def put_labels(
        self,
        horizon: int | str,
        data: np.ndarray | pd.Series,
        source_files: list[str] | None = None,
    ) -> bool:
        """Cache label data for a horizon."""
        return self.put("labels", str(horizon), data, source_files)

    def get_labels(
        self,
        horizon: int | str,
        validate_freshness: bool = True,
    ) -> np.ndarray | pd.Series | None:
        """Get cached label data for a horizon."""
        return self.get("labels", str(horizon), validate_freshness)

    def put_scaled(
        self,
        timeframe: str,
        data: np.ndarray | pd.DataFrame,
        source_files: list[str] | None = None,
    ) -> bool:
        """Cache scaled data for a timeframe."""
        return self.put("scaled", timeframe, data, source_files)

    def get_scaled(
        self,
        timeframe: str,
        validate_freshness: bool = True,
    ) -> np.ndarray | pd.DataFrame | None:
        """Get cached scaled data for a timeframe."""
        return self.get("scaled", timeframe, validate_freshness)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        mem_stats = self._memory_cache.get_stats()
        return {
            "memory_cache": {
                "hits": mem_stats.hits,
                "misses": mem_stats.misses,
                "evictions": mem_stats.evictions,
                "size_bytes": mem_stats.current_size_bytes,
                "entry_count": mem_stats.entry_count,
                "hit_rate": mem_stats.hit_rate,
            },
            "total_entries": len(self._metadata),
            "disk_enabled": self._config.enable_disk_fallback,
        }


def cached_result(
    cache_key: str | Callable[..., str] | None = None,
    ttl_seconds: float | None = None,
    max_size_bytes: int | None = None,
) -> Callable[[F], F]:
    """
    Decorator for memoizing expensive computations.

    Args:
        cache_key: Static key or function to compute key from args
        ttl_seconds: Time-to-live for cached result
        max_size_bytes: Maximum size to cache (skip caching if exceeded)

    Example:
        @cached_result(cache_key="features_15min", ttl_seconds=3600)
        def compute_features(df):
            ...

        @cached_result(cache_key=lambda df: f"features_{len(df)}")
        def compute_features(df):
            ...
    """
    # Use module-level cache
    _result_cache: dict[str, tuple[Any, float, float | None]] = {}
    _result_lock = threading.Lock()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Compute cache key
            if cache_key is None:
                key = f"{func.__module__}.{func.__name__}"
            elif callable(cache_key):
                key = cache_key(*args, **kwargs)
            else:
                key = cache_key

            with _result_lock:
                # Check cache
                if key in _result_cache:
                    result, timestamp, ttl = _result_cache[key]
                    # Check TTL
                    if ttl is None or time.time() - timestamp < ttl:
                        logger.debug(f"Cache hit for '{key}'")
                        return result
                    else:
                        logger.debug(f"Cache expired for '{key}'")
                        del _result_cache[key]

            # Compute result
            result = func(*args, **kwargs)

            # Check size before caching
            if max_size_bytes is not None:
                size = estimate_object_size(result)
                if size > max_size_bytes:
                    logger.debug(
                        f"Result for '{key}' exceeds size limit "
                        f"({size / 1024**2:.1f}MB > {max_size_bytes / 1024**2:.1f}MB), not caching"
                    )
                    return result

            # Cache result
            with _result_lock:
                _result_cache[key] = (result, time.time(), ttl_seconds)
                logger.debug(f"Cached result for '{key}'")

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# Global data cache instance (lazy initialization)
_global_data_cache: DataCache | None = None
_global_data_cache_lock = threading.Lock()


def get_global_data_cache() -> DataCache:
    """Get the global data cache instance."""
    global _global_data_cache
    if _global_data_cache is None:
        with _global_data_cache_lock:
            if _global_data_cache is None:
                _global_data_cache = DataCache()
    return _global_data_cache


__all__ = [
    # Data classes
    "CacheMetadata",
    "DataCacheConfig",
    # Classes
    "DataCache",
    # Decorators
    "cached_result",
    # Global cache
    "get_global_data_cache",
]
