"""
Memory management utilities for the ML Model Factory.

Provides memory tracking, estimation, and automatic cleanup to prevent
out-of-memory errors during training and inference.
"""

from __future__ import annotations

import functools
import gc
import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

# Try to import psutil for system memory checks
psutil: Any = None  # Type declaration for conditional import
try:
    import psutil  # type: ignore[no-redef]

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Memory monitoring will be limited.")


# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class MemoryInfo:
    """Information about current memory state."""

    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent_used: float

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    @property
    def available_gb(self) -> float:
        return self.available_bytes / (1024**3)

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024**3)


@dataclass
class CacheEntry:
    """Entry in the cache with metadata."""

    key: str
    data: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    version: int = 1
    ttl_seconds: float | None = None

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Time since entry was created."""
        return time.time() - self.created_at


def estimate_array_size(arr: np.ndarray | None) -> int:
    """
    Estimate memory usage of a numpy array in bytes.

    Args:
        arr: NumPy array to estimate size of

    Returns:
        Estimated size in bytes
    """
    if arr is None:
        return 0
    return int(arr.nbytes)


def estimate_object_size(obj: Any) -> int:
    """
    Estimate memory usage of an object.

    Handles numpy arrays specially for accuracy. For other objects,
    uses sys.getsizeof as a rough estimate.

    Args:
        obj: Object to estimate size of

    Returns:
        Estimated size in bytes
    """
    import sys

    if obj is None:
        return 0

    if isinstance(obj, np.ndarray):
        return estimate_array_size(obj)

    if isinstance(obj, (list, tuple)):
        return sys.getsizeof(obj) + sum(estimate_object_size(item) for item in obj)

    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(
            estimate_object_size(k) + estimate_object_size(v) for k, v in obj.items()
        )

    return sys.getsizeof(obj)


def get_memory_info() -> MemoryInfo:
    """
    Get current system memory information.

    Returns:
        MemoryInfo with current memory state
    """
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        return MemoryInfo(
            total_bytes=mem.total,
            available_bytes=mem.available,
            used_bytes=mem.used,
            percent_used=mem.percent,
        )
    else:
        # Fallback: Return placeholder values
        logger.warning("psutil not available, returning placeholder memory info")
        return MemoryInfo(
            total_bytes=16 * 1024**3,  # Assume 16GB
            available_bytes=8 * 1024**3,  # Assume 8GB available
            used_bytes=8 * 1024**3,
            percent_used=50.0,
        )


def check_available_memory() -> int:
    """
    Get available system memory in bytes.

    Returns:
        Available memory in bytes
    """
    return get_memory_info().available_bytes


def check_memory_sufficient(required_bytes: int, safety_margin: float = 0.2) -> bool:
    """
    Check if there is sufficient memory for an operation.

    Args:
        required_bytes: Memory required in bytes
        safety_margin: Fraction of available memory to keep free (default 20%)

    Returns:
        True if sufficient memory available
    """
    available = check_available_memory()
    # Keep safety_margin of available memory free
    usable = int(available * (1 - safety_margin))
    return required_bytes <= usable


@contextmanager
def log_memory_usage(context: str, log_level: int = logging.DEBUG):
    """
    Context manager that logs memory usage before and after a block.

    Args:
        context: Description of the operation being measured
        log_level: Logging level to use (default: DEBUG)

    Example:
        with log_memory_usage("Training model"):
            model.fit(X_train, y_train)
    """
    gc.collect()
    before = get_memory_info()
    logger.log(log_level, f"[MEMORY] {context} - Before: {before.used_gb:.2f}GB used")

    start_time = time.time()
    try:
        yield
    finally:
        gc.collect()
        after = get_memory_info()
        elapsed = time.time() - start_time
        delta = after.used_gb - before.used_gb

        logger.log(
            log_level,
            f"[MEMORY] {context} - After: {after.used_gb:.2f}GB used "
            f"(delta: {delta:+.2f}GB, time: {elapsed:.2f}s)",
        )


def memory_logged(context: str | None = None, log_level: int = logging.DEBUG) -> Callable[[F], F]:
    """
    Decorator that logs memory usage before and after function execution.

    Args:
        context: Description for logging (defaults to function name)
        log_level: Logging level to use

    Example:
        @memory_logged("Feature engineering")
        def compute_features(df):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = context or func.__name__
            with log_memory_usage(ctx, log_level):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@dataclass
class CacheStats:
    """Statistics about cache usage."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size_bytes: int = 0
    max_size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


@dataclass
class CacheConfig:
    """Configuration for CacheManager."""

    max_memory_bytes: int = 1024 * 1024 * 1024  # 1GB default
    low_memory_threshold_bytes: int = 512 * 1024 * 1024  # 512MB triggers cleanup
    default_ttl_seconds: float | None = None  # No expiry by default
    enable_staleness_detection: bool = True


class CacheManager:
    """
    LRU cache manager with memory limits and staleness detection.

    Features:
    - LRU eviction policy
    - Maximum memory limit
    - Automatic cleanup on low memory
    - Staleness detection via TTL or version
    - Thread-safe operations

    Example:
        cache = CacheManager(max_memory_bytes=1024**3)  # 1GB
        cache.put("features", X_train, ttl_seconds=3600)
        X_train = cache.get("features")
        cache.clear()
    """

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self._config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size_bytes=self._config.max_memory_bytes)
        self._versions: dict[str, int] = {}

    def put(
        self,
        key: str,
        data: Any,
        ttl_seconds: float | None = None,
        version: int | None = None,
    ) -> bool:
        """
        Store data in cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time-to-live in seconds (None for no expiry)
            version: Data version for staleness detection

        Returns:
            True if stored successfully, False if data too large
        """
        size = estimate_object_size(data)

        # Check if single item exceeds max cache size
        if size > self._config.max_memory_bytes:
            logger.warning(
                f"Cache item '{key}' size ({size / 1024**2:.1f}MB) exceeds "
                f"max cache size ({self._config.max_memory_bytes / 1024**2:.1f}MB). "
                f"Not caching."
            )
            return False

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Evict entries to make room
            self._evict_if_needed(size)

            # Check memory again after eviction
            if self._stats.current_size_bytes + size > self._config.max_memory_bytes:
                avail = self._config.max_memory_bytes - self._stats.current_size_bytes
                logger.warning(
                    f"Cannot cache '{key}': insufficient space after eviction. "
                    f"Required: {size / 1024**2:.1f}MB, Available: {avail / 1024**2:.1f}MB"
                )
                return False

            # Determine version
            if version is None:
                version = self._versions.get(key, 0) + 1
            self._versions[key] = version

            # Create and store entry
            entry = CacheEntry(
                key=key,
                data=data,
                size_bytes=size,
                created_at=time.time(),
                last_accessed=time.time(),
                version=version,
                ttl_seconds=ttl_seconds or self._config.default_ttl_seconds,
            )
            self._cache[key] = entry
            self._stats.current_size_bytes += size
            self._stats.entry_count += 1

            logger.debug(
                f"Cached '{key}': {size / 1024**2:.1f}MB, "
                f"total: {self._stats.current_size_bytes / 1024**2:.1f}MB"
            )
            return True

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve data from cache.

        Args:
            key: Cache key
            default: Value to return if key not found or expired

        Returns:
            Cached data or default
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            # Check expiry
            if self._config.enable_staleness_detection and entry.is_expired:
                logger.debug(f"Cache entry '{key}' expired (age: {entry.age_seconds:.1f}s)")
                self._remove_entry(key)
                self._stats.misses += 1
                return default

            # Update access time and move to end (LRU)
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
            self._stats.hits += 1

            return entry.data

    def get_if_valid(self, key: str, min_version: int) -> Any | None:
        """
        Get cached data only if version is at least min_version.

        Args:
            key: Cache key
            min_version: Minimum acceptable version

        Returns:
            Cached data if valid, None otherwise
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.version < min_version:
                logger.debug(
                    f"Cache entry '{key}' stale (version {entry.version} < {min_version})"
                )
                self._stats.misses += 1
                return None

            if self._config.enable_staleness_detection and entry.is_expired:
                self._remove_entry(key)
                self._stats.misses += 1
                return None

            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return entry.data

    def invalidate(self, key: str) -> bool:
        """
        Remove a specific entry from cache.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def invalidate_prefix(self, prefix: str) -> int:
        """
        Invalidate all entries with keys starting with prefix.

        Args:
            prefix: Key prefix to match

        Returns:
            Number of entries removed
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                self._remove_entry(key)
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._stats.current_size_bytes = 0
            self._stats.entry_count = 0
            gc.collect()
            logger.debug("Cache cleared")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                current_size_bytes=self._stats.current_size_bytes,
                max_size_bytes=self._stats.max_size_bytes,
                entry_count=self._stats.entry_count,
            )

    def _remove_entry(self, key: str) -> None:
        """Remove entry and update stats (must hold lock)."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats.current_size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1

    def _evict_if_needed(self, required_bytes: int) -> None:
        """Evict LRU entries to make room (must hold lock)."""
        target_size = self._config.max_memory_bytes - required_bytes

        while self._stats.current_size_bytes > target_size and self._cache:
            # Evict oldest (first) entry
            oldest_key = next(iter(self._cache))
            logger.debug(f"Evicting '{oldest_key}' to make room")
            self._remove_entry(oldest_key)
            self._stats.evictions += 1

    def cleanup_if_low_memory(self) -> int:
        """
        Check system memory and evict entries if running low.

        Returns:
            Number of entries evicted
        """
        available = check_available_memory()
        if available > self._config.low_memory_threshold_bytes:
            return 0

        logger.warning(
            f"Low system memory ({available / 1024**2:.1f}MB available). "
            f"Triggering cache cleanup."
        )

        evicted = 0
        with self._lock:
            # Evict half the cache entries
            target_evictions = max(1, self._stats.entry_count // 2)
            while evicted < target_evictions and self._cache:
                oldest_key = next(iter(self._cache))
                self._remove_entry(oldest_key)
                self._stats.evictions += 1
                evicted += 1

        gc.collect()
        return evicted


# Global cache manager instance (lazy initialization)
_global_cache: CacheManager | None = None
_global_cache_lock = threading.Lock()


def get_global_cache() -> CacheManager:
    """Get the global cache manager instance."""
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = CacheManager()
    return _global_cache


__all__ = [
    # Data classes
    "MemoryInfo",
    "CacheEntry",
    "CacheStats",
    "CacheConfig",
    # Functions
    "estimate_array_size",
    "estimate_object_size",
    "get_memory_info",
    "check_available_memory",
    "check_memory_sufficient",
    "log_memory_usage",
    "memory_logged",
    # Classes
    "CacheManager",
    # Global cache
    "get_global_cache",
    # Constants
    "PSUTIL_AVAILABLE",
]
