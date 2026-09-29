"""
Memory management utilities for the ML Model Factory.

Provides system-memory queries, array size estimation and memory-usage logging.
"""

from __future__ import annotations

import functools
import gc
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

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


def check_memory_sufficient(required_bytes: int, safety_margin: float = 0.2) -> bool:
    """
    Check if there is sufficient memory for an operation.

    Args:
        required_bytes: Memory required in bytes
        safety_margin: Fraction of available memory to keep free (default 20%)

    Returns:
        True if sufficient memory available
    """
    available = get_memory_info().available_bytes
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


__all__ = [
    # Data classes
    "MemoryInfo",
    # Functions
    "estimate_array_size",
    "get_memory_info",
    "check_memory_sufficient",
    "log_memory_usage",
    "memory_logged",
    # Constants
    "PSUTIL_AVAILABLE",
]
