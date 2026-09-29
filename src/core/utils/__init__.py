"""
Utility modules for the ML Model Factory.

Memory management, math helpers, JSON encoding and safe pickle I/O.
"""

from .json_utils import NumpyEncoder

# Math utilities (Phase 8A)
from .math_utils import (
    ema,
    normalize_series,
    safe_divide,
    sma,
)
from .memory import (
    PSUTIL_AVAILABLE,
    MemoryInfo,
    check_memory_sufficient,
    estimate_array_size,
    get_memory_info,
    log_memory_usage,
    memory_logged,
)
from .safe_pickle import safe_pickle_dump, safe_pickle_load

__all__ = [
    # Memory management (MOD-007)
    "MemoryInfo",
    "estimate_array_size",
    "get_memory_info",
    "check_memory_sufficient",
    "log_memory_usage",
    "memory_logged",
    "PSUTIL_AVAILABLE",
    # Math utilities (Phase 8A)
    "safe_divide",
    "sma",
    "ema",
    "normalize_series",
    # Safe pickle loading
    "NumpyEncoder",
    "safe_pickle_dump",
    "safe_pickle_load",
]
