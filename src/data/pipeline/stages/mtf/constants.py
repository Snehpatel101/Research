"""
Constants for Multi-Timeframe (MTF) Feature Integration.

Supports timeframes from 1min (base) up to daily.
All timeframes must be integer multiples of the base timeframe.

NOTE: This module imports canonical timeframe definitions from src.core.common.timeframes.
Do not redefine timeframe constants here - use the common module instead.
"""

# Import canonical timeframe definitions from the single source of truth
from src.core.common.timeframes import (
    CANONICAL_TIMEFRAMES,
    normalize_timeframe_list,
)
from src.core.common.timeframes import (
    TIMEFRAME_TO_FREQ as _COMMON_TIMEFRAME_TO_FREQ,
)
from src.core.common.timeframes import (
    TIMEFRAME_TO_MINUTES as _COMMON_TIMEFRAME_TO_MINUTES,
)


# MTFMode imported from canonical location (src/config/data)
from src.config.data import MTFMode


# Supported MTF timeframes with their minute equivalents
# Imported from common module - includes all canonical forms and aliases
MTF_TIMEFRAMES = _COMMON_TIMEFRAME_TO_MINUTES

# Required OHLCV columns
REQUIRED_OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Default MTF configuration - 7-timeframe intraday ladder (canonical forms)
# Provides comprehensive multi-scale analysis from 10min to 1h
# Excludes 5min since it's commonly used as base timeframe
# Daily excluded by default for intraday trading focus
# NOTE: Using canonical forms ("60min" instead of "1h") for consistency
DEFAULT_MTF_TIMEFRAMES = normalize_timeframe_list(
    [
        "10min",  # Short-term momentum
        "15min",  # Standard short-term
        "20min",  # Extended short-term
        "25min",  # Medium transition
        "30min",  # Standard medium-term
        "45min",  # Extended medium-term
        "60min",  # Hourly trend (canonical form)
    ]
)

# Full 9-timeframe ladder including 1min base (canonical forms)
FULL_MTF_TIMEFRAMES = CANONICAL_TIMEFRAMES.copy()

# Default mode is to generate both bars and indicators
DEFAULT_MTF_MODE = MTFMode.BOTH

# Base timeframe for resampling
DEFAULT_BASE_TIMEFRAME = "1min"

# Minimum data requirements
MIN_BASE_BARS = 100  # Minimum base TF bars required
MIN_MTF_BARS = 30  # Minimum MTF bars required per timeframe

# Pandas frequency aliases for resampling
# Imported from common module for consistency
PANDAS_FREQ_MAP = _COMMON_TIMEFRAME_TO_FREQ
