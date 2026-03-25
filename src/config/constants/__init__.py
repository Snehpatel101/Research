"""
Central constants - re-exports from src.core.common.

This module provides unified access to constants from src.core.common.

Usage:
    from src.config.constants import CANONICAL_TIMEFRAMES, DEFAULT_SPLIT_RATIOS
    from src.config.constants import is_valid_timeframe, get_timeframe_minutes
    from src.config.constants import HorizonConfig, validate_horizons
"""

from src.core.common.horizon_config import (
    # Horizon lists
    ACTIVE_HORIZONS,
    # Purge/embargo configuration
    DEFAULT_TIMEFRAME_MINUTES,
    EMBARGO_MULTIPLIER,
    EMBARGO_TIME_MINUTES,
    HORIZONS,
    LABEL_HORIZONS,
    LOOKBACK_HORIZONS,
    MIN_EMBARGO_BARS,
    PURGE_MULTIPLIER,
    SUPPORTED_HORIZONS,
    # HorizonConfig dataclass
    HorizonConfig,
    # Functions
    auto_scale_purge_embargo,
    compute_embargo_bars,
    get_default_barrier_params_for_horizon,
    get_scaled_horizons,
    validate_horizons,
)

# =============================================================================
# SPLIT RATIO CONFIGURATION
# =============================================================================
from src.core.common.split_ratios import (
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    validate_split_ratios,
)
from src.core.common.timeframes import (
    # Core timeframe lists
    ALL_CANONICAL_TIMEFRAMES,
    CANONICAL_TIMEFRAMES,
    EXTENDED_TIMEFRAMES,
    FULL_9TF_LADDER,
    SUPPORTED_TIMEFRAMES,
    # Mappings
    TIMEFRAME_ALIASES,
    TIMEFRAME_TO_FREQ,
    TIMEFRAME_TO_MINUTES,
    # Functions
    get_canonical_from_suffix,
    get_timeframe_minutes,
    get_timeframe_suffix,
    is_valid_timeframe,
    normalize_timeframe,
    normalize_timeframe_list,
    timeframe_to_minutes,
    validate_timeframe,
)

__all__ = [
    # Timeframe lists
    "CANONICAL_TIMEFRAMES",
    "FULL_9TF_LADDER",
    "EXTENDED_TIMEFRAMES",
    "ALL_CANONICAL_TIMEFRAMES",
    "SUPPORTED_TIMEFRAMES",
    # Timeframe mappings
    "TIMEFRAME_ALIASES",
    "TIMEFRAME_TO_MINUTES",
    "TIMEFRAME_TO_FREQ",
    # Timeframe functions
    "normalize_timeframe",
    "normalize_timeframe_list",
    "get_timeframe_minutes",
    "timeframe_to_minutes",
    "is_valid_timeframe",
    "validate_timeframe",
    "get_timeframe_suffix",
    "get_canonical_from_suffix",
    # Split ratios
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "validate_split_ratios",
    # Horizon configuration
    "HORIZONS",
    "SUPPORTED_HORIZONS",
    "ACTIVE_HORIZONS",
    "LABEL_HORIZONS",
    "LOOKBACK_HORIZONS",
    # Purge/embargo configuration
    "PURGE_MULTIPLIER",
    "EMBARGO_MULTIPLIER",
    "EMBARGO_TIME_MINUTES",
    "DEFAULT_TIMEFRAME_MINUTES",
    "MIN_EMBARGO_BARS",
    # HorizonConfig dataclass
    "HorizonConfig",
    # Horizon functions
    "validate_horizons",
    "get_scaled_horizons",
    "auto_scale_purge_embargo",
    "compute_embargo_bars",
    "get_default_barrier_params_for_horizon",
]
