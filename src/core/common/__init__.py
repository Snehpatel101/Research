"""Common utilities shared across phases.

Provides artifact manifest management, horizon configuration, timeframe utilities,
and default split ratios.
"""

from .horizon_config import (
    ACTIVE_HORIZONS,
    HORIZON_TIMEFRAME_MINUTES,
    HORIZONS,
    LABEL_HORIZONS,
    LOOKBACK_HORIZONS,
    SUPPORTED_HORIZONS,
    HorizonConfig,
    auto_scale_purge_embargo,
    get_default_barrier_params_for_horizon,
    get_scaled_horizons,
    validate_horizons,
)
from .manifest import ArtifactManifest
from .split_ratios import (
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    validate_split_ratios,
)
from .timeframes import (
    ALL_CANONICAL_TIMEFRAMES,
    CANONICAL_TIMEFRAMES,
    EXTENDED_TIMEFRAMES,
    FULL_9TF_LADDER,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_ALIASES,
    TIMEFRAME_TO_FREQ,
    TIMEFRAME_TO_MINUTES,
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
    # manifest
    "ArtifactManifest",
    # horizon_config
    "HORIZONS",
    "SUPPORTED_HORIZONS",
    "ACTIVE_HORIZONS",
    "LABEL_HORIZONS",
    "LOOKBACK_HORIZONS",
    "HORIZON_TIMEFRAME_MINUTES",
    "HorizonConfig",
    "validate_horizons",
    "get_scaled_horizons",
    "auto_scale_purge_embargo",
    "get_default_barrier_params_for_horizon",
    # split_ratios (CFG-010)
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_SPLIT_RATIOS",
    "validate_split_ratios",
    # timeframes
    "ALL_CANONICAL_TIMEFRAMES",
    "CANONICAL_TIMEFRAMES",
    "EXTENDED_TIMEFRAMES",
    "FULL_9TF_LADDER",
    "SUPPORTED_TIMEFRAMES",
    "TIMEFRAME_ALIASES",
    "TIMEFRAME_TO_FREQ",
    "TIMEFRAME_TO_MINUTES",
    "get_canonical_from_suffix",
    "get_timeframe_minutes",
    "get_timeframe_suffix",
    "is_valid_timeframe",
    "normalize_timeframe",
    "normalize_timeframe_list",
    "timeframe_to_minutes",
    "validate_timeframe",
]
