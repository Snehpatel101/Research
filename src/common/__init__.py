"""Common utilities shared across phases.

Provides artifact manifest management, horizon configuration, and timeframe utilities.
"""

from src.common.horizon_config import (
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
from src.common.manifest import ArtifactManifest
from src.common.timeframes import (
    CANONICAL_TIMEFRAMES,
    FULL_9TF_LADDER,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_ALIASES,
    TIMEFRAME_TO_FREQ,
    TIMEFRAME_TO_MINUTES,
    is_valid_timeframe,
    normalize_timeframe,
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
    # timeframes
    "CANONICAL_TIMEFRAMES",
    "FULL_9TF_LADDER",
    "SUPPORTED_TIMEFRAMES",
    "TIMEFRAME_ALIASES",
    "TIMEFRAME_TO_FREQ",
    "TIMEFRAME_TO_MINUTES",
    "is_valid_timeframe",
    "normalize_timeframe",
    "timeframe_to_minutes",
    "validate_timeframe",
]
