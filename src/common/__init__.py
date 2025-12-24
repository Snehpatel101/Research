"""Common utilities shared across phases.

Provides artifact manifest management and horizon configuration.
"""
from src.common.manifest import ArtifactManifest

from src.common.horizon_config import (
    HORIZONS,
    SUPPORTED_HORIZONS,
    ACTIVE_HORIZONS,
    LOOKBACK_HORIZONS,
    HORIZON_TIMEFRAME_MINUTES,
    HorizonConfig,
    validate_horizons,
    get_scaled_horizons,
    auto_scale_purge_embargo,
    get_default_barrier_params_for_horizon,
)

__all__ = [
    # manifest
    'ArtifactManifest',
    # horizon_config
    'HORIZONS',
    'SUPPORTED_HORIZONS',
    'ACTIVE_HORIZONS',
    'LOOKBACK_HORIZONS',
    'HORIZON_TIMEFRAME_MINUTES',
    'HorizonConfig',
    'validate_horizons',
    'get_scaled_horizons',
    'auto_scale_purge_embargo',
    'get_default_barrier_params_for_horizon',
]
