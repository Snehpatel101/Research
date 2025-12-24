"""
Horizon configuration for triple-barrier labeling.

This module re-exports horizon configuration and utilities from the
dedicated horizon_config module. It provides:
- Configurable horizons for triple-barrier labeling
- Timeframe-aware horizon scaling
- Auto-scaling of purge and embargo bars
- Dynamic barrier parameter generation

Note: The actual implementation is in src/horizon_config.py.
This module serves as a bridge for the config package structure.
"""

# Import and re-export all horizon configuration from dedicated module
from src.common.horizon_config import (
    # Horizon lists
    SUPPORTED_HORIZONS,
    HORIZONS,
    LOOKBACK_HORIZONS,
    ACTIVE_HORIZONS,
    # Timeframe configuration
    HORIZON_TIMEFRAME_MINUTES,
    HORIZON_TIMEFRAME_SCALING,
    # Multipliers
    PURGE_MULTIPLIER,
    EMBARGO_MULTIPLIER,
    # Utility functions
    validate_horizons,
    get_scaled_horizons,
    auto_scale_purge_embargo,
    get_default_barrier_params_for_horizon,
    # Dataclass
    HorizonConfig,
)

__all__ = [
    # Horizon lists
    'SUPPORTED_HORIZONS',
    'HORIZONS',
    'LOOKBACK_HORIZONS',
    'ACTIVE_HORIZONS',
    # Timeframe configuration
    'HORIZON_TIMEFRAME_MINUTES',
    'HORIZON_TIMEFRAME_SCALING',
    # Multipliers
    'PURGE_MULTIPLIER',
    'EMBARGO_MULTIPLIER',
    # Utility functions
    'validate_horizons',
    'get_scaled_horizons',
    'auto_scale_purge_embargo',
    'get_default_barrier_params_for_horizon',
    # Dataclass
    'HorizonConfig',
]
