"""
DEPRECATED: Import from 'src.core.common' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.common import HorizonConfig, CANONICAL_TIMEFRAMES

    # New (preferred)
    from src.core.common import HorizonConfig, CANONICAL_TIMEFRAMES
"""

import warnings

_COMMON_EXPORTS = [
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
    # split_ratios
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


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _COMMON_EXPORTS:
        warnings.warn(
            f"Importing '{name}' from 'src.common' is deprecated. "
            f"Use 'from src.core.common import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.core import common as common_module

        return getattr(common_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _COMMON_EXPORTS
