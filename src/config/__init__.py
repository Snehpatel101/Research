"""
Centralized configuration package.

This package provides unified access to configuration across the codebase
using the facade pattern. Original files remain in place; this package
re-exports from existing locations for convenience.

Import Patterns:
----------------

1. Top-level imports (most common items):
    from src.config import CANONICAL_TIMEFRAMES, DEFAULT_SPLIT_RATIOS
    from src.config import TrainerConfig, detect_environment
    from src.config import MODEL_DATA_REQUIREMENTS, ModelFamily

2. Subpackage imports (organized by domain):
    from src.config.constants import is_valid_timeframe, HorizonConfig
    from src.config.models import load_model_config, build_config
    from src.config.pipeline import BARRIER_PARAMS, get_barrier_params

Structure:
----------
    src/config/
        __init__.py         <- This file (central facade)
        constants/          <- Re-exports from src.common
            __init__.py     <- timeframes, split_ratios, horizon_config
        models/             <- Re-exports from src.models.config
            __init__.py     <- TrainerConfig, loaders, environment
        pipeline/           <- Re-exports from src.phase1.config
            __init__.py     <- barriers, labeling, model_config, runtime

Backward Compatibility:
-----------------------
Original imports continue to work:
    from src.common.timeframes import CANONICAL_TIMEFRAMES  # Still works
    from src.models.config import TrainerConfig              # Still works
    from src.phase1.config import MODEL_DATA_REQUIREMENTS    # Still works
"""

# =============================================================================
# GLOBAL CONFIGURATION (NEW - single source of truth)
# =============================================================================
from src.config.global_config import (
    GlobalConfig,
    load_global_config,
    get_global_config,
    set_global_config,
)

# =============================================================================
# COMMONLY USED CONSTANTS (from src.config.constants)
# =============================================================================
from src.config.constants import (
    # Timeframes
    CANONICAL_TIMEFRAMES,
    SUPPORTED_TIMEFRAMES,
    get_timeframe_minutes,
    is_valid_timeframe,
    normalize_timeframe,
    # Split ratios
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    validate_split_ratios,
    # Horizons
    ACTIVE_HORIZONS,
    HORIZONS,
    SUPPORTED_HORIZONS,
    HorizonConfig,
    auto_scale_purge_embargo,
    validate_horizons,
)

# =============================================================================
# COMMONLY USED MODEL CONFIG (from src.config.models)
# =============================================================================
from src.config.models import (
    TrainerConfig,
    detect_environment,
    is_colab,
    load_model_config,
    build_config,
    save_config_json,
)

# =============================================================================
# COMMONLY USED PIPELINE CONFIG (from src.config.pipeline)
# =============================================================================
from src.config.pipeline import (
    MODEL_DATA_REQUIREMENTS,
    ModelFamily,
    ModelDataRequirements,
    EnsembleConfig,
    BARRIER_PARAMS,
    get_barrier_params,
)

__all__ = [
    # Global config
    "GlobalConfig",
    "load_global_config",
    "get_global_config",
    "set_global_config",
    # Timeframes
    "CANONICAL_TIMEFRAMES",
    "SUPPORTED_TIMEFRAMES",
    "get_timeframe_minutes",
    "is_valid_timeframe",
    "normalize_timeframe",
    # Split ratios
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "validate_split_ratios",
    # Horizons
    "HORIZONS",
    "SUPPORTED_HORIZONS",
    "ACTIVE_HORIZONS",
    "HorizonConfig",
    "validate_horizons",
    "auto_scale_purge_embargo",
    # Model config
    "TrainerConfig",
    "detect_environment",
    "is_colab",
    "load_model_config",
    "build_config",
    "save_config_json",
    # Pipeline config
    "MODEL_DATA_REQUIREMENTS",
    "ModelFamily",
    "ModelDataRequirements",
    "EnsembleConfig",
    "BARRIER_PARAMS",
    "get_barrier_params",
]
