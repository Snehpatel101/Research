"""
Centralized configuration package - Single Source of Truth.

This package provides unified access to configuration across the codebase.
The primary interface is UnifiedConfig, which consolidates all configuration
from MLConfig, TrainerConfig, PipelineConfig, and HorizonConfig.

Primary Interface (Recommended):
--------------------------------
    from src.config import UnifiedConfig, get_unified_config

    # Load the global unified config
    config = get_unified_config()

    # Access values through sections
    batch_size = config.training.batch_size
    horizons = config.horizons.active
    train_ratio = config.splits.train

    # Convert to legacy configs when needed
    trainer_config = config.to_trainer_config(model_name="xgboost")
    pipeline_config = config.to_pipeline_config()

Configuration Access Utility:
-----------------------------
    from src.config import get_config_value

    # Replace all _get_global_or_default() calls with this:
    batch_size = get_config_value("training.batch_size", 256)
    horizons = get_config_value("horizons.active", [5, 10, 15, 20])

Validation:
-----------
    from src.config import validate_config, validate_config_file

    # Validate a config dictionary
    result = validate_config(config_dict)
    if not result.is_valid:
        for error in result.errors:
            print(error)

    # Validate a YAML file
    result = validate_config_file("config/global.yaml")

Backward Compatible Imports:
----------------------------
Original imports continue to work (with deprecation warnings for some):

    from src.config import CANONICAL_TIMEFRAMES, DEFAULT_SPLIT_RATIOS
    from src.config import TrainerConfig, detect_environment
    from src.config import MODEL_DATA_REQUIREMENTS, ModelFamily

Subpackage imports:
    from src.config.constants import is_valid_timeframe, HorizonConfig
    from src.config.models import load_model_config, build_config
    from src.config.pipeline import BARRIER_PARAMS, get_barrier_params

Package Structure:
------------------
    src/config/
        __init__.py         <- This file (central facade)
        unified.py          <- UnifiedConfig (primary interface)
        utils.py            <- get_config_value (single implementation)
        validators.py       <- Schema validation
        global_config.py    <- GlobalConfig (YAML loader)
        constants/          <- Re-exports from src.common
        models/             <- Re-exports from src.models.config
        pipeline/           <- Re-exports from src.pipeline.config

Deprecation Notes:
------------------
The following patterns are deprecated but still work:

1. Direct _get_global_or_default() implementations:
   - Use get_config_value() from src.config.utils instead

2. GlobalConfig.from_yaml() for config loading:
   - Use UnifiedConfig.from_yaml() for the new interface

3. Direct TrainerConfig/PipelineConfig construction:
   - Consider using UnifiedConfig.to_trainer_config() instead
"""

from __future__ import annotations

import warnings

# =============================================================================
# PRIMARY INTERFACE: UNIFIED CONFIGURATION
# =============================================================================
from src.config.unified import (
    # Main class
    UnifiedConfig,
    # Section classes
    TimeframesSection,
    SplitsSection,
    PurgeEmbargoSection,
    HorizonsSection,
    FeaturesSection,
    FeatureSelectionSection,
    FeatureGenerationSection,
    MTFSection,
    TrainingSection,
    CalibrationSection,
    OptimizationSection,
    GASection,
    OptunaSection,
    CrossValidationSection,
    ProcessingSection,
    ScalerSection,
    TrackingSection,
    OOMRecoverySection,
    # Singleton functions
    get_unified_config,
    set_unified_config,
    reset_unified_config,
)

# =============================================================================
# CONFIGURATION ACCESS UTILITIES
# =============================================================================
from src.config.utils import (
    # Primary function - USE THIS instead of _get_global_or_default()
    get_config_value,
    get_config_value_strict,
    clear_config_cache,
    validate_config_path,
    list_config_paths,
    # Logging/debugging
    ConfigAccessLog,
    ConfigAccessEntry,
    ConfigSource,
    ConfigValueError,
)

# =============================================================================
# VALIDATION
# =============================================================================
from src.config.validators import (
    # Main functions
    validate_config,
    validate_config_file,
    coerce_types,
    # Result types
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ConfigValidationError,
)

# =============================================================================
# GLOBAL CONFIGURATION (Legacy - prefer UnifiedConfig)
# =============================================================================
from src.config.global_config import (
    GlobalConfig,
    load_global_config,
    get_global_config,
    set_global_config,
)

# =============================================================================
# ML FOR DUMMIES: SIMPLE API
# =============================================================================
from src.config.smart_config import (
    # Core API
    train,
    SmartConfig,
    ResolvedModelConfig,
    # Configuration data
    MODEL_DEFAULTS,
    FEATURE_SETS,
    OPTIMIZATION_DEFAULTS,
    # Resolution
    resolve_model_config,
    resolve_config,
    # Helpers
    list_models,
    describe_model,
    show_defaults,
    preview_config,
    quick_compare,
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


# =============================================================================
# DEPRECATED: _get_global_or_default
# =============================================================================
def _get_global_or_default(attr_path: str, fallback):
    """
    DEPRECATED: Use get_config_value() instead.

    This function exists only for backward compatibility.
    """
    warnings.warn(
        "_get_global_or_default is deprecated. "
        "Use get_config_value() from src.config instead:\n"
        "  from src.config import get_config_value\n"
        f"  value = get_config_value('{attr_path}', {fallback!r})",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_config_value(attr_path, fallback)


__all__ = [
    # ==========================================================================
    # PRIMARY INTERFACE (use these)
    # ==========================================================================
    # Unified config
    "UnifiedConfig",
    "get_unified_config",
    "set_unified_config",
    "reset_unified_config",
    # Section classes
    "TimeframesSection",
    "SplitsSection",
    "PurgeEmbargoSection",
    "HorizonsSection",
    "FeaturesSection",
    "FeatureSelectionSection",
    "FeatureGenerationSection",
    "MTFSection",
    "TrainingSection",
    "CalibrationSection",
    "OptimizationSection",
    "GASection",
    "OptunaSection",
    "CrossValidationSection",
    "ProcessingSection",
    "ScalerSection",
    "TrackingSection",
    "OOMRecoverySection",
    # Config access utility
    "get_config_value",
    "get_config_value_strict",
    "clear_config_cache",
    "validate_config_path",
    "list_config_paths",
    "ConfigAccessLog",
    "ConfigAccessEntry",
    "ConfigSource",
    "ConfigValueError",
    # Validation
    "validate_config",
    "validate_config_file",
    "coerce_types",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "ConfigValidationError",
    # ==========================================================================
    # BACKWARD COMPATIBLE (still supported)
    # ==========================================================================
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
    # ==========================================================================
    # ML FOR DUMMIES: SIMPLE API
    # ==========================================================================
    "train",
    "SmartConfig",
    "ResolvedModelConfig",
    "MODEL_DEFAULTS",
    "FEATURE_SETS",
    "OPTIMIZATION_DEFAULTS",
    "resolve_model_config",
    "resolve_config",
    "list_models",
    "describe_model",
    "show_defaults",
    "preview_config",
    "quick_compare",
    # ==========================================================================
    # DEPRECATED (for migration)
    # ==========================================================================
    "_get_global_or_default",
]
