"""
Centralized configuration package.

Only configuration that the pipeline actually reads lives here — a class or
field that can be set but never reaches MLFactory is a bug waiting to happen
(Phase 116 removed every such "aspirational" config class and field).

    # Top-level experiment config (used by MLFactory)
    from src.config import ExperimentConfig

    # Its section building blocks
    from src.config import (
        FeatureConfig, LabelingConfig, SequenceConfig, MTFConfig, SplitConfig,
        OptunaConfig, CalibrationConfig, WalkForwardConfig,
    )

    # Contract specs per symbol
    from src.config import SymbolConfig

    # Process-wide defaults from config/global.yaml
    from src.config import get_config_value
    batch_size = get_config_value("training.batch_size", 512)

Package Structure:
------------------
    src/config/
        __init__.py         <- This file (central facade)
        base.py             <- BaseConfig (foundation for the config classes)
        experiment.py       <- ExperimentConfig + its sections
        data.py             <- Feature / Labeling / Sequence / MTF / Split configs
        training.py         <- Optuna / Calibration configs
        cv.py               <- WalkForwardConfig
        symbol.py           <- SymbolConfig (contract specs)
        global_config.py    <- GlobalConfig (config/global.yaml loader)
        utils.py            <- get_config_value (single implementation)
        validators.py       <- global.yaml schema validation
        constants/          <- Re-exports from src.core.common
        models/             <- Re-exports from src.models.config
        pipeline/           <- Re-exports from src.data.pipeline.config

Operational configs (PurgedKFoldConfig, the Backtester's BacktestConfig,
ProbabilityCalibrator's CalibrationConfig, ...) live next to their
implementations.
"""

from __future__ import annotations

# =============================================================================
# BASE CONFIG
# =============================================================================
from src.config.base import BaseConfig

# =============================================================================
# COMMONLY USED CONSTANTS (from src.config.constants)
# =============================================================================
from src.config.constants import (
    # Horizons
    ACTIVE_HORIZONS,
    # Timeframes
    CANONICAL_TIMEFRAMES,
    # Split ratios
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    HORIZONS,
    SUPPORTED_HORIZONS,
    SUPPORTED_TIMEFRAMES,
    HorizonConfig,
    auto_scale_purge_embargo,
    get_timeframe_minutes,
    is_valid_timeframe,
    normalize_timeframe,
    validate_horizons,
    validate_split_ratios,
)

# =============================================================================
# EXPERIMENT SECTION CONFIGS
# =============================================================================
from src.config.cv import WalkForwardConfig, WindowType
from src.config.data import (
    FeatureCategory,
    FeatureConfig,
    LabelingConfig,
    MTFConfig,
    MTFMode,
    ScalerType,
    SequenceConfig,
    SplitConfig,
)

# Top-level ExperimentConfig (used by MLFactory) — canonical location
from src.config.experiment import ExperimentConfig

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
from src.config.global_config import (
    GlobalConfig,
    get_global_config,
    load_global_config,
)

# =============================================================================
# COMMONLY USED MODEL CONFIG (from src.config.models)
# =============================================================================
from src.config.models import (
    TrainerConfig,
    build_config,
    detect_environment,
    is_colab,
    load_model_config,
    save_config_json,
)

# =============================================================================
# COMMONLY USED PIPELINE CONFIG (from src.config.pipeline)
# =============================================================================
from src.config.pipeline import (
    BARRIER_PARAMS,
    MODEL_DATA_REQUIREMENTS,
    ModelDataRequirements,
    get_barrier_params,
)

# =============================================================================
# SYMBOL CONFIGURATION
# =============================================================================
from src.config.symbol import SymbolConfig
from src.config.training import CalibrationConfig, OptunaConfig

# =============================================================================
# CONFIGURATION ACCESS UTILITIES
# =============================================================================
from src.config.utils import (
    ConfigAccessEntry,
    # Logging/debugging
    ConfigAccessLog,
    ConfigSource,
    ConfigValueError,
    clear_config_cache,
    get_config_value,
    get_config_value_strict,
    list_config_paths,
    validate_config_path,
)

# =============================================================================
# VALIDATION
# =============================================================================
from src.config.validators import (
    ConfigValidationError,
    ValidationIssue,
    # Result types
    ValidationResult,
    ValidationSeverity,
    coerce_types,
    # Main functions
    validate_config,
    validate_config_file,
)

__all__ = [
    # Base
    "BaseConfig",
    # Experiment config + sections
    "ExperimentConfig",
    "FeatureConfig",
    "LabelingConfig",
    "SequenceConfig",
    "MTFConfig",
    "SplitConfig",
    "OptunaConfig",
    "CalibrationConfig",
    "WalkForwardConfig",
    "WindowType",
    # Shared enums
    "ScalerType",
    "FeatureCategory",
    "MTFMode",
    # Symbol configuration
    "SymbolConfig",
    # Trainer config (src.models.config)
    "TrainerConfig",
    # Config access utilities
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
    # Global config
    "GlobalConfig",
    "load_global_config",
    "get_global_config",
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
    "detect_environment",
    "is_colab",
    "load_model_config",
    "build_config",
    "save_config_json",
    # Pipeline config
    "MODEL_DATA_REQUIREMENTS",
    "ModelDataRequirements",
    "BARRIER_PARAMS",
    "get_barrier_params",
]
