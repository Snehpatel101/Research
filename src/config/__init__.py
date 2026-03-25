"""
Centralized configuration package - Single Source of Truth.

This package provides unified access to configuration across the codebase.

CONSOLIDATED CONFIG ARCHITECTURE (~15 config classes instead of 55+)
====================================================================

Import configs from their canonical locations:

    # Base class for all configs
    from src.config import BaseConfig

    # Data-related configs
    from src.config import (
        FeatureConfig, LabelingConfig, ScalerConfig,
        SequenceConfig, MTFConfig, SplitConfig
    )

    # Training-related configs
    from src.config import (
        TrainerConfig, OptunaConfig, CalibrationConfig,
        CheckpointConfig, OOMConfig, GAConfig
    )

    # Cross-validation configs
    from src.config import (
        CVConfig, CPCVConfig, WalkForwardConfig,
        PurgedKFoldConfig, PBOConfig
    )

    # Model-specific configs
    from src.config import (
        XGBoostConfig, LightGBMConfig, LSTMConfig,
        TransformerConfig, PerModelConfig
    )

    # Ensemble configs
    from src.config import (
        EnsembleConfig, MetaLearnerConfig,
        StackingConfig, VotingConfig
    )

    # Inference configs
    from src.config import (
        InferenceConfig, BacktestConfig,
        BundleConfig, ServerConfig
    )

Configuration Access Utility:
-----------------------------
    from src.config import get_config_value

    # Replace all _get_global_or_default() calls with this:
    batch_size = get_config_value("training.batch_size", 256)
    horizons = get_config_value("horizons.active", [5, 10, 15, 20])

Package Structure:
------------------
    src/config/
        __init__.py         <- This file (central facade)
        base.py             <- BaseConfig (foundation for all configs)
        data.py             <- Data configs (Feature, Labeling, Scaler, etc.)
        training.py         <- Training configs (Trainer, Optuna, etc.)
        cv.py               <- CV configs (CPCV, WalkForward, etc.)
        model_configs.py    <- Model configs (XGBoost, LSTM, etc.)
        ensemble.py         <- Ensemble configs (Stacking, Voting, etc.)
        inference.py        <- Inference configs (Backtest, Server, etc.)
        utils.py            <- get_config_value (single implementation)
        validators.py       <- Schema validation
        global_config.py    <- GlobalConfig (YAML loader)
        constants/          <- Re-exports from src.core.common
        models/             <- Re-exports from src.models.config
        pipeline/           <- Re-exports from src.data.pipeline.config
"""

from __future__ import annotations

import warnings

# =============================================================================
# BASE CONFIG (foundation for all config classes)
# =============================================================================
from src.config.base import (
    BaseConfig,
    TimestampedConfigMixin,
    VersionedConfigMixin,
)

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
# CROSS-VALIDATION CONFIGS (consolidated from multiple locations)
# =============================================================================
from src.config.cv import (
    # Configs
    CPCVConfig,
    CVConfig,
    # Enums
    CVMethod,
    DSRConfig,
    PBOConfig,
    PurgedKFoldConfig,
    PurgeEmbargoConfig,
    WalkForwardConfig,
    WindowType,
)

# =============================================================================
# DATA CONFIGS (consolidated from multiple locations)
# =============================================================================
from src.config.data import (
    # Configs
    BarConfig,
    # Enums
    FeatureCategory,
    FeatureConfig,
    LabelingConfig,
    LabelingMethod,
    MTFConfig,
    MTFMode,
    MultiResolutionConfig,
    ScalerConfig,
    ScalerType,
    SequenceConfig,
    SessionConfig,
    SessionsConfig,
    SplitConfig,
)

# =============================================================================
# ENSEMBLE CONFIGS (consolidated from multiple locations)
# =============================================================================
from src.config.ensemble import (
    # Predefined
    PREDEFINED_ENSEMBLES,
    # Configs
    BlendingConfig,
    EnsembleConfig,
    # Enums
    EnsembleMethod,
    MetaLearnerConfig,
    MetaLearnerType,
    OOFAlignmentConfig,
    StackingConfig,
    VotingConfig,
    VotingType,
    get_predefined_ensemble,
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
    set_global_config,
)

# =============================================================================
# INFERENCE CONFIGS (consolidated from multiple locations)
# =============================================================================
from src.config.inference import (
    # Configs
    AlertConfig,
    BacktestConfig,
    BundleConfig,
    InferenceConfig,
    # Enums
    InferenceMode,
    PositionSizerConfig,
    PositionSizingMethod,
    PreprocessingGraphConfig,
    ServerConfig,
)

# =============================================================================
# MODEL CONFIGS (consolidated from multiple locations)
# =============================================================================
from src.config.model_configs import (
    # Enums
    ActivationType,
    # Boosting
    CatBoostConfig,
    GRUConfig,
    LightGBMConfig,
    # Neural
    LSTMConfig,
    # Base
    ModelConfig,
    ModelFamily,
    # Transformer
    PatchTSTConfig,
    # Per-model
    PerModelConfig,
    TCNConfig,
    TransformerConfig,
    XGBoostConfig,
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

# =============================================================================
# TRAINING CONFIGS (consolidated from multiple locations)
# =============================================================================
from src.config.training import (
    # Optimization
    CalibrationConfig,
    CheckpointConfig,
    ConformalConfig,
    ExperimentTrackingConfig,
    GAConfig,
    OOMConfig,
    OptunaConfig,
    ParallelTrainingConfig,
)

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
    # Primary function - USE THIS instead of _get_global_or_default()
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
    # BASE CONFIG (foundation)
    # ==========================================================================
    "BaseConfig",
    "TimestampedConfigMixin",
    "VersionedConfigMixin",
    # ==========================================================================
    # DATA CONFIGS (consolidated)
    # ==========================================================================
    # Enums
    "ScalerType",
    "FeatureCategory",
    "LabelingMethod",
    "MTFMode",
    # Configs
    "FeatureConfig",
    "LabelingConfig",
    "ScalerConfig",
    "SequenceConfig",
    "MTFConfig",
    "MultiResolutionConfig",
    "SessionConfig",
    "SessionsConfig",
    "SplitConfig",
    "BarConfig",
    # ==========================================================================
    # TRAINING CONFIGS (consolidated)
    # ==========================================================================
    "TrainerConfig",
    "OptunaConfig",
    "GAConfig",
    "CalibrationConfig",
    "ConformalConfig",
    "CheckpointConfig",
    "OOMConfig",
    "ParallelTrainingConfig",
    "ExperimentTrackingConfig",
    "ExperimentConfig",
    # ==========================================================================
    # CROSS-VALIDATION CONFIGS (consolidated)
    # ==========================================================================
    # Enums
    "CVMethod",
    "WindowType",
    # Configs
    "CVConfig",
    "PurgeEmbargoConfig",
    "PurgedKFoldConfig",
    "CPCVConfig",
    "WalkForwardConfig",
    "PBOConfig",
    "DSRConfig",
    # ==========================================================================
    # MODEL CONFIGS (consolidated)
    # ==========================================================================
    # Enums
    "ModelFamily",
    "ActivationType",
    # Configs
    "ModelConfig",
    "XGBoostConfig",
    "LightGBMConfig",
    "CatBoostConfig",
    "LSTMConfig",
    "GRUConfig",
    "TCNConfig",
    "TransformerConfig",
    "PatchTSTConfig",
    "PerModelConfig",
    # ==========================================================================
    # ENSEMBLE CONFIGS (consolidated)
    # ==========================================================================
    # Enums
    "EnsembleMethod",
    "VotingType",
    "MetaLearnerType",
    # Configs
    "EnsembleConfig",
    "MetaLearnerConfig",
    "StackingConfig",
    "VotingConfig",
    "BlendingConfig",
    "OOFAlignmentConfig",
    # Predefined
    "PREDEFINED_ENSEMBLES",
    "get_predefined_ensemble",
    # ==========================================================================
    # INFERENCE CONFIGS (consolidated)
    # ==========================================================================
    # Enums
    "InferenceMode",
    "PositionSizingMethod",
    # Configs
    "InferenceConfig",
    "BundleConfig",
    "ServerConfig",
    "BacktestConfig",
    "PositionSizerConfig",
    "PreprocessingGraphConfig",
    "AlertConfig",
    # ==========================================================================
    # SYMBOL CONFIGURATION
    # ==========================================================================
    "SymbolConfig",
    # ==========================================================================
    # CONFIG ACCESS UTILITIES
    # ==========================================================================
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
    # ==========================================================================
    # DEPRECATED (for migration)
    # ==========================================================================
    "_get_global_or_default",
]
