"""
Model configuration - re-exports from src.models.config.

This module provides unified access to model training configuration.
All config modules remain in their original locations; this is a facade.

Usage:
    from src.config.models import TrainerConfig, detect_environment
    from src.config.models import load_model_config, build_config
"""

# =============================================================================
# PATHS
# =============================================================================
from src.models.config import (
    CONFIG_DIR,
    CONFIG_ROOT,
    CV_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
)

# =============================================================================
# EXCEPTIONS
# =============================================================================
from src.models.config import (
    ConfigError,
    ConfigValidationError,
)

# =============================================================================
# ENVIRONMENT
# =============================================================================
from src.models.config import (
    Environment,
    detect_environment,
    is_colab,
    resolve_device,
)

# =============================================================================
# TRAINER CONFIG
# =============================================================================
from src.models.config import TrainerConfig

# =============================================================================
# VALIDATION
# =============================================================================
from src.models.config import (
    validate_config,
    validate_config_strict,
    validate_model_config_structure,
)

# =============================================================================
# LOADERS
# =============================================================================
from src.models.config import (
    find_model_config,
    flatten_model_config,
    get_environment_overrides,
    load_cv_config,
    load_model_config,
    load_training_config,
    load_yaml_config,
)

# =============================================================================
# MERGING
# =============================================================================
from src.models.config import (
    AppliedOverrides,
    ConfigBuildResult,
    build_config,
    create_trainer_config,
    get_applied_overrides,
    merge_configs,
)

# =============================================================================
# SERIALIZATION
# =============================================================================
from src.models.config import (
    save_config,
    save_config_json,
)

# =============================================================================
# UTILS
# =============================================================================
from src.models.config import (
    get_model_info,
    list_available_models,
)

__all__ = [
    # Paths
    "CONFIG_ROOT",
    "CONFIG_DIR",
    "TRAINING_CONFIG_PATH",
    "CV_CONFIG_PATH",
    # Exceptions
    "ConfigError",
    "ConfigValidationError",
    # Environment
    "Environment",
    "detect_environment",
    "is_colab",
    "resolve_device",
    # TrainerConfig
    "TrainerConfig",
    # Validation
    "validate_model_config_structure",
    "validate_config",
    "validate_config_strict",
    # Loaders
    "load_yaml_config",
    "load_model_config",
    "flatten_model_config",
    "find_model_config",
    "load_training_config",
    "load_cv_config",
    "get_environment_overrides",
    # Merging
    "merge_configs",
    "build_config",
    "create_trainer_config",
    "get_applied_overrides",
    "AppliedOverrides",
    "ConfigBuildResult",
    # Serialization
    "save_config",
    "save_config_json",
    # Utils
    "list_available_models",
    "get_model_info",
]
