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
# =============================================================================
# EXCEPTIONS
# =============================================================================
# =============================================================================
# ENVIRONMENT
# =============================================================================
# =============================================================================
# TRAINER CONFIG
# =============================================================================
# =============================================================================
# VALIDATION
# =============================================================================
# =============================================================================
# LOADERS
# =============================================================================
# =============================================================================
# MERGING
# =============================================================================
# =============================================================================
# SERIALIZATION
# =============================================================================
# =============================================================================
# UTILS
# =============================================================================
from src.models.config import (
    CONFIG_DIR,
    CONFIG_ROOT,
    AppliedOverrides,
    ConfigBuildResult,
    ConfigError,
    ConfigValidationError,
    Environment,
    TrainerConfig,
    build_config,
    create_trainer_config,
    detect_environment,
    find_model_config,
    flatten_model_config,
    get_applied_overrides,
    is_colab,
    load_model_config,
    load_yaml_config,
    merge_configs,
    resolve_device,
    save_config,
    save_config_json,
    validate_config,
)

__all__ = [
    # Paths
    "CONFIG_ROOT",
    "CONFIG_DIR",
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
    "validate_config",
    # Loaders
    "load_yaml_config",
    "load_model_config",
    "flatten_model_config",
    "find_model_config",
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
]
