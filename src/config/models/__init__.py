"""
Model configuration - re-exports from src.models.config.

This module provides unified access to model training configuration.
All config modules remain in their original locations; this is a facade.

Usage:
    from src.config.models import TrainerConfig, detect_environment
"""

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
# SERIALIZATION
# =============================================================================
# =============================================================================
# UTILS
# =============================================================================
from src.models.config import (
    ConfigError,
    ConfigValidationError,
    Environment,
    TrainerConfig,
    detect_environment,
    is_colab,
    resolve_device,
    save_config,
    save_config_json,
    validate_config,
)

__all__ = [
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
    # Merging
    # Serialization
    "save_config",
    "save_config_json",
]
