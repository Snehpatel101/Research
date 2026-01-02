"""
Model Configuration - YAML config loading and CLI arg merging.

Precedence: CLI args > YAML file > Environment overrides > Model defaults
"""

from .environment import Environment, detect_environment, is_colab, resolve_device
from .exceptions import ConfigError, ConfigValidationError
from .loaders import (
    find_model_config,
    flatten_model_config,
    get_environment_overrides,
    load_cv_config,
    load_model_config,
    load_training_config,
    load_yaml_config,
)
from .merging import build_config, create_trainer_config, merge_configs
from .paths import CONFIG_DIR, CONFIG_ROOT, CV_CONFIG_PATH, TRAINING_CONFIG_PATH
from .serialization import save_config, save_config_json
from .trainer_config import TrainerConfig
from .utils import get_model_info, list_available_models
from .validation import (
    validate_config,
    validate_config_strict,
    validate_model_config_structure,
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
    # Serialization
    "save_config",
    "save_config_json",
    # Utils
    "list_available_models",
    "get_model_info",
]
