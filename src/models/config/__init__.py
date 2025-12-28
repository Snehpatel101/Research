"""
Model Configuration - YAML config loading and CLI arg merging.

Precedence: CLI args > YAML file > Environment overrides > Model defaults
"""
from .paths import CONFIG_ROOT, CONFIG_DIR, TRAINING_CONFIG_PATH, CV_CONFIG_PATH
from .exceptions import ConfigError, ConfigValidationError
from .environment import Environment, detect_environment, is_colab, resolve_device
from .trainer_config import TrainerConfig
from .validation import (
    validate_model_config_structure,
    validate_config,
    validate_config_strict,
)
from .loaders import (
    load_yaml_config,
    load_model_config,
    flatten_model_config,
    find_model_config,
    load_training_config,
    load_cv_config,
    get_environment_overrides,
)
from .merging import merge_configs, build_config, create_trainer_config
from .serialization import save_config, save_config_json
from .utils import list_available_models, get_model_info

__all__ = [
    # Paths
    "CONFIG_ROOT", "CONFIG_DIR", "TRAINING_CONFIG_PATH", "CV_CONFIG_PATH",
    # Exceptions
    "ConfigError", "ConfigValidationError",
    # Environment
    "Environment", "detect_environment", "is_colab", "resolve_device",
    # TrainerConfig
    "TrainerConfig",
    # Validation
    "validate_model_config_structure", "validate_config", "validate_config_strict",
    # Loaders
    "load_yaml_config", "load_model_config", "flatten_model_config",
    "find_model_config", "load_training_config", "load_cv_config",
    "get_environment_overrides",
    # Merging
    "merge_configs", "build_config", "create_trainer_config",
    # Serialization
    "save_config", "save_config_json",
    # Utils
    "list_available_models", "get_model_info",
]
