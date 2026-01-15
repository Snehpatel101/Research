"""
Model Configuration - YAML config loading and CLI arg merging.

Precedence: CLI args > YAML file > Environment overrides > Model defaults

This package also contains the canonical MODEL_DATA_REQUIREMENTS for model
data preparation. Import from here:
    from src.models.config import MODEL_DATA_REQUIREMENTS, ModelFamily
"""

from .data_requirements import (
    ENSEMBLE_CONFIGS,
    MODEL_DATA_REQUIREMENTS,
    EnsembleConfig,
    ModelDataRequirements,
    ModelFamily,
    ScalerType,
    get_all_ensemble_names,
    get_all_model_names,
    get_combined_requirements,
    get_ensemble_config,
    get_model_requirements,
    get_models_by_family,
    validate_model_config,
)
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
from .merging import (
    AppliedOverrides,
    ConfigBuildResult,
    build_config,
    create_trainer_config,
    get_applied_overrides,
    merge_configs,
)
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
    # Data Requirements (canonical location)
    "ModelFamily",
    "ScalerType",
    "ModelDataRequirements",
    "EnsembleConfig",
    "MODEL_DATA_REQUIREMENTS",
    "ENSEMBLE_CONFIGS",
    "get_model_requirements",
    "get_ensemble_config",
    "get_models_by_family",
    "get_combined_requirements",
    "validate_model_config",
    "get_all_model_names",
    "get_all_ensemble_names",
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
