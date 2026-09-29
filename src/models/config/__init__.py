"""
Model Configuration - YAML config loading and CLI arg merging.

Precedence: CLI args > YAML file > Environment overrides > Model defaults

This package also contains the canonical MODEL_DATA_REQUIREMENTS for model
data preparation. Import from here:
    from src.models.config import MODEL_DATA_REQUIREMENTS, ModelFamily
"""

from src.core.types import ModelFamily

from .data_requirements import (
    ENSEMBLE_CONFIGS,
    MODEL_DATA_REQUIREMENTS,
    EnsembleConfig,
    ModelDataRequirements,
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
    load_model_config,
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
from .paths import CONFIG_DIR, CONFIG_ROOT
from .serialization import save_config, save_config_json
from .trainer_config import TrainerConfig
from .validation import validate_config

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
