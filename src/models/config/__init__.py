"""
Model Configuration - TrainerConfig and the canonical MODEL_DATA_REQUIREMENTS for
model data preparation. Import from here:
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
from .serialization import save_config, save_config_json
from .trainer_config import TrainerConfig
from .validation import validate_config

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
    # Serialization
    "save_config",
    "save_config_json",
]
