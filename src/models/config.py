"""
Model Configuration - YAML config loading and CLI arg merging.

This module handles:
- Loading model configurations from YAML files
- Merging CLI arguments with file configs
- Validating configuration values
- Providing default configurations
- Environment-specific overrides (Colab, local, etc.)

Configuration precedence (highest to lowest):
1. CLI arguments (--param value)
2. YAML config file
3. Environment-specific overrides
4. Model default config (from get_default_config)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION PATHS
# =============================================================================

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"
CONFIG_DIR = CONFIG_ROOT / "models"
TRAINING_CONFIG_PATH = CONFIG_ROOT / "training.yaml"
CV_CONFIG_PATH = CONFIG_ROOT / "cv.yaml"


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

class Environment(Enum):
    """Execution environment types."""
    COLAB = "colab"
    LOCAL_CPU = "local_cpu"
    LOCAL_GPU = "local_gpu"
    UNKNOWN = "unknown"


def detect_environment() -> Environment:
    """
    Detect the current execution environment.

    Returns:
        Environment enum indicating detected environment
    """
    # Check for Google Colab
    try:
        import google.colab  # noqa: F401
        return Environment.COLAB
    except ImportError:
        pass

    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            return Environment.LOCAL_GPU
    except ImportError:
        pass

    return Environment.LOCAL_CPU


def is_colab() -> bool:
    """Check if running in Google Colab."""
    return detect_environment() == Environment.COLAB


def resolve_device(device_setting: str = "auto") -> str:
    """
    Resolve device setting to actual device.

    Args:
        device_setting: Device setting ("auto", "cuda", "cpu")

    Returns:
        Resolved device string ("cuda" or "cpu")
    """
    if device_setting == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_setting


# =============================================================================
# TRAINER CONFIGURATION
# =============================================================================

@dataclass
class TrainerConfig:
    """
    Configuration for model training.

    Combines model hyperparameters with training settings like
    batch size, epochs, and early stopping.

    Attributes:
        model_name: Name of the model to train
        horizon: Label horizon (5, 10, 15, 20)
        feature_set: Feature set name for data loading
        sequence_length: Sequence length for sequential models
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        early_stopping_patience: Epochs without improvement before stopping
        random_seed: Random seed for reproducibility
        experiment_name: Name for experiment tracking
        output_dir: Directory for saving outputs
        model_config: Model-specific hyperparameters
        device: Device to use ('cuda', 'cpu', 'auto')
        mixed_precision: Whether to use mixed precision training
        num_workers: DataLoader workers for data loading
        pin_memory: Pin memory for GPU transfer
    """
    model_name: str
    horizon: int = 20
    feature_set: str = "boosting_optimal"
    sequence_length: int = 60
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 15
    random_seed: int = 42
    experiment_name: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))
    model_config: Dict[str, Any] = field(default_factory=dict)
    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self) -> None:
        """Validate and convert configuration values."""
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be non-negative, "
                f"got {self.early_stopping_patience}"
            )
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "feature_set": self.feature_set,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "random_seed": self.random_seed,
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "model_config": self.model_config,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainerConfig":
        """Create TrainerConfig from dictionary."""
        return cls(**data)

    def get_resolved_device(self) -> str:
        """Get the resolved device (auto -> cuda/cpu)."""
        return resolve_device(self.device)


# =============================================================================
# YAML CONFIG LOADING
# =============================================================================

def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        logger.warning(f"Empty config file: {path}")
        return {}

    logger.debug(f"Loaded config from {path}: {len(config)} keys")
    return config


def load_model_config(
    model_name: str,
    config_dir: Optional[Path] = None,
    flatten: bool = True,
) -> Dict[str, Any]:
    """
    Load model-specific configuration from YAML.

    Looks for config file at: {config_dir}/{model_name}.yaml
    Supports the new structured format with model/defaults/training/device sections.

    Args:
        model_name: Name of the model (e.g., "xgboost", "lstm")
        config_dir: Config directory (defaults to config/models/)
        flatten: If True, flatten the nested structure for backward compatibility

    Returns:
        Model configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_dir is None:
        config_dir = CONFIG_DIR

    config_path = config_dir / f"{model_name}.yaml"
    raw_config = load_yaml_config(config_path)

    if not flatten:
        return raw_config

    # Flatten the structured config for backward compatibility
    return flatten_model_config(raw_config)


def flatten_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten structured model config to flat dictionary.

    Converts:
        model:
          name: xgboost
          family: boosting
        defaults:
          n_estimators: 500
        training:
          batch_size: 256
        device:
          default: auto

    To:
        model_name: xgboost
        model_family: boosting
        n_estimators: 500
        batch_size: 256
        device: auto

    Args:
        config: Nested configuration dictionary

    Returns:
        Flattened configuration dictionary
    """
    result = {}

    # Extract model metadata
    if "model" in config:
        model_section = config["model"]
        if "name" in model_section:
            result["model_name"] = model_section["name"]
        if "family" in model_section:
            result["model_family"] = model_section["family"]
        if "description" in model_section:
            result["model_description"] = model_section["description"]

    # Extract defaults (hyperparameters)
    if "defaults" in config:
        result.update(config["defaults"])

    # Extract training settings
    if "training" in config:
        result.update(config["training"])

    # Extract device settings
    if "device" in config:
        device_section = config["device"]
        if "default" in device_section:
            result["device"] = device_section["default"]
        if "use_gpu" in device_section:
            result["use_gpu"] = device_section["use_gpu"]
        if "mixed_precision" in device_section:
            result["mixed_precision"] = device_section["mixed_precision"]

    # Include any top-level keys not in known sections
    known_sections = {"model", "defaults", "training", "device"}
    for key, value in config.items():
        if key not in known_sections:
            result[key] = value

    return result


def find_model_config(
    model_name: str,
    config_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Find model config file if it exists.

    Args:
        model_name: Name of the model
        config_dir: Config directory to search

    Returns:
        Path to config file if found, None otherwise
    """
    if config_dir is None:
        config_dir = CONFIG_DIR

    config_path = config_dir / f"{model_name}.yaml"
    return config_path if config_path.exists() else None


def load_training_config() -> Dict[str, Any]:
    """
    Load global training configuration.

    Returns:
        Training configuration dictionary
    """
    if TRAINING_CONFIG_PATH.exists():
        return load_yaml_config(TRAINING_CONFIG_PATH)
    logger.warning(f"Training config not found: {TRAINING_CONFIG_PATH}")
    return {}


def load_cv_config() -> Dict[str, Any]:
    """
    Load cross-validation configuration.

    Returns:
        CV configuration dictionary
    """
    if CV_CONFIG_PATH.exists():
        return load_yaml_config(CV_CONFIG_PATH)
    logger.warning(f"CV config not found: {CV_CONFIG_PATH}")
    return {}


def get_environment_overrides(
    training_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get environment-specific configuration overrides.

    Args:
        training_config: Training config with environments section

    Returns:
        Environment-specific overrides
    """
    if training_config is None:
        training_config = load_training_config()

    environments = training_config.get("environments", {})
    env = detect_environment()

    if env == Environment.COLAB:
        return environments.get("colab", {})
    elif env == Environment.LOCAL_GPU:
        return environments.get("local_gpu", {})
    elif env == Environment.LOCAL_CPU:
        return environments.get("local_cpu", {})

    return {}


# =============================================================================
# CONFIG MERGING
# =============================================================================

def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
    deep: bool = True,
) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Override values take precedence over base values.

    Args:
        base: Base configuration
        override: Override configuration (higher priority)
        deep: If True, merge nested dicts recursively

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if (
            deep
            and key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value, deep=True)
        else:
            result[key] = value

    return result


def build_config(
    model_name: str,
    cli_args: Optional[Dict[str, Any]] = None,
    config_file: Optional[Union[str, Path]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    apply_environment_overrides: bool = True,
) -> Dict[str, Any]:
    """
    Build complete model configuration from multiple sources.

    Configuration precedence (highest to lowest):
    1. CLI arguments
    2. Config file
    3. Environment-specific overrides
    4. Model-specific YAML (config/models/{model_name}.yaml)
    5. Provided defaults

    Args:
        model_name: Name of the model
        cli_args: Arguments from CLI (highest priority)
        config_file: Path to override config file
        defaults: Default configuration (lowest priority)
        apply_environment_overrides: Apply environment-specific settings

    Returns:
        Complete merged configuration
    """
    # Start with defaults
    config = defaults.copy() if defaults else {}

    # Try to load model-specific YAML from config/models/
    model_config_path = find_model_config(model_name)
    if model_config_path:
        try:
            model_yaml = load_model_config(model_name, flatten=True)
            config = merge_configs(config, model_yaml)
            logger.debug(f"Merged config from {model_config_path}")
        except Exception as e:
            logger.warning(f"Failed to load {model_config_path}: {e}")

    # Apply environment-specific overrides
    if apply_environment_overrides:
        env_overrides = get_environment_overrides()
        if env_overrides:
            # Flatten environment overrides
            flat_overrides = {}
            for section_name, section_data in env_overrides.items():
                if isinstance(section_data, dict):
                    flat_overrides.update(section_data)
                else:
                    flat_overrides[section_name] = section_data
            config = merge_configs(config, flat_overrides)
            logger.debug(f"Applied environment overrides: {detect_environment().value}")

    # Load explicit config file if provided
    if config_file:
        file_config = load_yaml_config(config_file)
        # Flatten if structured
        if any(k in file_config for k in ["model", "defaults", "training", "device"]):
            file_config = flatten_model_config(file_config)
        config = merge_configs(config, file_config)
        logger.debug(f"Merged config from {config_file}")

    # Apply CLI overrides (highest priority)
    if cli_args:
        # Filter out None values from CLI args
        cli_config = {k: v for k, v in cli_args.items() if v is not None}
        config = merge_configs(config, cli_config)
        logger.debug(f"Applied {len(cli_config)} CLI overrides")

    return config


def create_trainer_config(
    model_name: str,
    horizon: int = 20,
    cli_args: Optional[Dict[str, Any]] = None,
    config_file: Optional[Union[str, Path]] = None,
) -> TrainerConfig:
    """
    Create TrainerConfig from CLI args and config files.

    Convenience function that builds a complete TrainerConfig
    using the configuration loading and merging utilities.

    Args:
        model_name: Name of the model to train
        horizon: Label horizon
        cli_args: CLI arguments
        config_file: Optional config file path

    Returns:
        Configured TrainerConfig instance
    """
    # Build merged model config
    model_config = build_config(
        model_name=model_name,
        cli_args=cli_args,
        config_file=config_file,
    )

    # Extract trainer-level settings from model config
    trainer_kwargs = {
        "model_name": model_name,
        "horizon": horizon,
    }

    # Map config keys to TrainerConfig fields
    field_mapping = {
        "feature_set": "feature_set",
        "sequence_length": "sequence_length",
        "batch_size": "batch_size",
        "max_epochs": "max_epochs",
        "early_stopping_patience": "early_stopping_patience",
        "random_seed": "random_seed",
        "experiment_name": "experiment_name",
        "output_dir": "output_dir",
        "device": "device",
        "mixed_precision": "mixed_precision",
        "num_workers": "num_workers",
        "pin_memory": "pin_memory",
    }

    for config_key, trainer_key in field_mapping.items():
        if config_key in model_config:
            trainer_kwargs[trainer_key] = model_config.pop(config_key)

    # CLI args override
    if cli_args:
        for key in field_mapping.values():
            if key in cli_args and cli_args[key] is not None:
                trainer_kwargs[key] = cli_args[key]

    # Remaining config is model-specific hyperparameters
    trainer_kwargs["model_config"] = model_config

    return TrainerConfig(**trainer_kwargs)


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed: {errors}")


def validate_model_config_structure(config: Dict[str, Any]) -> List[str]:
    """
    Validate the structure of a model config file.

    Checks for required sections and valid values in the new structured format.

    Args:
        config: Raw (non-flattened) configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check for required sections
    if "model" not in config:
        errors.append("Missing required section: 'model'")
    else:
        model_section = config["model"]
        if "name" not in model_section:
            errors.append("Missing required field: model.name")
        if "family" not in model_section:
            errors.append("Missing required field: model.family")

        # Validate family
        valid_families = {"boosting", "neural", "transformer", "classical", "ensemble"}
        if "family" in model_section and model_section["family"] not in valid_families:
            errors.append(
                f"Invalid model.family: {model_section['family']}. "
                f"Must be one of: {valid_families}"
            )

    # Check device section
    if "device" in config:
        device_section = config["device"]
        if "default" in device_section:
            valid_devices = {"auto", "cuda", "cpu"}
            if device_section["default"] not in valid_devices:
                errors.append(
                    f"Invalid device.default: {device_section['default']}. "
                    f"Must be one of: {valid_devices}"
                )

    return errors


def validate_config(config: Dict[str, Any], model_name: str) -> List[str]:
    """
    Validate model configuration (flattened format).

    Args:
        config: Configuration to validate
        model_name: Model name for context

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate numeric ranges
    numeric_ranges = {
        "n_estimators": (1, 10000),
        "iterations": (1, 10000),
        "max_depth": (1, 100),
        "depth": (1, 16),
        "learning_rate": (0.0001, 1.0),
        "dropout": (0.0, 1.0),
        "hidden_size": (1, 4096),
        "d_model": (1, 4096),
        "num_layers": (1, 100),
        "n_layers": (1, 100),
        "batch_size": (1, 65536),
        "max_epochs": (1, 10000),
        "sequence_length": (1, 4096),
        "n_heads": (1, 64),
        "early_stopping_patience": (0, 1000),
        "early_stopping_rounds": (1, 1000),
    }

    for field_name, (min_val, max_val) in numeric_ranges.items():
        if field_name in config:
            value = config[field_name]
            if value is None:
                continue  # Allow None values
            if not isinstance(value, (int, float)):
                errors.append(f"{field_name} must be numeric, got {type(value).__name__}")
            elif value < min_val or value > max_val:
                errors.append(
                    f"{field_name}={value} out of range [{min_val}, {max_val}]"
                )

    # Validate device
    if "device" in config:
        valid_devices = {"auto", "cuda", "cpu"}
        if config["device"] not in valid_devices:
            errors.append(
                f"Invalid device: {config['device']}. Must be one of: {valid_devices}"
            )

    return errors


def validate_config_strict(
    config: Dict[str, Any],
    model_name: str,
    raise_on_error: bool = True,
) -> List[str]:
    """
    Strictly validate configuration and optionally raise on errors.

    Args:
        config: Configuration to validate
        model_name: Model name for context
        raise_on_error: If True, raise ConfigValidationError on validation failure

    Returns:
        List of validation error messages

    Raises:
        ConfigValidationError: If raise_on_error is True and validation fails
    """
    errors = validate_config(config, model_name)

    if errors and raise_on_error:
        raise ConfigValidationError(errors)

    return errors


# =============================================================================
# CONFIG SERIALIZATION
# =============================================================================

def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {path}")


def save_config_json(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Saved config to {path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_available_models() -> List[str]:
    """
    List all models with configuration files.

    Returns:
        List of model names
    """
    if not CONFIG_DIR.exists():
        return []

    models = []
    for config_file in CONFIG_DIR.glob("*.yaml"):
        models.append(config_file.stem)

    return sorted(models)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get basic info about a model from its config.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model info (name, family, description)
    """
    config_path = find_model_config(model_name)
    if not config_path:
        return {"name": model_name, "family": "unknown", "description": "No config found"}

    try:
        raw_config = load_yaml_config(config_path)
        model_section = raw_config.get("model", {})
        return {
            "name": model_section.get("name", model_name),
            "family": model_section.get("family", "unknown"),
            "description": model_section.get("description", ""),
        }
    except Exception as e:
        logger.warning(f"Failed to load model info for {model_name}: {e}")
        return {"name": model_name, "family": "unknown", "description": str(e)}


__all__ = [
    # Classes
    "TrainerConfig",
    "Environment",
    "ConfigValidationError",
    # Paths
    "CONFIG_ROOT",
    "CONFIG_DIR",
    "TRAINING_CONFIG_PATH",
    "CV_CONFIG_PATH",
    # Environment detection
    "detect_environment",
    "is_colab",
    "resolve_device",
    # Loading
    "load_yaml_config",
    "load_model_config",
    "find_model_config",
    "load_training_config",
    "load_cv_config",
    "flatten_model_config",
    "get_environment_overrides",
    # Building
    "merge_configs",
    "build_config",
    "create_trainer_config",
    # Validation
    "validate_model_config_structure",
    "validate_config",
    "validate_config_strict",
    # Serialization
    "save_config",
    "save_config_json",
    # Utilities
    "list_available_models",
    "get_model_info",
]
