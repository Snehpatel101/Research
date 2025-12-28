"""
Model Configuration - YAML config loading and CLI arg merging.

Precedence: CLI args > YAML file > Environment overrides > Model defaults
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
    """Detect execution environment (Colab/GPU/CPU)."""
    try:
        import google.colab  # noqa: F401
        return Environment.COLAB
    except ImportError:
        pass
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
    """Resolve device setting to actual device ("cuda"/"cpu")."""
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
    """Configuration for model training (hyperparameters + training settings)."""
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
    # Calibration settings
    use_calibration: bool = True
    calibration_method: str = "auto"  # "auto", "isotonic", "sigmoid"
    # Test set evaluation (default True, but marked as one-shot)
    evaluate_test_set: bool = True

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
            "use_calibration": self.use_calibration,
            "calibration_method": self.calibration_method,
            "evaluate_test_set": self.evaluate_test_set,
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

def load_yaml_config(
    path: Union[str, Path],
    explicit: bool = False,
) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to YAML configuration file
        explicit: If True, raise ConfigError on any failure (user-requested config).
                 If False, allow FileNotFoundError to propagate (auto-discovery).

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If file doesn't exist and explicit=False
        ConfigError: If explicit=True and loading/parsing fails
        yaml.YAMLError: If file is not valid YAML and explicit=False
    """
    path = Path(path)

    if not path.exists():
        if explicit:
            raise ConfigError(
                f"Configuration file not found: {path.absolute()}\n"
                f"Suggestion: Check that the file exists and the path is correct."
            )
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        error_msg = (
            f"Failed to parse YAML configuration from {path.absolute()}\n"
            f"Error: {e}\n"
            f"Suggestion: Check that the file contains valid YAML syntax."
        )
        if explicit:
            raise ConfigError(error_msg) from e
        raise

    if config is None:
        logger.warning(f"Empty config file: {path}")
        return {}

    logger.debug(f"Loaded config from {path}: {len(config)} keys")
    return config


def load_model_config(
    model_name: str,
    config_dir: Optional[Path] = None,
    flatten: bool = True,
    explicit: bool = False,
) -> Dict[str, Any]:
    """
    Load model-specific configuration from YAML.

    Looks for config file at: {config_dir}/{model_name}.yaml

    Args:
        model_name: Name of the model (e.g., "xgboost", "lstm")
        config_dir: Config directory (defaults to config/models/)
        flatten: If True, flatten the nested structure for backward compatibility
        explicit: If True, fail hard on any loading error (user-requested config).
                 If False, warnings are logged for missing configs (auto-discovery).

    Returns:
        Model configuration dictionary

    Raises:
        ConfigError: If explicit=True and loading fails
        FileNotFoundError: If config file doesn't exist (when not explicit)
    """
    if config_dir is None:
        config_dir = CONFIG_DIR

    config_path = config_dir / f"{model_name}.yaml"

    try:
        raw_config = load_yaml_config(config_path, explicit=explicit)
    except FileNotFoundError:
        if explicit:
            raise ConfigError(
                f"Model configuration not found for '{model_name}'\n"
                f"Expected location: {config_path.absolute()}\n"
                f"Suggestion: Check that the model name is correct and the config file exists."
            )
        raise
    except Exception as e:
        if explicit:
            raise ConfigError(
                f"Failed to load model configuration for '{model_name}' from {config_path.absolute()}\n"
                f"Error: {e}\n"
                f"Suggestion: Check that the file exists and has valid YAML syntax."
            ) from e
        raise

    if not flatten:
        return raw_config

    # Flatten the structured config for backward compatibility
    return flatten_model_config(raw_config)


def flatten_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten structured config (model/defaults/training/device sections) to flat dict."""
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
    """Find model config file if it exists."""
    if config_dir is None:
        config_dir = CONFIG_DIR

    config_path = config_dir / f"{model_name}.yaml"
    return config_path if config_path.exists() else None


def load_training_config() -> Dict[str, Any]:
    """Load global training configuration."""
    if TRAINING_CONFIG_PATH.exists():
        return load_yaml_config(TRAINING_CONFIG_PATH)
    logger.warning(f"Training config not found: {TRAINING_CONFIG_PATH}")
    return {}


def load_cv_config() -> Dict[str, Any]:
    """Load cross-validation configuration."""
    if CV_CONFIG_PATH.exists():
        return load_yaml_config(CV_CONFIG_PATH)
    logger.warning(f"CV config not found: {CV_CONFIG_PATH}")
    return {}


def get_environment_overrides(
    training_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get environment-specific configuration overrides."""
    if training_config is None:
        training_config = load_training_config()

    environments = training_config.get("environments", {})
    env = detect_environment()

    env_map = {
        Environment.COLAB: "colab",
        Environment.LOCAL_GPU: "local_gpu",
        Environment.LOCAL_CPU: "local_cpu",
    }
    return environments.get(env_map.get(env, ""), {})


# =============================================================================
# CONFIG MERGING
# =============================================================================

def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
    deep: bool = True,
) -> Dict[str, Any]:
    """Merge configs (override takes precedence, supports deep merge)."""
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
    2. Config file (if explicitly provided, FAIL HARD on errors)
    3. Environment-specific overrides
    4. Model-specific YAML (config/models/{model_name}.yaml - warn on errors)
    5. Provided defaults

    Args:
        model_name: Name of the model
        cli_args: Arguments from CLI (highest priority)
        config_file: Path to override config file (if provided, FAIL HARD on errors)
        defaults: Default configuration (lowest priority)
        apply_environment_overrides: Apply environment-specific settings

    Returns:
        Complete merged configuration

    Raises:
        ConfigError: If config_file is explicitly provided and loading fails
    """
    # Start with defaults
    config = defaults.copy() if defaults else {}

    # Try to load model-specific YAML from config/models/ (auto-discovery, warn on failure)
    model_config_path = find_model_config(model_name)
    if model_config_path:
        try:
            model_yaml = load_model_config(model_name, flatten=True, explicit=False)
            config = merge_configs(config, model_yaml)
            logger.debug(f"Merged config from {model_config_path}")
        except Exception as e:
            # Auto-discovery: warn and continue
            logger.warning(
                f"Failed to load default config from {model_config_path}: {e}. "
                f"Using built-in defaults."
            )

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

    # Load explicit config file if provided (FAIL HARD on errors)
    if config_file:
        try:
            file_config = load_yaml_config(config_file, explicit=True)
            # Flatten if structured
            if any(k in file_config for k in ["model", "defaults", "training", "device"]):
                file_config = flatten_model_config(file_config)
            config = merge_configs(config, file_config)
            logger.debug(f"Merged config from {config_file}")
        except ConfigError:
            # Re-raise ConfigError as-is (already has good error message)
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise ConfigError(
                f"Failed to load configuration from {Path(config_file).absolute()}\n"
                f"Error: {e}\n"
                f"Suggestion: Check that the file exists and has valid YAML syntax."
            ) from e

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

    Args:
        model_name: Name of the model to train
        horizon: Label horizon
        cli_args: CLI arguments
        config_file: Optional config file path (FAIL HARD if provided and invalid)

    Returns:
        Configured TrainerConfig instance

    Raises:
        ConfigError: If config_file is provided and loading fails
    """
    # Build merged model config (will raise ConfigError if config_file is invalid)
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
# EXCEPTIONS
# =============================================================================

class ConfigError(Exception):
    """Raised when configuration loading or parsing fails."""
    pass


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed: {errors}")


def validate_model_config_structure(config: Dict[str, Any]) -> List[str]:
    """Validate structured config (checks model/defaults/training/device sections)."""
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
    """Validate flattened config (checks ranges and valid values)."""
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
    """Validate config and optionally raise ConfigValidationError."""
    errors = validate_config(config, model_name)
    if errors and raise_on_error:
        raise ConfigValidationError(errors)
    return errors


# =============================================================================
# CONFIG SERIALIZATION
# =============================================================================

def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {path}")


def save_config_json(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"Saved config to {path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_available_models() -> List[str]:
    """List all models with configuration files."""
    if not CONFIG_DIR.exists():
        return []
    return sorted([f.stem for f in CONFIG_DIR.glob("*.yaml")])


def get_model_info(
    model_name: str,
    explicit: bool = False,
) -> Dict[str, Any]:
    """
    Get basic info about a model from its config.

    Args:
        model_name: Name of the model
        explicit: If True, fail hard if config cannot be loaded

    Returns:
        Dictionary with model info (name, family, description)

    Raises:
        ConfigError: If explicit=True and config loading fails
    """
    config_path = find_model_config(model_name)
    if not config_path:
        error_msg = (
            f"Model configuration not found for '{model_name}'\n"
            f"Suggestion: Check that the model name is correct or use --list-models to see available models."
        )
        if explicit:
            raise ConfigError(error_msg)
        return {"name": model_name, "family": "unknown", "description": "No config found"}

    try:
        raw_config = load_yaml_config(config_path, explicit=explicit)
        model_section = raw_config.get("model", {})
        return {
            "name": model_section.get("name", model_name),
            "family": model_section.get("family", "unknown"),
            "description": model_section.get("description", ""),
        }
    except ConfigError:
        # Re-raise ConfigError as-is
        raise
    except Exception as e:
        error_msg = (
            f"Failed to load model info for '{model_name}' from {config_path.absolute()}\n"
            f"Error: {e}\n"
            f"Suggestion: Check that the config file has valid YAML syntax."
        )
        if explicit:
            raise ConfigError(error_msg) from e
        logger.warning(error_msg)
        return {"name": model_name, "family": "unknown", "description": str(e)}


__all__ = [
    "TrainerConfig", "Environment", "ConfigError", "ConfigValidationError",
    "CONFIG_ROOT", "CONFIG_DIR", "TRAINING_CONFIG_PATH", "CV_CONFIG_PATH",
    "detect_environment", "is_colab", "resolve_device",
    "load_yaml_config", "load_model_config", "find_model_config",
    "load_training_config", "load_cv_config", "flatten_model_config",
    "get_environment_overrides", "merge_configs", "build_config",
    "create_trainer_config", "validate_model_config_structure",
    "validate_config", "validate_config_strict", "save_config",
    "save_config_json", "list_available_models", "get_model_info",
]
