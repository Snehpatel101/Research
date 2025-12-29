"""YAML configuration loading functions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .environment import Environment, detect_environment
from .exceptions import ConfigError
from .paths import CONFIG_DIR, CV_CONFIG_PATH, TRAINING_CONFIG_PATH

logger = logging.getLogger(__name__)


def load_yaml_config(
    path: str | Path,
    explicit: bool = False,
) -> dict[str, Any]:
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
        with open(path) as f:
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
    config_dir: Path | None = None,
    flatten: bool = True,
    explicit: bool = False,
) -> dict[str, Any]:
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


def flatten_model_config(config: dict[str, Any]) -> dict[str, Any]:
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
    config_dir: Path | None = None,
) -> Path | None:
    """Find model config file if it exists."""
    if config_dir is None:
        config_dir = CONFIG_DIR

    config_path = config_dir / f"{model_name}.yaml"
    return config_path if config_path.exists() else None


def load_training_config() -> dict[str, Any]:
    """Load global training configuration."""
    if TRAINING_CONFIG_PATH.exists():
        return load_yaml_config(TRAINING_CONFIG_PATH)
    logger.warning(f"Training config not found: {TRAINING_CONFIG_PATH}")
    return {}


def load_cv_config() -> dict[str, Any]:
    """Load cross-validation configuration."""
    if CV_CONFIG_PATH.exists():
        return load_yaml_config(CV_CONFIG_PATH)
    logger.warning(f"CV config not found: {CV_CONFIG_PATH}")
    return {}


def get_environment_overrides(
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
