"""YAML configuration loading functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

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
    return dict(config)


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
    logger.info(
        f"Training config not found at {TRAINING_CONFIG_PATH}. "
        f"Using built-in defaults. To customize, create this file."
    )
    return {}


def load_cv_config() -> dict[str, Any]:
    """Load cross-validation configuration."""
    if CV_CONFIG_PATH.exists():
        return load_yaml_config(CV_CONFIG_PATH)
    logger.info(
        f"CV config not found at {CV_CONFIG_PATH}. "
        f"Using built-in defaults. To customize, create this file."
    )
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
    return dict(environments.get(env_map.get(env, ""), {}))


def validate_ensemble_base_models(
    config: dict[str, Any],
    ensemble_type: str | None = None,
) -> tuple[bool, str | None]:
    """
    Validate ensemble base model compatibility at config load time (MOD-007).

    Performs early validation of base model compatibility before training begins.
    This catches configuration errors before expensive model instantiation.

    Args:
        config: Ensemble model configuration containing 'base_model_names'
        ensemble_type: Type of ensemble ('voting', 'stacking', 'blending')
                      If None, inferred from config or defaults to strict validation

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if configuration is valid
        - (False, error_message) if configuration is invalid

    Example:
        >>> config = {'base_model_names': ['xgboost', 'lstm']}
        >>> is_valid, error = validate_ensemble_base_models(config, 'voting')
        >>> if not is_valid:
        ...     logger.error(f"Invalid ensemble config: {error}")
    """
    base_models = config.get("base_model_names", [])

    if not base_models:
        return True, None  # No base models to validate

    if len(base_models) < 2:
        return False, f"Ensemble requires at least 2 base models, got {len(base_models)}"

    # Import validator lazily to avoid circular imports
    try:
        from ..ensemble.validator import is_heterogeneous_ensemble, validate_ensemble_config
    except ImportError:
        logger.debug("Ensemble validator not available, skipping validation")
        return True, None

    # Determine ensemble type from config if not provided
    if ensemble_type is None:
        ensemble_type = config.get("ensemble_type")

    # Run validation
    is_valid, error_msg = validate_ensemble_config(
        base_model_names=base_models,
        ensemble_type=ensemble_type,  # type: ignore[arg-type]
    )

    if is_valid:
        # Log info about heterogeneous configuration
        if is_heterogeneous_ensemble(base_models):
            logger.info(
                f"Heterogeneous stacking ensemble detected: "
                f"base_models={base_models}. Meta-learner will receive 2D OOF predictions."
            )

    return is_valid, error_msg if not is_valid else None


def load_ensemble_config(
    model_name: str,
    config_dir: Path | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Load ensemble configuration with optional base model validation (MOD-007).

    Wraps load_model_config with ensemble-specific validation.

    Args:
        model_name: Ensemble model name ('voting', 'stacking', 'blending')
        config_dir: Config directory (defaults to config/models/)
        validate: If True, validate base model compatibility

    Returns:
        Ensemble configuration dictionary

    Raises:
        ConfigError: If validation fails and validate=True
    """
    config = load_model_config(model_name, config_dir=config_dir)

    if validate:
        is_valid, error_msg = validate_ensemble_base_models(
            config,
            ensemble_type=model_name.lower(),
        )
        if not is_valid:
            raise ConfigError(
                f"Ensemble configuration validation failed for '{model_name}':\n" f"{error_msg}"
            )

    return config
