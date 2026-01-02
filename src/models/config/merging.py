"""Configuration merging and building functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .environment import detect_environment
from .exceptions import ConfigError
from .loaders import (
    find_model_config,
    flatten_model_config,
    get_environment_overrides,
    load_model_config,
    load_yaml_config,
)
from .trainer_config import TrainerConfig

logger = logging.getLogger(__name__)


def merge_configs(
    base: dict[str, Any],
    override: dict[str, Any],
    deep: bool = True,
) -> dict[str, Any]:
    """Merge configs (override takes precedence, supports deep merge)."""
    result = base.copy()

    for key, value in override.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value, deep=True)
        else:
            result[key] = value

    return result


def build_config(
    model_name: str,
    cli_args: dict[str, Any] | None = None,
    config_file: str | Path | None = None,
    defaults: dict[str, Any] | None = None,
    apply_environment_overrides: bool = True,
) -> dict[str, Any]:
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
    cli_args: dict[str, Any] | None = None,
    config_file: str | Path | None = None,
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
