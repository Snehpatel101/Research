"""
Configuration merging and building functions.

Configuration Precedence (Highest to Lowest)
============================================

When building model configuration, values are merged in this order,
with higher-precedence sources overriding lower ones:

1. **CLI Arguments** (Highest Priority)
   - Explicit user intent via command-line flags
   - Example: `--batch-size 128`, `--max-epochs 50`
   - None values are ignored (don't override)

2. **Explicit Config File** (if provided via --config)
   - User-provided YAML file path
   - FAIL HARD on errors (user explicitly requested this file)
   - Example: `--config my_experiment.yaml`

3. **Environment-Specific Overrides**
   - From config/training.yaml `environments:` section
   - Auto-detected based on execution environment:
     * colab: Google Colab environment
     * local_gpu: Local machine with CUDA available
     * local_cpu: Local machine without GPU
   - Common overrides: batch_size, num_workers, mixed_precision

4. **Model-Specific YAML** (auto-discovered)
   - Located at config/models/{model_name}.yaml
   - Contains model-specific hyperparameters and defaults
   - WARN on errors (not user-requested, auto-discovery)

5. **Provided Defaults** (Lowest Priority)
   - Built-in defaults passed to build_config()
   - Fallback values when no other source specifies a key

Example Override Flow:
----------------------
```
defaults = {"batch_size": 64, "max_epochs": 100}
model.yaml = {"batch_size": 256}  # Overrides default
environment (GPU) = {"batch_size": 512}  # Overrides model.yaml
CLI args = {"batch_size": 128}  # Final value: 128
```

Applied Overrides Tracking:
---------------------------
Use `get_applied_overrides()` to retrieve a dictionary showing which
configuration sources were applied for debugging and reproducibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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


@dataclass
class AppliedOverrides:
    """
    Tracks which configuration sources were applied during config building.

    Used for debugging, reproducibility, and surfacing in training artifacts.

    Attributes:
        environment: Detected environment (colab, local_gpu, local_cpu)
        environment_overrides: Keys that came from environment-specific config
        model_yaml_path: Path to model-specific YAML if loaded
        model_yaml_keys: Keys that came from model YAML
        explicit_config_path: Path to explicit config file if provided
        explicit_config_keys: Keys from explicit config file
        cli_overrides: Keys that came from CLI arguments
    """

    environment: str = ""
    environment_overrides: dict[str, Any] = field(default_factory=dict)
    model_yaml_path: str | None = None
    model_yaml_keys: list[str] = field(default_factory=list)
    explicit_config_path: str | None = None
    explicit_config_keys: list[str] = field(default_factory=list)
    cli_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "environment": self.environment,
            "environment_overrides": self.environment_overrides,
            "model_yaml_path": self.model_yaml_path,
            "model_yaml_keys": self.model_yaml_keys,
            "explicit_config_path": self.explicit_config_path,
            "explicit_config_keys": self.explicit_config_keys,
            "cli_overrides": self.cli_overrides,
        }


@dataclass
class ConfigBuildResult:
    """
    Result of build_config() containing both config and applied overrides.

    This replaces the previous global state pattern (CFG-001 fix).

    Attributes:
        config: The merged configuration dictionary
        applied_overrides: Tracking of which sources contributed to the config
    """

    config: dict[str, Any]
    applied_overrides: AppliedOverrides

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config,
            "applied_overrides": self.applied_overrides.to_dict(),
        }


# Module-level storage for the last applied overrides (deprecated, kept for backward compatibility)
_last_applied_overrides: AppliedOverrides | None = None
_deprecation_warning_shown: bool = False


def get_applied_overrides() -> dict[str, Any]:
    """
    Get information about which configuration overrides were applied.

    .. deprecated::
        This function relies on global state and is deprecated.
        Use the `applied_overrides` attribute from the `ConfigBuildResult`
        returned by `build_config()` instead.

    Returns a dictionary containing:
    - environment: Detected execution environment
    - environment_overrides: Settings that came from environment config
    - model_yaml_path: Path to model YAML if loaded
    - model_yaml_keys: Keys from model YAML
    - explicit_config_path: User-provided config file path
    - explicit_config_keys: Keys from explicit config
    - cli_overrides: CLI argument values that were applied

    Returns:
        Dictionary with override tracking information, or empty dict if
        build_config() hasn't been called yet.
    """
    global _deprecation_warning_shown
    if not _deprecation_warning_shown:
        import warnings

        warnings.warn(
            "get_applied_overrides() is deprecated. Use the 'applied_overrides' "
            "attribute from ConfigBuildResult returned by build_config() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _deprecation_warning_shown = True

    if _last_applied_overrides is None:
        return {}
    return _last_applied_overrides.to_dict()


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
) -> ConfigBuildResult:
    """
    Build complete model configuration from multiple sources.

    Configuration precedence (highest to lowest):
    1. CLI arguments
    2. Config file (if explicitly provided, FAIL HARD on errors)
    3. Environment-specific overrides
    4. Model-specific YAML (config/models/{model_name}.yaml - warn on errors)
    5. Provided defaults

    See module docstring for detailed precedence documentation.

    The returned ConfigBuildResult contains both the merged configuration
    and tracking information about which sources contributed to it.

    Args:
        model_name: Name of the model
        cli_args: Arguments from CLI (highest priority)
        config_file: Path to override config file (if provided, FAIL HARD on errors)
        defaults: Default configuration (lowest priority)
        apply_environment_overrides: Apply environment-specific settings

    Returns:
        ConfigBuildResult containing:
        - config: The merged configuration dictionary
        - applied_overrides: AppliedOverrides tracking which sources were used

    Raises:
        ConfigError: If config_file is explicitly provided and loading fails
    """
    global _last_applied_overrides

    # Initialize override tracking
    applied = AppliedOverrides()
    env = detect_environment()
    applied.environment = env.value

    # Start with defaults
    config = defaults.copy() if defaults else {}

    # Try to load model-specific YAML from config/models/ (auto-discovery, warn on failure)
    model_config_path = find_model_config(model_name)
    if model_config_path:
        try:
            model_yaml = load_model_config(model_name, flatten=True, explicit=False)
            applied.model_yaml_path = str(model_config_path)
            applied.model_yaml_keys = list(model_yaml.keys())
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

            # Track environment overrides
            applied.environment_overrides = flat_overrides.copy()
            config = merge_configs(config, flat_overrides)
            logger.info(
                f"Applied environment overrides for '{env.value}': "
                f"{list(flat_overrides.keys())}"
            )

    # Load explicit config file if provided (FAIL HARD on errors)
    if config_file:
        try:
            file_config = load_yaml_config(config_file, explicit=True)
            # Flatten if structured
            if any(k in file_config for k in ["model", "defaults", "training", "device"]):
                file_config = flatten_model_config(file_config)

            # Track explicit config
            applied.explicit_config_path = str(Path(config_file).absolute())
            applied.explicit_config_keys = list(file_config.keys())
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
        # Track CLI overrides
        applied.cli_overrides = cli_config.copy()
        config = merge_configs(config, cli_config)
        logger.debug(f"Applied {len(cli_config)} CLI overrides")

    # Store applied overrides for backward compatibility (deprecated)
    _last_applied_overrides = applied

    return ConfigBuildResult(config=config, applied_overrides=applied)


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
    result = build_config(
        model_name=model_name,
        cli_args=cli_args,
        config_file=config_file,
    )
    model_config = result.config

    # Extract trainer-level settings from model config
    trainer_kwargs: dict[str, Any] = {
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
