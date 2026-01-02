"""Utility functions for model configuration."""

import logging
from typing import Any

from .exceptions import ConfigError
from .loaders import find_model_config, load_yaml_config
from .paths import CONFIG_DIR

logger = logging.getLogger(__name__)


def list_available_models() -> list[str]:
    """List all models with configuration files."""
    if not CONFIG_DIR.exists():
        return []
    return sorted([f.stem for f in CONFIG_DIR.glob("*.yaml")])


def get_model_info(
    model_name: str,
    explicit: bool = False,
) -> dict[str, Any]:
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
