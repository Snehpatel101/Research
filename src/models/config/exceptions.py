"""
Configuration exceptions for src.models.config.

ConfigError is re-exported from its canonical location, src.core.exceptions;
ConfigValidationError is the model-config validation failure.
"""

from src.core.exceptions import ConfigError

# Re-export ConfigError from canonical location
__all__ = ["ConfigError", "ConfigValidationError"]


class ConfigValidationError(ConfigError):
    """
    Raised when configuration validation fails.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed: {errors}")
