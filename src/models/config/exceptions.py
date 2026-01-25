"""
Configuration exceptions for src.models.config.

Note: These are local definitions to avoid circular imports.
The canonical ConfigError is in src.core.exceptions.
The canonical ConfigValidationError is in src.config.validators.

For external use, prefer importing from those canonical locations.
This module exists for src.models.config internal use only.
"""

from src.core.exceptions import ConfigError

# Re-export ConfigError from canonical location
__all__ = ["ConfigError", "ConfigValidationError"]


class ConfigValidationError(ConfigError):
    """
    Raised when configuration validation fails.

    This is a local definition to avoid circular imports.
    The canonical version is in src.config.validators.ConfigValidationError.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed: {errors}")
