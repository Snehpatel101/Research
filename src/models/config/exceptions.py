"""Configuration exceptions."""


class ConfigError(Exception):
    """Raised when configuration loading or parsing fails."""

    pass


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed: {errors}")
