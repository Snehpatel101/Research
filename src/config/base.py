"""
BaseConfig - Foundation for all configuration classes.

This module provides the base class that all config classes should inherit from.
It provides common functionality like:
- JSON/YAML serialization
- Validation framework
- Deep copy support
- Dictionary conversion

Usage:
    from src.config.base import BaseConfig

    @dataclass
    class MyConfig(BaseConfig):
        setting1: str
        setting2: int = 10

        def validate(self) -> list[str]:
            issues = super().validate()
            if self.setting2 < 0:
                issues.append("setting2 must be non-negative")
            return issues
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Self, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """
    Base configuration class with common functionality.

    All configuration classes should inherit from this base to ensure
    consistent serialization, validation, and factory methods.

    Features:
    - validate(): Returns list of validation issues
    - to_dict(): Convert to dictionary for serialization
    - from_dict(): Create instance from dictionary
    - save(): Save to JSON file
    - load(): Load from JSON file
    - copy(): Create a deep copy
    - merge(): Merge with another config (overrides)

    Example:
        @dataclass
        class TrainingConfig(BaseConfig):
            batch_size: int = 32
            learning_rate: float = 0.001

            def validate(self) -> list[str]:
                issues = super().validate()
                if self.batch_size <= 0:
                    issues.append("batch_size must be positive")
                return issues
    """

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Override in subclasses to add specific validation.
        Always call super().validate() first.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues: list[str] = []

        # Check for None values in required fields (no default)
        for field in fields(self):
            value = getattr(self, field.name)
            # Check if field has no default and is None
            if (
                field.default is field.default_factory  # Both are MISSING
                and value is None
            ):
                issues.append(f"Required field '{field.name}' is None")

        return issues

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Handles Path objects, datetime, and nested configs.

        Returns:
            Dictionary representation of the config
        """
        result = {}

        for field in fields(self):
            value = getattr(self, field.name)
            result[field.name] = self._serialize_value(value)

        return result

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value for JSON compatibility."""
        if value is None:
            return None
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, BaseConfig):
            return value.to_dict()
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, "value"):  # Enum
            return value.value
        else:
            return value

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """
        Create configuration instance from dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            New configuration instance
        """
        # Filter to only known fields
        known_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        # Log unknown fields
        unknown = set(data.keys()) - known_fields
        if unknown:
            logger.debug(f"Ignoring unknown fields in {cls.__name__}: {unknown}")

        return cls(**filtered_data)

    def save(self, path: str | Path, indent: int = 2) -> Path:
        """
        Save configuration to JSON file.

        Args:
            path: File path to save to
            indent: JSON indentation level

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)

        logger.debug(f"Saved {self.__class__.__name__} to {path}")
        return path

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Loaded configuration instance

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is invalid JSON
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        logger.debug(f"Loaded {cls.__name__} from {path}")
        return cls.from_dict(data)

    def copy(self: Self) -> Self:
        """
        Create a deep copy of this configuration.

        Returns:
            New configuration instance with copied values
        """
        return copy.deepcopy(self)

    def merge(self: Self, other: Self | dict[str, Any], in_place: bool = False) -> Self:
        """
        Merge another config or dict into this one.

        Values from 'other' override values in self.
        Only known fields are merged.

        Args:
            other: Another config instance or dictionary
            in_place: If True, modify self; else return new instance

        Returns:
            Merged configuration (self if in_place, else new instance)
        """
        target = self if in_place else self.copy()

        if isinstance(other, BaseConfig):
            other = other.to_dict()

        known_fields = {f.name for f in fields(self)}

        for key, value in other.items():
            if key in known_fields and value is not None:
                setattr(target, key, value)

        return target

    def diff(self, other: Self) -> dict[str, tuple[Any, Any]]:
        """
        Compare this config with another and return differences.

        Args:
            other: Another config instance to compare

        Returns:
            Dict mapping field name -> (self_value, other_value)
        """
        differences = {}

        for field in fields(self):
            self_val = getattr(self, field.name)
            other_val = getattr(other, field.name)

            if self_val != other_val:
                differences[field.name] = (self_val, other_val)

        return differences

    def summary(self) -> str:
        """
        Generate a human-readable summary of the configuration.

        Override in subclasses for custom formatting.

        Returns:
            Summary string
        """
        lines = [f"{self.__class__.__name__}:"]

        for field in fields(self):
            value = getattr(self, field.name)
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 60:
                value_str = value_str[:57] + "..."
            lines.append(f"  {field.name}: {value_str}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation showing key fields."""
        field_strs = []
        for field in fields(self)[:5]:  # Show first 5 fields
            value = getattr(self, field.name)
            if isinstance(value, str) and len(value) > 20:
                value = value[:17] + "..."
            field_strs.append(f"{field.name}={value!r}")

        if len(fields(self)) > 5:
            field_strs.append("...")

        return f"{self.__class__.__name__}({', '.join(field_strs)})"


# =============================================================================
# MIXIN CLASSES FOR OPTIONAL FUNCTIONALITY
# =============================================================================


class TimestampedConfigMixin:
    """Mixin that adds created_at timestamp to configs."""

    created_at: str

    def __post_init__(self) -> None:
        """Set created_at if not provided."""
        if not hasattr(self, "created_at") or self.created_at is None:
            self.created_at = datetime.now().isoformat()


class VersionedConfigMixin:
    """Mixin that adds version tracking to configs."""

    version: str

    def __post_init__(self) -> None:
        """Set version if not provided."""
        if not hasattr(self, "version") or self.version is None:
            self.version = "1.0.0"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "BaseConfig",
    "TimestampedConfigMixin",
    "VersionedConfigMixin",
]
