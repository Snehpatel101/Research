"""
Centralized configuration value access utilities.

This module provides the SINGLE implementation of config value retrieval
that should be used throughout the codebase. It replaces the 71+ duplicated
`_get_global_or_default()` implementations.

Usage:
    from src.config.utils import get_config_value, ConfigAccessLog

    # Simple usage with fallback
    batch_size = get_config_value("training.batch_size", 256)

    # Get access log for debugging
    log = ConfigAccessLog.get_log()
    for entry in log:
        print(f"{entry.path}: {entry.source} -> {entry.value}")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path  # noqa: F401 - used for type hints in docstrings
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigSource(Enum):
    """Source of a configuration value."""

    YAML_FILE = "yaml_file"
    FALLBACK = "fallback"
    OVERRIDE = "override"
    CACHED = "cached"
    ERROR = "error"


@dataclass
class ConfigAccessEntry:
    """Record of a single configuration value access."""

    path: str
    value: Any
    source: ConfigSource
    fallback_used: bool
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: str | None = None


class ConfigAccessLog:
    """
    Thread-safe log of configuration accesses for debugging and auditing.

    This helps identify which config values are being accessed, whether
    they came from the YAML config or fell back to defaults, and any
    errors that occurred during access.
    """

    _instance: ConfigAccessLog | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._entries: list[ConfigAccessEntry] = []
        self._entry_lock = threading.Lock()
        self._enabled = True

    @classmethod
    def get_instance(cls) -> ConfigAccessLog:
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ConfigAccessLog()
        return cls._instance

    def log_access(
        self,
        path: str,
        value: Any,
        source: ConfigSource,
        fallback_used: bool,
        error_message: str | None = None,
    ) -> None:
        """Log a configuration access."""
        if not self._enabled:
            return

        entry = ConfigAccessEntry(
            path=path,
            value=value,
            source=source,
            fallback_used=fallback_used,
            error_message=error_message,
        )

        with self._entry_lock:
            self._entries.append(entry)

    def get_log(self) -> list[ConfigAccessEntry]:
        """Get a copy of the access log."""
        with self._entry_lock:
            return list(self._entries)

    def get_fallback_accesses(self) -> list[ConfigAccessEntry]:
        """Get only entries where fallback was used."""
        with self._entry_lock:
            return [e for e in self._entries if e.fallback_used]

    def get_errors(self) -> list[ConfigAccessEntry]:
        """Get entries with errors."""
        with self._entry_lock:
            return [e for e in self._entries if e.source == ConfigSource.ERROR]

    def clear(self) -> None:
        """Clear the access log."""
        with self._entry_lock:
            self._entries.clear()

    def enable(self) -> None:
        """Enable logging."""
        self._enabled = True

    def disable(self) -> None:
        """Disable logging."""
        self._enabled = False

    def summary(self) -> dict[str, Any]:
        """Get summary statistics of config accesses."""
        with self._entry_lock:
            total = len(self._entries)
            fallbacks = sum(1 for e in self._entries if e.fallback_used)
            errors = sum(1 for e in self._entries if e.source == ConfigSource.ERROR)
            unique_paths = len({e.path for e in self._entries})

            return {
                "total_accesses": total,
                "unique_paths": unique_paths,
                "fallback_count": fallbacks,
                "error_count": errors,
                "fallback_rate": fallbacks / total if total > 0 else 0.0,
            }


# Module-level cache for the global config to avoid repeated loading
_global_config_cache: Any = None
_global_config_lock = threading.Lock()


def _load_global_config() -> Any:
    """
    Load the global config with caching.

    Returns the GlobalConfig instance, loading it only once per process.
    """
    global _global_config_cache

    if _global_config_cache is not None:
        return _global_config_cache

    with _global_config_lock:
        # Double-check pattern
        if _global_config_cache is not None:
            return _global_config_cache

        try:
            from src.config.global_config import get_global_config

            _global_config_cache = get_global_config()
            return _global_config_cache
        except Exception as e:
            logger.warning(f"Failed to load global config: {e}")
            return None


def clear_config_cache() -> None:
    """
    Clear the global config cache.

    Call this if the config file has been modified and needs to be reloaded.
    """
    global _global_config_cache

    with _global_config_lock:
        _global_config_cache = None


def get_config_value(attr_path: str, fallback: T, log_access: bool = True) -> T:
    """
    Get a configuration value from the global config with proper error handling.

    This is the SINGLE canonical implementation that replaces all the duplicated
    `_get_global_or_default()` functions throughout the codebase.

    Key improvements over the duplicated implementations:
    1. Proper error logging (not silent swallowing)
    2. Caching of the global config
    3. Access tracking for debugging
    4. Type-safe fallback return

    Parameters
    ----------
    attr_path : str
        Dot-separated path to the config attribute (e.g., "training.batch_size")
    fallback : T
        Default value to return if the config value is not found
    log_access : bool, default True
        Whether to log this access for debugging/auditing

    Returns
    -------
    T
        The config value or the fallback if not found

    Examples
    --------
    >>> batch_size = get_config_value("training.batch_size", 256)
    >>> horizons = get_config_value("horizons.active", [5, 10, 15, 20])
    >>> seed = get_config_value("random_seed", 42)
    """
    access_log = ConfigAccessLog.get_instance()

    try:
        config = _load_global_config()

        if config is None:
            if log_access:
                access_log.log_access(
                    path=attr_path,
                    value=fallback,
                    source=ConfigSource.FALLBACK,
                    fallback_used=True,
                    error_message="Global config not loaded",
                )
            return fallback

        # Navigate the attribute path
        parts = attr_path.split(".")
        value = config

        for part in parts:
            try:
                value = getattr(value, part)
            except AttributeError:
                # Try dict-style access as fallback
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    logger.debug(
                        f"Config path '{attr_path}' not found at '{part}', "
                        f"using fallback: {fallback}"
                    )
                    if log_access:
                        access_log.log_access(
                            path=attr_path,
                            value=fallback,
                            source=ConfigSource.FALLBACK,
                            fallback_used=True,
                            error_message=f"Attribute '{part}' not found",
                        )
                    return fallback

        # Handle None values - use fallback
        if value is None:
            if log_access:
                access_log.log_access(
                    path=attr_path,
                    value=fallback,
                    source=ConfigSource.FALLBACK,
                    fallback_used=True,
                    error_message="Value is None",
                )
            return fallback

        if log_access:
            access_log.log_access(
                path=attr_path,
                value=value,
                source=ConfigSource.YAML_FILE,
                fallback_used=False,
            )

        return value

    except Exception as e:
        # Log the error but don't crash - return fallback
        logger.warning(
            f"Error accessing config path '{attr_path}': {e}. Using fallback: {fallback}"
        )
        if log_access:
            access_log.log_access(
                path=attr_path,
                value=fallback,
                source=ConfigSource.ERROR,
                fallback_used=True,
                error_message=str(e),
            )
        return fallback


def get_config_value_strict(attr_path: str) -> Any:
    """
    Get a configuration value, raising an error if not found.

    Unlike get_config_value(), this function does not accept a fallback
    and will raise ConfigValueError if the value is not found.

    Parameters
    ----------
    attr_path : str
        Dot-separated path to the config attribute

    Returns
    -------
    Any
        The config value

    Raises
    ------
    ConfigValueError
        If the config value is not found
    """
    try:
        config = _load_global_config()

        if config is None:
            raise ConfigValueError(
                f"Cannot get '{attr_path}': Global config not loaded"
            )

        parts = attr_path.split(".")
        value = config

        for part in parts:
            try:
                value = getattr(value, part)
            except AttributeError:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    raise ConfigValueError(
                        f"Config path '{attr_path}' not found: "
                        f"attribute '{part}' does not exist"
                    )

        if value is None:
            raise ConfigValueError(f"Config value at '{attr_path}' is None")

        return value

    except ConfigValueError:
        raise
    except Exception as e:
        raise ConfigValueError(f"Error accessing '{attr_path}': {e}") from e


class ConfigValueError(Exception):
    """Raised when a required configuration value cannot be retrieved."""

    pass


def validate_config_path(attr_path: str) -> tuple[bool, str | None]:
    """
    Validate that a configuration path exists.

    Parameters
    ----------
    attr_path : str
        Dot-separated path to check

    Returns
    -------
    tuple[bool, str | None]
        (is_valid, error_message)
    """
    try:
        get_config_value_strict(attr_path)
        return True, None
    except ConfigValueError as e:
        return False, str(e)


def list_config_paths(prefix: str = "") -> list[str]:
    """
    List all available configuration paths.

    Parameters
    ----------
    prefix : str
        Optional prefix to filter paths

    Returns
    -------
    list[str]
        List of dot-separated configuration paths
    """
    config = _load_global_config()
    if config is None:
        return []

    paths = []
    _collect_paths(config, "", paths)

    if prefix:
        paths = [p for p in paths if p.startswith(prefix)]

    return sorted(paths)


def _collect_paths(obj: Any, prefix: str, paths: list[str]) -> None:
    """Recursively collect configuration paths."""
    if hasattr(obj, "__dataclass_fields__"):
        for field_name in obj.__dataclass_fields__:
            full_path = f"{prefix}.{field_name}" if prefix else field_name
            value = getattr(obj, field_name, None)
            if hasattr(value, "__dataclass_fields__"):
                _collect_paths(value, full_path, paths)
            else:
                paths.append(full_path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            full_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _collect_paths(value, full_path, paths)
            else:
                paths.append(full_path)


# Backward compatibility aliases
def _get_global_or_default(attr_path: str, fallback: T) -> T:
    """
    DEPRECATED: Use get_config_value() instead.

    This function exists only for backward compatibility during migration.
    It will be removed in a future version.
    """
    import warnings

    warnings.warn(
        "_get_global_or_default is deprecated. Use get_config_value() from "
        "src.config.utils instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_config_value(attr_path, fallback)


__all__ = [
    "get_config_value",
    "get_config_value_strict",
    "clear_config_cache",
    "validate_config_path",
    "list_config_paths",
    "ConfigAccessLog",
    "ConfigAccessEntry",
    "ConfigSource",
    "ConfigValueError",
    # Deprecated
    "_get_global_or_default",
]
