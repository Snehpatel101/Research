"""
Configuration validation utilities.

This module provides comprehensive validation for configuration objects,
including schema validation, cross-field validation, and type coercion.

Usage:
    from src.config.validators import (
        validate_config,
        ValidationResult,
        ValidationError,
    )

    result = validate_config(config_dict)
    if not result.is_valid:
        for error in result.errors:
            print(f"Error: {error}")
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Must be fixed, will prevent operation
    WARNING = "warning"  # Should be fixed, may cause issues
    INFO = "info"  # Informational, may indicate suboptimal config


@dataclass
class ValidationIssue:
    """A single validation issue."""

    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    suggestion: str | None = None

    def __str__(self) -> str:
        base = f"[{self.severity.value.upper()}] {self.field}: {self.message}"
        if self.suggestion:
            base += f" (Suggestion: {self.suggestion})"
        return base


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if no errors (warnings/info are allowed)."""
        return not any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def add_error(
        self,
        field: str,
        message: str,
        value: Any = None,
        suggestion: str | None = None,
    ) -> None:
        """Add an error-level issue."""
        self.issues.append(
            ValidationIssue(
                field=field,
                message=message,
                severity=ValidationSeverity.ERROR,
                value=value,
                suggestion=suggestion,
            )
        )

    def add_warning(
        self,
        field: str,
        message: str,
        value: Any = None,
        suggestion: str | None = None,
    ) -> None:
        """Add a warning-level issue."""
        self.issues.append(
            ValidationIssue(
                field=field,
                message=message,
                severity=ValidationSeverity.WARNING,
                value=value,
                suggestion=suggestion,
            )
        )

    def add_info(
        self,
        field: str,
        message: str,
        value: Any = None,
        suggestion: str | None = None,
    ) -> None:
        """Add an info-level issue."""
        self.issues.append(
            ValidationIssue(
                field=field,
                message=message,
                severity=ValidationSeverity.INFO,
                value=value,
                suggestion=suggestion,
            )
        )

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge another result into this one."""
        self.issues.extend(other.issues)
        return self

    def summary(self) -> str:
        """Get a summary of validation results."""
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        info_count = len([i for i in self.issues if i.severity == ValidationSeverity.INFO])

        if not self.issues:
            return "Configuration is valid."

        parts = []
        if error_count:
            parts.append(f"{error_count} error(s)")
        if warning_count:
            parts.append(f"{warning_count} warning(s)")
        if info_count:
            parts.append(f"{info_count} info message(s)")

        status = "INVALID" if error_count else "VALID with warnings"
        return f"Configuration {status}: {', '.join(parts)}"


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, result: ValidationResult):
        self.result = result
        errors = [str(e) for e in result.errors]
        super().__init__("Configuration validation failed:\n" + "\n".join(errors))


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# Required fields for each config section
REQUIRED_FIELDS = {
    "root": ["random_seed"],
    "timeframes": ["default_primary", "canonical_ladder"],
    "splits": ["train", "val", "test"],
    "purge_embargo": ["purge_multiplier", "embargo_time_minutes"],
    "horizons": ["supported", "active"],
    "training": ["batch_size", "max_epochs"],
}

# Field type specifications
FIELD_TYPES = {
    "random_seed": int,
    "timeframes.default_primary": str,
    "timeframes.canonical_ladder": list,
    "splits.train": float,
    "splits.val": float,
    "splits.test": float,
    "purge_embargo.purge_multiplier": float,
    "purge_embargo.embargo_time_minutes": int,
    "purge_embargo.min_embargo_bars": int,
    "horizons.supported": list,
    "horizons.active": list,
    "horizons.default": list,
    "training.sequence_length": int,
    "training.batch_size": int,
    "training.max_epochs": int,
    "training.early_stopping_patience": int,
    "training.device": str,
    "training.mixed_precision": bool,
    "training.num_workers": int,
    "training.pin_memory": bool,
    "calibration.enabled": bool,
    "calibration.method": str,
    "mtf.enabled": bool,
    "mtf.default_mode": str,
    "mtf.default_timeframes": list,
}

# Valid values for enum-like fields
VALID_VALUES = {
    "training.device": ["auto", "cpu", "cuda", "mps"],
    "calibration.method": ["auto", "isotonic", "sigmoid", "none"],
    "mtf.default_mode": ["indicators", "bars", "both"],
    "scaler.default": ["robust", "standard", "minmax", "none"],
    "tracking.backend": ["local", "mlflow", "wandb", "disabled"],
}

# Numeric bounds
NUMERIC_BOUNDS = {
    "random_seed": (0, 2**31 - 1),
    "splits.train": (0.0, 1.0),
    "splits.val": (0.0, 1.0),
    "splits.test": (0.0, 1.0),
    "purge_embargo.purge_multiplier": (1.0, 10.0),
    "purge_embargo.embargo_time_minutes": (60, 20160),  # 1 hour to 2 weeks
    "training.batch_size": (1, 8192),
    "training.max_epochs": (1, 10000),
    "training.early_stopping_patience": (0, 1000),
    "training.num_workers": (0, 64),
    "horizons.supported": (1, 500),  # Element bounds
    "horizons.active": (1, 500),  # Element bounds
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_required_fields(
    data: dict[str, Any],
    section: str = "root",
    prefix: str = "",
) -> ValidationResult:
    """Validate that all required fields are present."""
    result = ValidationResult()

    required = REQUIRED_FIELDS.get(section, [])
    for field_name in required:
        full_path = f"{prefix}.{field_name}" if prefix else field_name
        if field_name not in data:
            result.add_error(
                field=full_path,
                message=f"Required field '{field_name}' is missing",
                suggestion=f"Add '{field_name}' to your configuration",
            )

    return result


def validate_field_types(
    data: dict[str, Any],
    prefix: str = "",
) -> ValidationResult:
    """Validate that fields have the correct types."""
    result = ValidationResult()

    for path, expected_type in FIELD_TYPES.items():
        if prefix and not path.startswith(prefix):
            continue

        # Navigate to the value
        keys = path.split(".")
        value = data
        found = True

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                found = False
                break

        if not found:
            continue  # Missing fields are handled by required validation

        # Check type
        if not isinstance(value, expected_type):
            # Allow int for float fields
            if expected_type == float and isinstance(value, int):
                continue

            result.add_error(
                field=path,
                message=f"Expected type {expected_type.__name__}, got {type(value).__name__}",
                value=value,
                suggestion=f"Convert value to {expected_type.__name__}",
            )

    return result


def validate_enum_values(
    data: dict[str, Any],
    prefix: str = "",
) -> ValidationResult:
    """Validate that enum-like fields have valid values."""
    result = ValidationResult()

    for path, valid_values in VALID_VALUES.items():
        if prefix and not path.startswith(prefix):
            continue

        # Navigate to the value
        keys = path.split(".")
        value: Any = data
        found = True

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                found = False
                break

        if not found:
            continue

        if not isinstance(value, dict) and value not in valid_values:
            result.add_error(
                field=path,
                message=f"Invalid value '{value}'. Must be one of: {valid_values}",
                value=value,
                suggestion=f"Use one of: {', '.join(str(v) for v in valid_values)}",
            )

    return result


def validate_numeric_bounds(
    data: dict[str, Any],
    prefix: str = "",
) -> ValidationResult:
    """Validate that numeric fields are within acceptable bounds."""
    result = ValidationResult()

    for path, (min_val, max_val) in NUMERIC_BOUNDS.items():
        if prefix and not path.startswith(prefix):
            continue

        # Navigate to the value
        keys = path.split(".")
        value = data
        found = True

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                found = False
                break

        if not found:
            continue

        # Handle list fields (check each element)
        if isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, (int, float)):
                    if item < min_val or item > max_val:
                        result.add_warning(
                            field=f"{path}[{i}]",
                            message=f"Value {item} outside typical range [{min_val}, {max_val}]",
                            value=item,
                        )
        elif isinstance(value, (int, float)):
            if value < min_val or value > max_val:
                result.add_warning(
                    field=path,
                    message=f"Value {value} outside typical range [{min_val}, {max_val}]",
                    value=value,
                )

    return result


def validate_split_ratios(data: dict[str, Any]) -> ValidationResult:
    """Validate that split ratios sum to 1.0."""
    result = ValidationResult()

    splits = data.get("splits", {})
    train = splits.get("train", 0.0)
    val = splits.get("val", 0.0)
    test = splits.get("test", 0.0)

    total = train + val + test
    if abs(total - 1.0) > 0.01:
        result.add_error(
            field="splits",
            message=f"Split ratios must sum to 1.0, got {total:.3f} "
            f"(train={train}, val={val}, test={test})",
            value={"train": train, "val": val, "test": test},
            suggestion="Adjust ratios so they sum to 1.0",
        )

    # Warn if train ratio is too low
    if train < 0.5:
        result.add_warning(
            field="splits.train",
            message=f"Train ratio {train} is below 0.5, which may lead to underfitting",
            value=train,
            suggestion="Consider using at least 0.5 for train ratio",
        )

    return result


def validate_horizons(data: dict[str, Any]) -> ValidationResult:
    """Validate horizon configuration."""
    result = ValidationResult()

    horizons = data.get("horizons", {})
    supported = horizons.get("supported", [])
    active = horizons.get("active", [])

    # Active must be subset of supported
    for h in active:
        if h not in supported:
            result.add_error(
                field="horizons.active",
                message=f"Active horizon {h} is not in supported horizons {supported}",
                value=h,
                suggestion=f"Add {h} to horizons.supported or remove from active",
            )

    # Check for reasonable horizon values
    if active:
        max_horizon = max(active)
        if max_horizon > 120:
            result.add_warning(
                field="horizons.active",
                message=f"Maximum horizon {max_horizon} is very large, "
                f"may require significant data",
                value=max_horizon,
            )

    return result


def validate_purge_embargo(data: dict[str, Any]) -> ValidationResult:
    """Validate purge/embargo configuration and cross-field dependencies."""
    result = ValidationResult()

    purge_embargo = data.get("purge_embargo", {})
    horizons = data.get("horizons", {})
    timeframes = data.get("timeframes", {})

    purge_mult = purge_embargo.get("purge_multiplier", 3.0)
    embargo_minutes = purge_embargo.get("embargo_time_minutes", 7200)
    active_horizons = horizons.get("active", [])
    default_tf = timeframes.get("default_primary", "5min")

    # Calculate expected purge bars
    if active_horizons:
        max_horizon = max(active_horizons)
        expected_purge = int(max_horizon * purge_mult)

        # Check if purge is sufficient for max_bars calculation
        # max_bars is typically 2.5x horizon
        max_bars = int(max_horizon * 2.5)
        if expected_purge < max_bars:
            result.add_warning(
                field="purge_embargo.purge_multiplier",
                message=f"Purge ({expected_purge} bars) may be insufficient for "
                f"max_bars ({max_bars}). Consider multiplier >= 2.5",
                value=purge_mult,
            )

    # Calculate embargo bars for the default timeframe
    tf_minutes = _parse_timeframe_minutes(default_tf)
    if tf_minutes:
        embargo_bars = embargo_minutes // tf_minutes
        if embargo_bars < 100:
            result.add_warning(
                field="purge_embargo.embargo_time_minutes",
                message=f"Embargo of {embargo_bars} bars at {default_tf} may be "
                f"insufficient for decorrelation",
                value=embargo_minutes,
                suggestion="Consider at least 288 bars (~1 day at 5min)",
            )

    return result


def validate_timeframes(data: dict[str, Any]) -> ValidationResult:
    """Validate timeframe configuration."""
    result = ValidationResult()

    from src.core.common.timeframes import CANONICAL_TIMEFRAMES as VALID_TFS

    timeframes = data.get("timeframes", {})
    canonical_ladder = timeframes.get("canonical_ladder", [])
    default_primary = timeframes.get("default_primary", "")

    # Validate canonical ladder
    for tf in canonical_ladder:
        if tf not in VALID_TFS:
            result.add_warning(
                field="timeframes.canonical_ladder",
                message=f"Timeframe '{tf}' is not a standard canonical timeframe",
                value=tf,
                suggestion=f"Standard timeframes: {VALID_TFS}",
            )

    # Validate default primary
    if default_primary and default_primary not in canonical_ladder:
        result.add_error(
            field="timeframes.default_primary",
            message=f"Default primary '{default_primary}' is not in canonical_ladder",
            value=default_primary,
            suggestion="Add it to canonical_ladder or choose a different default",
        )

    return result


def validate_training(data: dict[str, Any]) -> ValidationResult:
    """Validate training configuration."""
    result = ValidationResult()

    training = data.get("training", {})
    batch_size = training.get("batch_size", 256)
    max_epochs = training.get("max_epochs", 100)
    patience = training.get("early_stopping_patience", 15)

    # Warn if patience > max_epochs
    if patience >= max_epochs:
        result.add_warning(
            field="training.early_stopping_patience",
            message=f"Patience ({patience}) >= max_epochs ({max_epochs}), "
            f"early stopping will never trigger",
            value=patience,
            suggestion=f"Set patience < max_epochs (e.g., {max_epochs // 5})",
        )

    # Warn about power-of-2 batch sizes for GPU efficiency
    if batch_size > 0 and (batch_size & (batch_size - 1)) != 0:
        # Not a power of 2
        next_power = 2 ** (batch_size.bit_length())
        prev_power = next_power // 2
        result.add_info(
            field="training.batch_size",
            message=f"Batch size {batch_size} is not a power of 2",
            value=batch_size,
            suggestion=f"Consider {prev_power} or {next_power} for GPU efficiency",
        )

    return result


def validate_unknown_keys(
    data: dict[str, Any],
    known_keys: set[str],
    prefix: str = "",
) -> ValidationResult:
    """Warn about unknown configuration keys."""
    result = ValidationResult()

    for key in data:
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in known_keys:
            result.add_warning(
                field=full_key,
                message=f"Unknown configuration key '{key}'",
                value=data[key],
                suggestion="Check spelling or remove if not needed",
            )

    return result


def _parse_timeframe_minutes(tf: str) -> int | None:
    """Parse timeframe string to minutes."""
    try:
        if tf.endswith("min"):
            return int(tf[:-3])
        elif tf.endswith("h"):
            return int(tf[:-1]) * 60
        elif tf.endswith("d"):
            return int(tf[:-1]) * 1440
        else:
            return None
    except ValueError:
        return None


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================


def validate_config(
    data: dict[str, Any],
    strict: bool = False,
) -> ValidationResult:
    """
    Validate a configuration dictionary.

    Runs all validation checks and returns a combined result.

    Parameters
    ----------
    data : dict[str, Any]
        Configuration dictionary (typically loaded from YAML)
    strict : bool, default False
        If True, raise ConfigValidationError on any errors

    Returns
    -------
    ValidationResult
        Combined validation result

    Raises
    ------
    ConfigValidationError
        If strict=True and validation errors are found

    Examples
    --------
    >>> config = {"random_seed": 42, "splits": {"train": 0.7, "val": 0.15, "test": 0.15}}
    >>> result = validate_config(config)
    >>> if not result.is_valid:
    ...     for error in result.errors:
    ...         print(error)
    """
    result = ValidationResult()

    # Run all validators
    validators: list[Callable[[dict[str, Any]], ValidationResult]] = [
        lambda d: validate_required_fields(d, "root"),
        validate_field_types,
        validate_enum_values,
        validate_numeric_bounds,
        validate_split_ratios,
        validate_horizons,
        validate_purge_embargo,
        validate_timeframes,
        validate_training,
    ]

    for validator in validators:
        try:
            result.merge(validator(data))
        except Exception as e:
            logger.warning(f"Validator {validator.__name__} raised exception: {e}")
            result.add_error(
                field="validator",
                message=f"Validation error in {validator.__name__}: {e}",
            )

    # Log summary
    if result.errors:
        logger.error(f"Configuration validation failed: {len(result.errors)} error(s)")
    elif result.warnings:
        logger.warning(f"Configuration valid with {len(result.warnings)} warning(s)")

    # Strict mode
    if strict and not result.is_valid:
        raise ConfigValidationError(result)

    return result


def validate_config_file(
    path: str | Path,
    strict: bool = False,
) -> ValidationResult:
    """
    Validate a YAML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file
    strict : bool, default False
        If True, raise ConfigValidationError on any errors

    Returns
    -------
    ValidationResult
        Combined validation result
    """
    import yaml

    path = Path(path)
    result = ValidationResult()

    if not path.exists():
        result.add_error(
            field="file",
            message=f"Configuration file not found: {path}",
        )
        return result

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result.add_error(
            field="file",
            message=f"Invalid YAML syntax: {e}",
        )
        return result

    if data is None:
        result.add_error(
            field="file",
            message="Configuration file is empty",
        )
        return result

    return validate_config(data, strict=strict)


# =============================================================================
# TYPE COERCION
# =============================================================================


def coerce_types(
    data: dict[str, Any],
    warn_on_coercion: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """
    Coerce configuration values to expected types.

    Parameters
    ----------
    data : dict[str, Any]
        Configuration dictionary
    warn_on_coercion : bool, default True
        Emit warnings when values are coerced

    Returns
    -------
    tuple[dict[str, Any], list[str]]
        (coerced_data, coercion_warnings)
    """
    coerced = data.copy()
    coercion_log: list[str] = []

    for path, expected_type in FIELD_TYPES.items():
        keys = path.split(".")
        parent = coerced
        found = True

        # Navigate to parent
        for key in keys[:-1]:
            if isinstance(parent, dict) and key in parent:
                parent = parent[key]
            else:
                found = False
                break

        if not found:
            continue

        final_key = keys[-1]
        if final_key not in parent:
            continue

        value = parent[final_key]

        # Skip if already correct type
        if isinstance(value, expected_type):
            continue

        # Attempt coercion
        try:
            if expected_type == int and isinstance(value, float):
                coerced_value = int(value)
                if coerced_value != value:
                    msg = f"Coerced {path}: {value} -> {coerced_value} (truncated)"
                    coercion_log.append(msg)
                    if warn_on_coercion:
                        warnings.warn(msg, UserWarning)
                parent[final_key] = coerced_value

            elif expected_type == float and isinstance(value, int):
                parent[final_key] = float(value)
                coercion_log.append(f"Coerced {path}: {value} (int) -> {float(value)} (float)")

            elif expected_type == bool and isinstance(value, (int, str)):
                if isinstance(value, int):
                    parent[final_key] = bool(value)
                elif isinstance(value, str):
                    parent[final_key] = value.lower() in ("true", "yes", "1", "on")
                coercion_log.append(f"Coerced {path}: {value} -> {parent[final_key]}")

            elif expected_type == list and isinstance(value, (tuple, set)):
                parent[final_key] = list(value)
                coercion_log.append(f"Coerced {path}: {type(value).__name__} -> list")

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to coerce {path} to {expected_type.__name__}: {e}")

    return coerced, coercion_log


__all__ = [
    # Main functions
    "validate_config",
    "validate_config_file",
    "coerce_types",
    # Result types
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    # Exceptions
    "ConfigValidationError",
    # Individual validators (for custom validation pipelines)
    "validate_required_fields",
    "validate_field_types",
    "validate_enum_values",
    "validate_numeric_bounds",
    "validate_split_ratios",
    "validate_horizons",
    "validate_purge_embargo",
    "validate_timeframes",
    "validate_training",
    "validate_unknown_keys",
]
