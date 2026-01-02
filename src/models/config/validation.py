"""Configuration validation functions."""

from typing import Any

from .exceptions import ConfigValidationError


def validate_model_config_structure(config: dict[str, Any]) -> list[str]:
    """Validate structured config (checks model/defaults/training/device sections)."""
    errors = []

    # Check for required sections
    if "model" not in config:
        errors.append("Missing required section: 'model'")
    else:
        model_section = config["model"]
        if "name" not in model_section:
            errors.append("Missing required field: model.name")
        if "family" not in model_section:
            errors.append("Missing required field: model.family")

        # Validate family
        valid_families = {"boosting", "neural", "transformer", "classical", "ensemble"}
        if "family" in model_section and model_section["family"] not in valid_families:
            errors.append(
                f"Invalid model.family: {model_section['family']}. "
                f"Must be one of: {valid_families}"
            )

    # Check device section
    if "device" in config:
        device_section = config["device"]
        if "default" in device_section:
            valid_devices = {"auto", "cuda", "cpu"}
            if device_section["default"] not in valid_devices:
                errors.append(
                    f"Invalid device.default: {device_section['default']}. "
                    f"Must be one of: {valid_devices}"
                )

    return errors


def validate_config(config: dict[str, Any], model_name: str) -> list[str]:
    """Validate flattened config (checks ranges and valid values)."""
    errors = []

    # Validate numeric ranges
    numeric_ranges = {
        "n_estimators": (1, 10000),
        "iterations": (1, 10000),
        "max_depth": (1, 100),
        "depth": (1, 16),
        "learning_rate": (0.0001, 1.0),
        "dropout": (0.0, 1.0),
        "hidden_size": (1, 4096),
        "d_model": (1, 4096),
        "num_layers": (1, 100),
        "n_layers": (1, 100),
        "batch_size": (1, 65536),
        "max_epochs": (1, 10000),
        "sequence_length": (1, 4096),
        "n_heads": (1, 64),
        "early_stopping_patience": (0, 1000),
        "early_stopping_rounds": (1, 1000),
    }

    for field_name, (min_val, max_val) in numeric_ranges.items():
        if field_name in config:
            value = config[field_name]
            if value is None:
                continue  # Allow None values
            if not isinstance(value, (int, float)):
                errors.append(f"{field_name} must be numeric, got {type(value).__name__}")
            elif value < min_val or value > max_val:
                errors.append(f"{field_name}={value} out of range [{min_val}, {max_val}]")

    # Validate device
    if "device" in config:
        valid_devices = {"auto", "cuda", "cpu"}
        if config["device"] not in valid_devices:
            errors.append(f"Invalid device: {config['device']}. Must be one of: {valid_devices}")

    return errors


def validate_config_strict(
    config: dict[str, Any],
    model_name: str,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate config and optionally raise ConfigValidationError."""
    errors = validate_config(config, model_name)
    if errors and raise_on_error:
        raise ConfigValidationError(errors)
    return errors
