"""
Unified exception hierarchy for ML Factory.

This module provides a consistent exception hierarchy for all ML Factory errors.
All custom exceptions inherit from MLFactoryError, making it easy to catch
any ML Factory-related error.

Exception Hierarchy:
    MLFactoryError (base)
    ├── ValidationError - Data/model validation failures
    ├── ConfigError - Configuration parsing/validation failures
    ├── ContractViolation - Data or model contract violations
    ├── DataError - Data processing failures
    ├── TrainingError - Model training failures
    └── InferenceError - Model inference failures

Usage:
    from src.core.exceptions import ValidationError, ContractViolation

    # Catch specific error
    try:
        validate_data(df)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")

    # Catch any ML Factory error
    try:
        run_pipeline()
    except MLFactoryError as e:
        logger.error(f"Pipeline failed: {e}")
"""

from __future__ import annotations

from typing import Any


class MLFactoryError(Exception):
    """
    Base exception for all ML Factory errors.

    All custom exceptions in ML Factory inherit from this class,
    making it easy to catch any ML Factory-specific error.
    """

    pass


class ValidationError(MLFactoryError):
    """
    Raised when data or model validation fails.

    Attributes
    ----------
    field : str | None
        The field/parameter that failed validation
    expected : Any | None
        What was expected
    actual : Any | None
        What was received
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        expected: Any | None = None,
        actual: Any | None = None,
    ):
        self.field = field
        self.expected = expected
        self.actual = actual

        # Build detailed message
        full_message = message
        if field:
            full_message = f"[{field}] {message}"
        if expected is not None and actual is not None:
            full_message += f" (expected: {expected}, got: {actual})"

        super().__init__(full_message)


class ConfigError(MLFactoryError):
    """
    Raised when configuration is invalid.

    This covers both parsing errors (invalid YAML/JSON) and
    semantic errors (invalid parameter combinations).
    """

    pass


class ContractViolation(MLFactoryError):
    """
    Raised when a data or model contract is violated.

    Contract violations indicate that data or model outputs
    don't meet the required specifications (shape, type, range, etc.).

    Attributes
    ----------
    contract_name : str | None
        Name of the violated contract
    violation_details : str | None
        Specific details about what was violated
    """

    def __init__(
        self,
        message: str,
        contract_name: str | None = None,
        violation_details: str | None = None,
    ):
        self.contract_name = contract_name
        self.violation_details = violation_details

        full_message = message
        if contract_name:
            full_message = f"[{contract_name}] {message}"
        if violation_details:
            full_message += f": {violation_details}"

        super().__init__(full_message)


class DataError(MLFactoryError):
    """
    Raised when data processing fails.

    This covers errors during feature engineering, data transformation,
    adapter processing, and other data pipeline operations.
    """

    pass


class TrainingError(MLFactoryError):
    """
    Raised when model training fails.

    This covers errors during model fitting, optimization,
    cross-validation, and hyperparameter tuning.
    """

    pass


class InferenceError(MLFactoryError):
    """
    Raised when model inference fails.

    This covers errors during prediction, probability estimation,
    and model serving operations.
    """

    pass


class LeakageError(MLFactoryError):
    """
    Raised when data leakage is detected.

    Data leakage occurs when information from the future
    inadvertently influences model training or feature computation.
    """

    pass


class LookaheadError(MLFactoryError):
    """
    Raised when lookahead bias is detected.

    Lookahead bias occurs when features are computed using
    data that wouldn't be available at prediction time.
    """

    pass


__all__ = [
    "MLFactoryError",
    "ValidationError",
    "ConfigError",
    "ContractViolation",
    "DataError",
    "TrainingError",
    "InferenceError",
    "LeakageError",
    "LookaheadError",
]
