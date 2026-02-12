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
    │   ├── DataContractViolation - Data contract violations
    │   └── ModelContractViolation - Model contract violations
    ├── DataError - Data processing failures
    │   ├── LeakageError - Data leakage detected
    │   ├── LookaheadError - Lookahead bias detected
    │   └── ChronologicalSortError - Chronological sorting failures
    ├── TrainingError - Model training failures
    │   ├── PreTrainingValidationError - Pre-training validation failures
    │   └── ScalerFitError - Scaler fitting failures
    ├── InferenceError - Model inference failures
    └── ResilienceError - Resilience mechanism failures (in src.core.resilience)
        ├── ResilienceTimeoutError - Operation timeout
        ├── CircuitOpenError - Circuit breaker open
        └── RetryExhaustedError - All retry attempts exhausted

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


class DataContractViolation(ContractViolation):
    """
    Raised when a data contract is violated.

    Data contracts define the expected shape, type, and range of data
    flowing through the pipeline.

    Attributes
    ----------
    issues : list[str]
        List of specific contract violation issues
    """

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Contract violations", violation_details=str(issues))


# NOTE: ModelContractViolation moved to contracts/model_contract.py (Phase 27)
# Re-import for backward compatibility
from src.core.contracts.model_contract import ModelContractViolation  # noqa: E402


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


class FeatureStoreError(MLFactoryError):
    """Raised when feature store operations fail."""

    pass


class FeatureNotFoundError(FeatureStoreError):
    """Raised when requested feature is not found in store."""

    pass


class FeatureIntegrityError(FeatureStoreError):
    """Raised when feature integrity check fails."""

    pass


class RawMTFStoreError(MLFactoryError):
    """Raised when raw MTF store operations fail."""

    pass


class TimeframeNotFoundError(RawMTFStoreError):
    """Raised when requested timeframe is not found."""

    pass


class InvalidTimeframeError(RawMTFStoreError):
    """Raised when timeframe specification is invalid."""

    pass


class InvalidSplitError(RawMTFStoreError):
    """Raised when split specification is invalid."""

    pass


class NumericalInstabilityError(MLFactoryError):
    """Raised when numerical instability is detected in neural network operations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ScalerFitError(MLFactoryError):
    """Raised when scaler fitting fails."""

    pass


class ChronologicalSortError(MLFactoryError):
    """Raised when chronological sorting fails or data is not sorted."""

    pass


class FeatureSchemaError(MLFactoryError):
    """Raised when feature schema validation fails."""

    pass


class EnsembleCompatibilityError(MLFactoryError):
    """Raised when ensemble models are incompatible."""

    pass


class SecurityError(MLFactoryError):
    """Raised when security validation fails (e.g., path traversal)."""

    pass


class StageValidationError(MLFactoryError):
    """Raised when pipeline stage validation fails."""

    def __init__(self, stage_name: str, issues: list[str]) -> None:
        self.stage_name = stage_name
        self.issues = issues
        issue_str = "\n".join(f"  - {issue}" for issue in issues)
        super().__init__(f"Stage '{stage_name}' validation failed:\n{issue_str}")


class ConfigValueError(MLFactoryError):
    """Raised when configuration value is invalid."""

    pass


class PreTrainingValidationError(MLFactoryError):
    """Raised when pre-training validation fails."""

    pass


__all__ = [
    # Base error
    "MLFactoryError",
    # Validation errors
    "ValidationError",
    "ConfigError",
    "ConfigValueError",
    # Contract violations
    "ContractViolation",
    "DataContractViolation",
    "ModelContractViolation",
    # Data errors
    "DataError",
    "LeakageError",
    "LookaheadError",
    "ChronologicalSortError",
    # Training errors
    "TrainingError",
    "PreTrainingValidationError",
    "ScalerFitError",
    # Inference errors
    "InferenceError",
    # Feature store errors
    "FeatureStoreError",
    "FeatureNotFoundError",
    "FeatureIntegrityError",
    "FeatureSchemaError",
    # Raw MTF store errors
    "RawMTFStoreError",
    "TimeframeNotFoundError",
    "InvalidTimeframeError",
    "InvalidSplitError",
    # Other errors
    "NumericalInstabilityError",
    "EnsembleCompatibilityError",
    "SecurityError",
    "StageValidationError",
    # Note: ResilienceError hierarchy is in src.core.resilience to avoid
    # circular imports (resilience module imports from exceptions)
]
