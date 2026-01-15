"""
Numerical stability validation for neural network training.

Provides utilities to detect NaN/Inf values in:
- Input tensors (before training)
- Forward pass outputs (logits)
- Loss values (during backprop)
- Gradients (after backward pass)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class NumericalInstabilityError(RuntimeError):
    """Raised when numerical instability is detected during training."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


@dataclass
class NumericalCheckResult:
    """Result of a numerical stability check."""

    is_valid: bool
    has_nan: bool
    has_inf: bool
    nan_count: int
    inf_count: int
    total_elements: int
    message: str

    @property
    def nan_ratio(self) -> float:
        """Ratio of NaN values."""
        return self.nan_count / self.total_elements if self.total_elements > 0 else 0.0

    @property
    def inf_ratio(self) -> float:
        """Ratio of Inf values."""
        return self.inf_count / self.total_elements if self.total_elements > 0 else 0.0


class NumericalValidator:
    """
    Validates numerical stability of tensors during neural network training.

    Usage:
        validator = NumericalValidator(raise_on_error=True)

        # Check input data
        validator.check_array(X_train, "X_train")

        # In training loop
        logits = model(X_batch)
        validator.check_tensor(logits, "logits")

        loss = criterion(logits, y_batch)
        validator.check_loss(loss)

        loss.backward()
        validator.check_gradients(model)
    """

    def __init__(
        self,
        raise_on_error: bool = True,
        log_warnings: bool = True,
        nan_threshold: float = 0.0,  # Max allowed NaN ratio before error
        inf_threshold: float = 0.0,  # Max allowed Inf ratio before error
    ) -> None:
        """
        Initialize numerical validator.

        Args:
            raise_on_error: Raise NumericalInstabilityError on invalid values
            log_warnings: Log warnings when issues detected
            nan_threshold: Maximum allowed ratio of NaN values (0.0 = none allowed)
            inf_threshold: Maximum allowed ratio of Inf values (0.0 = none allowed)
        """
        self.raise_on_error = raise_on_error
        self.log_warnings = log_warnings
        self.nan_threshold = nan_threshold
        self.inf_threshold = inf_threshold
        self._check_count = 0
        self._issue_count = 0

    def check_tensor(
        self,
        tensor: torch.Tensor,
        name: str = "tensor",
        raise_on_error: bool | None = None,
    ) -> NumericalCheckResult:
        """
        Check a PyTorch tensor for NaN/Inf values.

        Args:
            tensor: Tensor to check
            name: Name for logging/error messages
            raise_on_error: Override instance setting for this check

        Returns:
            NumericalCheckResult with validation details

        Raises:
            NumericalInstabilityError: If invalid values found and raise_on_error=True
        """
        self._check_count += 1
        should_raise = raise_on_error if raise_on_error is not None else self.raise_on_error

        # Count NaN and Inf
        nan_mask = torch.isnan(tensor)
        inf_mask = torch.isinf(tensor)
        nan_count = nan_mask.sum().item()
        inf_count = inf_mask.sum().item()
        total = tensor.numel()

        has_nan = nan_count > 0
        has_inf = inf_count > 0
        nan_ratio = nan_count / total if total > 0 else 0.0
        inf_ratio = inf_count / total if total > 0 else 0.0

        # Check against thresholds
        is_valid = nan_ratio <= self.nan_threshold and inf_ratio <= self.inf_threshold

        # Build message
        if not is_valid:
            self._issue_count += 1
            issues = []
            if has_nan:
                issues.append(f"NaN: {nan_count}/{total} ({nan_ratio:.2%})")
            if has_inf:
                issues.append(f"Inf: {inf_count}/{total} ({inf_ratio:.2%})")
            message = f"{name} contains invalid values: {', '.join(issues)}"
        else:
            message = f"{name} is valid"

        result = NumericalCheckResult(
            is_valid=is_valid,
            has_nan=has_nan,
            has_inf=has_inf,
            nan_count=int(nan_count),
            inf_count=int(inf_count),
            total_elements=total,
            message=message,
        )

        # Handle invalid values
        if not is_valid:
            if self.log_warnings:
                logger.warning(message)
            if should_raise:
                raise NumericalInstabilityError(
                    message,
                    details={
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "nan_count": nan_count,
                        "inf_count": inf_count,
                        "total": total,
                    },
                )

        return result

    def check_array(
        self,
        array: np.ndarray,
        name: str = "array",
        raise_on_error: bool | None = None,
    ) -> NumericalCheckResult:
        """
        Check a NumPy array for NaN/Inf values.

        Args:
            array: Array to check
            name: Name for logging/error messages
            raise_on_error: Override instance setting for this check

        Returns:
            NumericalCheckResult with validation details
        """
        self._check_count += 1
        should_raise = raise_on_error if raise_on_error is not None else self.raise_on_error

        nan_count = int(np.isnan(array).sum())
        inf_count = int(np.isinf(array).sum())
        total = array.size

        has_nan = nan_count > 0
        has_inf = inf_count > 0
        nan_ratio = nan_count / total if total > 0 else 0.0
        inf_ratio = inf_count / total if total > 0 else 0.0

        is_valid = nan_ratio <= self.nan_threshold and inf_ratio <= self.inf_threshold

        if not is_valid:
            self._issue_count += 1
            issues = []
            if has_nan:
                issues.append(f"NaN: {nan_count}/{total} ({nan_ratio:.2%})")
            if has_inf:
                issues.append(f"Inf: {inf_count}/{total} ({inf_ratio:.2%})")
            message = f"{name} contains invalid values: {', '.join(issues)}"
        else:
            message = f"{name} is valid"

        result = NumericalCheckResult(
            is_valid=is_valid,
            has_nan=has_nan,
            has_inf=has_inf,
            nan_count=nan_count,
            inf_count=inf_count,
            total_elements=total,
            message=message,
        )

        if not is_valid:
            if self.log_warnings:
                logger.warning(message)
            if should_raise:
                raise NumericalInstabilityError(
                    message,
                    details={
                        "name": name,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "nan_count": nan_count,
                        "inf_count": inf_count,
                    },
                )

        return result

    def check_loss(
        self,
        loss: torch.Tensor,
        raise_on_error: bool | None = None,
    ) -> NumericalCheckResult:
        """
        Check loss value for NaN/Inf.

        Args:
            loss: Loss tensor (scalar)
            raise_on_error: Override instance setting

        Returns:
            NumericalCheckResult

        Raises:
            NumericalInstabilityError: If loss is NaN or Inf
        """
        return self.check_tensor(loss, "loss", raise_on_error)

    def check_gradients(
        self,
        model: nn.Module,
        raise_on_error: bool | None = None,
    ) -> dict[str, NumericalCheckResult]:
        """
        Check all gradients in a model for NaN/Inf.

        Args:
            model: PyTorch model to check
            raise_on_error: Override instance setting

        Returns:
            Dict mapping parameter names to their check results

        Raises:
            NumericalInstabilityError: If any gradient has NaN/Inf
        """
        results: dict[str, NumericalCheckResult] = {}
        invalid_params: list[str] = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                result = self.check_tensor(
                    param.grad,
                    f"grad_{name}",
                    raise_on_error=False,  # Collect all before raising
                )
                results[name] = result
                if not result.is_valid:
                    invalid_params.append(name)

        # Raise after collecting all invalid gradients
        should_raise = raise_on_error if raise_on_error is not None else self.raise_on_error
        if invalid_params and should_raise:
            message = f"Invalid gradients in {len(invalid_params)} parameters: {invalid_params[:5]}"
            if len(invalid_params) > 5:
                message += f" (and {len(invalid_params) - 5} more)"
            raise NumericalInstabilityError(
                message,
                details={
                    "invalid_params": invalid_params,
                    "results": {k: v.message for k, v in results.items() if not v.is_valid},
                },
            )

        return results

    def check_gradient_norm(
        self,
        model: nn.Module,
        max_norm: float = 1000.0,
        raise_on_error: bool | None = None,
    ) -> tuple[float, bool]:
        """
        Check total gradient norm and detect explosion.

        Args:
            model: PyTorch model
            max_norm: Maximum allowed total gradient norm
            raise_on_error: Override instance setting

        Returns:
            Tuple of (total_norm, is_valid)
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                if not np.isfinite(param_norm):
                    should_raise = (
                        raise_on_error if raise_on_error is not None else self.raise_on_error
                    )
                    if should_raise:
                        raise NumericalInstabilityError(
                            f"Non-finite gradient norm detected: {param_norm}"
                        )
                    return float("inf"), False
                total_norm += param_norm**2

        total_norm = total_norm**0.5
        is_valid = total_norm <= max_norm

        if not is_valid:
            should_raise = raise_on_error if raise_on_error is not None else self.raise_on_error
            if self.log_warnings:
                logger.warning(f"Gradient norm {total_norm:.2f} exceeds max {max_norm:.2f}")
            if should_raise:
                raise NumericalInstabilityError(
                    f"Gradient explosion detected: norm={total_norm:.2f} > max={max_norm:.2f}"
                )

        return total_norm, is_valid

    @property
    def stats(self) -> dict[str, int]:
        """Get validation statistics."""
        return {
            "checks_performed": self._check_count,
            "issues_found": self._issue_count,
        }

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self._check_count = 0
        self._issue_count = 0


def validate_training_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> None:
    """
    Validate all training inputs before starting training.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        sample_weights: Optional sample weights

    Raises:
        NumericalInstabilityError: If any input contains NaN/Inf
    """
    validator = NumericalValidator(raise_on_error=True)

    validator.check_array(X_train, "X_train")
    validator.check_array(y_train, "y_train")
    validator.check_array(X_val, "X_val")
    validator.check_array(y_val, "y_val")

    if sample_weights is not None:
        validator.check_array(sample_weights, "sample_weights")

    logger.debug(
        f"Training inputs validated: X_train={X_train.shape}, "
        f"X_val={X_val.shape}, sample_weights={'present' if sample_weights is not None else 'none'}"
    )


__all__ = [
    "NumericalInstabilityError",
    "NumericalCheckResult",
    "NumericalValidator",
    "validate_training_inputs",
]
