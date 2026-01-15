"""Tests for numerical stability validation in neural network training."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.neural.numerical_stability import (
    NumericalCheckResult,
    NumericalInstabilityError,
    NumericalValidator,
    validate_training_inputs,
)


class TestNumericalCheckResult:
    """Tests for NumericalCheckResult dataclass."""

    def test_valid_result(self):
        """Test valid result properties."""
        result = NumericalCheckResult(
            is_valid=True,
            has_nan=False,
            has_inf=False,
            nan_count=0,
            inf_count=0,
            total_elements=100,
            message="tensor is valid",
        )
        assert result.is_valid
        assert result.nan_ratio == 0.0
        assert result.inf_ratio == 0.0

    def test_nan_ratio(self):
        """Test NaN ratio calculation."""
        result = NumericalCheckResult(
            is_valid=False,
            has_nan=True,
            has_inf=False,
            nan_count=10,
            inf_count=0,
            total_elements=100,
            message="tensor has NaN",
        )
        assert result.nan_ratio == 0.1
        assert result.inf_ratio == 0.0

    def test_inf_ratio(self):
        """Test Inf ratio calculation."""
        result = NumericalCheckResult(
            is_valid=False,
            has_nan=False,
            has_inf=True,
            nan_count=0,
            inf_count=5,
            total_elements=100,
            message="tensor has Inf",
        )
        assert result.nan_ratio == 0.0
        assert result.inf_ratio == 0.05


class TestNumericalValidator:
    """Tests for NumericalValidator class."""

    def test_valid_tensor(self):
        """Test validation of valid tensor."""
        validator = NumericalValidator(raise_on_error=True)
        tensor = torch.randn(10, 10)
        result = validator.check_tensor(tensor, "test_tensor")
        assert result.is_valid
        assert not result.has_nan
        assert not result.has_inf

    def test_tensor_with_nan(self):
        """Test detection of NaN in tensor."""
        validator = NumericalValidator(raise_on_error=False)
        tensor = torch.randn(10, 10)
        tensor[5, 5] = float("nan")
        result = validator.check_tensor(tensor, "test_tensor")
        assert not result.is_valid
        assert result.has_nan
        assert result.nan_count == 1

    def test_tensor_with_inf(self):
        """Test detection of Inf in tensor."""
        validator = NumericalValidator(raise_on_error=False)
        tensor = torch.randn(10, 10)
        tensor[0, 0] = float("inf")
        tensor[0, 1] = float("-inf")
        result = validator.check_tensor(tensor, "test_tensor")
        assert not result.is_valid
        assert result.has_inf
        assert result.inf_count == 2

    def test_raise_on_nan(self):
        """Test that NumericalInstabilityError is raised for NaN."""
        validator = NumericalValidator(raise_on_error=True)
        tensor = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(NumericalInstabilityError) as exc_info:
            validator.check_tensor(tensor, "test_tensor")
        assert "NaN" in str(exc_info.value)

    def test_raise_on_inf(self):
        """Test that NumericalInstabilityError is raised for Inf."""
        validator = NumericalValidator(raise_on_error=True)
        tensor = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(NumericalInstabilityError) as exc_info:
            validator.check_tensor(tensor, "test_tensor")
        assert "Inf" in str(exc_info.value)

    def test_valid_array(self):
        """Test validation of valid NumPy array."""
        validator = NumericalValidator(raise_on_error=True)
        array = np.random.randn(10, 10)
        result = validator.check_array(array, "test_array")
        assert result.is_valid

    def test_array_with_nan(self):
        """Test detection of NaN in NumPy array."""
        validator = NumericalValidator(raise_on_error=False)
        array = np.random.randn(10, 10)
        array[5, 5] = np.nan
        result = validator.check_array(array, "test_array")
        assert not result.is_valid
        assert result.has_nan

    def test_loss_check_valid(self):
        """Test loss validation with valid value."""
        validator = NumericalValidator(raise_on_error=True)
        loss = torch.tensor(0.5)
        result = validator.check_loss(loss)
        assert result.is_valid

    def test_loss_check_nan(self):
        """Test loss validation with NaN."""
        validator = NumericalValidator(raise_on_error=True)
        loss = torch.tensor(float("nan"))
        with pytest.raises(NumericalInstabilityError):
            validator.check_loss(loss)

    def test_gradient_check(self):
        """Test gradient validation on a simple model."""
        validator = NumericalValidator(raise_on_error=True)

        # Create simple model and compute gradients
        model = nn.Linear(10, 3)
        x = torch.randn(5, 10)
        y = torch.randint(0, 3, (5,))
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()

        # Check gradients
        results = validator.check_gradients(model)
        assert len(results) > 0
        for name, result in results.items():
            assert result.is_valid, f"Gradient {name} should be valid"

    def test_gradient_check_with_nan_gradients(self):
        """Test gradient validation detects NaN gradients."""
        validator = NumericalValidator(raise_on_error=True)

        # Create model and manually set NaN gradient
        model = nn.Linear(10, 3)
        model.weight.grad = torch.full_like(model.weight, float("nan"))

        with pytest.raises(NumericalInstabilityError) as exc_info:
            validator.check_gradients(model)
        assert "gradient" in str(exc_info.value).lower()

    def test_gradient_norm_check(self):
        """Test gradient norm checking."""
        validator = NumericalValidator(raise_on_error=False)

        model = nn.Linear(10, 3)
        x = torch.randn(5, 10)
        y = torch.randint(0, 3, (5,))
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()

        norm, is_valid = validator.check_gradient_norm(model, max_norm=1000.0)
        assert is_valid
        assert norm > 0

    def test_gradient_norm_explosion(self):
        """Test gradient explosion detection."""
        validator = NumericalValidator(raise_on_error=True)

        # Create model with very large gradients
        model = nn.Linear(10, 3)
        model.weight.grad = torch.full_like(model.weight, 1000.0)

        with pytest.raises(NumericalInstabilityError) as exc_info:
            validator.check_gradient_norm(model, max_norm=1.0)
        assert "explosion" in str(exc_info.value).lower()

    def test_stats_tracking(self):
        """Test validation statistics tracking."""
        validator = NumericalValidator(raise_on_error=False)

        # Perform some checks
        validator.check_tensor(torch.randn(10), "t1")
        validator.check_tensor(torch.randn(10), "t2")

        nan_tensor = torch.randn(10)
        nan_tensor[0] = float("nan")
        validator.check_tensor(nan_tensor, "t3")

        stats = validator.stats
        assert stats["checks_performed"] == 3
        assert stats["issues_found"] == 1

    def test_stats_reset(self):
        """Test statistics reset."""
        validator = NumericalValidator(raise_on_error=False)
        validator.check_tensor(torch.randn(10), "t1")
        validator.reset_stats()
        assert validator.stats["checks_performed"] == 0
        assert validator.stats["issues_found"] == 0


class TestValidateTrainingInputs:
    """Tests for validate_training_inputs function."""

    def test_valid_inputs(self):
        """Test validation of valid training inputs."""
        X_train = np.random.randn(100, 10, 5)
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.randn(20, 10, 5)
        y_val = np.random.randint(0, 3, 20)

        # Should not raise
        validate_training_inputs(X_train, y_train, X_val, y_val)

    def test_nan_in_x_train(self):
        """Test detection of NaN in X_train."""
        X_train = np.random.randn(100, 10, 5)
        X_train[50, 5, 2] = np.nan
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.randn(20, 10, 5)
        y_val = np.random.randint(0, 3, 20)

        with pytest.raises(NumericalInstabilityError) as exc_info:
            validate_training_inputs(X_train, y_train, X_val, y_val)
        assert "X_train" in str(exc_info.value)

    def test_inf_in_x_val(self):
        """Test detection of Inf in X_val."""
        X_train = np.random.randn(100, 10, 5)
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.randn(20, 10, 5)
        X_val[10, 3, 1] = np.inf
        y_val = np.random.randint(0, 3, 20)

        with pytest.raises(NumericalInstabilityError) as exc_info:
            validate_training_inputs(X_train, y_train, X_val, y_val)
        assert "X_val" in str(exc_info.value)

    def test_sample_weights_validation(self):
        """Test validation of sample weights."""
        X_train = np.random.randn(100, 10, 5)
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.randn(20, 10, 5)
        y_val = np.random.randint(0, 3, 20)
        sample_weights = np.ones(100)
        sample_weights[50] = np.nan

        with pytest.raises(NumericalInstabilityError) as exc_info:
            validate_training_inputs(X_train, y_train, X_val, y_val, sample_weights)
        assert "sample_weights" in str(exc_info.value)


class TestNumericalInstabilityError:
    """Tests for NumericalInstabilityError exception."""

    def test_error_message(self):
        """Test error message is preserved."""
        error = NumericalInstabilityError("Test error message")
        assert str(error) == "Test error message"

    def test_error_details(self):
        """Test error details are preserved."""
        details = {"name": "tensor", "shape": [10, 10]}
        error = NumericalInstabilityError("Test error", details=details)
        assert error.details == details

    def test_error_default_details(self):
        """Test default empty details."""
        error = NumericalInstabilityError("Test error")
        assert error.details == {}
