"""
Shared fixtures for error handling tests.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def invalid_model_path() -> Path:
    """Return a path that doesn't exist."""
    return Path("/nonexistent/path/to/model")


@pytest.fixture
def valid_tabular_data() -> dict:
    """Valid tabular data for error handling tests."""
    np.random.seed(42)
    n_samples = 100

    return {
        'X_train': np.random.randn(n_samples, 10).astype(np.float32),
        'y_train': np.random.choice([-1, 0, 1], n_samples),
        'X_val': np.random.randn(20, 10).astype(np.float32),
        'y_val': np.random.choice([-1, 0, 1], 20),
    }


@pytest.fixture
def data_with_nan() -> dict:
    """Data containing NaN values."""
    np.random.seed(42)
    n_samples = 100

    X = np.random.randn(n_samples, 10).astype(np.float32)
    X[50, 5] = np.nan  # Introduce NaN

    return {
        'X_train': X,
        'y_train': np.random.choice([-1, 0, 1], n_samples),
        'X_val': np.random.randn(20, 10).astype(np.float32),
        'y_val': np.random.choice([-1, 0, 1], 20),
    }


@pytest.fixture
def data_with_inf() -> dict:
    """Data containing infinity values."""
    np.random.seed(42)
    n_samples = 100

    X = np.random.randn(n_samples, 10).astype(np.float32)
    X[50, 5] = np.inf  # Introduce infinity

    return {
        'X_train': X,
        'y_train': np.random.choice([-1, 0, 1], n_samples),
        'X_val': np.random.randn(20, 10).astype(np.float32),
        'y_val': np.random.choice([-1, 0, 1], 20),
    }
