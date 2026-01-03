"""
Shared fixtures for integration tests.

Imports fixtures from models conftest and adds integration-specific ones.
"""
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest


# =============================================================================
# MOCK DATA CONTAINER
# =============================================================================

def create_mock_container(
    n_train: int = 100,
    n_val: int = 20,
    n_features: int = 10,
    seq_len: int = 10,
    horizon: int = 20,
) -> MagicMock:
    """
    Create a mock TimeSeriesDataContainer for testing.

    Can be used as a function or via the fixture.
    """
    np.random.seed(42)

    # Generate tabular data
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.random.choice([-1, 0, 1], size=n_train)
    weights_train = np.ones(n_train, dtype=np.float32)

    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = np.random.choice([-1, 0, 1], size=n_val)
    weights_val = np.ones(n_val, dtype=np.float32)

    # Generate sequence data
    X_train_seq = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
    X_val_seq = np.random.randn(n_val, seq_len, n_features).astype(np.float32)

    mock = MagicMock()

    # Mock tabular data access
    def get_sklearn_arrays(split: str, return_df: bool = False):
        import pandas as pd
        if split == "train":
            data = (X_train, y_train, weights_train)
        elif split == "val":
            data = (X_val, y_val, weights_val)
        else:
            data = (X_val, y_val, weights_val)

        if return_df:
            # Return as DataFrame/Series like real container
            feature_cols = [f"feature_{i}" for i in range(data[0].shape[1])]
            return (
                pd.DataFrame(data[0], columns=feature_cols),
                pd.Series(data[1], name="label"),
                pd.Series(data[2], name="weight"),
            )
        return data

    mock.get_sklearn_arrays = get_sklearn_arrays

    # Mock get_array method (used by some tests)
    def get_array(split: str, array_type: str):
        if split == "train":
            if array_type == "features":
                return X_train
            elif array_type == "labels":
                return y_train
            elif array_type == "weights":
                return weights_train
        else:
            if array_type == "features":
                return X_val
            elif array_type == "labels":
                return y_val
            elif array_type == "weights":
                return weights_val

    mock.get_array = get_array

    # Mock sequence data access
    class MockSequenceDataset:
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X
            self.y = y
            self.weights = np.ones(len(y), dtype=np.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.weights[idx]

    def get_pytorch_sequences(split: str, seq_len: int = 30, symbol_isolated: bool = True):
        if split == "train":
            return MockSequenceDataset(X_train_seq, y_train)
        else:
            return MockSequenceDataset(X_val_seq, y_val)

    mock.get_pytorch_sequences = get_pytorch_sequences

    # Add metadata
    mock.horizons = [5, 10, 15, 20]
    mock.symbols = ["MES", "MGC"]
    mock.n_features = n_features
    mock.horizon = horizon

    return mock


@pytest.fixture
def mock_data_container() -> MagicMock:
    """Provide a mock TimeSeriesDataContainer for testing."""
    return create_mock_container(
        n_train=200,
        n_val=50,
        n_features=20,
    )


@pytest.fixture
def mock_data_container_factory():
    """Factory to create mock containers for different horizons."""
    def _factory(horizon: int = 20, n_train: int = 200, n_val: int = 50):
        return create_mock_container(
            n_train=n_train,
            n_val=n_val,
            n_features=20,
            horizon=horizon,
        )

    return _factory


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Provide a temporary directory for trainer output."""
    output_dir = tmp_path / "experiments" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
