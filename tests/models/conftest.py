"""
Shared fixtures for Model Factory tests.

Provides:
- Synthetic classification data (tabular and sequential)
- Mock TimeSeriesDataContainer
- GPU availability skip markers
- Pre-trained model fixtures
"""
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest


# =============================================================================
# SKIP MARKERS
# =============================================================================

def _torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    """Check if CUDA is available for PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _lightgbm_available() -> bool:
    """Check if LightGBM is installed."""
    try:
        import lightgbm
        return True
    except ImportError:
        return False


def _catboost_available() -> bool:
    """Check if CatBoost is installed."""
    try:
        import catboost
        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch not installed"
)

requires_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA not available"
)

requires_lightgbm = pytest.mark.skipif(
    not _lightgbm_available(),
    reason="LightGBM not installed"
)

requires_catboost = pytest.mark.skipif(
    not _catboost_available(),
    reason="CatBoost not installed"
)


# =============================================================================
# DATA FIXTURES - TABULAR
# =============================================================================

@pytest.fixture
def synthetic_tabular_data() -> Dict[str, np.ndarray]:
    """
    Generate synthetic 3-class tabular classification data.

    Returns dict with:
        - X_train: (500, 30) training features
        - y_train: (500,) training labels in {-1, 0, 1}
        - weights: (500,) sample weights
        - X_val: (100,) validation features
        - y_val: (100,) validation labels
        - X_test: (100,) test features
        - y_test: (100,) test labels
    """
    np.random.seed(42)
    n_train, n_val, n_test, n_features = 500, 100, 100, 30

    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.random.choice([-1, 0, 1], size=n_train)
    weights = np.random.uniform(0.5, 1.5, size=n_train).astype(np.float32)

    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = np.random.choice([-1, 0, 1], size=n_val)

    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    y_test = np.random.choice([-1, 0, 1], size=n_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "weights": weights,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def small_tabular_data() -> Dict[str, np.ndarray]:
    """
    Generate small synthetic data for fast tests.

    Returns dict with:
        - X_train: (100, 10) training features
        - y_train: (100,) training labels
        - X_val: (20, 10) validation features
        - y_val: (20,) validation labels
    """
    np.random.seed(42)
    n_train, n_val, n_features = 100, 20, 10

    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.random.choice([-1, 0, 1], size=n_train)

    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = np.random.choice([-1, 0, 1], size=n_val)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }


# =============================================================================
# DATA FIXTURES - SEQUENTIAL
# =============================================================================

@pytest.fixture
def synthetic_sequence_data() -> Dict[str, np.ndarray]:
    """
    Generate synthetic 3-class sequential classification data.

    Returns dict with:
        - X_train: (200, 30, 20) training sequences
        - y_train: (200,) training labels in {-1, 0, 1}
        - weights: (200,) sample weights
        - X_val: (50, 30, 20) validation sequences
        - y_val: (50,) validation labels
        - X_test: (50, 30, 20) test sequences
        - y_test: (50,) test labels
    """
    np.random.seed(42)
    n_train, n_val, n_test = 200, 50, 50
    seq_len, n_features = 30, 20

    X_train = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
    y_train = np.random.choice([-1, 0, 1], size=n_train)
    weights = np.random.uniform(0.5, 1.5, size=n_train).astype(np.float32)

    X_val = np.random.randn(n_val, seq_len, n_features).astype(np.float32)
    y_val = np.random.choice([-1, 0, 1], size=n_val)

    X_test = np.random.randn(n_test, seq_len, n_features).astype(np.float32)
    y_test = np.random.choice([-1, 0, 1], size=n_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "weights": weights,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def small_sequence_data() -> Dict[str, np.ndarray]:
    """
    Generate small sequential data for fast tests.

    Returns dict with:
        - X_train: (50, 10, 8) training sequences
        - y_train: (50,) training labels
        - X_val: (10, 10, 8) validation sequences
        - y_val: (10,) validation labels
    """
    np.random.seed(42)
    n_train, n_val = 50, 10
    seq_len, n_features = 10, 8

    X_train = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
    y_train = np.random.choice([-1, 0, 1], size=n_train)

    X_val = np.random.randn(n_val, seq_len, n_features).astype(np.float32)
    y_val = np.random.choice([-1, 0, 1], size=n_val)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }


# =============================================================================
# MOCK DATA CONTAINER
# =============================================================================

def create_mock_container(
    n_train: int = 100,
    n_val: int = 20,
    n_test: int = 20,
    n_features: int = 10,
    seq_len: int = 10,
    horizon: int = 20,
) -> MagicMock:
    """
    Create a mock TimeSeriesDataContainer for testing.

    Can be used as a function or via the fixture.
    Supports both array and DataFrame return types for feature selection integration.
    """
    import pandas as pd

    np.random.seed(42)

    # Generate feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Generate tabular data
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.random.choice([-1, 0, 1], size=n_train)
    weights_train = np.ones(n_train, dtype=np.float32)

    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = np.random.choice([-1, 0, 1], size=n_val)
    weights_val = np.ones(n_val, dtype=np.float32)

    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    y_test = np.random.choice([-1, 0, 1], size=n_test)
    weights_test = np.ones(n_test, dtype=np.float32)

    # Create DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    y_train_series = pd.Series(y_train)
    y_val_series = pd.Series(y_val)
    y_test_series = pd.Series(y_test)

    weights_train_series = pd.Series(weights_train)
    weights_val_series = pd.Series(weights_val)
    weights_test_series = pd.Series(weights_test)

    # Generate sequence data
    X_train_seq = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
    X_val_seq = np.random.randn(n_val, seq_len, n_features).astype(np.float32)
    X_test_seq = np.random.randn(n_test, seq_len, n_features).astype(np.float32)

    mock = MagicMock()

    # Mock tabular data access - supports return_df parameter
    def get_sklearn_arrays(split: str, return_df: bool = False):
        if split == "train":
            if return_df:
                return (X_train_df, y_train_series, weights_train_series)
            return (X_train, y_train, weights_train)
        elif split == "val":
            if return_df:
                return (X_val_df, y_val_series, weights_val_series)
            return (X_val, y_val, weights_val)
        else:  # test
            if return_df:
                return (X_test_df, y_test_series, weights_test_series)
            return (X_test, y_test, weights_test)

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
        elif split == "val":
            if array_type == "features":
                return X_val
            elif array_type == "labels":
                return y_val
            elif array_type == "weights":
                return weights_val
        else:  # test
            if array_type == "features":
                return X_test
            elif array_type == "labels":
                return y_test
            elif array_type == "weights":
                return weights_test

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
        elif split == "val":
            return MockSequenceDataset(X_val_seq, y_val)
        else:
            return MockSequenceDataset(X_test_seq, y_test)

    mock.get_pytorch_sequences = get_pytorch_sequences

    # Mock get_label_end_times (returns None by default)
    mock.get_label_end_times = MagicMock(return_value=None)

    # Add metadata
    mock.horizons = [5, 10, 15, 20]
    mock.symbols = ["MES", "MGC"]
    mock.n_features = n_features
    mock.horizon = horizon
    mock.feature_columns = feature_names

    return mock


@pytest.fixture
def mock_data_container(small_tabular_data, small_sequence_data):
    """
    Create a mock TimeSeriesDataContainer for testing Trainer.

    Provides:
        - get_sklearn_arrays(split) -> (X, y, weights)
        - get_pytorch_sequences(split, seq_len, symbol_isolated) -> Dataset
        - get_array(split, array_type) -> array
    """
    return create_mock_container(
        n_train=len(small_tabular_data["y_train"]),
        n_val=len(small_tabular_data["y_val"]),
        n_features=small_tabular_data["X_train"].shape[1],
    )


# =============================================================================
# TEMPORARY DIRECTORY FIXTURES
# =============================================================================

@pytest.fixture
def tmp_model_dir(tmp_path) -> Path:
    """Provide a temporary directory for model save/load tests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Provide a temporary directory for trainer output."""
    output_dir = tmp_path / "experiments" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# MODEL CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def fast_xgboost_config() -> Dict[str, Any]:
    """Fast XGBoost config for tests."""
    return {
        "n_estimators": 10,
        "max_depth": 3,
        "early_stopping_rounds": 3,
        "verbosity": 0,
        "use_gpu": False,
    }


@pytest.fixture
def fast_lightgbm_config() -> Dict[str, Any]:
    """Fast LightGBM config for tests."""
    return {
        "n_estimators": 10,
        "max_depth": 3,
        "early_stopping_rounds": 3,
        "verbosity": -1,
        "use_gpu": False,
    }


@pytest.fixture
def fast_catboost_config() -> Dict[str, Any]:
    """Fast CatBoost config for tests."""
    return {
        "iterations": 10,
        "depth": 3,
        "early_stopping_rounds": 3,
        "verbose": False,
        "use_gpu": False,
    }


@pytest.fixture
def fast_lstm_config() -> Dict[str, Any]:
    """Fast LSTM config for tests."""
    return {
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_size": 32,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "mixed_precision": False,
        "num_workers": 0,
    }


@pytest.fixture
def fast_gru_config() -> Dict[str, Any]:
    """Fast GRU config for tests."""
    return {
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_size": 32,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "mixed_precision": False,
        "num_workers": 0,
    }


@pytest.fixture
def fast_tcn_config() -> Dict[str, Any]:
    """Fast TCN config for tests."""
    return {
        "num_channels": [8, 8],
        "kernel_size": 2,
        "dropout": 0.0,
        "batch_size": 32,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "mixed_precision": False,
        "num_workers": 0,
    }


# =============================================================================
# TRAINED MODEL FIXTURES
# =============================================================================

@pytest.fixture
def trained_xgboost(small_tabular_data, fast_xgboost_config):
    """Provide a trained XGBoost model."""
    from src.models.boosting import XGBoostModel

    model = XGBoostModel(config=fast_xgboost_config)
    model.fit(
        small_tabular_data["X_train"],
        small_tabular_data["y_train"],
        small_tabular_data["X_val"],
        small_tabular_data["y_val"],
    )
    return model


@pytest.fixture
def trained_lightgbm(small_tabular_data, fast_lightgbm_config):
    """Provide a trained LightGBM model."""
    if not _lightgbm_available():
        pytest.skip("LightGBM not installed")

    from src.models.boosting import LightGBMModel

    model = LightGBMModel(config=fast_lightgbm_config)
    model.fit(
        small_tabular_data["X_train"],
        small_tabular_data["y_train"],
        small_tabular_data["X_val"],
        small_tabular_data["y_val"],
    )
    return model


@pytest.fixture
def trained_catboost(small_tabular_data, fast_catboost_config):
    """Provide a trained CatBoost model."""
    if not _catboost_available():
        pytest.skip("CatBoost not installed")

    from src.models.boosting import CatBoostModel

    model = CatBoostModel(config=fast_catboost_config)
    model.fit(
        small_tabular_data["X_train"],
        small_tabular_data["y_train"],
        small_tabular_data["X_val"],
        small_tabular_data["y_val"],
    )
    return model


@pytest.fixture
def trained_lstm(small_sequence_data, fast_lstm_config):
    """Provide a trained LSTM model."""
    if not _torch_available():
        pytest.skip("PyTorch not installed")

    from src.models.neural import LSTMModel

    model = LSTMModel(config=fast_lstm_config)
    model.fit(
        small_sequence_data["X_train"],
        small_sequence_data["y_train"],
        small_sequence_data["X_val"],
        small_sequence_data["y_val"],
    )
    return model


@pytest.fixture
def trained_gru(small_sequence_data, fast_gru_config):
    """Provide a trained GRU model."""
    if not _torch_available():
        pytest.skip("PyTorch not installed")

    from src.models.neural import GRUModel

    model = GRUModel(config=fast_gru_config)
    model.fit(
        small_sequence_data["X_train"],
        small_sequence_data["y_train"],
        small_sequence_data["X_val"],
        small_sequence_data["y_val"],
    )
    return model


@pytest.fixture
def trained_tcn(small_sequence_data, fast_tcn_config):
    """Provide a trained TCN model."""
    if not _torch_available():
        pytest.skip("PyTorch not installed")

    from src.models.neural import TCNModel

    model = TCNModel(config=fast_tcn_config)
    model.fit(
        small_sequence_data["X_train"],
        small_sequence_data["y_train"],
        small_sequence_data["X_val"],
        small_sequence_data["y_val"],
    )
    return model
