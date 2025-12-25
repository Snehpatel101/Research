"""
Shared fixtures for Cross-Validation tests.

Provides:
- Synthetic time series data with DatetimeIndex
- Mock models for OOF generation
- PurgedKFold configurations
- Feature importance test data
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig


# =============================================================================
# TIME SERIES DATA FIXTURES
# =============================================================================

@pytest.fixture
def time_series_data() -> Dict[str, Any]:
    """
    Generate synthetic time series data with DatetimeIndex.

    Returns dict with:
        - X: DataFrame (1000, 20) with DatetimeIndex at 5-min intervals
        - y: Series (1000,) with labels in {-1, 0, 1}
        - weights: Series (1000,) with sample weights
        - feature_names: List of feature names
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Create DatetimeIndex (5-min bars)
    start_time = datetime(2023, 1, 2, 9, 30)  # Market open
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    # Generate features
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features).astype(np.float32),
        index=dates,
        columns=feature_names,
    )

    # Generate labels (slightly imbalanced)
    y = pd.Series(
        np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.4, 0.3]),
        index=dates,
        name="label",
    )

    # Generate sample weights
    weights = pd.Series(
        np.random.uniform(0.5, 1.5, size=n_samples).astype(np.float32),
        index=dates,
        name="weight",
    )

    return {
        "X": X,
        "y": y,
        "weights": weights,
        "feature_names": feature_names,
    }


@pytest.fixture
def small_time_series_data() -> Dict[str, Any]:
    """
    Generate small synthetic time series data for fast tests.

    Returns dict with:
        - X: DataFrame (200, 10) with DatetimeIndex
        - y: Series (200,) with labels
        - weights: Series (200,) with sample weights
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    start_time = datetime(2023, 1, 2, 9, 30)
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    feature_names = [f"feature_{i}" for i in range(n_features)]
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features).astype(np.float32),
        index=dates,
        columns=feature_names,
    )

    y = pd.Series(
        np.random.choice([-1, 0, 1], size=n_samples),
        index=dates,
        name="label",
    )

    weights = pd.Series(
        np.random.uniform(0.5, 1.5, size=n_samples).astype(np.float32),
        index=dates,
        name="weight",
    )

    return {
        "X": X,
        "y": y,
        "weights": weights,
        "feature_names": feature_names,
    }


@pytest.fixture
def correlated_features_data() -> Dict[str, Any]:
    """
    Generate data with correlated feature groups for testing clustered importance.

    Creates:
    - Group 1: 5 highly correlated features (rho > 0.8)
    - Group 2: 5 moderately correlated features (rho ~ 0.5)
    - Group 3: 5 independent features
    """
    np.random.seed(42)
    n_samples = 500
    n_features = 15

    start_time = datetime(2023, 1, 2, 9, 30)
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    # Base signals
    base_1 = np.random.randn(n_samples)
    base_2 = np.random.randn(n_samples)

    features = {}

    # Group 1: Highly correlated (add small noise)
    for i in range(5):
        noise = np.random.randn(n_samples) * 0.2
        features[f"corr_high_{i}"] = base_1 + noise

    # Group 2: Moderately correlated
    for i in range(5):
        noise = np.random.randn(n_samples) * 0.7
        features[f"corr_mod_{i}"] = base_2 + noise

    # Group 3: Independent
    for i in range(5):
        features[f"independent_{i}"] = np.random.randn(n_samples)

    X = pd.DataFrame(features, index=dates).astype(np.float32)

    # Label: depends on group 3 features (to test importance)
    y_prob = 1 / (1 + np.exp(-X["independent_0"]))
    y = pd.Series(
        np.where(y_prob > 0.6, 1, np.where(y_prob < 0.4, -1, 0)),
        index=dates,
        name="label",
    )

    return {
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "high_corr_group": [f"corr_high_{i}" for i in range(5)],
        "mod_corr_group": [f"corr_mod_{i}" for i in range(5)],
        "independent_group": [f"independent_{i}" for i in range(5)],
    }


@pytest.fixture
def label_end_times_data() -> Dict[str, Any]:
    """
    Generate data with label_end_times for testing overlapping label purging.

    Each label has an end time that is horizon bars in the future.
    """
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    horizon = 20  # Label depends on prices 20 bars ahead

    start_time = datetime(2023, 1, 2, 9, 30)
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features).astype(np.float32),
        index=dates,
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = pd.Series(
        np.random.choice([-1, 0, 1], size=n_samples),
        index=dates,
        name="label",
    )

    # Label end times: each label is resolved horizon bars later
    label_end_times = pd.Series(
        [dates[min(i + horizon, n_samples - 1)] for i in range(n_samples)],
        index=dates,
        name="label_end_time",
    )

    return {
        "X": X,
        "y": y,
        "label_end_times": label_end_times,
        "horizon": horizon,
    }


# =============================================================================
# CV CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def default_cv_config() -> PurgedKFoldConfig:
    """Default PurgedKFold configuration."""
    return PurgedKFoldConfig(
        n_splits=5,
        purge_bars=60,
        embargo_bars=100,
        min_train_size=0.3,
    )


@pytest.fixture
def small_cv_config() -> PurgedKFoldConfig:
    """Small CV config for fast tests (3 folds, minimal purge/embargo)."""
    return PurgedKFoldConfig(
        n_splits=3,
        purge_bars=10,
        embargo_bars=20,
        min_train_size=0.2,
    )


@pytest.fixture
def strict_cv_config() -> PurgedKFoldConfig:
    """Strict CV config with large purge/embargo."""
    return PurgedKFoldConfig(
        n_splits=5,
        purge_bars=100,
        embargo_bars=200,
        min_train_size=0.2,
    )


@pytest.fixture
def default_cv(default_cv_config) -> PurgedKFold:
    """PurgedKFold instance with default config."""
    return PurgedKFold(default_cv_config)


@pytest.fixture
def small_cv(small_cv_config) -> PurgedKFold:
    """PurgedKFold instance with small config for fast tests."""
    return PurgedKFold(small_cv_config)


# =============================================================================
# MOCK MODEL FIXTURES
# =============================================================================

class MockModel:
    """
    Mock model for testing OOF generation without requiring real model training.

    Provides:
    - fit(): No-op, just stores data shape
    - predict(): Returns random predictions
    - predict_proba(): Returns random probabilities
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_fitted = False
        self._n_features = None

    @property
    def model_family(self) -> str:
        return "mock"

    @property
    def requires_scaling(self) -> bool:
        return False

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> Dict[str, Any]:
        return {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        sample_weights: np.ndarray = None,
    ):
        """Mock fit that records data shape."""
        self._n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
        self.is_fitted = True

        # Return mock TrainingMetrics
        from src.models.base import TrainingMetrics
        return TrainingMetrics(
            train_loss=0.5,
            val_loss=0.6,
            train_accuracy=0.6,
            val_accuracy=0.55,
            train_f1=0.55,
            val_f1=0.5,
            epochs_trained=10,
            training_time_seconds=1.0,
            early_stopped=False,
            best_epoch=10,
            history={},
        )

    def predict(self, X: np.ndarray):
        """Return mock predictions."""
        np.random.seed(42)
        n_samples = X.shape[0]

        # Generate random probabilities
        probs = np.random.dirichlet([1, 1, 1], size=n_samples)

        # Return PredictionOutput
        from src.models.base import PredictionOutput
        return PredictionOutput(
            class_predictions=np.argmax(probs, axis=1) - 1,  # -1, 0, 1
            class_probabilities=probs,
            confidence=probs.max(axis=1),
            metadata={},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return mock probabilities."""
        np.random.seed(42)
        n_samples = X.shape[0]
        return np.random.dirichlet([1, 1, 1], size=n_samples)


@pytest.fixture
def mock_model():
    """Provide a fresh MockModel instance."""
    return MockModel()


@pytest.fixture
def mock_model_factory():
    """Factory for creating mock models."""
    def _create_model(config: Dict[str, Any] = None):
        return MockModel(config)
    return _create_model


def register_mock_model():
    """Register MockModel with ModelRegistry for testing."""
    from src.models.registry import ModelRegistry

    # Only register if not already registered
    if "mock" not in ModelRegistry._models:
        @ModelRegistry.register(name="mock", family="mock", description="Mock model for testing")
        class RegisteredMockModel(MockModel):
            pass


# =============================================================================
# MOCK DATA CONTAINER FIXTURE
# =============================================================================

@pytest.fixture
def mock_cv_data_container(time_series_data):
    """
    Create a mock TimeSeriesDataContainer for CV testing.

    Returns a mock that provides:
    - get_sklearn_arrays(split, return_df=True) -> (X, y, weights)
    - horizons, symbols, n_features attributes
    """
    mock = MagicMock()

    X = time_series_data["X"]
    y = time_series_data["y"]
    weights = time_series_data["weights"]

    def get_sklearn_arrays(split: str, return_df: bool = False):
        if return_df:
            return X, y, weights
        else:
            return X.values, y.values, weights.values

    mock.get_sklearn_arrays = get_sklearn_arrays
    mock.horizons = [5, 10, 15, 20]
    mock.symbols = ["MES"]
    mock.n_features = X.shape[1]

    return mock


# =============================================================================
# CV SPLITS FIXTURES
# =============================================================================

@pytest.fixture
def cv_splits(small_cv, small_time_series_data) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pre-computed CV splits for small data."""
    X = small_time_series_data["X"]
    return list(small_cv.split(X))


# =============================================================================
# TEMPORARY DIRECTORY FIXTURES
# =============================================================================

@pytest.fixture
def tmp_cv_output_dir(tmp_path):
    """Temporary directory for CV output files."""
    output_dir = tmp_path / "cv_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
