"""
Smoke tests for all 12 ML Factory models.

Tests that every registered model can:
1. Be instantiated via ModelRegistry
2. Accept synthetic data with correct shape
3. Train (fit) without errors
4. Generate predictions (predict) that return valid PredictionResult
5. Produce outputs with correct shapes and value ranges

Uses minimal data sizes for speed. Not testing accuracy - just that the
pipeline works end-to-end for each model.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from src.models import ModelRegistry
from src.models.base import BaseModel, PredictionResult, TrainingMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

N_SAMPLES_TRAIN = 200
N_SAMPLES_VAL = 60
N_FEATURES = 10
SEQ_LEN = 20
N_TIMEFRAMES = 2
N_CLASSES = 3


def _make_labels(n: int) -> np.ndarray:
    """Generate random 3-class labels (-1, 0, 1) matching ML Factory convention."""
    rng = np.random.RandomState(42)
    return rng.choice([-1, 0, 1], size=n)


def _make_tabular(n: int) -> np.ndarray:
    """2D data: (n_samples, n_features)."""
    rng = np.random.RandomState(42)
    return rng.randn(n, N_FEATURES).astype(np.float32)


def _make_sequential(n: int) -> np.ndarray:
    """3D data: (n_samples, seq_len, n_features)."""
    rng = np.random.RandomState(42)
    return rng.randn(n, SEQ_LEN, N_FEATURES).astype(np.float32)


def _make_4d(n: int) -> np.ndarray:
    """4D data: (n_samples, n_timeframes, seq_len, n_features)."""
    rng = np.random.RandomState(42)
    return rng.randn(n, N_TIMEFRAMES, SEQ_LEN, N_FEATURES).astype(np.float32)


def _get_data_for_model(model: BaseModel, n: int) -> np.ndarray:
    """Return synthetic data with the right shape for the model."""
    if model.requires_4d:
        return _make_4d(n)
    elif model.requires_sequences:
        return _make_sequential(n)
    else:
        return _make_tabular(n)


# ---------------------------------------------------------------------------
# Minimal configs to make training fast
# ---------------------------------------------------------------------------

FAST_CONFIGS: dict[str, dict] = {
    # Boosting - minimal iterations
    "xgboost": {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.3},
    "lightgbm": {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.3},
    "catboost": {"iterations": 5, "depth": 3, "learning_rate": 0.3, "verbose": 0},
    # Neural RNN - minimal epochs/size
    "lstm": {"hidden_size": 16, "num_layers": 1, "max_epochs": 2, "batch_size": 32},
    "gru": {"hidden_size": 16, "num_layers": 1, "max_epochs": 2, "batch_size": 32},
    # Neural CNN
    "tcn": {"hidden_size": 16, "num_layers": 1, "max_epochs": 2, "batch_size": 32},
    "inceptiontime": {"n_filters": 8, "max_epochs": 2, "batch_size": 32},
    "resnet1d": {"n_filters": 8, "max_epochs": 2, "batch_size": 32},
    # Transformers
    "patchtst": {
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "max_epochs": 2,
        "batch_size": 32,
        "patch_len": 4,
    },
    "itransformer": {
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "max_epochs": 2,
        "batch_size": 32,
    },
    "tft": {
        "hidden_size": 16,
        "n_heads": 2,
        "num_layers": 1,
        "max_epochs": 2,
        "batch_size": 32,
    },
    # MLP
    "nbeats": {
        "hidden_size": 16,
        "n_stacks": 1,
        "n_blocks": 1,
        "max_epochs": 2,
        "batch_size": 32,
    },
    # Classical
    "random_forest": {"n_estimators": 5, "max_depth": 3},
    "logistic": {"max_iter": 50},
    "svm": {"max_iter": 50},
}

# ---------------------------------------------------------------------------
# The 12 production models (+ 3 classical baselines)
# ---------------------------------------------------------------------------

PRODUCTION_MODELS = [
    "xgboost",
    "lightgbm",
    "catboost",
    "lstm",
    "gru",
    "tcn",
    "inceptiontime",
    "resnet1d",
    "patchtst",
    "itransformer",
    "tft",
    "nbeats",
]

CLASSICAL_MODELS = [
    "random_forest",
    "logistic",
    "svm",
]

ALL_MODELS = PRODUCTION_MODELS + CLASSICAL_MODELS


# ---------------------------------------------------------------------------
# Test: Registry lists all expected models
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    """Verify all models are registered."""

    def test_all_production_models_registered(self):
        """All 12 production models should be in the registry."""
        for name in PRODUCTION_MODELS:
            assert ModelRegistry.is_registered(name), f"Model '{name}' not registered"

    def test_list_models_has_families(self):
        """Registry should have boosting, neural, and other families."""
        families = ModelRegistry.list_models()
        assert "boosting" in families, f"Missing 'boosting' family. Got: {list(families.keys())}"

    def test_model_count(self):
        """Registry should have at least 12 models."""
        count = ModelRegistry.count()
        assert count >= 12, f"Expected >= 12 models, got {count}"


# ---------------------------------------------------------------------------
# Parametrized smoke tests
# ---------------------------------------------------------------------------


def _get_available_models():
    """Return list of models that can actually be instantiated."""
    available = []
    for name in ALL_MODELS:
        if ModelRegistry.is_registered(name) and ModelRegistry.is_available(name):
            available.append(name)
        elif ModelRegistry.is_registered(name):
            # Registered but not available (missing dependency like catboost)
            available.append(pytest.param(name, marks=pytest.mark.skip(
                reason=f"{name} registered but dependencies not available"
            )))
    return available


@pytest.fixture(params=_get_available_models())
def model_name(request):
    """Parametrize over all available models."""
    return request.param


class TestModelSmoke:
    """Smoke test every available model: instantiate, fit, predict."""

    def test_instantiate(self, model_name: str):
        """Model can be created from the registry."""
        config = FAST_CONFIGS.get(model_name, {})
        model = ModelRegistry.create(model_name, config=config)
        assert isinstance(model, BaseModel)
        assert not model.is_fitted

    def test_properties(self, model_name: str):
        """Model reports its properties correctly."""
        model = ModelRegistry.create(model_name)
        assert isinstance(model.model_family, str)
        assert isinstance(model.requires_scaling, bool)
        assert isinstance(model.requires_sequences, bool)
        assert isinstance(model.requires_4d, bool)
        assert isinstance(model.get_default_config(), dict)

    def test_fit_and_predict(self, model_name: str):
        """Model can fit on synthetic data and produce valid predictions."""
        config = FAST_CONFIGS.get(model_name, {})
        model = ModelRegistry.create(model_name, config=config)

        # Generate data
        X_train = _get_data_for_model(model, N_SAMPLES_TRAIN)
        y_train = _make_labels(N_SAMPLES_TRAIN)
        X_val = _get_data_for_model(model, N_SAMPLES_VAL)
        y_val = _make_labels(N_SAMPLES_VAL)

        # Fit
        t0 = time.time()
        metrics = model.fit(X_train, y_train, X_val, y_val)
        fit_time = time.time() - t0
        logger.info(f"{model_name} fit in {fit_time:.1f}s")

        # Check training metrics
        assert isinstance(metrics, TrainingMetrics), (
            f"fit() should return TrainingMetrics, got {type(metrics)}"
        )
        assert metrics.epochs_trained >= 0
        assert metrics.training_time_seconds >= 0
        assert model.is_fitted

        # Predict
        result = model.predict(X_val)

        # Check prediction result
        assert isinstance(result, PredictionResult), (
            f"predict() should return PredictionResult, got {type(result)}"
        )

        # Check predictions shape
        assert result.class_predictions is not None, "class_predictions should not be None"
        preds = np.asarray(result.class_predictions)
        assert preds.shape[0] == N_SAMPLES_VAL, (
            f"Expected {N_SAMPLES_VAL} predictions, got {preds.shape[0]}"
        )

        # Check probabilities if available
        if result.class_probabilities is not None:
            probs = np.asarray(result.class_probabilities)
            assert probs.shape[0] == N_SAMPLES_VAL
            # Probabilities should be in [0, 1]
            assert np.all(probs >= -0.01), f"Probabilities below 0: {probs.min()}"
            assert np.all(probs <= 1.01), f"Probabilities above 1: {probs.max()}"


# ---------------------------------------------------------------------------
# Targeted family tests (for parallel team execution)
# ---------------------------------------------------------------------------


class TestBoostingModels:
    """Smoke test boosting models specifically."""

    @pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
    def test_boosting_fit_predict(self, name: str):
        config = FAST_CONFIGS[name]
        model = ModelRegistry.create(name, config=config)
        X_train, y_train = _make_tabular(N_SAMPLES_TRAIN), _make_labels(N_SAMPLES_TRAIN)
        X_val, y_val = _make_tabular(N_SAMPLES_VAL), _make_labels(N_SAMPLES_VAL)
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert isinstance(metrics, TrainingMetrics)
        result = model.predict(X_val)
        assert isinstance(result, PredictionResult)

    def test_catboost_if_available(self):
        if not ModelRegistry.is_available("catboost"):
            pytest.skip("CatBoost not installed")
        config = FAST_CONFIGS["catboost"]
        model = ModelRegistry.create("catboost", config=config)
        X_train, y_train = _make_tabular(N_SAMPLES_TRAIN), _make_labels(N_SAMPLES_TRAIN)
        X_val, y_val = _make_tabular(N_SAMPLES_VAL), _make_labels(N_SAMPLES_VAL)
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert isinstance(metrics, TrainingMetrics)


class TestNeuralRNNModels:
    """Smoke test LSTM and GRU."""

    @pytest.mark.parametrize("name", ["lstm", "gru"])
    def test_rnn_fit_predict(self, name: str):
        config = FAST_CONFIGS[name]
        model = ModelRegistry.create(name, config=config)
        X_train, y_train = _make_sequential(N_SAMPLES_TRAIN), _make_labels(N_SAMPLES_TRAIN)
        X_val, y_val = _make_sequential(N_SAMPLES_VAL), _make_labels(N_SAMPLES_VAL)
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert isinstance(metrics, TrainingMetrics)
        result = model.predict(X_val)
        assert isinstance(result, PredictionResult)


class TestNeuralCNNModels:
    """Smoke test TCN, InceptionTime, ResNet1D."""

    @pytest.mark.parametrize("name", ["tcn", "inceptiontime", "resnet1d"])
    def test_cnn_fit_predict(self, name: str):
        if not ModelRegistry.is_available(name):
            pytest.skip(f"{name} not available")
        config = FAST_CONFIGS[name]
        model = ModelRegistry.create(name, config=config)
        X_train, y_train = _make_sequential(N_SAMPLES_TRAIN), _make_labels(N_SAMPLES_TRAIN)
        X_val, y_val = _make_sequential(N_SAMPLES_VAL), _make_labels(N_SAMPLES_VAL)
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert isinstance(metrics, TrainingMetrics)
        result = model.predict(X_val)
        assert isinstance(result, PredictionResult)


class TestTransformerModels:
    """Smoke test PatchTST, iTransformer, TFT."""

    @pytest.mark.parametrize("name", ["patchtst", "itransformer", "tft"])
    def test_transformer_fit_predict(self, name: str):
        if not ModelRegistry.is_available(name):
            pytest.skip(f"{name} not available")
        config = FAST_CONFIGS[name]
        model = ModelRegistry.create(name, config=config)
        X_train, y_train = _get_data_for_model(model, N_SAMPLES_TRAIN), _make_labels(N_SAMPLES_TRAIN)
        X_val, y_val = _get_data_for_model(model, N_SAMPLES_VAL), _make_labels(N_SAMPLES_VAL)
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert isinstance(metrics, TrainingMetrics)
        result = model.predict(X_val)
        assert isinstance(result, PredictionResult)


class TestMLPModels:
    """Smoke test N-BEATS."""

    def test_nbeats_fit_predict(self):
        if not ModelRegistry.is_available("nbeats"):
            pytest.skip("N-BEATS not available")
        config = FAST_CONFIGS["nbeats"]
        model = ModelRegistry.create("nbeats", config=config)
        X_train, y_train = _get_data_for_model(model, N_SAMPLES_TRAIN), _make_labels(N_SAMPLES_TRAIN)
        X_val, y_val = _get_data_for_model(model, N_SAMPLES_VAL), _make_labels(N_SAMPLES_VAL)
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert isinstance(metrics, TrainingMetrics)
        result = model.predict(X_val)
        assert isinstance(result, PredictionResult)


class TestClassicalModels:
    """Smoke test classical baselines."""

    @pytest.mark.parametrize("name", ["random_forest", "logistic", "svm"])
    def test_classical_fit_predict(self, name: str):
        if not ModelRegistry.is_available(name):
            pytest.skip(f"{name} not available")
        config = FAST_CONFIGS.get(name, {})
        model = ModelRegistry.create(name, config=config)
        X_train, y_train = _make_tabular(N_SAMPLES_TRAIN), _make_labels(N_SAMPLES_TRAIN)
        X_val, y_val = _make_tabular(N_SAMPLES_VAL), _make_labels(N_SAMPLES_VAL)
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert isinstance(metrics, TrainingMetrics)
        result = model.predict(X_val)
        assert isinstance(result, PredictionResult)
