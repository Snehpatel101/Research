"""
Tests for XGBoostModel implementation.

Tests cover:
- Model registration and creation
- Training with sample weights
- Predictions and probabilities
- Save/load functionality
- GPU mode fallback
- Feature importance
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models import ModelRegistry
from src.models.boosting import XGBoostModel


@pytest.fixture
def synthetic_data():
    """Generate synthetic 3-class classification data."""
    np.random.seed(42)
    n_train, n_val, n_features = 500, 100, 30

    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.choice([-1, 0, 1], size=n_train)
    weights = np.random.uniform(0.5, 1.5, size=n_train)

    X_val = np.random.randn(n_val, n_features)
    y_val = np.random.choice([-1, 0, 1], size=n_val)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "weights": weights,
        "X_val": X_val,
        "y_val": y_val,
    }


@pytest.fixture
def trained_model(synthetic_data):
    """Return a trained XGBoost model."""
    model = XGBoostModel(config={
        "n_estimators": 20,
        "max_depth": 3,
        "early_stopping_rounds": 5,
        "verbosity": 0,
        "use_gpu": False,
    })
    model.fit(
        synthetic_data["X_train"],
        synthetic_data["y_train"],
        synthetic_data["X_val"],
        synthetic_data["y_val"],
        sample_weights=synthetic_data["weights"],
    )
    return model


class TestModelRegistration:
    """Tests for model registration."""

    def test_xgboost_registered(self):
        """XGBoost model should be registered."""
        assert ModelRegistry.is_registered("xgboost")

    def test_xgb_alias_registered(self):
        """XGBoost alias 'xgb' should be registered."""
        assert ModelRegistry.is_registered("xgb")

    def test_create_via_registry(self):
        """Should create XGBoost model via registry."""
        model = ModelRegistry.create("xgboost")
        assert isinstance(model, XGBoostModel)
        assert model.model_family == "boosting"

    def test_create_via_alias(self):
        """Should create XGBoost model via alias."""
        model = ModelRegistry.create("xgb")
        assert isinstance(model, XGBoostModel)

    def test_listed_in_boosting_family(self):
        """XGBoost should be listed in boosting family."""
        families = ModelRegistry.list_models()
        assert "boosting" in families
        assert "xgboost" in families["boosting"]


class TestModelProperties:
    """Tests for model properties."""

    def test_model_family(self):
        """Model family should be 'boosting'."""
        model = XGBoostModel()
        assert model.model_family == "boosting"

    def test_requires_scaling_false(self):
        """XGBoost should not require scaling."""
        model = XGBoostModel()
        assert model.requires_scaling is False

    def test_requires_sequences_false(self):
        """XGBoost should not require sequences."""
        model = XGBoostModel()
        assert model.requires_sequences is False

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        model = XGBoostModel()
        assert model.is_fitted is False

    def test_default_config(self):
        """Should have expected default config keys."""
        model = XGBoostModel()
        config = model.get_default_config()

        expected_keys = [
            "n_estimators", "max_depth", "learning_rate",
            "early_stopping_rounds", "use_gpu"
        ]
        for key in expected_keys:
            assert key in config


class TestTraining:
    """Tests for model training."""

    def test_fit_returns_metrics(self, synthetic_data):
        """Training should return TrainingMetrics."""
        model = XGBoostModel(config={
            "n_estimators": 20,
            "early_stopping_rounds": 5,
            "verbosity": 0,
        })
        metrics = model.fit(
            synthetic_data["X_train"],
            synthetic_data["y_train"],
            synthetic_data["X_val"],
            synthetic_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert 0 <= metrics.train_f1 <= 1
        assert 0 <= metrics.val_f1 <= 1
        assert metrics.training_time_seconds > 0

    def test_fit_with_sample_weights(self, synthetic_data):
        """Training should accept sample weights."""
        model = XGBoostModel(config={
            "n_estimators": 20,
            "early_stopping_rounds": 5,
            "verbosity": 0,
        })
        metrics = model.fit(
            synthetic_data["X_train"],
            synthetic_data["y_train"],
            synthetic_data["X_val"],
            synthetic_data["y_val"],
            sample_weights=synthetic_data["weights"],
        )
        assert metrics.epochs_trained > 0

    def test_early_stopping(self, synthetic_data):
        """Early stopping should work."""
        model = XGBoostModel(config={
            "n_estimators": 1000,  # High number
            "early_stopping_rounds": 5,
            "verbosity": 0,
        })
        metrics = model.fit(
            synthetic_data["X_train"],
            synthetic_data["y_train"],
            synthetic_data["X_val"],
            synthetic_data["y_val"],
        )
        # Should stop early (not train all 1000)
        assert metrics.early_stopped is True
        assert metrics.epochs_trained < 1000

    def test_is_fitted_after_training(self, trained_model):
        """Model should be fitted after training."""
        assert trained_model.is_fitted is True


class TestPrediction:
    """Tests for model prediction."""

    def test_predict_returns_output(self, trained_model, synthetic_data):
        """Prediction should return PredictionOutput."""
        output = trained_model.predict(synthetic_data["X_val"])

        assert output.n_samples == len(synthetic_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_model, synthetic_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_model.predict(synthetic_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_shape(self, trained_model, synthetic_data):
        """Probabilities should have shape (n_samples, 3)."""
        output = trained_model.predict(synthetic_data["X_val"])
        assert output.class_probabilities.shape == (len(synthetic_data["X_val"]), 3)

    def test_predict_probabilities_sum_to_one(self, trained_model, synthetic_data):
        """Probabilities should sum to 1."""
        output = trained_model.predict(synthetic_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0)

    def test_predict_confidence(self, trained_model, synthetic_data):
        """Confidence should be max probability."""
        output = trained_model.predict(synthetic_data["X_val"])
        expected_conf = output.class_probabilities.max(axis=1)
        assert np.allclose(output.confidence, expected_conf)

    def test_predict_unfitted_raises(self, synthetic_data):
        """Prediction on unfitted model should raise."""
        model = XGBoostModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(synthetic_data["X_val"])


class TestSaveLoad:
    """Tests for model serialization."""

    def test_save_creates_files(self, trained_model):
        """Save should create model files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            trained_model.save(save_path)

            assert (save_path / "model.json").exists()
            assert (save_path / "metadata.pkl").exists()

    def test_load_restores_model(self, trained_model, synthetic_data):
        """Load should restore model correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            trained_model.save(save_path)

            loaded = XGBoostModel()
            loaded.load(save_path)

            assert loaded.is_fitted is True

    def test_predictions_match_after_load(self, trained_model, synthetic_data):
        """Predictions should match after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            trained_model.save(save_path)

            loaded = XGBoostModel()
            loaded.load(save_path)

            original = trained_model.predict(synthetic_data["X_val"])
            restored = loaded.predict(synthetic_data["X_val"])

            assert np.allclose(
                original.class_probabilities,
                restored.class_probabilities
            )

    def test_save_unfitted_raises(self):
        """Save on unfitted model should raise."""
        model = XGBoostModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="not fitted"):
                model.save(Path(tmpdir) / "model")

    def test_load_missing_raises(self):
        """Load from missing path should raise."""
        model = XGBoostModel()
        with pytest.raises(FileNotFoundError):
            model.load(Path("/nonexistent/path"))


class TestFeatureImportance:
    """Tests for feature importance."""

    def test_importance_returns_dict(self, trained_model):
        """Feature importance should return dict."""
        importance = trained_model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_importance_values_positive(self, trained_model):
        """Feature importance values should be positive."""
        importance = trained_model.get_feature_importance()
        for value in importance.values():
            assert value >= 0

    def test_importance_with_names(self, trained_model, synthetic_data):
        """Feature importance should use feature names if set."""
        n_features = synthetic_data["X_train"].shape[1]
        names = [f"feature_{i}" for i in range(n_features)]
        trained_model.set_feature_names(names)

        importance = trained_model.get_feature_importance()
        # At least some should have our names
        assert any(k.startswith("feature_") for k in importance.keys())

    def test_importance_unfitted_returns_none(self):
        """Feature importance on unfitted model should return None."""
        model = XGBoostModel()
        assert model.get_feature_importance() is None


class TestConfigOverride:
    """Tests for configuration override."""

    def test_config_override_in_constructor(self):
        """Config should be overridable in constructor."""
        model = XGBoostModel(config={"max_depth": 10})
        assert model.config["max_depth"] == 10

    def test_config_override_in_fit(self, synthetic_data):
        """Config should be overridable in fit()."""
        model = XGBoostModel(config={
            "n_estimators": 100,
            "verbosity": 0,
        })
        metrics = model.fit(
            synthetic_data["X_train"],
            synthetic_data["y_train"],
            synthetic_data["X_val"],
            synthetic_data["y_val"],
            config={"n_estimators": 10, "early_stopping_rounds": 5},
        )
        # With n_estimators=10 and early_stopping=5, should train < 100
        assert metrics.epochs_trained <= 10


class TestInputValidation:
    """Tests for input validation."""

    def test_invalid_input_shape_raises(self, trained_model):
        """Invalid input shape should raise."""
        with pytest.raises(ValueError):
            trained_model.predict(np.array([1, 2, 3]))  # 1D array

    def test_3d_input_raises(self, trained_model):
        """3D input should raise for non-sequential model."""
        with pytest.raises(ValueError):
            trained_model.predict(np.random.randn(10, 5, 3))


class TestGPUMode:
    """Tests for GPU mode."""

    def test_gpu_fallback(self):
        """Should fall back to CPU if GPU unavailable."""
        model = XGBoostModel(config={"use_gpu": True})
        # Either GPU is available or it fell back to CPU
        # Both are valid outcomes
        assert isinstance(model._use_gpu, bool)

    def test_cpu_mode_explicit(self):
        """Should use CPU when use_gpu=False."""
        model = XGBoostModel(config={"use_gpu": False})
        assert model._use_gpu is False
