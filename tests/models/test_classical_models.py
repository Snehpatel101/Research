"""
Tests for Classical ML Models - Random Forest, Logistic Regression, SVM.

Tests cover:
- Model initialization and properties
- Default config values
- Training and validation
- Prediction output format
- Save/load roundtrip
- Feature importance (where applicable)
- Model-specific functionality
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.models.classical import RandomForestModel, LogisticModel, SVMModel


# =============================================================================
# RANDOM FOREST TESTS
# =============================================================================

class TestRandomForestModelProperties:
    """Tests for RandomForestModel properties."""

    def test_model_family(self):
        """Model family should be 'classical'."""
        model = RandomForestModel()
        assert model.model_family == "classical"

    def test_requires_scaling_false(self):
        """RandomForest should not require scaling."""
        model = RandomForestModel()
        assert model.requires_scaling is False

    def test_requires_sequences_false(self):
        """RandomForest should not require sequences."""
        model = RandomForestModel()
        assert model.requires_sequences is False

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        model = RandomForestModel()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        model = RandomForestModel()
        config = model.get_default_config()

        expected_keys = [
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "max_features", "class_weight",
            "n_jobs", "random_state", "bootstrap", "oob_score",
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"

    def test_config_override(self):
        """Config should be overridable in constructor."""
        model = RandomForestModel(config={"n_estimators": 50, "max_depth": 5})
        assert model.config["n_estimators"] == 50
        assert model.config["max_depth"] == 5


class TestRandomForestTraining:
    """Tests for RandomForestModel training."""

    def test_fit_returns_metrics(self, small_tabular_data, fast_rf_config):
        """Training should return TrainingMetrics."""
        model = RandomForestModel(config=fast_rf_config)
        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
        )

        assert metrics.epochs_trained > 0  # Trees trained
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert 0 <= metrics.train_f1 <= 1
        assert 0 <= metrics.val_f1 <= 1
        assert metrics.training_time_seconds >= 0

    def test_fit_with_sample_weights(self, small_tabular_data, fast_rf_config):
        """Training should accept sample weights."""
        weights = np.random.uniform(0.5, 1.5, size=len(small_tabular_data["y_train"]))
        model = RandomForestModel(config=fast_rf_config)

        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
            sample_weights=weights.astype(np.float32),
        )

        assert metrics.epochs_trained > 0

    def test_is_fitted_after_training(self, trained_rf):
        """Model should be fitted after training."""
        assert trained_rf.is_fitted is True

    def test_metadata_includes_oob_score(self, small_tabular_data, fast_rf_config):
        """Training metadata should include OOB score if enabled."""
        config = {**fast_rf_config, "oob_score": True, "bootstrap": True}
        model = RandomForestModel(config=config)
        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
        )

        assert "oob_score" in metrics.metadata


class TestRandomForestPrediction:
    """Tests for RandomForestModel prediction."""

    def test_predict_returns_output(self, trained_rf, small_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_rf.predict(small_tabular_data["X_val"])

        assert output.n_samples == len(small_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_rf, small_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_rf.predict(small_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_shape(self, trained_rf, small_tabular_data):
        """Probabilities should have shape (n_samples, 3)."""
        output = trained_rf.predict(small_tabular_data["X_val"])
        assert output.class_probabilities.shape == (len(small_tabular_data["X_val"]), 3)

    def test_predict_probabilities_sum_to_one(self, trained_rf, small_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_rf.predict(small_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_predict_unfitted_raises(self, small_tabular_data):
        """Prediction on unfitted model should raise."""
        model = RandomForestModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(small_tabular_data["X_val"])


class TestRandomForestSaveLoad:
    """Tests for RandomForestModel serialization."""

    def test_save_creates_files(self, trained_rf, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "rf_model"
        trained_rf.save(save_path)

        assert (save_path / "model.joblib").exists()
        assert (save_path / "metadata.joblib").exists()

    def test_load_restores_model(self, trained_rf, tmp_model_dir):
        """Load should restore model correctly."""
        save_path = tmp_model_dir / "rf_model"
        trained_rf.save(save_path)

        loaded = RandomForestModel()
        loaded.load(save_path)

        assert loaded.is_fitted is True

    def test_predictions_match_after_load(self, trained_rf, small_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        save_path = tmp_model_dir / "rf_model"
        trained_rf.save(save_path)

        loaded = RandomForestModel()
        loaded.load(save_path)

        original = trained_rf.predict(small_tabular_data["X_val"])
        restored = loaded.predict(small_tabular_data["X_val"])

        assert np.array_equal(original.class_predictions, restored.class_predictions)

    def test_load_missing_raises(self):
        """Load from missing path should raise."""
        model = RandomForestModel()
        with pytest.raises(FileNotFoundError):
            model.load(Path("/nonexistent/path"))


class TestRandomForestFeatureImportance:
    """Tests for RandomForestModel feature importance."""

    def test_get_feature_importance(self, trained_rf):
        """Should return feature importances."""
        importance = trained_rf.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_feature_importance_values_nonnegative(self, trained_rf):
        """Feature importances should be non-negative."""
        importance = trained_rf.get_feature_importance()

        for value in importance.values():
            assert value >= 0

    def test_feature_importance_unfitted_returns_none(self):
        """Feature importance on unfitted model should return None."""
        model = RandomForestModel()
        assert model.get_feature_importance() is None

    def test_set_feature_names(self, trained_rf):
        """Should allow setting feature names."""
        names = [f"feature_{i}" for i in range(10)]
        trained_rf.set_feature_names(names)

        importance = trained_rf.get_feature_importance()
        assert all(name.startswith("feature_") for name in importance.keys())


# =============================================================================
# LOGISTIC REGRESSION TESTS
# =============================================================================

class TestLogisticModelProperties:
    """Tests for LogisticModel properties."""

    def test_model_family(self):
        """Model family should be 'classical'."""
        model = LogisticModel()
        assert model.model_family == "classical"

    def test_requires_scaling_true(self):
        """Logistic should require scaling."""
        model = LogisticModel()
        assert model.requires_scaling is True

    def test_requires_sequences_false(self):
        """Logistic should not require sequences."""
        model = LogisticModel()
        assert model.requires_sequences is False

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        model = LogisticModel()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        model = LogisticModel()
        config = model.get_default_config()

        expected_keys = ["penalty", "C", "solver", "max_iter", "multi_class"]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


class TestLogisticTraining:
    """Tests for LogisticModel training."""

    def test_fit_returns_metrics(self, small_tabular_data, fast_logistic_config):
        """Training should return TrainingMetrics."""
        model = LogisticModel(config=fast_logistic_config)
        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
        )

        assert metrics.epochs_trained >= 0  # iterations
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert metrics.training_time_seconds >= 0

    def test_fit_with_sample_weights(self, small_tabular_data, fast_logistic_config):
        """Training should accept sample weights."""
        weights = np.random.uniform(0.5, 1.5, size=len(small_tabular_data["y_train"]))
        model = LogisticModel(config=fast_logistic_config)

        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
            sample_weights=weights.astype(np.float32),
        )

        assert metrics.train_accuracy > 0

    def test_is_fitted_after_training(self, trained_logistic):
        """Model should be fitted after training."""
        assert trained_logistic.is_fitted is True


class TestLogisticPrediction:
    """Tests for LogisticModel prediction."""

    def test_predict_returns_output(self, trained_logistic, small_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_logistic.predict(small_tabular_data["X_val"])

        assert output.n_samples == len(small_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_logistic, small_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_logistic.predict(small_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_logistic, small_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_logistic.predict(small_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_predict_unfitted_raises(self, small_tabular_data):
        """Prediction on unfitted model should raise."""
        model = LogisticModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(small_tabular_data["X_val"])


class TestLogisticSaveLoad:
    """Tests for LogisticModel serialization."""

    def test_save_creates_files(self, trained_logistic, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "logistic_model"
        trained_logistic.save(save_path)

        assert (save_path / "model.joblib").exists()

    def test_predictions_match_after_load(self, trained_logistic, small_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        save_path = tmp_model_dir / "logistic_model"
        trained_logistic.save(save_path)

        loaded = LogisticModel()
        loaded.load(save_path)

        original = trained_logistic.predict(small_tabular_data["X_val"])
        restored = loaded.predict(small_tabular_data["X_val"])

        assert np.array_equal(original.class_predictions, restored.class_predictions)


class TestLogisticCoefficients:
    """Tests for LogisticModel coefficients access."""

    def test_get_coefficients(self, trained_logistic):
        """Should return model coefficients."""
        coefs = trained_logistic.get_coefficients()

        assert coefs is not None
        assert "coef" in coefs
        assert "intercept" in coefs
        assert coefs["coef"].shape[0] == 3  # 3 classes

    def test_coefficients_unfitted_returns_none(self):
        """Coefficients on unfitted model should return None."""
        model = LogisticModel()
        assert model.get_coefficients() is None


# =============================================================================
# SVM TESTS
# =============================================================================

class TestSVMModelProperties:
    """Tests for SVMModel properties."""

    def test_model_family(self):
        """Model family should be 'classical'."""
        model = SVMModel()
        assert model.model_family == "classical"

    def test_requires_scaling_true(self):
        """SVM should require scaling."""
        model = SVMModel()
        assert model.requires_scaling is True

    def test_requires_sequences_false(self):
        """SVM should not require sequences."""
        model = SVMModel()
        assert model.requires_sequences is False

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        model = SVMModel()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        model = SVMModel()
        config = model.get_default_config()

        expected_keys = ["kernel", "C", "gamma", "class_weight", "probability"]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


class TestSVMTraining:
    """Tests for SVMModel training."""

    def test_fit_returns_metrics(self, small_tabular_data, fast_svm_config):
        """Training should return TrainingMetrics."""
        model = SVMModel(config=fast_svm_config)
        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
        )

        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert metrics.training_time_seconds >= 0

    def test_fit_with_sample_weights(self, small_tabular_data, fast_svm_config):
        """Training should accept sample weights."""
        weights = np.random.uniform(0.5, 1.5, size=len(small_tabular_data["y_train"]))
        model = SVMModel(config=fast_svm_config)

        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
            sample_weights=weights.astype(np.float32),
        )

        assert metrics.train_accuracy > 0

    def test_is_fitted_after_training(self, trained_svm):
        """Model should be fitted after training."""
        assert trained_svm.is_fitted is True

    def test_metadata_includes_support_vectors(self, small_tabular_data, fast_svm_config):
        """Training metadata should include support vector counts."""
        model = SVMModel(config=fast_svm_config)
        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
        )

        assert "n_support_vectors" in metrics.metadata
        assert "n_support_per_class" in metrics.metadata


class TestSVMPrediction:
    """Tests for SVMModel prediction."""

    def test_predict_returns_output(self, trained_svm, small_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_svm.predict(small_tabular_data["X_val"])

        assert output.n_samples == len(small_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_svm, small_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_svm.predict(small_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_svm, small_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_svm.predict(small_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_predict_unfitted_raises(self, small_tabular_data):
        """Prediction on unfitted model should raise."""
        model = SVMModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(small_tabular_data["X_val"])


class TestSVMSaveLoad:
    """Tests for SVMModel serialization."""

    def test_save_creates_files(self, trained_svm, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "svm_model"
        trained_svm.save(save_path)

        assert (save_path / "model.joblib").exists()

    def test_predictions_match_after_load(self, trained_svm, small_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        save_path = tmp_model_dir / "svm_model"
        trained_svm.save(save_path)

        loaded = SVMModel()
        loaded.load(save_path)

        original = trained_svm.predict(small_tabular_data["X_val"])
        restored = loaded.predict(small_tabular_data["X_val"])

        assert np.array_equal(original.class_predictions, restored.class_predictions)


class TestSVMSupportVectors:
    """Tests for SVM support vector access."""

    def test_get_support_vectors(self, trained_svm):
        """Should return support vectors."""
        sv = trained_svm.get_support_vectors()

        assert sv is not None
        assert len(sv) > 0

    def test_get_support_indices(self, trained_svm):
        """Should return support vector indices."""
        indices = trained_svm.get_support_indices()

        assert indices is not None
        assert len(indices) > 0

    def test_support_vectors_unfitted_returns_none(self):
        """Support vectors on unfitted model should return None."""
        model = SVMModel()
        assert model.get_support_vectors() is None


# =============================================================================
# CROSS-MODEL TESTS
# =============================================================================

class TestClassicalModelConsistency:
    """Tests for consistent behavior across classical models."""

    def test_all_models_in_registry(self):
        """All classical models should be in registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "classical" in families
        assert "random_forest" in families["classical"]
        assert "logistic" in families["classical"]
        assert "svm" in families["classical"]

    def test_all_not_require_sequences(self):
        """All classical models should not require sequences."""
        for ModelClass in [RandomForestModel, LogisticModel, SVMModel]:
            model = ModelClass()
            assert model.requires_sequences is False

    def test_prediction_output_format_consistent(
        self, trained_rf, trained_logistic, trained_svm, small_tabular_data
    ):
        """All models should return same prediction format."""
        for model in [trained_rf, trained_logistic, trained_svm]:
            output = model.predict(small_tabular_data["X_val"])

            # Check required attributes
            assert hasattr(output, "class_predictions")
            assert hasattr(output, "class_probabilities")
            assert hasattr(output, "confidence")

            # Check shapes
            n_samples = len(small_tabular_data["X_val"])
            assert output.class_predictions.shape == (n_samples,)
            assert output.class_probabilities.shape == (n_samples, 3)
            assert output.confidence.shape == (n_samples,)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fast_rf_config() -> Dict[str, Any]:
    """Fast Random Forest config for tests."""
    return {
        "n_estimators": 10,
        "max_depth": 3,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "n_jobs": 1,
        "random_state": 42,
        "verbose": 0,
    }


@pytest.fixture
def fast_logistic_config() -> Dict[str, Any]:
    """Fast Logistic Regression config for tests."""
    return {
        "max_iter": 100,
        "solver": "lbfgs",
        "C": 1.0,
        "random_state": 42,
        "verbose": 0,
    }


@pytest.fixture
def fast_svm_config() -> Dict[str, Any]:
    """Fast SVM config for tests."""
    return {
        "kernel": "rbf",
        "C": 1.0,
        "max_iter": 1000,
        "probability": True,
        "random_state": 42,
        "verbose": False,
    }


@pytest.fixture
def trained_rf(small_tabular_data, fast_rf_config):
    """Provide a trained Random Forest model."""
    model = RandomForestModel(config=fast_rf_config)
    model.fit(
        small_tabular_data["X_train"],
        small_tabular_data["y_train"],
        small_tabular_data["X_val"],
        small_tabular_data["y_val"],
    )
    return model


@pytest.fixture
def trained_logistic(small_tabular_data, fast_logistic_config):
    """Provide a trained Logistic Regression model."""
    model = LogisticModel(config=fast_logistic_config)
    model.fit(
        small_tabular_data["X_train"],
        small_tabular_data["y_train"],
        small_tabular_data["X_val"],
        small_tabular_data["y_val"],
    )
    return model


@pytest.fixture
def trained_svm(small_tabular_data, fast_svm_config):
    """Provide a trained SVM model."""
    model = SVMModel(config=fast_svm_config)
    model.fit(
        small_tabular_data["X_train"],
        small_tabular_data["y_train"],
        small_tabular_data["X_val"],
        small_tabular_data["y_val"],
    )
    return model
