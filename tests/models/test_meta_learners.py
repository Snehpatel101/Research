"""
Tests for Meta-Learner Models - RidgeMeta, MLPMeta, CalibratedMeta, XGBoostMeta.

Tests cover:
- Model initialization and properties
- Model registration in registry
- Default configuration validation
- Training on OOF predictions (stacking datasets)
- Prediction output format
- Save/load roundtrip
- Feature importance (where applicable)
- Sample weight support
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


# =============================================================================
# FIXTURES FOR META-LEARNER TESTS
# =============================================================================


@pytest.fixture
def oof_tabular_data() -> Dict[str, np.ndarray]:
    """
    Generate synthetic OOF prediction data for meta-learner testing.

    Simulates OOF predictions from 3 base models, each producing
    3-class probabilities, giving 9 features total.

    Returns dict with:
        - X_train: (200, 9) OOF predictions from base models
        - y_train: (200,) true labels in {-1, 0, 1}
        - X_val: (50, 9) validation OOF predictions
        - y_val: (50,) validation labels
        - weights: (200,) sample weights
    """
    np.random.seed(42)
    n_train, n_val = 200, 50
    n_base_models = 3
    n_classes = 3
    n_features = n_base_models * n_classes  # 9

    # Generate pseudo-probabilities (softmax-like for each base model)
    def generate_probs(n_samples: int) -> np.ndarray:
        probs = np.random.rand(n_samples, n_features).astype(np.float32)
        # Normalize each base model's predictions to sum to 1
        for i in range(n_base_models):
            start = i * n_classes
            end = start + n_classes
            probs[:, start:end] /= probs[:, start:end].sum(axis=1, keepdims=True)
        return probs

    X_train = generate_probs(n_train)
    y_train = np.random.choice([-1, 0, 1], size=n_train)
    weights = np.random.uniform(0.5, 1.5, size=n_train).astype(np.float32)

    X_val = generate_probs(n_val)
    y_val = np.random.choice([-1, 0, 1], size=n_val)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "weights": weights,
        "X_val": X_val,
        "y_val": y_val,
    }


@pytest.fixture
def fast_ridge_meta_config() -> Dict[str, Any]:
    """Fast RidgeMeta config for tests."""
    return {
        "alpha": 1.0,
        "fit_intercept": True,
        "class_weight": "balanced",
        "scale_features": True,
        "random_state": 42,
    }


@pytest.fixture
def fast_mlp_meta_config() -> Dict[str, Any]:
    """Fast MLPMeta config for tests."""
    return {
        "hidden_layer_sizes": (16, 8),
        "activation": "relu",
        "alpha": 0.01,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 5,
        "max_iter": 50,
        "random_state": 42,
        "scale_features": True,
        "verbose": False,
    }


@pytest.fixture
def fast_calibrated_meta_config() -> Dict[str, Any]:
    """Fast CalibratedMeta config for tests."""
    return {
        "base_estimator": "logistic",
        "base_estimator_config": {"C": 1.0, "max_iter": 100},
        "method": "sigmoid",  # Faster than isotonic for small datasets
        "cv": 3,
        "ensemble": True,
        "scale_features": True,
        "random_state": 42,
    }


@pytest.fixture
def fast_xgboost_meta_config() -> Dict[str, Any]:
    """Fast XGBoostMeta config for tests."""
    return {
        "n_estimators": 20,
        "max_depth": 2,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "learning_rate": 0.1,
        "early_stopping_rounds": 5,
        "use_gpu": False,
        "random_state": 42,
        "verbosity": 0,
    }


@pytest.fixture
def trained_ridge_meta(oof_tabular_data, fast_ridge_meta_config):
    """Provide a trained RidgeMeta model."""
    from src.models.ensemble import RidgeMetaLearner

    model = RidgeMetaLearner(config=fast_ridge_meta_config)
    model.fit(
        oof_tabular_data["X_train"],
        oof_tabular_data["y_train"],
        oof_tabular_data["X_val"],
        oof_tabular_data["y_val"],
    )
    return model


@pytest.fixture
def trained_mlp_meta(oof_tabular_data, fast_mlp_meta_config):
    """Provide a trained MLPMeta model."""
    from src.models.ensemble import MLPMetaLearner

    model = MLPMetaLearner(config=fast_mlp_meta_config)
    model.fit(
        oof_tabular_data["X_train"],
        oof_tabular_data["y_train"],
        oof_tabular_data["X_val"],
        oof_tabular_data["y_val"],
    )
    return model


@pytest.fixture
def trained_calibrated_meta(oof_tabular_data, fast_calibrated_meta_config):
    """Provide a trained CalibratedMeta model."""
    from src.models.ensemble import CalibratedMetaLearner

    model = CalibratedMetaLearner(config=fast_calibrated_meta_config)
    model.fit(
        oof_tabular_data["X_train"],
        oof_tabular_data["y_train"],
        oof_tabular_data["X_val"],
        oof_tabular_data["y_val"],
    )
    return model


@pytest.fixture
def trained_xgboost_meta(oof_tabular_data, fast_xgboost_meta_config):
    """Provide a trained XGBoostMeta model."""
    from src.models.ensemble import XGBoostMeta

    model = XGBoostMeta(config=fast_xgboost_meta_config)
    model.fit(
        oof_tabular_data["X_train"],
        oof_tabular_data["y_train"],
        oof_tabular_data["X_val"],
        oof_tabular_data["y_val"],
    )
    return model


# =============================================================================
# RIDGE META-LEARNER TESTS
# =============================================================================


class TestRidgeMetaRegistration:
    """Tests for RidgeMeta model registration."""

    def test_model_in_registry(self):
        """RidgeMeta should be registered in the model registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "ensemble" in families
        assert "ridge_meta" in families["ensemble"]

    def test_registry_create(self):
        """Should be able to create RidgeMeta via registry."""
        from src.models import ModelRegistry

        model = ModelRegistry.create("ridge_meta")
        assert model is not None
        assert model.model_family == "ensemble"


class TestRidgeMetaProperties:
    """Tests for RidgeMeta model properties."""

    def test_model_family(self):
        """Model family should be 'ensemble'."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner()
        assert model.model_family == "ensemble"

    def test_requires_scaling_false(self):
        """RidgeMeta handles scaling internally."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner()
        assert model.requires_scaling is False

    def test_requires_sequences_false(self):
        """RidgeMeta should not require sequences."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner()
        assert model.requires_sequences is False

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner()
        config = model.get_default_config()

        expected_keys = ["alpha", "fit_intercept", "class_weight", "scale_features"]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


class TestRidgeMetaTraining:
    """Tests for RidgeMeta training."""

    def test_fit_returns_metrics(self, oof_tabular_data, fast_ridge_meta_config):
        """Training should return TrainingMetrics."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner(config=fast_ridge_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
        )

        assert metrics.epochs_trained == 1  # Ridge is single-pass
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert 0 <= metrics.train_f1 <= 1
        assert 0 <= metrics.val_f1 <= 1
        assert metrics.training_time_seconds > 0

    def test_fit_with_sample_weights(self, oof_tabular_data, fast_ridge_meta_config):
        """Training should accept sample weights."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner(config=fast_ridge_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
            sample_weights=oof_tabular_data["weights"],
        )

        assert model.is_fitted is True
        assert metrics.epochs_trained == 1

    def test_is_fitted_after_training(self, trained_ridge_meta):
        """Model should be fitted after training."""
        assert trained_ridge_meta.is_fitted is True

    def test_metadata_includes_meta_learner_type(self, oof_tabular_data, fast_ridge_meta_config):
        """Training metadata should include meta-learner type."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner(config=fast_ridge_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
        )

        assert "meta_learner" in metrics.metadata
        assert metrics.metadata["meta_learner"] == "ridge"


class TestRidgeMetaPrediction:
    """Tests for RidgeMeta prediction."""

    def test_predict_returns_output(self, trained_ridge_meta, oof_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_ridge_meta.predict(oof_tabular_data["X_val"])

        assert output.n_samples == len(oof_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_ridge_meta, oof_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_ridge_meta.predict(oof_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_shape(self, trained_ridge_meta, oof_tabular_data):
        """Probabilities should have correct shape."""
        output = trained_ridge_meta.predict(oof_tabular_data["X_val"])
        assert output.class_probabilities.shape == (len(oof_tabular_data["X_val"]), 3)

    def test_predict_probabilities_sum_to_one(self, trained_ridge_meta, oof_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_ridge_meta.predict(oof_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_predict_confidence_is_max_prob(self, trained_ridge_meta, oof_tabular_data):
        """Confidence should be max probability."""
        output = trained_ridge_meta.predict(oof_tabular_data["X_val"])
        expected_conf = output.class_probabilities.max(axis=1)
        assert np.allclose(output.confidence, expected_conf)

    def test_predict_unfitted_raises(self, oof_tabular_data):
        """Prediction on unfitted model should raise."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(oof_tabular_data["X_val"])

    def test_predict_proba_method(self, trained_ridge_meta, oof_tabular_data):
        """predict_proba should return class probabilities."""
        proba = trained_ridge_meta.predict_proba(oof_tabular_data["X_val"])
        assert proba.shape == (len(oof_tabular_data["X_val"]), 3)


class TestRidgeMetaSaveLoad:
    """Tests for RidgeMeta serialization."""

    def test_save_creates_files(self, trained_ridge_meta, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "ridge_meta_model"
        trained_ridge_meta.save(save_path)

        assert (save_path / "model.joblib").exists()
        assert (save_path / "metadata.joblib").exists()

    def test_predictions_match_after_load(self, trained_ridge_meta, oof_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.ensemble import RidgeMetaLearner

        save_path = tmp_model_dir / "ridge_meta_model"
        trained_ridge_meta.save(save_path)

        loaded = RidgeMetaLearner()
        loaded.load(save_path)

        original = trained_ridge_meta.predict(oof_tabular_data["X_val"])
        restored = loaded.predict(oof_tabular_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-5)

    def test_save_unfitted_raises(self, tmp_model_dir):
        """Save on unfitted model should raise."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save(tmp_model_dir / "model")

    def test_load_missing_raises(self):
        """Load from missing path should raise."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner()
        with pytest.raises(FileNotFoundError):
            model.load(Path("/nonexistent/path"))


class TestRidgeMetaFeatureImportance:
    """Tests for RidgeMeta feature importance."""

    def test_get_feature_importance(self, trained_ridge_meta):
        """Should return coefficient-based feature importance."""
        importance = trained_ridge_meta.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_feature_importance_unfitted_returns_none(self):
        """Feature importance on unfitted model should return None."""
        from src.models.ensemble import RidgeMetaLearner

        model = RidgeMetaLearner()
        assert model.get_feature_importance() is None


# =============================================================================
# MLP META-LEARNER TESTS
# =============================================================================


class TestMLPMetaRegistration:
    """Tests for MLPMeta model registration."""

    def test_model_in_registry(self):
        """MLPMeta should be registered in the model registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "ensemble" in families
        assert "mlp_meta" in families["ensemble"]

    def test_registry_create(self):
        """Should be able to create MLPMeta via registry."""
        from src.models import ModelRegistry

        model = ModelRegistry.create("mlp_meta")
        assert model is not None
        assert model.model_family == "ensemble"


class TestMLPMetaProperties:
    """Tests for MLPMeta model properties."""

    def test_model_family(self):
        """Model family should be 'ensemble'."""
        from src.models.ensemble import MLPMetaLearner

        model = MLPMetaLearner()
        assert model.model_family == "ensemble"

    def test_requires_scaling_false(self):
        """MLPMeta handles scaling internally."""
        from src.models.ensemble import MLPMetaLearner

        model = MLPMetaLearner()
        assert model.requires_scaling is False

    def test_requires_sequences_false(self):
        """MLPMeta should not require sequences."""
        from src.models.ensemble import MLPMetaLearner

        model = MLPMetaLearner()
        assert model.requires_sequences is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.ensemble import MLPMetaLearner

        model = MLPMetaLearner()
        config = model.get_default_config()

        expected_keys = [
            "hidden_layer_sizes", "activation", "alpha",
            "early_stopping", "max_iter", "scale_features"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


class TestMLPMetaTraining:
    """Tests for MLPMeta training."""

    def test_fit_returns_metrics(self, oof_tabular_data, fast_mlp_meta_config):
        """Training should return TrainingMetrics."""
        from src.models.ensemble import MLPMetaLearner

        model = MLPMetaLearner(config=fast_mlp_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_is_fitted_after_training(self, trained_mlp_meta):
        """Model should be fitted after training."""
        assert trained_mlp_meta.is_fitted is True

    def test_history_tracked(self, oof_tabular_data, fast_mlp_meta_config):
        """Training history should be tracked."""
        from src.models.ensemble import MLPMetaLearner

        model = MLPMetaLearner(config=fast_mlp_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
        )

        assert "loss_curve" in metrics.history


class TestMLPMetaPrediction:
    """Tests for MLPMeta prediction."""

    def test_predict_returns_output(self, trained_mlp_meta, oof_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_mlp_meta.predict(oof_tabular_data["X_val"])

        assert output.n_samples == len(oof_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_mlp_meta, oof_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_mlp_meta.predict(oof_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_mlp_meta, oof_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_mlp_meta.predict(oof_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_predict_unfitted_raises(self, oof_tabular_data):
        """Prediction on unfitted model should raise."""
        from src.models.ensemble import MLPMetaLearner

        model = MLPMetaLearner()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(oof_tabular_data["X_val"])


class TestMLPMetaSaveLoad:
    """Tests for MLPMeta serialization."""

    def test_save_creates_files(self, trained_mlp_meta, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "mlp_meta_model"
        trained_mlp_meta.save(save_path)

        assert (save_path / "model.joblib").exists()
        assert (save_path / "metadata.joblib").exists()

    def test_predictions_match_after_load(self, trained_mlp_meta, oof_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.ensemble import MLPMetaLearner

        save_path = tmp_model_dir / "mlp_meta_model"
        trained_mlp_meta.save(save_path)

        loaded = MLPMetaLearner()
        loaded.load(save_path)

        original = trained_mlp_meta.predict(oof_tabular_data["X_val"])
        restored = loaded.predict(oof_tabular_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-4)


class TestMLPMetaFeatureImportance:
    """Tests for MLPMeta feature importance."""

    def test_get_feature_importance(self, trained_mlp_meta):
        """Should return input layer weight-based feature importance."""
        importance = trained_mlp_meta.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_feature_importance_unfitted_returns_none(self):
        """Feature importance on unfitted model should return None."""
        from src.models.ensemble import MLPMetaLearner

        model = MLPMetaLearner()
        assert model.get_feature_importance() is None


# =============================================================================
# CALIBRATED META-LEARNER TESTS
# =============================================================================


class TestCalibratedMetaRegistration:
    """Tests for CalibratedMeta model registration."""

    def test_model_in_registry(self):
        """CalibratedMeta should be registered in the model registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "ensemble" in families
        assert "calibrated_meta" in families["ensemble"]

    def test_registry_create(self):
        """Should be able to create CalibratedMeta via registry."""
        from src.models import ModelRegistry

        model = ModelRegistry.create("calibrated_meta")
        assert model is not None
        assert model.model_family == "ensemble"


class TestCalibratedMetaProperties:
    """Tests for CalibratedMeta model properties."""

    def test_model_family(self):
        """Model family should be 'ensemble'."""
        from src.models.ensemble import CalibratedMetaLearner

        model = CalibratedMetaLearner()
        assert model.model_family == "ensemble"

    def test_requires_scaling_false(self):
        """CalibratedMeta handles scaling internally."""
        from src.models.ensemble import CalibratedMetaLearner

        model = CalibratedMetaLearner()
        assert model.requires_scaling is False

    def test_requires_sequences_false(self):
        """CalibratedMeta should not require sequences."""
        from src.models.ensemble import CalibratedMetaLearner

        model = CalibratedMetaLearner()
        assert model.requires_sequences is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.ensemble import CalibratedMetaLearner

        model = CalibratedMetaLearner()
        config = model.get_default_config()

        expected_keys = [
            "base_estimator", "base_estimator_config", "method",
            "cv", "ensemble", "scale_features"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


class TestCalibratedMetaTraining:
    """Tests for CalibratedMeta training."""

    def test_fit_returns_metrics(self, oof_tabular_data, fast_calibrated_meta_config):
        """Training should return TrainingMetrics."""
        from src.models.ensemble import CalibratedMetaLearner

        model = CalibratedMetaLearner(config=fast_calibrated_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
        )

        assert metrics.epochs_trained == 1  # Calibration is single-pass
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_fit_with_sample_weights(self, oof_tabular_data, fast_calibrated_meta_config):
        """Training should accept sample weights."""
        from src.models.ensemble import CalibratedMetaLearner

        model = CalibratedMetaLearner(config=fast_calibrated_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
            sample_weights=oof_tabular_data["weights"],
        )

        assert model.is_fitted is True
        assert metrics is not None

    def test_is_fitted_after_training(self, trained_calibrated_meta):
        """Model should be fitted after training."""
        assert trained_calibrated_meta.is_fitted is True

    def test_metadata_includes_calibration_method(self, oof_tabular_data, fast_calibrated_meta_config):
        """Training metadata should include calibration method."""
        from src.models.ensemble import CalibratedMetaLearner

        model = CalibratedMetaLearner(config=fast_calibrated_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
        )

        assert "calibration_method" in metrics.metadata
        assert "base_estimator" in metrics.metadata


class TestCalibratedMetaPrediction:
    """Tests for CalibratedMeta prediction."""

    def test_predict_returns_output(self, trained_calibrated_meta, oof_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_calibrated_meta.predict(oof_tabular_data["X_val"])

        assert output.n_samples == len(oof_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_calibrated_meta, oof_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_calibrated_meta.predict(oof_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_calibrated_meta, oof_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_calibrated_meta.predict(oof_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_predict_unfitted_raises(self, oof_tabular_data):
        """Prediction on unfitted model should raise."""
        from src.models.ensemble import CalibratedMetaLearner

        model = CalibratedMetaLearner()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(oof_tabular_data["X_val"])


class TestCalibratedMetaSaveLoad:
    """Tests for CalibratedMeta serialization."""

    def test_save_creates_files(self, trained_calibrated_meta, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "calibrated_meta_model"
        trained_calibrated_meta.save(save_path)

        assert (save_path / "model.joblib").exists()
        assert (save_path / "metadata.joblib").exists()

    def test_predictions_match_after_load(self, trained_calibrated_meta, oof_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.ensemble import CalibratedMetaLearner

        save_path = tmp_model_dir / "calibrated_meta_model"
        trained_calibrated_meta.save(save_path)

        loaded = CalibratedMetaLearner()
        loaded.load(save_path)

        original = trained_calibrated_meta.predict(oof_tabular_data["X_val"])
        restored = loaded.predict(oof_tabular_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-4)


class TestCalibratedMetaFeatureImportance:
    """Tests for CalibratedMeta feature importance."""

    def test_feature_importance_returns_none(self, trained_calibrated_meta):
        """CalibratedMeta does not support feature importance."""
        importance = trained_calibrated_meta.get_feature_importance()
        assert importance is None


# =============================================================================
# XGBOOST META-LEARNER TESTS
# =============================================================================


class TestXGBoostMetaRegistration:
    """Tests for XGBoostMeta model registration."""

    def test_model_in_registry(self):
        """XGBoostMeta should be registered in the model registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "ensemble" in families
        assert "xgboost_meta" in families["ensemble"]

    def test_registry_create(self):
        """Should be able to create XGBoostMeta via registry."""
        from src.models import ModelRegistry

        model = ModelRegistry.create("xgboost_meta")
        assert model is not None
        assert model.model_family == "ensemble"


class TestXGBoostMetaProperties:
    """Tests for XGBoostMeta model properties."""

    def test_model_family(self):
        """Model family should be 'ensemble'."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta()
        assert model.model_family == "ensemble"

    def test_requires_scaling_false(self):
        """XGBoostMeta does not require scaling."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta()
        assert model.requires_scaling is False

    def test_requires_sequences_false(self):
        """XGBoostMeta should not require sequences."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta()
        assert model.requires_sequences is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta()
        config = model.get_default_config()

        expected_keys = [
            "n_estimators", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "early_stopping_rounds"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


class TestXGBoostMetaTraining:
    """Tests for XGBoostMeta training."""

    def test_fit_returns_metrics(self, oof_tabular_data, fast_xgboost_meta_config):
        """Training should return TrainingMetrics."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta(config=fast_xgboost_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_fit_with_sample_weights(self, oof_tabular_data, fast_xgboost_meta_config):
        """Training should accept sample weights."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta(config=fast_xgboost_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
            sample_weights=oof_tabular_data["weights"],
        )

        assert model.is_fitted is True
        assert metrics.epochs_trained > 0

    def test_is_fitted_after_training(self, trained_xgboost_meta):
        """Model should be fitted after training."""
        assert trained_xgboost_meta.is_fitted is True

    def test_early_stopping(self, oof_tabular_data, fast_xgboost_meta_config):
        """Early stopping should work."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta(config=fast_xgboost_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
        )

        # May stop early
        assert metrics.epochs_trained <= fast_xgboost_meta_config["n_estimators"]

    def test_history_tracked(self, oof_tabular_data, fast_xgboost_meta_config):
        """Training history should be tracked."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta(config=fast_xgboost_meta_config)
        metrics = model.fit(
            oof_tabular_data["X_train"],
            oof_tabular_data["y_train"],
            oof_tabular_data["X_val"],
            oof_tabular_data["y_val"],
        )

        assert "train_mlogloss" in metrics.history or "val_mlogloss" in metrics.history


class TestXGBoostMetaPrediction:
    """Tests for XGBoostMeta prediction."""

    def test_predict_returns_output(self, trained_xgboost_meta, oof_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_xgboost_meta.predict(oof_tabular_data["X_val"])

        assert output.n_samples == len(oof_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_xgboost_meta, oof_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_xgboost_meta.predict(oof_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_shape(self, trained_xgboost_meta, oof_tabular_data):
        """Probabilities should have correct shape."""
        output = trained_xgboost_meta.predict(oof_tabular_data["X_val"])
        assert output.class_probabilities.shape == (len(oof_tabular_data["X_val"]), 3)

    def test_predict_probabilities_sum_to_one(self, trained_xgboost_meta, oof_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_xgboost_meta.predict(oof_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_predict_unfitted_raises(self, oof_tabular_data):
        """Prediction on unfitted model should raise."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(oof_tabular_data["X_val"])


class TestXGBoostMetaSaveLoad:
    """Tests for XGBoostMeta serialization."""

    def test_save_creates_files(self, trained_xgboost_meta, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "xgboost_meta_model"
        trained_xgboost_meta.save(save_path)

        assert (save_path / "model.json").exists()
        assert (save_path / "metadata.pkl").exists()

    def test_predictions_match_after_load(self, trained_xgboost_meta, oof_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        from src.models.ensemble import XGBoostMeta

        save_path = tmp_model_dir / "xgboost_meta_model"
        trained_xgboost_meta.save(save_path)

        loaded = XGBoostMeta()
        loaded.load(save_path)

        original = trained_xgboost_meta.predict(oof_tabular_data["X_val"])
        restored = loaded.predict(oof_tabular_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-5)

    def test_save_unfitted_raises(self, tmp_model_dir):
        """Save on unfitted model should raise."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save(tmp_model_dir / "model")

    def test_load_missing_raises(self):
        """Load from missing path should raise."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta()
        with pytest.raises(FileNotFoundError):
            model.load(Path("/nonexistent/path"))


class TestXGBoostMetaFeatureImportance:
    """Tests for XGBoostMeta feature importance."""

    def test_get_feature_importance(self, trained_xgboost_meta):
        """Should return gain-based feature importance."""
        importance = trained_xgboost_meta.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, dict)
        # May be empty if model didn't use all features

    def test_feature_importance_unfitted_returns_none(self):
        """Feature importance on unfitted model should return None."""
        from src.models.ensemble import XGBoostMeta

        model = XGBoostMeta()
        assert model.get_feature_importance() is None


# =============================================================================
# CROSS-META-LEARNER TESTS
# =============================================================================


class TestMetaLearnerConsistency:
    """Tests for consistent behavior across meta-learner models."""

    def test_all_models_in_registry(self):
        """All meta-learner models should be in registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "ensemble" in families

        expected_models = ["ridge_meta", "mlp_meta", "calibrated_meta", "xgboost_meta"]
        for model_name in expected_models:
            assert model_name in families["ensemble"], f"Missing model: {model_name}"

    def test_all_not_require_sequences(self):
        """All meta-learners should not require sequences."""
        from src.models.ensemble import (
            RidgeMetaLearner, MLPMetaLearner, CalibratedMetaLearner, XGBoostMeta
        )

        for ModelClass in [RidgeMetaLearner, MLPMetaLearner, CalibratedMetaLearner, XGBoostMeta]:
            model = ModelClass()
            assert model.requires_sequences is False, f"{ModelClass.__name__} should not require sequences"

    def test_all_not_require_scaling(self):
        """All meta-learners should not require external scaling."""
        from src.models.ensemble import (
            RidgeMetaLearner, MLPMetaLearner, CalibratedMetaLearner, XGBoostMeta
        )

        for ModelClass in [RidgeMetaLearner, MLPMetaLearner, CalibratedMetaLearner, XGBoostMeta]:
            model = ModelClass()
            assert model.requires_scaling is False, f"{ModelClass.__name__} should not require external scaling"

    def test_all_have_ensemble_family(self):
        """All meta-learners should have 'ensemble' family."""
        from src.models.ensemble import (
            RidgeMetaLearner, MLPMetaLearner, CalibratedMetaLearner, XGBoostMeta
        )

        for ModelClass in [RidgeMetaLearner, MLPMetaLearner, CalibratedMetaLearner, XGBoostMeta]:
            model = ModelClass()
            assert model.model_family == "ensemble", f"{ModelClass.__name__} should have 'ensemble' family"

    def test_prediction_output_format_consistent(
        self, trained_ridge_meta, trained_mlp_meta,
        trained_calibrated_meta, trained_xgboost_meta, oof_tabular_data
    ):
        """All meta-learners should return same prediction format."""
        for model in [trained_ridge_meta, trained_mlp_meta, trained_calibrated_meta, trained_xgboost_meta]:
            output = model.predict(oof_tabular_data["X_val"])

            # Check required attributes
            assert hasattr(output, "class_predictions")
            assert hasattr(output, "class_probabilities")
            assert hasattr(output, "confidence")

            # Check shapes
            n_samples = len(oof_tabular_data["X_val"])
            assert output.class_predictions.shape == (n_samples,)
            assert output.class_probabilities.shape == (n_samples, 3)
            assert output.confidence.shape == (n_samples,)

            # Check class labels
            assert set(output.class_predictions).issubset({-1, 0, 1})

            # Check probabilities sum to 1
            sums = output.class_probabilities.sum(axis=1)
            assert np.allclose(sums, 1.0, atol=1e-5)
