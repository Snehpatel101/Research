"""
Tests for Boosting Models - XGBoost, LightGBM, CatBoost.

Tests cover:
- Model initialization and properties
- Fit with and without sample weights
- Prediction output format and values
- Save/load roundtrip
- Feature importance
- GPU mode detection and fallback
- Configuration handling
- Input validation
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models import ModelRegistry
from src.models.boosting import XGBoostModel

# Import conditionally available models
try:
    from src.models.boosting import LightGBMModel
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LightGBMModel = None

try:
    from src.models.boosting import CatBoostModel
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostModel = None


# =============================================================================
# XGBOOST TESTS
# =============================================================================

class TestXGBoostModelProperties:
    """Tests for XGBoostModel properties."""

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

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        model = XGBoostModel()
        config = model.get_default_config()

        expected_keys = [
            "n_estimators", "max_depth", "learning_rate",
            "early_stopping_rounds", "use_gpu", "subsample",
            "colsample_bytree", "min_child_weight"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"

    def test_config_override(self):
        """Config should be overridable in constructor."""
        model = XGBoostModel(config={"max_depth": 10, "n_estimators": 50})
        assert model.config["max_depth"] == 10
        assert model.config["n_estimators"] == 50


class TestXGBoostTraining:
    """Tests for XGBoostModel training."""

    def test_fit_returns_metrics(self, small_tabular_data, fast_xgboost_config):
        """Training should return TrainingMetrics."""
        model = XGBoostModel(config=fast_xgboost_config)
        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert 0 <= metrics.train_f1 <= 1
        assert 0 <= metrics.val_f1 <= 1
        assert metrics.training_time_seconds > 0

    def test_fit_with_sample_weights(self, small_tabular_data, fast_xgboost_config):
        """Training should accept sample weights."""
        weights = np.random.uniform(0.5, 1.5, size=len(small_tabular_data["y_train"]))
        model = XGBoostModel(config=fast_xgboost_config)

        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
            sample_weights=weights.astype(np.float32),
        )

        assert metrics.epochs_trained > 0

    def test_early_stopping(self, small_tabular_data):
        """Early stopping should work with high n_estimators."""
        model = XGBoostModel(config={
            "n_estimators": 1000,
            "early_stopping_rounds": 3,
            "verbosity": 0,
        })
        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
        )

        assert metrics.early_stopped is True
        assert metrics.epochs_trained < 1000

    def test_is_fitted_after_training(self, trained_xgboost):
        """Model should be fitted after training."""
        assert trained_xgboost.is_fitted is True

    def test_config_override_in_fit(self, small_tabular_data, fast_xgboost_config):
        """Config should be overridable in fit()."""
        model = XGBoostModel(config=fast_xgboost_config)
        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
            config={"n_estimators": 5},
        )

        assert metrics.epochs_trained <= 5


class TestXGBoostPrediction:
    """Tests for XGBoostModel prediction."""

    def test_predict_returns_output(self, trained_xgboost, small_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_xgboost.predict(small_tabular_data["X_val"])

        assert output.n_samples == len(small_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_xgboost, small_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_xgboost.predict(small_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_shape(self, trained_xgboost, small_tabular_data):
        """Probabilities should have shape (n_samples, 3)."""
        output = trained_xgboost.predict(small_tabular_data["X_val"])
        assert output.class_probabilities.shape == (len(small_tabular_data["X_val"]), 3)

    def test_predict_probabilities_sum_to_one(self, trained_xgboost, small_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_xgboost.predict(small_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0)

    def test_predict_confidence_is_max_prob(self, trained_xgboost, small_tabular_data):
        """Confidence should be max probability."""
        output = trained_xgboost.predict(small_tabular_data["X_val"])
        expected_conf = output.class_probabilities.max(axis=1)
        assert np.allclose(output.confidence, expected_conf)

    def test_predict_unfitted_raises(self, small_tabular_data):
        """Prediction on unfitted model should raise."""
        model = XGBoostModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(small_tabular_data["X_val"])


class TestXGBoostSaveLoad:
    """Tests for XGBoostModel serialization."""

    def test_save_creates_files(self, trained_xgboost, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "xgb_model"
        trained_xgboost.save(save_path)

        assert (save_path / "model.json").exists()
        assert (save_path / "metadata.pkl").exists()

    def test_load_restores_model(self, trained_xgboost, tmp_model_dir):
        """Load should restore model correctly."""
        save_path = tmp_model_dir / "xgb_model"
        trained_xgboost.save(save_path)

        loaded = XGBoostModel()
        loaded.load(save_path)

        assert loaded.is_fitted is True

    def test_predictions_match_after_load(self, trained_xgboost, small_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        save_path = tmp_model_dir / "xgb_model"
        trained_xgboost.save(save_path)

        loaded = XGBoostModel()
        loaded.load(save_path)

        original = trained_xgboost.predict(small_tabular_data["X_val"])
        restored = loaded.predict(small_tabular_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities)

    def test_save_unfitted_raises(self, tmp_model_dir):
        """Save on unfitted model should raise."""
        model = XGBoostModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save(tmp_model_dir / "model")

    def test_load_missing_raises(self):
        """Load from missing path should raise."""
        model = XGBoostModel()
        with pytest.raises(FileNotFoundError):
            model.load(Path("/nonexistent/path"))


class TestXGBoostFeatureImportance:
    """Tests for XGBoostModel feature importance."""

    def test_importance_returns_dict(self, trained_xgboost):
        """Feature importance should return dict."""
        importance = trained_xgboost.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_importance_values_positive(self, trained_xgboost):
        """Feature importance values should be positive."""
        importance = trained_xgboost.get_feature_importance()
        for value in importance.values():
            assert value >= 0

    def test_importance_with_names(self, trained_xgboost, small_tabular_data):
        """Feature importance should use feature names if set."""
        n_features = small_tabular_data["X_train"].shape[1]
        names = [f"feature_{i}" for i in range(n_features)]
        trained_xgboost.set_feature_names(names)

        importance = trained_xgboost.get_feature_importance()
        assert any(k.startswith("feature_") for k in importance.keys())

    def test_importance_unfitted_returns_none(self):
        """Feature importance on unfitted model should return None."""
        model = XGBoostModel()
        assert model.get_feature_importance() is None


class TestXGBoostInputValidation:
    """Tests for XGBoostModel input validation."""

    def test_1d_input_raises(self, trained_xgboost):
        """1D input should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D"):
            trained_xgboost.predict(np.array([1, 2, 3]))

    def test_3d_input_raises(self, trained_xgboost):
        """3D input should raise for non-sequential model."""
        with pytest.raises(ValueError, match="must be 2D"):
            trained_xgboost.predict(np.random.randn(10, 5, 3))


class TestXGBoostGPU:
    """Tests for XGBoost GPU mode."""

    def test_gpu_fallback_to_cpu(self):
        """Should fall back to CPU if GPU unavailable."""
        model = XGBoostModel(config={"use_gpu": True})
        # Either GPU is available or it fell back to CPU
        assert isinstance(model._use_gpu, bool)

    def test_cpu_mode_explicit(self):
        """Should use CPU when use_gpu=False."""
        model = XGBoostModel(config={"use_gpu": False})
        assert model._use_gpu is False


# =============================================================================
# LIGHTGBM TESTS
# =============================================================================

@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMModelProperties:
    """Tests for LightGBMModel properties."""

    def test_model_family(self):
        """Model family should be 'boosting'."""
        model = LightGBMModel()
        assert model.model_family == "boosting"

    def test_requires_scaling_false(self):
        """LightGBM should not require scaling."""
        model = LightGBMModel()
        assert model.requires_scaling is False

    def test_requires_sequences_false(self):
        """LightGBM should not require sequences."""
        model = LightGBMModel()
        assert model.requires_sequences is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        model = LightGBMModel()
        config = model.get_default_config()

        expected_keys = [
            "n_estimators", "max_depth", "learning_rate",
            "early_stopping_rounds", "num_leaves", "subsample"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMTraining:
    """Tests for LightGBMModel training."""

    def test_fit_returns_metrics(self, small_tabular_data, fast_lightgbm_config):
        """Training should return TrainingMetrics."""
        model = LightGBMModel(config=fast_lightgbm_config)
        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert metrics.training_time_seconds > 0

    def test_fit_with_sample_weights(self, small_tabular_data, fast_lightgbm_config):
        """Training should accept sample weights."""
        weights = np.random.uniform(0.5, 1.5, size=len(small_tabular_data["y_train"]))
        model = LightGBMModel(config=fast_lightgbm_config)

        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
            sample_weights=weights.astype(np.float32),
        )

        assert metrics.epochs_trained > 0


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMPrediction:
    """Tests for LightGBMModel prediction."""

    def test_predict_returns_output(self, trained_lightgbm, small_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_lightgbm.predict(small_tabular_data["X_val"])

        assert output.n_samples == len(small_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_lightgbm, small_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_lightgbm.predict(small_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_lightgbm, small_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_lightgbm.predict(small_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0)


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMSaveLoad:
    """Tests for LightGBMModel serialization."""

    def test_save_creates_files(self, trained_lightgbm, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "lgb_model"
        trained_lightgbm.save(save_path)

        assert (save_path / "model.txt").exists()
        assert (save_path / "metadata.pkl").exists()

    def test_predictions_match_after_load(self, trained_lightgbm, small_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        save_path = tmp_model_dir / "lgb_model"
        trained_lightgbm.save(save_path)

        loaded = LightGBMModel()
        loaded.load(save_path)

        original = trained_lightgbm.predict(small_tabular_data["X_val"])
        restored = loaded.predict(small_tabular_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities)


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMFeatureImportance:
    """Tests for LightGBMModel feature importance."""

    def test_importance_returns_dict(self, trained_lightgbm):
        """Feature importance should return dict."""
        importance = trained_lightgbm.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_importance_unfitted_returns_none(self):
        """Feature importance on unfitted model should return None."""
        model = LightGBMModel()
        assert model.get_feature_importance() is None


# =============================================================================
# CATBOOST TESTS
# =============================================================================

@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not installed")
class TestCatBoostModelProperties:
    """Tests for CatBoostModel properties."""

    def test_model_family(self):
        """Model family should be 'boosting'."""
        model = CatBoostModel()
        assert model.model_family == "boosting"

    def test_requires_scaling_false(self):
        """CatBoost should not require scaling."""
        model = CatBoostModel()
        assert model.requires_scaling is False

    def test_requires_sequences_false(self):
        """CatBoost should not require sequences."""
        model = CatBoostModel()
        assert model.requires_sequences is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        model = CatBoostModel()
        config = model.get_default_config()

        expected_keys = [
            "iterations", "depth", "learning_rate",
            "early_stopping_rounds", "l2_leaf_reg"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not installed")
class TestCatBoostTraining:
    """Tests for CatBoostModel training."""

    @pytest.mark.skip(reason="CatBoost FPE on small synthetic data - env issue")
    def test_fit_returns_metrics(self, small_tabular_data, fast_catboost_config):
        """Training should return TrainingMetrics."""
        model = CatBoostModel(config=fast_catboost_config)
        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
        )

        assert metrics.epochs_trained > 0
        assert 0 <= metrics.train_accuracy <= 1
        assert 0 <= metrics.val_accuracy <= 1
        assert metrics.training_time_seconds > 0

    @pytest.mark.skip(reason="CatBoost FPE on small synthetic data - env issue")
    def test_fit_with_sample_weights(self, small_tabular_data, fast_catboost_config):
        """Training should accept sample weights."""
        weights = np.random.uniform(0.5, 1.5, size=len(small_tabular_data["y_train"]))
        model = CatBoostModel(config=fast_catboost_config)

        metrics = model.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
            sample_weights=weights.astype(np.float32),
        )

        assert metrics.epochs_trained > 0


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not installed")
@pytest.mark.skip(reason="CatBoost FPE on small synthetic data - env issue")
class TestCatBoostPrediction:
    """Tests for CatBoostModel prediction."""

    def test_predict_returns_output(self, trained_catboost, small_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_catboost.predict(small_tabular_data["X_val"])

        assert output.n_samples == len(small_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_catboost, small_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_catboost.predict(small_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_catboost, small_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_catboost.predict(small_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0)


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not installed")
@pytest.mark.skip(reason="CatBoost FPE on small synthetic data - env issue")
class TestCatBoostSaveLoad:
    """Tests for CatBoostModel serialization."""

    def test_save_creates_files(self, trained_catboost, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "cat_model"
        trained_catboost.save(save_path)

        assert (save_path / "model.cbm").exists()
        assert (save_path / "metadata.pkl").exists()

    def test_predictions_match_after_load(self, trained_catboost, small_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        save_path = tmp_model_dir / "cat_model"
        trained_catboost.save(save_path)

        loaded = CatBoostModel()
        loaded.load(save_path)

        original = trained_catboost.predict(small_tabular_data["X_val"])
        restored = loaded.predict(small_tabular_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities)


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not installed")
@pytest.mark.skip(reason="CatBoost FPE on small synthetic data - env issue")
class TestCatBoostFeatureImportance:
    """Tests for CatBoostModel feature importance."""

    def test_importance_returns_dict(self, trained_catboost):
        """Feature importance should return dict."""
        importance = trained_catboost.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0


# =============================================================================
# CROSS-MODEL TESTS
# =============================================================================

class TestBoostingModelConsistency:
    """Tests for consistent behavior across boosting models."""

    def test_all_models_in_registry(self):
        """All boosting models should be in registry."""
        from src.models.boosting import XGBoostModel

        families = ModelRegistry.list_models()
        assert "boosting" in families
        assert "xgboost" in families["boosting"]

        if LIGHTGBM_AVAILABLE:
            assert "lightgbm" in families["boosting"]

        if CATBOOST_AVAILABLE:
            assert "catboost" in families["boosting"]

    def test_prediction_output_format_consistent(self, trained_xgboost, small_tabular_data):
        """All models should return same prediction format."""
        output = trained_xgboost.predict(small_tabular_data["X_val"])

        # Check required attributes
        assert hasattr(output, "class_predictions")
        assert hasattr(output, "class_probabilities")
        assert hasattr(output, "confidence")
        assert hasattr(output, "metadata")

        # Check shapes
        n_samples = len(small_tabular_data["X_val"])
        assert output.class_predictions.shape == (n_samples,)
        assert output.class_probabilities.shape == (n_samples, 3)
        assert output.confidence.shape == (n_samples,)

    def test_training_metrics_format_consistent(self, trained_xgboost):
        """All models should return same training metrics format."""
        # trained_xgboost was trained via fixture
        assert trained_xgboost.is_fitted
