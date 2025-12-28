"""
Tests for Ensemble Models - Voting, Stacking, Blending.

Tests cover:
- Model initialization and properties
- Training with pre-trained base models
- Training from scratch
- Prediction output format
- Save/load roundtrip
- Different voting strategies
- Ensemble-specific functionality
"""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.models.ensemble import VotingEnsemble, StackingEnsemble, BlendingEnsemble
from src.models.boosting import XGBoostModel
from src.models.classical import RandomForestModel, LogisticModel


# =============================================================================
# VOTING ENSEMBLE TESTS
# =============================================================================

class TestVotingEnsembleProperties:
    """Tests for VotingEnsemble properties."""

    def test_model_family(self):
        """Model family should be 'ensemble'."""
        model = VotingEnsemble()
        assert model.model_family == "ensemble"

    def test_requires_scaling_false_default(self):
        """VotingEnsemble should not require scaling by default."""
        model = VotingEnsemble()
        assert model.requires_scaling is False

    def test_requires_sequences_false_default(self):
        """VotingEnsemble should not require sequences by default."""
        model = VotingEnsemble()
        assert model.requires_sequences is False

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        model = VotingEnsemble()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        model = VotingEnsemble()
        config = model.get_default_config()

        expected_keys = ["voting", "weights", "base_model_names", "base_model_configs"]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


class TestVotingEnsembleWithPretrainedModels:
    """Tests for VotingEnsemble with pre-trained base models."""

    def test_set_base_models(self, trained_base_models):
        """Should accept pre-trained models."""
        ensemble = VotingEnsemble(config={"voting": "soft"})
        ensemble.set_base_models(trained_base_models)

        assert ensemble.is_fitted is True
        assert len(ensemble._base_models) == len(trained_base_models)

    def test_set_base_models_with_weights(self, trained_base_models):
        """Should accept weights for models."""
        ensemble = VotingEnsemble(config={"voting": "soft"})
        weights = [0.5, 0.5]
        ensemble.set_base_models(trained_base_models, weights=weights)

        assert ensemble.is_fitted is True
        assert ensemble._weights is not None

    def test_set_base_models_empty_raises(self):
        """Empty model list should raise."""
        ensemble = VotingEnsemble()
        with pytest.raises(ValueError, match="at least one"):
            ensemble.set_base_models([])

    def test_set_base_models_unfitted_raises(self, small_tabular_data):
        """Unfitted models should raise."""
        ensemble = VotingEnsemble()
        unfitted_model = XGBoostModel()

        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.set_base_models([unfitted_model])


class TestVotingEnsembleTraining:
    """Tests for VotingEnsemble training from scratch."""

    def test_fit_with_base_model_names(self, small_tabular_data, fast_voting_config):
        """Should train when base_model_names provided."""
        ensemble = VotingEnsemble(config=fast_voting_config)
        metrics = ensemble.fit(
            small_tabular_data["X_train"],
            small_tabular_data["y_train"],
            small_tabular_data["X_val"],
            small_tabular_data["y_val"],
        )

        assert ensemble.is_fitted is True
        assert metrics.epochs_trained >= 1
        assert 0 <= metrics.val_accuracy <= 1

    def test_fit_no_base_model_names_raises(self, small_tabular_data):
        """Fit without base_model_names should raise."""
        ensemble = VotingEnsemble(config={"voting": "soft"})

        with pytest.raises(ValueError, match="No base_model_names"):
            ensemble.fit(
                small_tabular_data["X_train"],
                small_tabular_data["y_train"],
                small_tabular_data["X_val"],
                small_tabular_data["y_val"],
            )


class TestVotingEnsemblePrediction:
    """Tests for VotingEnsemble prediction."""

    def test_predict_soft_voting(self, trained_voting_soft, small_tabular_data):
        """Soft voting should return averaged probabilities."""
        output = trained_voting_soft.predict(small_tabular_data["X_val"])

        assert output.n_samples == len(small_tabular_data["X_val"])
        assert output.n_classes == 3
        assert output.metadata["voting"] == "soft"

    def test_predict_hard_voting(self, trained_voting_hard, small_tabular_data):
        """Hard voting should return majority vote."""
        output = trained_voting_hard.predict(small_tabular_data["X_val"])

        assert output.n_samples == len(small_tabular_data["X_val"])
        assert output.metadata["voting"] == "hard"

    def test_predict_class_labels(self, trained_voting_soft, small_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_voting_soft.predict(small_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_voting_soft, small_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_voting_soft.predict(small_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_predict_unfitted_raises(self, small_tabular_data):
        """Prediction on unfitted model should raise."""
        model = VotingEnsemble()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(small_tabular_data["X_val"])


class TestVotingEnsembleSaveLoad:
    """Tests for VotingEnsemble serialization."""

    def test_save_creates_files(self, trained_voting_from_names, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "voting_model"
        trained_voting_from_names.save(save_path)

        assert (save_path / "ensemble_metadata.joblib").exists()
        assert (save_path / "base_model_0").exists()
        assert (save_path / "base_model_1").exists()

    def test_predictions_match_after_load(self, trained_voting_from_names, small_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        save_path = tmp_model_dir / "voting_model"
        trained_voting_from_names.save(save_path)

        loaded = VotingEnsemble()
        loaded.load(save_path)

        original = trained_voting_from_names.predict(small_tabular_data["X_val"])
        restored = loaded.predict(small_tabular_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-5)


class TestVotingEnsembleFeatureImportance:
    """Tests for VotingEnsemble feature importance."""

    def test_get_feature_importance(self, trained_voting_soft):
        """Should aggregate feature importances from base models."""
        importance = trained_voting_soft.get_feature_importance()

        # XGBoost and RF both provide feature importance
        assert importance is not None
        assert isinstance(importance, dict)

    def test_feature_importance_unfitted_returns_none(self):
        """Feature importance on unfitted model should return None."""
        model = VotingEnsemble()
        assert model.get_feature_importance() is None


# =============================================================================
# STACKING ENSEMBLE TESTS
# =============================================================================

class TestStackingEnsembleProperties:
    """Tests for StackingEnsemble properties."""

    def test_model_family(self):
        """Model family should be 'ensemble'."""
        model = StackingEnsemble()
        assert model.model_family == "ensemble"

    def test_requires_scaling_false(self):
        """StackingEnsemble handles scaling internally."""
        model = StackingEnsemble()
        assert model.requires_scaling is False

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        model = StackingEnsemble()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        model = StackingEnsemble()
        config = model.get_default_config()

        expected_keys = [
            "base_model_names", "meta_learner_name",
            "n_folds", "use_probabilities", "passthrough"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


class TestStackingEnsembleTraining:
    """Tests for StackingEnsemble training."""

    def test_fit_returns_metrics(self, medium_tabular_data, fast_stacking_config):
        """Training should return TrainingMetrics."""
        ensemble = StackingEnsemble(config=fast_stacking_config)
        metrics = ensemble.fit(
            medium_tabular_data["X_train"],
            medium_tabular_data["y_train"],
            medium_tabular_data["X_val"],
            medium_tabular_data["y_val"],
        )

        assert ensemble.is_fitted is True
        assert 0 <= metrics.val_accuracy <= 1
        assert "n_folds" in metrics.metadata

    def test_fit_no_base_model_names_raises(self, small_tabular_data):
        """Fit without base_model_names should raise."""
        ensemble = StackingEnsemble(config={"base_model_names": []})

        with pytest.raises(ValueError, match="No base_model_names"):
            ensemble.fit(
                small_tabular_data["X_train"],
                small_tabular_data["y_train"],
                small_tabular_data["X_val"],
                small_tabular_data["y_val"],
            )

    def test_metadata_includes_fold_info(self, medium_tabular_data, fast_stacking_config):
        """Training metadata should include fold information."""
        ensemble = StackingEnsemble(config=fast_stacking_config)
        metrics = ensemble.fit(
            medium_tabular_data["X_train"],
            medium_tabular_data["y_train"],
            medium_tabular_data["X_val"],
            medium_tabular_data["y_val"],
        )

        assert "n_folds" in metrics.metadata
        assert metrics.metadata["n_folds"] == fast_stacking_config["n_folds"]


class TestStackingEnsemblePrediction:
    """Tests for StackingEnsemble prediction."""

    def test_predict_returns_output(self, trained_stacking, medium_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_stacking.predict(medium_tabular_data["X_val"])

        assert output.n_samples == len(medium_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_stacking, medium_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_stacking.predict(medium_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_stacking, medium_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_stacking.predict(medium_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


class TestStackingEnsembleSaveLoad:
    """Tests for StackingEnsemble serialization."""

    def test_save_creates_files(self, trained_stacking, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "stacking_model"
        trained_stacking.save(save_path)

        assert (save_path / "ensemble_metadata.joblib").exists()
        assert (save_path / "meta_learner").exists()
        assert (save_path / "base_models").exists()

    def test_predictions_match_after_load(self, trained_stacking, medium_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        save_path = tmp_model_dir / "stacking_model"
        trained_stacking.save(save_path)

        loaded = StackingEnsemble()
        loaded.load(save_path)

        original = trained_stacking.predict(medium_tabular_data["X_val"])
        restored = loaded.predict(medium_tabular_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-4)


# =============================================================================
# BLENDING ENSEMBLE TESTS
# =============================================================================

class TestBlendingEnsembleProperties:
    """Tests for BlendingEnsemble properties."""

    def test_model_family(self):
        """Model family should be 'ensemble'."""
        model = BlendingEnsemble()
        assert model.model_family == "ensemble"

    def test_requires_scaling_false(self):
        """BlendingEnsemble should not require scaling."""
        model = BlendingEnsemble()
        assert model.requires_scaling is False

    def test_not_fitted_initially(self):
        """Model should not be fitted initially."""
        model = BlendingEnsemble()
        assert model.is_fitted is False

    def test_default_config_keys(self):
        """Should have expected default config keys."""
        model = BlendingEnsemble()
        config = model.get_default_config()

        expected_keys = [
            "base_model_names", "meta_learner_name",
            "holdout_fraction", "use_probabilities", "retrain_on_full"
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"


class TestBlendingEnsembleTraining:
    """Tests for BlendingEnsemble training."""

    def test_fit_returns_metrics(self, medium_tabular_data, fast_blending_config):
        """Training should return TrainingMetrics."""
        ensemble = BlendingEnsemble(config=fast_blending_config)
        metrics = ensemble.fit(
            medium_tabular_data["X_train"],
            medium_tabular_data["y_train"],
            medium_tabular_data["X_val"],
            medium_tabular_data["y_val"],
        )

        assert ensemble.is_fitted is True
        assert 0 <= metrics.val_accuracy <= 1
        assert "holdout_fraction" in metrics.metadata

    def test_fit_no_base_model_names_raises(self, small_tabular_data):
        """Fit without base_model_names should raise."""
        ensemble = BlendingEnsemble(config={"base_model_names": []})

        with pytest.raises(ValueError, match="No base_model_names"):
            ensemble.fit(
                small_tabular_data["X_train"],
                small_tabular_data["y_train"],
                small_tabular_data["X_val"],
                small_tabular_data["y_val"],
            )


class TestBlendingEnsemblePrediction:
    """Tests for BlendingEnsemble prediction."""

    def test_predict_returns_output(self, trained_blending, medium_tabular_data):
        """Prediction should return PredictionOutput."""
        output = trained_blending.predict(medium_tabular_data["X_val"])

        assert output.n_samples == len(medium_tabular_data["X_val"])
        assert output.n_classes == 3

    def test_predict_class_labels(self, trained_blending, medium_tabular_data):
        """Predictions should be in {-1, 0, 1}."""
        output = trained_blending.predict(medium_tabular_data["X_val"])
        assert set(output.class_predictions).issubset({-1, 0, 1})

    def test_predict_probabilities_sum_to_one(self, trained_blending, medium_tabular_data):
        """Probabilities should sum to 1."""
        output = trained_blending.predict(medium_tabular_data["X_val"])
        sums = output.class_probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


class TestBlendingEnsembleSaveLoad:
    """Tests for BlendingEnsemble serialization."""

    def test_save_creates_files(self, trained_blending, tmp_model_dir):
        """Save should create model files."""
        save_path = tmp_model_dir / "blending_model"
        trained_blending.save(save_path)

        assert (save_path / "ensemble_metadata.joblib").exists()
        assert (save_path / "meta_learner").exists()
        assert (save_path / "base_model_0").exists()

    def test_predictions_match_after_load(self, trained_blending, medium_tabular_data, tmp_model_dir):
        """Predictions should match after save/load."""
        save_path = tmp_model_dir / "blending_model"
        trained_blending.save(save_path)

        loaded = BlendingEnsemble()
        loaded.load(save_path)

        original = trained_blending.predict(medium_tabular_data["X_val"])
        restored = loaded.predict(medium_tabular_data["X_val"])

        assert np.allclose(original.class_probabilities, restored.class_probabilities, atol=1e-4)


# =============================================================================
# CROSS-ENSEMBLE TESTS
# =============================================================================

class TestEnsembleModelConsistency:
    """Tests for consistent behavior across ensemble models."""

    def test_all_models_in_registry(self):
        """All ensemble models should be in registry."""
        from src.models import ModelRegistry

        families = ModelRegistry.list_models()
        assert "ensemble" in families
        assert "voting" in families["ensemble"]
        assert "stacking" in families["ensemble"]
        assert "blending" in families["ensemble"]

    def test_all_not_require_sequences(self):
        """All ensemble models should not require sequences (by default)."""
        for ModelClass in [VotingEnsemble, StackingEnsemble, BlendingEnsemble]:
            model = ModelClass()
            assert model.requires_sequences is False


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def trained_base_models(small_tabular_data):
    """Provide trained base models for ensemble testing."""
    xgb = XGBoostModel(config={
        "n_estimators": 10,
        "max_depth": 3,
        "verbosity": 0,
        "early_stopping_rounds": 3,
    })
    xgb.fit(
        small_tabular_data["X_train"],
        small_tabular_data["y_train"],
        small_tabular_data["X_val"],
        small_tabular_data["y_val"],
    )

    rf = RandomForestModel(config={
        "n_estimators": 10,
        "max_depth": 3,
        "n_jobs": 1,
    })
    rf.fit(
        small_tabular_data["X_train"],
        small_tabular_data["y_train"],
        small_tabular_data["X_val"],
        small_tabular_data["y_val"],
    )

    return [xgb, rf]


@pytest.fixture
def fast_voting_config() -> Dict[str, Any]:
    """Fast voting ensemble config."""
    return {
        "voting": "soft",
        "base_model_names": ["random_forest", "logistic"],
        "base_model_configs": {
            "random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1},
            "logistic": {"max_iter": 100},
        },
    }


@pytest.fixture
def fast_stacking_config() -> Dict[str, Any]:
    """Fast stacking ensemble config."""
    return {
        "base_model_names": ["random_forest", "logistic"],
        "base_model_configs": {
            "random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1},
            "logistic": {"max_iter": 100},
        },
        "meta_learner_name": "logistic",
        "meta_learner_config": {"max_iter": 100},
        "n_folds": 2,
        "use_probabilities": True,
        # Override purge/embargo for small test data
        "purge_bars": 5,
        "embargo_bars": 10,
    }


@pytest.fixture
def fast_blending_config() -> Dict[str, Any]:
    """Fast blending ensemble config."""
    return {
        "base_model_names": ["random_forest", "logistic"],
        "base_model_configs": {
            "random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1},
            "logistic": {"max_iter": 100},
        },
        "meta_learner_name": "logistic",
        "meta_learner_config": {"max_iter": 100},
        "holdout_fraction": 0.3,
        "use_probabilities": True,
        "retrain_on_full": False,  # Faster for tests
    }


@pytest.fixture
def medium_tabular_data() -> Dict[str, np.ndarray]:
    """Generate medium-sized synthetic data for stacking/blending tests."""
    np.random.seed(42)
    n_train, n_val, n_features = 300, 60, 15

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


@pytest.fixture
def trained_voting_soft(trained_base_models):
    """Provide a trained soft-voting ensemble (from pre-trained models)."""
    ensemble = VotingEnsemble(config={"voting": "soft"})
    ensemble.set_base_models(trained_base_models)
    return ensemble


@pytest.fixture
def trained_voting_hard(trained_base_models):
    """Provide a trained hard-voting ensemble (from pre-trained models)."""
    ensemble = VotingEnsemble(config={"voting": "hard"})
    ensemble.set_base_models(trained_base_models)
    return ensemble


@pytest.fixture
def trained_voting_from_names(small_tabular_data, fast_voting_config):
    """Provide a voting ensemble trained from base_model_names (supports save/load)."""
    ensemble = VotingEnsemble(config=fast_voting_config)
    ensemble.fit(
        small_tabular_data["X_train"],
        small_tabular_data["y_train"],
        small_tabular_data["X_val"],
        small_tabular_data["y_val"],
    )
    return ensemble


@pytest.fixture
def trained_stacking(medium_tabular_data, fast_stacking_config):
    """Provide a trained stacking ensemble."""
    ensemble = StackingEnsemble(config=fast_stacking_config)
    ensemble.fit(
        medium_tabular_data["X_train"],
        medium_tabular_data["y_train"],
        medium_tabular_data["X_val"],
        medium_tabular_data["y_val"],
    )
    return ensemble


@pytest.fixture
def trained_blending(medium_tabular_data, fast_blending_config):
    """Provide a trained blending ensemble."""
    ensemble = BlendingEnsemble(config=fast_blending_config)
    ensemble.fit(
        medium_tabular_data["X_train"],
        medium_tabular_data["y_train"],
        medium_tabular_data["X_val"],
        medium_tabular_data["y_val"],
    )
    return ensemble
