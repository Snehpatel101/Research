"""
Integration tests for full training pipeline.

Tests end-to-end workflows:
- Training with mock data container
- Multiple model types through same pipeline
- Model save/load/predict cycle
- Ensemble training with base models
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.models import ModelRegistry
from src.models.config import TrainerConfig
from src.models.trainer import Trainer


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================

class TestTrainingPipeline:
    """Test complete training workflow from config to trained model."""

    def test_xgboost_full_pipeline(self, mock_data_container, tmp_output_dir):
        """XGBoost should train, evaluate, and save successfully."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 10,
                "max_depth": 3,
                "early_stopping_rounds": 3,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=False)

        # Check training succeeded
        assert "run_id" in results
        assert results["model_name"] == "xgboost"
        assert "training_metrics" in results
        assert "evaluation_metrics" in results

        # Check files created
        output_path = Path(results["output_path"])
        assert output_path.exists()
        assert (output_path / "checkpoints" / "best_model").exists()
        assert (output_path / "metrics" / "evaluation_metrics.json").exists()

        # Check predictions saved
        assert (output_path / "predictions" / "val_predictions.npz").exists()

    def test_random_forest_full_pipeline(self, mock_data_container, tmp_output_dir):
        """Random Forest should train through same pipeline."""
        config = TrainerConfig(
            model_name="random_forest",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 10,
                "max_depth": 3,
                "n_jobs": 1,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=True)

        assert results["model_name"] == "random_forest"
        assert 0 <= results["evaluation_metrics"]["accuracy"] <= 1

    def test_logistic_full_pipeline(self, mock_data_container, tmp_output_dir):
        """Logistic Regression should train through same pipeline."""
        config = TrainerConfig(
            model_name="logistic",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "max_iter": 100,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=True)

        assert results["model_name"] == "logistic"
        assert 0 <= results["evaluation_metrics"]["accuracy"] <= 1


class TestModelSaveLoadCycle:
    """Test complete save/load/predict cycle."""

    def test_model_save_load_predict(self, mock_data_container, tmp_output_dir):
        """Saved model should load and produce identical predictions."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 10,
                "max_depth": 3,
                "early_stopping_rounds": 3,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=False)

        # Get original predictions
        original_model = trainer.model
        X_val = mock_data_container.get_array("val", "features")
        original_preds = original_model.predict(X_val)

        # Load saved model
        saved_path = Path(results["output_path"]) / "checkpoints" / "best_model"
        loaded_model = ModelRegistry.create("xgboost")
        loaded_model.load(saved_path)

        # Get predictions from loaded model
        loaded_preds = loaded_model.predict(X_val)

        # Predictions should match
        assert np.array_equal(
            original_preds.class_predictions,
            loaded_preds.class_predictions
        )


class TestMultipleHorizons:
    """Test training across multiple horizons."""

    def test_train_multiple_horizons(self, mock_data_container_factory, tmp_output_dir):
        """Should train same model type across different horizons."""
        results = {}

        for horizon in [5, 10, 20]:
            container = mock_data_container_factory(horizon=horizon)
            config = TrainerConfig(
                model_name="xgboost",
                horizon=horizon,
                output_dir=tmp_output_dir,
                model_config={
                    "n_estimators": 10,
                    "max_depth": 3,
                    "early_stopping_rounds": 3,
                    "verbosity": 0,
                },
            )

            trainer = Trainer(config)
            result = trainer.run(container, skip_save=True)
            results[horizon] = result

        # All horizons should produce results
        assert len(results) == 3
        assert all(h in results for h in [5, 10, 20])

        # Each should have evaluation metrics
        for h, r in results.items():
            assert r["horizon"] == h
            assert "accuracy" in r["evaluation_metrics"]


# =============================================================================
# ENSEMBLE PIPELINE TESTS
# =============================================================================

class TestEnsemblePipeline:
    """Test ensemble training workflows."""

    def test_voting_ensemble_from_scratch(self, mock_data_container, tmp_output_dir):
        """Voting ensemble should train base models and combine."""
        config = TrainerConfig(
            model_name="voting",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "voting": "soft",
                "base_model_names": ["random_forest", "logistic"],
                "base_model_configs": {
                    "random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1},
                    "logistic": {"max_iter": 100},
                },
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=True)

        assert results["model_name"] == "voting"
        assert 0 <= results["evaluation_metrics"]["accuracy"] <= 1

    def test_stacking_ensemble_pipeline(self, mock_data_container, tmp_output_dir):
        """Stacking ensemble should train with OOF predictions."""
        config = TrainerConfig(
            model_name="stacking",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "base_model_names": ["random_forest", "logistic"],
                "base_model_configs": {
                    "random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1},
                    "logistic": {"max_iter": 100},
                },
                "meta_learner_name": "logistic",
                "meta_learner_config": {"max_iter": 100},
                "n_folds": 2,
                # Override purge/embargo for small test data
                "purge_bars": 5,
                "embargo_bars": 10,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=True)

        assert results["model_name"] == "stacking"
        assert "n_folds" in results["training_metrics"]["metadata"]


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestPipelineErrorHandling:
    """Test pipeline error handling."""

    def test_invalid_model_raises(self, tmp_output_dir):
        """Unknown model name should raise ValueError."""
        config = TrainerConfig(
            model_name="nonexistent_model",
            horizon=20,
            output_dir=tmp_output_dir,
        )

        with pytest.raises(ValueError, match="Unknown model"):
            Trainer(config)

    def test_empty_data_raises(self, tmp_output_dir):
        """Empty data container should raise."""
        from tests.models.conftest import create_mock_container

        # Create container with no data
        empty_container = create_mock_container(n_train=0, n_val=0)

        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={"n_estimators": 5, "verbosity": 0},
        )

        trainer = Trainer(config)

        with pytest.raises(Exception):  # Various exceptions possible
            trainer.run(empty_container, skip_save=True)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_data_container_factory():
    """Factory to create mock containers for different horizons."""
    from tests.models.conftest import create_mock_container

    def _factory(horizon: int = 20, n_train: int = 200, n_val: int = 50):
        return create_mock_container(
            n_train=n_train,
            n_val=n_val,
            n_features=20,
            horizon=horizon,
        )

    return _factory
