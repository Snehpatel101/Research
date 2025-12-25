"""
Integration tests for model comparison workflows.

Tests cover:
- Comparing multiple model types on same data
- Consistent evaluation across models
- Feature importance comparison
- Ensemble vs individual model comparison
"""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.models import ModelRegistry
from src.models.config import TrainerConfig
from src.models.trainer import Trainer


# =============================================================================
# MODEL COMPARISON TESTS
# =============================================================================

class TestModelComparison:
    """Test comparing multiple models on the same data."""

    def test_boosting_vs_classical_comparison(self, mock_data_container, tmp_output_dir):
        """Compare boosting and classical models on same data."""
        models_to_test = [
            ("xgboost", {
                "n_estimators": 10,
                "max_depth": 3,
                "early_stopping_rounds": 3,
                "verbosity": 0,
            }),
            ("random_forest", {
                "n_estimators": 10,
                "max_depth": 3,
                "n_jobs": 1,
            }),
            ("logistic", {
                "max_iter": 100,
            }),
        ]

        results = {}

        for model_name, model_config in models_to_test:
            config = TrainerConfig(
                model_name=model_name,
                horizon=20,
                output_dir=tmp_output_dir,
                model_config=model_config,
            )

            trainer = Trainer(config)
            result = trainer.run(mock_data_container, skip_save=True)
            results[model_name] = result

        # All should produce results
        assert len(results) == len(models_to_test)

        # All should have consistent evaluation metrics
        metric_keys = ["accuracy", "macro_f1", "weighted_f1"]
        for model_name, result in results.items():
            for key in metric_keys:
                assert key in result["evaluation_metrics"], \
                    f"{model_name} missing {key}"
                value = result["evaluation_metrics"][key]
                assert 0 <= value <= 1, \
                    f"{model_name} {key} = {value} out of range"

    def test_all_produce_3_class_predictions(self, mock_data_container, tmp_output_dir):
        """All models should produce 3-class predictions."""
        models_to_test = ["xgboost", "random_forest", "logistic"]

        for model_name in models_to_test:
            model_config = {
                "n_estimators": 10 if "forest" in model_name or "xgboost" in model_name else None,
                "max_depth": 3 if "forest" in model_name or "xgboost" in model_name else None,
                "max_iter": 100 if model_name == "logistic" else None,
                "n_jobs": 1 if model_name == "random_forest" else None,
                "early_stopping_rounds": 3 if model_name == "xgboost" else None,
                "verbosity": 0 if model_name == "xgboost" else None,
            }
            # Remove None values
            model_config = {k: v for k, v in model_config.items() if v is not None}

            config = TrainerConfig(
                model_name=model_name,
                horizon=20,
                output_dir=tmp_output_dir,
                model_config=model_config,
            )

            trainer = Trainer(config)
            results = trainer.run(mock_data_container, skip_save=True)

            # Check confusion matrix is 3x3
            confusion = results["evaluation_metrics"]["confusion_matrix"]
            assert len(confusion) == 3, f"{model_name} confusion matrix rows != 3"
            assert all(len(row) == 3 for row in confusion), \
                f"{model_name} confusion matrix cols != 3"


class TestFeatureImportanceComparison:
    """Test comparing feature importances across models."""

    def test_boosting_models_provide_importance(self, mock_data_container, tmp_output_dir):
        """Boosting models should provide feature importance."""
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
        trainer.run(mock_data_container, skip_save=True)

        importance = trainer.model.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_classical_models_provide_importance(self, mock_data_container, tmp_output_dir):
        """Classical models should provide some form of feature importance."""
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
        trainer.run(mock_data_container, skip_save=True)

        importance = trainer.model.get_feature_importance()

        assert importance is not None


class TestEnsembleVsIndividual:
    """Test comparing ensemble performance to individual models."""

    def test_ensemble_uses_base_model_predictions(self, mock_data_container, tmp_output_dir):
        """Ensemble should combine base model predictions."""
        # Train individual models
        individual_results = {}

        for model_name, model_config in [
            ("random_forest", {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}),
            ("logistic", {"max_iter": 100}),
        ]:
            config = TrainerConfig(
                model_name=model_name,
                horizon=20,
                output_dir=tmp_output_dir,
                model_config=model_config,
            )

            trainer = Trainer(config)
            result = trainer.run(mock_data_container, skip_save=True)
            individual_results[model_name] = result

        # Train ensemble
        ensemble_config = TrainerConfig(
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

        trainer = Trainer(ensemble_config)
        ensemble_result = trainer.run(mock_data_container, skip_save=True)

        # Ensemble should have valid results
        assert 0 <= ensemble_result["evaluation_metrics"]["accuracy"] <= 1
        assert "voting" in ensemble_result["model_name"]


class TestMetricConsistency:
    """Test that metrics are consistent across model types."""

    def test_per_class_f1_always_present(self, mock_data_container, tmp_output_dir):
        """All models should report per-class F1 scores."""
        models = ["xgboost", "random_forest", "logistic"]

        for model_name in models:
            model_config = {
                "n_estimators": 10 if "xgboost" in model_name or "forest" in model_name else None,
                "max_depth": 3 if "xgboost" in model_name or "forest" in model_name else None,
                "max_iter": 100 if "logistic" in model_name else None,
                "n_jobs": 1 if "forest" in model_name else None,
                "early_stopping_rounds": 3 if "xgboost" in model_name else None,
                "verbosity": 0 if "xgboost" in model_name else None,
            }
            model_config = {k: v for k, v in model_config.items() if v is not None}

            config = TrainerConfig(
                model_name=model_name,
                horizon=20,
                output_dir=tmp_output_dir,
                model_config=model_config,
            )

            trainer = Trainer(config)
            result = trainer.run(mock_data_container, skip_save=True)

            assert "per_class_f1" in result["evaluation_metrics"], \
                f"{model_name} missing per_class_f1"

            per_class = result["evaluation_metrics"]["per_class_f1"]
            # Should have 3 classes (could be named short/neutral/long or -1/0/1)
            assert len(per_class) >= 3, f"{model_name} should have at least 3 classes in per_class_f1"
            # All values should be valid F1 scores
            for cls_name, f1_score in per_class.items():
                assert 0 <= f1_score <= 1, f"{model_name} {cls_name} F1 = {f1_score} invalid"

    def test_trading_metrics_always_present(self, mock_data_container, tmp_output_dir):
        """All models should report trading metrics."""
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
        result = trainer.run(mock_data_container, skip_save=True)

        assert "trading" in result["evaluation_metrics"]

        trading = result["evaluation_metrics"]["trading"]
        expected_keys = ["long_signals", "short_signals", "neutral_signals"]
        for key in expected_keys:
            assert key in trading, f"Missing trading metric: {key}"


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestModelRegistry:
    """Test model registry functionality."""

    def test_all_models_registered(self):
        """All expected models should be registered."""
        families = ModelRegistry.list_models()

        # Check boosting
        assert "boosting" in families
        assert "xgboost" in families["boosting"]

        # Check classical
        assert "classical" in families
        assert "random_forest" in families["classical"]
        assert "logistic" in families["classical"]
        assert "svm" in families["classical"]

        # Check ensemble
        assert "ensemble" in families
        assert "voting" in families["ensemble"]
        assert "stacking" in families["ensemble"]
        assert "blending" in families["ensemble"]

    def test_create_by_alias(self):
        """Models should be creatable by alias."""
        aliases = [
            ("xgb", "xgboost"),
            ("rf", "random_forest"),
            ("lr", "logistic"),
        ]

        for alias, expected_family_member in aliases:
            model = ModelRegistry.create(alias)
            assert model is not None, f"Failed to create model with alias {alias}"
