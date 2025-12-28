"""
Tests for Trainer - Model training orchestration.

Tests cover:
- Trainer initialization
- Training with mock data container
- Config merging (defaults + yaml + cli)
- Output directory creation
- Metrics saving
- Model saving
- TrainerConfig validation
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.config import (
    TrainerConfig,
    merge_configs,
    build_config,
    validate_config,
    save_config,
    save_config_json,
)
from src.models.trainer import Trainer, compute_classification_metrics


# =============================================================================
# TRAINER CONFIG TESTS
# =============================================================================

class TestTrainerConfig:
    """Tests for TrainerConfig dataclass."""

    def test_basic_creation(self):
        """Should create TrainerConfig with required fields."""
        config = TrainerConfig(model_name="xgboost")
        assert config.model_name == "xgboost"
        assert config.horizon == 20  # default

    def test_all_defaults(self):
        """Should have sensible defaults."""
        config = TrainerConfig(model_name="xgboost")
        assert config.horizon == 20
        assert config.batch_size == 256
        assert config.max_epochs == 100
        assert config.early_stopping_patience == 15
        assert config.random_seed == 42
        assert config.mixed_precision is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = TrainerConfig(
            model_name="lstm",
            horizon=10,
            batch_size=128,
            max_epochs=50,
            sequence_length=30,
        )
        assert config.model_name == "lstm"
        assert config.horizon == 10
        assert config.batch_size == 128
        assert config.max_epochs == 50
        assert config.sequence_length == 30

    def test_model_config_dict(self):
        """Should store model-specific config."""
        model_config = {"hidden_size": 256, "num_layers": 2}
        config = TrainerConfig(
            model_name="lstm",
            model_config=model_config,
        )
        assert config.model_config == model_config

    def test_invalid_horizon_raises(self):
        """Should raise ValueError for invalid horizon."""
        with pytest.raises(ValueError, match="horizon must be positive"):
            TrainerConfig(model_name="xgboost", horizon=0)

    def test_invalid_batch_size_raises(self):
        """Should raise ValueError for invalid batch_size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainerConfig(model_name="xgboost", batch_size=0)

    def test_invalid_max_epochs_raises(self):
        """Should raise ValueError for invalid max_epochs."""
        with pytest.raises(ValueError, match="max_epochs must be positive"):
            TrainerConfig(model_name="xgboost", max_epochs=0)

    def test_to_dict(self):
        """Should convert to dict for serialization."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=15,
            model_config={"n_estimators": 100},
        )
        result = config.to_dict()

        assert result["model_name"] == "xgboost"
        assert result["horizon"] == 15
        assert result["model_config"]["n_estimators"] == 100
        assert isinstance(result["output_dir"], str)

    def test_from_dict(self):
        """Should create from dict."""
        data = {
            "model_name": "lstm",
            "horizon": 10,
            "batch_size": 64,
        }
        config = TrainerConfig.from_dict(data)

        assert config.model_name == "lstm"
        assert config.horizon == 10
        assert config.batch_size == 64


# =============================================================================
# CONFIG MERGING TESTS
# =============================================================================

class TestConfigMerging:
    """Tests for configuration merging utilities."""

    def test_merge_configs_basic(self):
        """Should merge two configs with override priority."""
        base = {"a": 1, "b": 2, "c": 3}
        override = {"b": 20, "d": 4}

        result = merge_configs(base, override)

        assert result["a"] == 1
        assert result["b"] == 20  # overridden
        assert result["c"] == 3
        assert result["d"] == 4  # added

    def test_merge_configs_deep(self):
        """Should merge nested dicts recursively."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 20, "c": 3}}

        result = merge_configs(base, override, deep=True)

        assert result["outer"]["a"] == 1
        assert result["outer"]["b"] == 20
        assert result["outer"]["c"] == 3

    def test_merge_configs_shallow(self):
        """Should replace nested dicts when deep=False."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"c": 3}}

        result = merge_configs(base, override, deep=False)

        assert result["outer"] == {"c": 3}

    def test_build_config_with_cli_args(self):
        """CLI args should have highest priority."""
        defaults = {"param1": 1, "param2": 2}
        cli_args = {"param1": 100}

        with patch("src.models.config.find_model_config", return_value=None):
            result = build_config(
                model_name="test",
                cli_args=cli_args,
                defaults=defaults,
            )

        assert result["param1"] == 100
        assert result["param2"] == 2

    def test_build_config_filters_none(self):
        """Should filter out None values from CLI args."""
        defaults = {"param1": 1, "param2": 2}
        cli_args = {"param1": None, "param2": 20}

        with patch("src.models.config.find_model_config", return_value=None):
            result = build_config(
                model_name="test",
                cli_args=cli_args,
                defaults=defaults,
            )

        assert result["param1"] == 1  # Not overridden (was None)
        assert result["param2"] == 20


# =============================================================================
# CONFIG VALIDATION TESTS
# =============================================================================

class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_valid_config(self):
        """Should return empty list for valid config."""
        config = {
            "model_name": "xgboost",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
        }
        errors = validate_config(config, "xgboost")
        assert len(errors) == 0

    def test_validate_out_of_range(self):
        """Should return errors for out-of-range values."""
        config = {
            "model_name": "xgboost",
            "n_estimators": 100000,  # Too high
            "learning_rate": 5.0,    # Too high
        }
        errors = validate_config(config, "xgboost")
        assert any("n_estimators" in e for e in errors)
        assert any("learning_rate" in e for e in errors)

    def test_validate_invalid_type(self):
        """Should return errors for invalid types."""
        config = {
            "model_name": "xgboost",
            "n_estimators": "one hundred",  # Should be numeric
        }
        errors = validate_config(config, "xgboost")
        assert any("n_estimators" in e and "numeric" in e for e in errors)


# =============================================================================
# CONFIG SAVING TESTS
# =============================================================================

class TestConfigSaving:
    """Tests for config saving utilities."""

    def test_save_config_yaml(self, tmp_path):
        """Should save config to YAML file."""
        config = {"param1": 1, "param2": "value"}
        path = tmp_path / "config.yaml"

        save_config(config, path)

        assert path.exists()
        with open(path) as f:
            content = f.read()
        assert "param1: 1" in content

    def test_save_config_json(self, tmp_path):
        """Should save config to JSON file."""
        config = {"param1": 1, "param2": "value"}
        path = tmp_path / "config.json"

        save_config_json(config, path)

        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["param1"] == 1
        assert loaded["param2"] == "value"

    def test_save_creates_parent_dirs(self, tmp_path):
        """Should create parent directories."""
        config = {"param": 1}
        path = tmp_path / "nested" / "dir" / "config.json"

        save_config_json(config, path)

        assert path.exists()


# =============================================================================
# CLASSIFICATION METRICS TESTS
# =============================================================================

class TestClassificationMetrics:
    """Tests for compute_classification_metrics function."""

    def test_basic_metrics(self):
        """Should compute basic classification metrics."""
        # Trading labels: -1=short, 0=neutral, 1=long
        y_true = np.array([-1, -1, 0, 0, 1, 1])
        y_pred = np.array([-1, -1, 0, 0, 1, 1])  # Perfect predictions
        y_proba = np.array([
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.05, 0.9, 0.05],
            [0.1, 0.1, 0.8],
            [0.05, 0.05, 0.9],
        ])

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["n_samples"] == 6

    def test_imperfect_predictions(self):
        """Should compute metrics for imperfect predictions."""
        # Trading labels: -1=short, 0=neutral, 1=long
        y_true = np.array([-1, -1, 0, 0, 1, 1])
        y_pred = np.array([-1, 0, 0, 1, 1, -1])  # Some errors
        y_proba = np.ones((6, 3)) / 3

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        assert 0 < metrics["accuracy"] < 1
        assert 0 < metrics["macro_f1"] < 1

    def test_confusion_matrix_included(self):
        """Should include confusion matrix."""
        # Trading labels: -1=short, 0=neutral, 1=long
        y_true = np.array([-1, -1, 0, 0])
        y_pred = np.array([-1, -1, 0, 0])
        y_proba = np.ones((4, 3)) / 3

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        assert "confusion_matrix" in metrics
        assert isinstance(metrics["confusion_matrix"], list)

    def test_per_class_f1_included(self):
        """Should include per-class F1 scores."""
        # Trading labels: -1=short, 0=neutral, 1=long
        y_true = np.array([-1, -1, 0, 0, 1, 1])
        y_pred = np.array([-1, -1, 0, 0, 1, 1])
        y_proba = np.ones((6, 3)) / 3

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        assert "per_class_f1" in metrics
        assert "short" in metrics["per_class_f1"]
        assert "neutral" in metrics["per_class_f1"]
        assert "long" in metrics["per_class_f1"]


# =============================================================================
# TRAINER TESTS
# =============================================================================

class TestTrainerInitialization:
    """Tests for Trainer initialization."""

    def test_basic_init(self, tmp_output_dir):
        """Should initialize trainer with config."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={"n_estimators": 10, "verbosity": 0},
        )

        trainer = Trainer(config)

        assert trainer.config == config
        assert trainer.run_id is not None
        assert "xgboost" in trainer.run_id
        assert "h20" in trainer.run_id

    def test_model_created_from_registry(self, tmp_output_dir):
        """Should create model from registry."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={"n_estimators": 10},
        )

        trainer = Trainer(config)

        from src.models.boosting import XGBoostModel
        assert isinstance(trainer.model, XGBoostModel)

    def test_unknown_model_raises(self, tmp_output_dir):
        """Should raise for unknown model."""
        config = TrainerConfig(
            model_name="nonexistent_model",
            horizon=20,
            output_dir=tmp_output_dir,
        )

        with pytest.raises(ValueError, match="Unknown model"):
            Trainer(config)


class TestTrainerRun:
    """Tests for Trainer.run() method."""

    def test_run_creates_output_dirs(self, mock_data_container, tmp_output_dir):
        """Training run should create output directory structure."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=False)

        output_path = Path(results["output_path"])
        assert (output_path / "config").exists()
        assert (output_path / "checkpoints").exists()
        assert (output_path / "predictions").exists()
        assert (output_path / "metrics").exists()

    def test_run_returns_results_dict(self, mock_data_container, tmp_output_dir):
        """Training run should return results dict."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=True)

        assert "run_id" in results
        assert "model_name" in results
        assert "horizon" in results
        assert "training_metrics" in results
        assert "evaluation_metrics" in results
        assert "total_time_seconds" in results

    def test_run_saves_model(self, mock_data_container, tmp_output_dir):
        """Training run should save model checkpoint."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=False)

        output_path = Path(results["output_path"])
        assert (output_path / "checkpoints" / "best_model").exists()

    def test_run_saves_metrics(self, mock_data_container, tmp_output_dir):
        """Training run should save metrics files."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=False)

        output_path = Path(results["output_path"])
        assert (output_path / "metrics" / "training_metrics.json").exists()
        assert (output_path / "metrics" / "evaluation_metrics.json").exists()

    def test_run_saves_predictions(self, mock_data_container, tmp_output_dir):
        """Training run should save validation predictions."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=False)

        output_path = Path(results["output_path"])
        assert (output_path / "predictions" / "val_predictions.npz").exists()

    def test_run_saves_feature_importance(self, mock_data_container, tmp_output_dir):
        """Training run should save feature importance for boosting models."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=False)

        output_path = Path(results["output_path"])
        assert (output_path / "metrics" / "feature_importance.json").exists()

    def test_skip_save_skips_artifacts(self, mock_data_container, tmp_output_dir):
        """skip_save=True should skip saving artifacts."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=True)

        output_path = Path(results["output_path"])
        # Config is still saved, but model and predictions are not
        assert (output_path / "config").exists()
        assert not (output_path / "predictions" / "val_predictions.npz").exists()


class TestTrainerEvaluationMetrics:
    """Tests for evaluation metrics in Trainer."""

    def test_evaluation_metrics_structure(self, mock_data_container, tmp_output_dir):
        """Evaluation metrics should have expected structure."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=True)

        eval_metrics = results["evaluation_metrics"]
        assert "accuracy" in eval_metrics
        assert "macro_f1" in eval_metrics
        assert "weighted_f1" in eval_metrics
        assert "precision" in eval_metrics
        assert "recall" in eval_metrics
        assert "per_class_f1" in eval_metrics
        assert "confusion_matrix" in eval_metrics
        assert "trading" in eval_metrics

    def test_trading_metrics_included(self, mock_data_container, tmp_output_dir):
        """Should include basic trading metrics."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=True)

        trading = results["evaluation_metrics"]["trading"]
        assert "long_signals" in trading
        assert "short_signals" in trading
        assert "neutral_signals" in trading
        assert "position_win_rate" in trading


class TestTrainerWithDifferentModels:
    """Tests for Trainer with different model types."""

    def test_trainer_with_xgboost(self, mock_data_container, tmp_output_dir):
        """Should train XGBoost model."""
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=True)

        assert results["model_name"] == "xgboost"
        assert "accuracy" in results["evaluation_metrics"]

    @pytest.mark.skipif(
        True,  # Skip LightGBM/CatBoost trainer tests to keep test fast
        reason="Optional boosting model tests"
    )
    def test_trainer_with_lightgbm(self, mock_data_container, tmp_output_dir):
        """Should train LightGBM model."""
        try:
            import lightgbm
        except ImportError:
            pytest.skip("LightGBM not installed")

        config = TrainerConfig(
            model_name="lightgbm",
            horizon=20,
            output_dir=tmp_output_dir,
            model_config={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": -1,
            },
        )

        trainer = Trainer(config)
        results = trainer.run(mock_data_container, skip_save=True)

        assert results["model_name"] == "lightgbm"


# =============================================================================
# TRAINER CONVENIENCE FUNCTIONS TESTS
# =============================================================================

class TestTrainModelFunction:
    """Tests for train_model convenience function."""

    def test_train_model_basic(self, mock_data_container, tmp_output_dir):
        """Should train model using convenience function."""
        from src.models.trainer import train_model

        results = train_model(
            model_name="xgboost",
            container=mock_data_container,
            horizon=20,
            config_overrides={
                "n_estimators": 5,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
            output_dir=tmp_output_dir,
        )

        assert "run_id" in results
        assert results["model_name"] == "xgboost"
        assert results["horizon"] == 20
