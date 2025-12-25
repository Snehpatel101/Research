"""
Tests for Cross-Validation Runner.

Tests:
- CVRunner initialization
- Fold metrics computation
- Result aggregation
- Stacking dataset building
- Result serialization
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.cross_validation.cv_runner import (
    CrossValidationRunner,
    CVResult,
    FoldMetrics,
    TimeSeriesOptunaTuner,
    analyze_cv_stability,
    _grade_stability,
)
from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig


# =============================================================================
# FOLD METRICS TESTS
# =============================================================================

class TestFoldMetrics:
    """Tests for FoldMetrics dataclass."""

    def test_fold_metrics_creation(self):
        """Test FoldMetrics creation with all fields."""
        metrics = FoldMetrics(
            fold=0,
            train_size=800,
            val_size=200,
            accuracy=0.65,
            f1=0.60,
            precision=0.62,
            recall=0.58,
            training_time=10.5,
            val_loss=0.45,
        )

        assert metrics.fold == 0
        assert metrics.train_size == 800
        assert metrics.accuracy == 0.65
        assert metrics.f1 == 0.60

    def test_fold_metrics_to_dict(self):
        """Test FoldMetrics conversion to dictionary."""
        metrics = FoldMetrics(
            fold=1,
            train_size=800,
            val_size=200,
            accuracy=0.65,
            f1=0.60,
            precision=0.62,
            recall=0.58,
            training_time=10.5,
        )

        d = metrics.to_dict()

        assert d["fold"] == 1
        assert d["train_size"] == 800
        assert d["accuracy"] == 0.65
        assert "training_time" in d


# =============================================================================
# CV RESULT TESTS
# =============================================================================

class TestCVResult:
    """Tests for CVResult dataclass."""

    @pytest.fixture
    def sample_fold_metrics(self):
        """Create sample fold metrics for testing."""
        return [
            FoldMetrics(fold=i, train_size=800, val_size=200,
                       accuracy=0.6 + 0.02*i, f1=0.55 + 0.02*i,
                       precision=0.58, recall=0.52, training_time=10.0)
            for i in range(5)
        ]

    def test_cv_result_creation(self, sample_fold_metrics):
        """Test CVResult creation."""
        result = CVResult(
            model_name="xgboost",
            horizon=20,
            fold_metrics=sample_fold_metrics,
            oos_predictions=pd.DataFrame({"pred": [1, 0, -1]}),
        )

        assert result.model_name == "xgboost"
        assert result.horizon == 20
        assert result.n_folds == 5

    def test_cv_result_mean_accuracy(self, sample_fold_metrics):
        """Test mean_accuracy property."""
        result = CVResult(
            model_name="xgboost",
            horizon=20,
            fold_metrics=sample_fold_metrics,
            oos_predictions=pd.DataFrame(),
        )

        expected = np.mean([m.accuracy for m in sample_fold_metrics])
        assert np.isclose(result.mean_accuracy, expected)

    def test_cv_result_mean_f1(self, sample_fold_metrics):
        """Test mean_f1 property."""
        result = CVResult(
            model_name="xgboost",
            horizon=20,
            fold_metrics=sample_fold_metrics,
            oos_predictions=pd.DataFrame(),
        )

        expected = np.mean([m.f1 for m in sample_fold_metrics])
        assert np.isclose(result.mean_f1, expected)

    def test_cv_result_std_f1(self, sample_fold_metrics):
        """Test std_f1 property."""
        result = CVResult(
            model_name="xgboost",
            horizon=20,
            fold_metrics=sample_fold_metrics,
            oos_predictions=pd.DataFrame(),
        )

        expected = np.std([m.f1 for m in sample_fold_metrics])
        assert np.isclose(result.std_f1, expected)

    def test_cv_result_stability_score(self, sample_fold_metrics):
        """Test get_stability_score method."""
        result = CVResult(
            model_name="xgboost",
            horizon=20,
            fold_metrics=sample_fold_metrics,
            oos_predictions=pd.DataFrame(),
        )

        stability = result.get_stability_score()

        # Stability = std / mean (coefficient of variation)
        expected = result.std_f1 / result.mean_f1
        assert np.isclose(stability, expected)

    def test_cv_result_to_dict(self, sample_fold_metrics):
        """Test CVResult conversion to dictionary."""
        result = CVResult(
            model_name="xgboost",
            horizon=20,
            fold_metrics=sample_fold_metrics,
            oos_predictions=pd.DataFrame(),
            tuned_params={"max_depth": 6},
            selected_features=["f1", "f2", "f3"],
            total_time=120.5,
        )

        d = result.to_dict()

        assert d["model_name"] == "xgboost"
        assert d["horizon"] == 20
        assert d["n_folds"] == 5
        assert d["tuned_params"] == {"max_depth": 6}
        assert d["n_selected_features"] == 3
        assert d["total_time"] == 120.5


# =============================================================================
# CV RUNNER INITIALIZATION TESTS
# =============================================================================

class TestCVRunnerInitialization:
    """Tests for CrossValidationRunner initialization."""

    def test_cv_runner_creation(self, default_cv):
        """Test CVRunner creation with valid parameters."""
        runner = CrossValidationRunner(
            cv=default_cv,
            models=["xgboost", "lightgbm"],
            horizons=[5, 10, 20],
        )

        assert runner.models == ["xgboost", "lightgbm"]
        assert runner.horizons == [5, 10, 20]
        assert runner.cv == default_cv

    def test_cv_runner_default_options(self, default_cv):
        """Test CVRunner default options."""
        runner = CrossValidationRunner(
            cv=default_cv,
            models=["xgboost"],
            horizons=[20],
        )

        assert runner.tune_hyperparams is True
        assert runner.select_features is True
        assert runner.n_features_to_select == 50
        assert runner.tuning_trials == 50

    def test_cv_runner_custom_options(self, default_cv):
        """Test CVRunner with custom options."""
        runner = CrossValidationRunner(
            cv=default_cv,
            models=["xgboost"],
            horizons=[20],
            tune_hyperparams=False,
            select_features=False,
            n_features_to_select=30,
            tuning_trials=25,
        )

        assert runner.tune_hyperparams is False
        assert runner.select_features is False
        assert runner.n_features_to_select == 30
        assert runner.tuning_trials == 25


# =============================================================================
# TIME SERIES OPTUNA TUNER TESTS
# =============================================================================

class TestTimeSeriesOptunaTuner:
    """Tests for TimeSeriesOptunaTuner."""

    def test_tuner_creation(self, default_cv):
        """Test tuner creation."""
        tuner = TimeSeriesOptunaTuner(
            model_name="xgboost",
            cv=default_cv,
            n_trials=10,
        )

        assert tuner.model_name == "xgboost"
        assert tuner.n_trials == 10
        assert tuner.direction == "maximize"
        assert tuner.metric == "f1"

    def test_tuner_returns_result_without_optuna(self, default_cv, time_series_data):
        """Test tuner handles missing optuna gracefully."""
        X = time_series_data["X"]
        y = time_series_data["y"]

        tuner = TimeSeriesOptunaTuner(
            model_name="unknown_model",  # No param space
            cv=default_cv,
            n_trials=5,
        )

        result = tuner.tune(X, y, param_space={})

        assert "best_params" in result
        assert result["skipped"] is True


# =============================================================================
# STABILITY ANALYSIS TESTS
# =============================================================================

class TestCVStabilityAnalysis:
    """Tests for CV stability analysis."""

    @pytest.fixture
    def sample_cv_results(self):
        """Create sample CV results for stability analysis."""
        results = {}

        for model in ["xgboost", "lightgbm"]:
            for horizon in [10, 20]:
                fold_metrics = [
                    FoldMetrics(
                        fold=i,
                        train_size=800,
                        val_size=200,
                        accuracy=0.6 + np.random.uniform(-0.05, 0.05),
                        f1=0.55 + np.random.uniform(-0.05, 0.05),
                        precision=0.58,
                        recall=0.52,
                        training_time=10.0,
                    )
                    for i in range(5)
                ]

                results[(model, horizon)] = CVResult(
                    model_name=model,
                    horizon=horizon,
                    fold_metrics=fold_metrics,
                    oos_predictions=pd.DataFrame(),
                )

        return results

    def test_analyze_stability_returns_dataframe(self, sample_cv_results):
        """Test stability analysis returns DataFrame."""
        result = analyze_cv_stability(sample_cv_results)

        assert isinstance(result, pd.DataFrame)
        assert "model" in result.columns
        assert "horizon" in result.columns
        assert "metric" in result.columns
        assert "mean" in result.columns
        assert "std" in result.columns
        assert "cv" in result.columns

    def test_analyze_stability_covers_all_models(self, sample_cv_results):
        """Test stability analysis covers all models and horizons."""
        result = analyze_cv_stability(sample_cv_results)

        # Should have entries for each model/horizon/metric combination
        models = result["model"].unique()
        assert "xgboost" in models
        assert "lightgbm" in models

        horizons = result["horizon"].unique()
        assert 10 in horizons
        assert 20 in horizons

    def test_analyze_stability_includes_grades(self, sample_cv_results):
        """Test stability analysis includes stability grades."""
        result = analyze_cv_stability(sample_cv_results)

        assert "stability_grade" in result.columns

        valid_grades = ["Excellent", "Good", "Acceptable", "Poor", "Unstable"]
        assert result["stability_grade"].isin(valid_grades).all()


# =============================================================================
# STABILITY GRADING TESTS
# =============================================================================

class TestStabilityGrading:
    """Tests for stability grading function."""

    def test_excellent_stability(self):
        """Test excellent grade for low CV."""
        assert _grade_stability(0.1) == "Excellent"
        assert _grade_stability(0.14) == "Excellent"

    def test_good_stability(self):
        """Test good grade for moderate-low CV."""
        assert _grade_stability(0.16) == "Good"
        assert _grade_stability(0.24) == "Good"

    def test_acceptable_stability(self):
        """Test acceptable grade for moderate CV."""
        assert _grade_stability(0.3) == "Acceptable"
        assert _grade_stability(0.39) == "Acceptable"

    def test_poor_stability(self):
        """Test poor grade for high CV."""
        assert _grade_stability(0.45) == "Poor"
        assert _grade_stability(0.59) == "Poor"

    def test_unstable(self):
        """Test unstable grade for very high CV."""
        assert _grade_stability(0.7) == "Unstable"
        assert _grade_stability(1.0) == "Unstable"


# =============================================================================
# SAVE RESULTS TESTS
# =============================================================================

class TestSaveResults:
    """Tests for saving CV results."""

    @pytest.fixture
    def sample_results_for_save(self):
        """Create sample results for save tests."""
        fold_metrics = [
            FoldMetrics(
                fold=i, train_size=800, val_size=200,
                accuracy=0.65, f1=0.60, precision=0.62,
                recall=0.58, training_time=10.0
            )
            for i in range(3)
        ]

        results = {
            ("xgboost", 20): CVResult(
                model_name="xgboost",
                horizon=20,
                fold_metrics=fold_metrics,
                oos_predictions=pd.DataFrame({"pred": [1, 0, -1]}),
                tuned_params={"max_depth": 6},
            )
        }

        return results

    def test_save_creates_results_json(
        self, small_cv, sample_results_for_save, tmp_cv_output_dir
    ):
        """Test saving creates cv_results.json."""
        runner = CrossValidationRunner(
            cv=small_cv,
            models=["xgboost"],
            horizons=[20],
        )

        runner.save_results(
            sample_results_for_save,
            stacking_datasets={},
            output_dir=tmp_cv_output_dir,
        )

        results_path = tmp_cv_output_dir / "cv_results.json"
        assert results_path.exists()

    def test_save_creates_tuned_params(
        self, small_cv, sample_results_for_save, tmp_cv_output_dir
    ):
        """Test saving creates tuned params files."""
        runner = CrossValidationRunner(
            cv=small_cv,
            models=["xgboost"],
            horizons=[20],
        )

        runner.save_results(
            sample_results_for_save,
            stacking_datasets={},
            output_dir=tmp_cv_output_dir,
        )

        params_dir = tmp_cv_output_dir / "tuned_params"
        assert params_dir.exists()

        # Should have params file for xgboost h20
        params_file = params_dir / "xgboost_h20.json"
        assert params_file.exists()


# =============================================================================
# INTEGRATION-LIKE TESTS (MOCKED)
# =============================================================================

class TestCVRunnerRun:
    """Integration-like tests for CVRunner.run with mocked models."""

    def test_cv_runner_returns_results(
        self, small_cv, mock_cv_data_container
    ):
        """Test CVRunner.run returns results dictionary."""
        from src.models.base import TrainingMetrics, PredictionOutput

        # Create mock model
        mock_model = MagicMock()
        mock_model.fit.return_value = TrainingMetrics(
            train_loss=0.5, val_loss=0.6,
            train_accuracy=0.6, val_accuracy=0.55,
            train_f1=0.55, val_f1=0.5,
            epochs_trained=10, training_time_seconds=1.0,
            early_stopped=False, best_epoch=10, history={},
        )

        def mock_predict(X):
            n = X.shape[0]
            np.random.seed(42)
            probs = np.random.dirichlet([1, 1, 1], size=n)
            return PredictionOutput(
                class_predictions=np.argmax(probs, axis=1) - 1,
                class_probabilities=probs,
                confidence=probs.max(axis=1),
                metadata={},
            )

        mock_model.predict.side_effect = mock_predict

        # Patch both locations where ModelRegistry is used
        with patch("src.cross_validation.cv_runner.ModelRegistry") as mock_cv_reg, \
             patch("src.cross_validation.oof_generator.ModelRegistry") as mock_oof_reg:

            mock_cv_reg.get_model_info.return_value = {
                "family": "boosting",
                "default_config": {"max_depth": 6},
            }
            mock_cv_reg.create.return_value = mock_model
            mock_oof_reg.create.return_value = mock_model

            runner = CrossValidationRunner(
                cv=small_cv,
                models=["mock"],
                horizons=[20],
                tune_hyperparams=False,
                select_features=False,
            )

            results = runner.run(mock_cv_data_container)

            assert isinstance(results, dict)
            assert ("mock", 20) in results

    def test_cv_runner_results_have_correct_structure(
        self, small_cv, mock_cv_data_container
    ):
        """Test CVRunner results have correct structure."""
        from src.models.base import TrainingMetrics, PredictionOutput

        # Create mock model
        mock_model = MagicMock()
        mock_model.fit.return_value = TrainingMetrics(
            train_loss=0.5, val_loss=0.6,
            train_accuracy=0.6, val_accuracy=0.55,
            train_f1=0.55, val_f1=0.5,
            epochs_trained=10, training_time_seconds=1.0,
            early_stopped=False, best_epoch=10, history={},
        )

        def mock_predict(X):
            n = X.shape[0]
            np.random.seed(42)
            probs = np.random.dirichlet([1, 1, 1], size=n)
            return PredictionOutput(
                class_predictions=np.argmax(probs, axis=1) - 1,
                class_probabilities=probs,
                confidence=probs.max(axis=1),
                metadata={},
            )

        mock_model.predict.side_effect = mock_predict

        # Patch both locations where ModelRegistry is used
        with patch("src.cross_validation.cv_runner.ModelRegistry") as mock_cv_reg, \
             patch("src.cross_validation.oof_generator.ModelRegistry") as mock_oof_reg:

            mock_cv_reg.get_model_info.return_value = {
                "family": "boosting",
                "default_config": {"max_depth": 6},
            }
            mock_cv_reg.create.return_value = mock_model
            mock_oof_reg.create.return_value = mock_model

            runner = CrossValidationRunner(
                cv=small_cv,
                models=["mock"],
                horizons=[20],
                tune_hyperparams=False,
                select_features=False,
            )

            results = runner.run(mock_cv_data_container)
            result = results[("mock", 20)]

            assert isinstance(result, CVResult)
            assert result.model_name == "mock"
            assert result.horizon == 20
            assert len(result.fold_metrics) == small_cv.config.n_splits


# =============================================================================
# PARAM SPACE TESTS
# =============================================================================

class TestParamSpaces:
    """Tests for hyperparameter search spaces."""

    def test_xgboost_param_space_exists(self):
        """Test XGBoost param space is defined."""
        from src.cross_validation.param_spaces import PARAM_SPACES

        assert "xgboost" in PARAM_SPACES
        assert "n_estimators" in PARAM_SPACES["xgboost"]
        assert "max_depth" in PARAM_SPACES["xgboost"]

    def test_lightgbm_param_space_exists(self):
        """Test LightGBM param space is defined."""
        from src.cross_validation.param_spaces import PARAM_SPACES

        assert "lightgbm" in PARAM_SPACES
        assert "n_estimators" in PARAM_SPACES["lightgbm"]
        assert "num_leaves" in PARAM_SPACES["lightgbm"]

    def test_lstm_param_space_exists(self):
        """Test LSTM param space is defined."""
        from src.cross_validation.param_spaces import PARAM_SPACES

        assert "lstm" in PARAM_SPACES
        assert "hidden_size" in PARAM_SPACES["lstm"]
        assert "dropout" in PARAM_SPACES["lstm"]

    def test_get_param_space_function(self):
        """Test get_param_space helper function."""
        from src.cross_validation.param_spaces import get_param_space

        space = get_param_space("xgboost")
        assert "n_estimators" in space

        # Unknown model returns empty
        empty_space = get_param_space("unknown_model")
        assert empty_space == {}
