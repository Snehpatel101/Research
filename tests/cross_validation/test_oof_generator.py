"""
Tests for Out-of-Fold (OOF) Prediction Generator.

Tests:
- OOF prediction generation
- 100% coverage validation
- Stacking dataset creation
- Derived stacking features
- Prediction correlation analysis
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.cross_validation.oof_generator import (
    OOFGenerator,
    OOFPrediction,
    StackingDataset,
    analyze_prediction_correlation,
    _grade_diversity,
)
from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig


# =============================================================================
# OOF PREDICTION DATA CLASS TESTS
# =============================================================================

class TestOOFPrediction:
    """Tests for OOFPrediction dataclass."""

    def test_get_probabilities(self):
        """Test get_probabilities extracts probability columns."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=10, freq="5min"),
            "model1_prob_short": np.random.rand(10),
            "model1_prob_neutral": np.random.rand(10),
            "model1_prob_long": np.random.rand(10),
            "model1_pred": np.random.choice([-1, 0, 1], size=10),
            "model1_confidence": np.random.rand(10),
        })

        oof_pred = OOFPrediction(
            model_name="model1",
            predictions=df,
            fold_info=[],
            coverage=1.0,
        )

        probs = oof_pred.get_probabilities()

        assert probs.shape == (10, 3)
        assert probs.dtype == np.float64

    def test_get_class_predictions(self):
        """Test get_class_predictions extracts prediction column."""
        predictions = np.array([-1, 0, 1, 0, -1, 1, 0, 0, 1, -1])
        df = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=10, freq="5min"),
            "model1_prob_short": np.random.rand(10),
            "model1_prob_neutral": np.random.rand(10),
            "model1_prob_long": np.random.rand(10),
            "model1_pred": predictions,
            "model1_confidence": np.random.rand(10),
        })

        oof_pred = OOFPrediction(
            model_name="model1",
            predictions=df,
            fold_info=[],
        )

        preds = oof_pred.get_class_predictions()

        np.testing.assert_array_equal(preds, predictions)


# =============================================================================
# STACKING DATASET TESTS
# =============================================================================

class TestStackingDataset:
    """Tests for StackingDataset dataclass."""

    def test_n_samples_property(self):
        """Test n_samples property returns correct count."""
        df = pd.DataFrame({
            "model1_prob_short": np.random.rand(100),
            "model1_prob_neutral": np.random.rand(100),
            "model1_prob_long": np.random.rand(100),
            "y_true": np.random.choice([-1, 0, 1], size=100),
        })

        stacking_ds = StackingDataset(
            data=df,
            model_names=["model1"],
            horizon=20,
        )

        assert stacking_ds.n_samples == 100

    def test_n_models_property(self):
        """Test n_models property returns correct count."""
        df = pd.DataFrame({
            "model1_prob_short": np.random.rand(100),
            "model2_prob_short": np.random.rand(100),
            "y_true": np.random.choice([-1, 0, 1], size=100),
        })

        stacking_ds = StackingDataset(
            data=df,
            model_names=["model1", "model2"],
            horizon=20,
        )

        assert stacking_ds.n_models == 2

    def test_get_features_excludes_y_true(self):
        """Test get_features excludes y_true column."""
        df = pd.DataFrame({
            "model1_prob_short": np.random.rand(100),
            "model1_prob_neutral": np.random.rand(100),
            "model1_prob_long": np.random.rand(100),
            "y_true": np.random.choice([-1, 0, 1], size=100),
        })

        stacking_ds = StackingDataset(
            data=df,
            model_names=["model1"],
            horizon=20,
        )

        features = stacking_ds.get_features()

        assert "y_true" not in features.columns
        assert "model1_prob_short" in features.columns

    def test_get_labels_returns_y_true(self):
        """Test get_labels returns y_true column."""
        labels = np.random.choice([-1, 0, 1], size=100)
        df = pd.DataFrame({
            "model1_prob_short": np.random.rand(100),
            "y_true": labels,
        })

        stacking_ds = StackingDataset(
            data=df,
            model_names=["model1"],
            horizon=20,
        )

        y = stacking_ds.get_labels()

        np.testing.assert_array_equal(y.values, labels)


# =============================================================================
# OOF GENERATOR TESTS
# =============================================================================

class TestOOFGenerator:
    """Tests for OOFGenerator class."""

    @pytest.fixture
    def mock_model_registry(self):
        """Mock ModelRegistry for testing."""
        with patch("src.cross_validation.oof_generator.ModelRegistry") as mock_reg:
            # Create mock model
            mock_model = MagicMock()

            # Setup fit to return TrainingMetrics
            from src.models.base import TrainingMetrics, PredictionOutput
            mock_model.fit.return_value = TrainingMetrics(
                train_loss=0.5,
                val_loss=0.6,
                train_accuracy=0.6,
                val_accuracy=0.55,
                train_f1=0.55,
                val_f1=0.5,
                epochs_trained=10,
                training_time_seconds=1.0,
                early_stopped=False,
                best_epoch=10,
                history={},
            )

            # Setup predict to return PredictionOutput
            def mock_predict(X):
                n_samples = X.shape[0]
                np.random.seed(42)
                probs = np.random.dirichlet([1, 1, 1], size=n_samples)
                return PredictionOutput(
                    class_predictions=np.argmax(probs, axis=1) - 1,
                    class_probabilities=probs,
                    confidence=probs.max(axis=1),
                    metadata={},
                )

            mock_model.predict.side_effect = mock_predict
            mock_reg.create.return_value = mock_model

            yield mock_reg

    def test_generate_oof_predictions_structure(
        self, small_cv, small_time_series_data, mock_model_registry
    ):
        """Test OOF prediction generation returns correct structure."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        oof_gen = OOFGenerator(small_cv)
        oof_predictions = oof_gen.generate_oof_predictions(
            X=X,
            y=y,
            model_configs={"mock_model": {}},
        )

        assert "mock_model" in oof_predictions
        assert isinstance(oof_predictions["mock_model"], OOFPrediction)

    def test_generate_oof_predictions_coverage(
        self, small_cv, small_time_series_data, mock_model_registry
    ):
        """Test OOF predictions have 100% coverage."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        oof_gen = OOFGenerator(small_cv)
        oof_predictions = oof_gen.generate_oof_predictions(
            X=X,
            y=y,
            model_configs={"mock_model": {}},
        )

        oof_pred = oof_predictions["mock_model"]

        # Coverage should be 100%
        assert oof_pred.coverage == 1.0

        # No NaN predictions
        assert not oof_pred.predictions["mock_model_pred"].isna().any()

    def test_generate_oof_predictions_with_feature_subset(
        self, small_cv, small_time_series_data, mock_model_registry
    ):
        """Test OOF generation with feature subset."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]
        feature_subset = list(X.columns[:5])

        oof_gen = OOFGenerator(small_cv)
        oof_predictions = oof_gen.generate_oof_predictions(
            X=X,
            y=y,
            model_configs={"mock_model": {}},
            feature_subset=feature_subset,
        )

        assert "mock_model" in oof_predictions

    def test_generate_oof_predictions_fold_info(
        self, small_cv, small_time_series_data, mock_model_registry
    ):
        """Test fold info is recorded correctly."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        oof_gen = OOFGenerator(small_cv)
        oof_predictions = oof_gen.generate_oof_predictions(
            X=X,
            y=y,
            model_configs={"mock_model": {}},
        )

        fold_info = oof_predictions["mock_model"].fold_info

        assert len(fold_info) == small_cv.config.n_splits

        for fi in fold_info:
            assert "fold" in fi
            assert "train_size" in fi
            assert "val_size" in fi
            assert "val_accuracy" in fi
            assert "val_f1" in fi


# =============================================================================
# COVERAGE VALIDATION TESTS
# =============================================================================

class TestOOFCoverageValidation:
    """Tests for OOF coverage validation."""

    def test_validate_coverage_all_covered(self, small_cv, small_time_series_data):
        """Test validation passes when all samples are covered."""
        X = small_time_series_data["X"]

        # Create complete OOF predictions (no NaN)
        df = pd.DataFrame({
            "datetime": X.index,
            "model1_prob_short": np.random.rand(len(X)),
            "model1_prob_neutral": np.random.rand(len(X)),
            "model1_prob_long": np.random.rand(len(X)),
            "model1_pred": np.random.choice([-1, 0, 1], size=len(X)),
            "model1_confidence": np.random.rand(len(X)),
        })

        oof_pred = OOFPrediction(
            model_name="model1",
            predictions=df,
            fold_info=[],
            coverage=1.0,
        )

        oof_gen = OOFGenerator(small_cv)
        validation = oof_gen.validate_oof_coverage(
            {"model1": oof_pred},
            X.index,
        )

        assert validation["passed"] is True
        assert len(validation["issues"]) == 0
        assert validation["coverage"]["model1"] == 1.0

    def test_validate_coverage_missing_samples(self, small_cv, small_time_series_data):
        """Test validation fails when samples are missing."""
        X = small_time_series_data["X"]

        # Create OOF predictions with some NaN
        predictions = np.random.choice([-1, 0, 1], size=len(X)).astype(float)
        predictions[:10] = np.nan  # Missing 10 samples

        df = pd.DataFrame({
            "datetime": X.index,
            "model1_prob_short": np.random.rand(len(X)),
            "model1_prob_neutral": np.random.rand(len(X)),
            "model1_prob_long": np.random.rand(len(X)),
            "model1_pred": predictions,
            "model1_confidence": np.random.rand(len(X)),
        })

        oof_pred = OOFPrediction(
            model_name="model1",
            predictions=df,
            fold_info=[],
            coverage=1 - 10 / len(X),
        )

        oof_gen = OOFGenerator(small_cv)
        validation = oof_gen.validate_oof_coverage(
            {"model1": oof_pred},
            X.index,
        )

        assert validation["passed"] is False
        assert len(validation["issues"]) == 1
        assert validation["issues"][0]["model"] == "model1"
        assert validation["issues"][0]["missing_samples"] == 10


# =============================================================================
# STACKING DATASET CREATION TESTS
# =============================================================================

class TestStackingDatasetCreation:
    """Tests for building stacking datasets from OOF predictions."""

    def test_build_stacking_dataset_structure(self, small_cv, small_time_series_data):
        """Test stacking dataset has correct structure."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        # Create mock OOF predictions
        oof_preds = {}
        for model in ["model1", "model2"]:
            df = pd.DataFrame({
                "datetime": X.index,
                f"{model}_prob_short": np.random.rand(len(X)),
                f"{model}_prob_neutral": np.random.rand(len(X)),
                f"{model}_prob_long": np.random.rand(len(X)),
                f"{model}_pred": np.random.choice([-1, 0, 1], size=len(X)),
                f"{model}_confidence": np.random.rand(len(X)),
            })
            oof_preds[model] = OOFPrediction(
                model_name=model,
                predictions=df,
                fold_info=[],
            )

        oof_gen = OOFGenerator(small_cv)
        stacking_ds = oof_gen.build_stacking_dataset(
            oof_preds, y, horizon=20
        )

        assert isinstance(stacking_ds, StackingDataset)
        assert stacking_ds.horizon == 20
        assert stacking_ds.n_models == 2
        assert "model1" in stacking_ds.model_names
        assert "model2" in stacking_ds.model_names

    def test_build_stacking_dataset_contains_probabilities(
        self, small_cv, small_time_series_data
    ):
        """Test stacking dataset contains probability columns."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        df = pd.DataFrame({
            "datetime": X.index,
            "model1_prob_short": np.random.rand(len(X)),
            "model1_prob_neutral": np.random.rand(len(X)),
            "model1_prob_long": np.random.rand(len(X)),
            "model1_pred": np.random.choice([-1, 0, 1], size=len(X)),
            "model1_confidence": np.random.rand(len(X)),
        })
        oof_pred = OOFPrediction(
            model_name="model1",
            predictions=df,
            fold_info=[],
        )

        oof_gen = OOFGenerator(small_cv)
        stacking_ds = oof_gen.build_stacking_dataset(
            {"model1": oof_pred}, y, horizon=20
        )

        assert "model1_prob_short" in stacking_ds.data.columns
        assert "model1_prob_neutral" in stacking_ds.data.columns
        assert "model1_prob_long" in stacking_ds.data.columns

    def test_build_stacking_dataset_contains_y_true(
        self, small_cv, small_time_series_data
    ):
        """Test stacking dataset contains y_true column."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        df = pd.DataFrame({
            "datetime": X.index,
            "model1_prob_short": np.random.rand(len(X)),
            "model1_prob_neutral": np.random.rand(len(X)),
            "model1_prob_long": np.random.rand(len(X)),
            "model1_pred": np.random.choice([-1, 0, 1], size=len(X)),
            "model1_confidence": np.random.rand(len(X)),
        })
        oof_pred = OOFPrediction(
            model_name="model1",
            predictions=df,
            fold_info=[],
        )

        oof_gen = OOFGenerator(small_cv)
        stacking_ds = oof_gen.build_stacking_dataset(
            {"model1": oof_pred}, y, horizon=20
        )

        assert "y_true" in stacking_ds.data.columns
        np.testing.assert_array_equal(
            stacking_ds.data["y_true"].values,
            y.values
        )


# =============================================================================
# DERIVED STACKING FEATURES TESTS
# =============================================================================

class TestDerivedStackingFeatures:
    """Tests for derived stacking features."""

    @pytest.fixture
    def two_model_oof(self, small_time_series_data):
        """Create OOF predictions from two models."""
        X = small_time_series_data["X"]

        oof_preds = {}
        for model in ["model1", "model2"]:
            df = pd.DataFrame({
                "datetime": X.index,
                f"{model}_prob_short": np.random.rand(len(X)),
                f"{model}_prob_neutral": np.random.rand(len(X)),
                f"{model}_prob_long": np.random.rand(len(X)),
                f"{model}_pred": np.random.choice([-1, 0, 1], size=len(X)),
                f"{model}_confidence": np.random.rand(len(X)),
            })
            oof_preds[model] = OOFPrediction(
                model_name=model,
                predictions=df,
                fold_info=[],
            )

        return oof_preds

    def test_adds_agreement_features(
        self, small_cv, small_time_series_data, two_model_oof
    ):
        """Test that agreement features are added."""
        y = small_time_series_data["y"]

        oof_gen = OOFGenerator(small_cv)
        stacking_ds = oof_gen.build_stacking_dataset(
            two_model_oof, y, horizon=20, add_derived_features=True
        )

        assert "models_agree" in stacking_ds.data.columns
        assert "agreement_count" in stacking_ds.data.columns

        # models_agree should be 0 or 1
        assert stacking_ds.data["models_agree"].isin([0, 1]).all()

    def test_adds_confidence_features(
        self, small_cv, small_time_series_data, two_model_oof
    ):
        """Test that confidence features are added."""
        y = small_time_series_data["y"]

        oof_gen = OOFGenerator(small_cv)
        stacking_ds = oof_gen.build_stacking_dataset(
            two_model_oof, y, horizon=20, add_derived_features=True
        )

        assert "avg_confidence" in stacking_ds.data.columns
        assert "min_confidence" in stacking_ds.data.columns
        assert "max_confidence" in stacking_ds.data.columns

    def test_adds_entropy_features(
        self, small_cv, small_time_series_data, two_model_oof
    ):
        """Test that entropy features are added."""
        y = small_time_series_data["y"]

        oof_gen = OOFGenerator(small_cv)
        stacking_ds = oof_gen.build_stacking_dataset(
            two_model_oof, y, horizon=20, add_derived_features=True
        )

        assert "model1_entropy" in stacking_ds.data.columns
        assert "model2_entropy" in stacking_ds.data.columns
        assert "avg_entropy" in stacking_ds.data.columns

        # Entropy should be non-negative
        assert (stacking_ds.data["model1_entropy"] >= 0).all()

    def test_adds_prediction_std(
        self, small_cv, small_time_series_data, two_model_oof
    ):
        """Test that prediction standard deviation is added."""
        y = small_time_series_data["y"]

        oof_gen = OOFGenerator(small_cv)
        stacking_ds = oof_gen.build_stacking_dataset(
            two_model_oof, y, horizon=20, add_derived_features=True
        )

        assert "prediction_std" in stacking_ds.data.columns

    def test_no_derived_features_when_disabled(
        self, small_cv, small_time_series_data, two_model_oof
    ):
        """Test that derived features are not added when disabled."""
        y = small_time_series_data["y"]

        oof_gen = OOFGenerator(small_cv)
        stacking_ds = oof_gen.build_stacking_dataset(
            two_model_oof, y, horizon=20, add_derived_features=False
        )

        assert "models_agree" not in stacking_ds.data.columns
        assert "avg_entropy" not in stacking_ds.data.columns


# =============================================================================
# PREDICTION CORRELATION ANALYSIS TESTS
# =============================================================================

class TestPredictionCorrelationAnalysis:
    """Tests for prediction correlation analysis."""

    def test_analyze_correlation_returns_dataframe(self):
        """Test correlation analysis returns DataFrame."""
        stacking_df = pd.DataFrame({
            "model1_pred": np.random.choice([-1, 0, 1], size=100),
            "model2_pred": np.random.choice([-1, 0, 1], size=100),
            "model3_pred": np.random.choice([-1, 0, 1], size=100),
        })

        result = analyze_prediction_correlation(
            stacking_df,
            ["model1", "model2", "model3"]
        )

        assert isinstance(result, pd.DataFrame)
        assert "model_1" in result.columns
        assert "model_2" in result.columns
        assert "correlation" in result.columns
        assert "diversity_grade" in result.columns

    def test_analyze_correlation_pairwise(self):
        """Test correlation analysis covers all pairs."""
        stacking_df = pd.DataFrame({
            "model1_pred": np.random.choice([-1, 0, 1], size=100),
            "model2_pred": np.random.choice([-1, 0, 1], size=100),
            "model3_pred": np.random.choice([-1, 0, 1], size=100),
        })

        result = analyze_prediction_correlation(
            stacking_df,
            ["model1", "model2", "model3"]
        )

        # 3 models = 3 pairs
        assert len(result) == 3

        pairs = [(row["model_1"], row["model_2"]) for _, row in result.iterrows()]
        expected_pairs = [("model1", "model2"), ("model1", "model3"), ("model2", "model3")]

        for expected in expected_pairs:
            assert expected in pairs

    def test_high_correlation_low_diversity(self):
        """Test that high correlation gives low diversity grade."""
        # Create identical predictions
        stacking_df = pd.DataFrame({
            "model1_pred": [1, 1, 1, -1, -1, -1, 0, 0, 0, 0],
            "model2_pred": [1, 1, 1, -1, -1, -1, 0, 0, 0, 0],  # Same as model1
        })

        result = analyze_prediction_correlation(
            stacking_df,
            ["model1", "model2"]
        )

        assert result.iloc[0]["correlation"] == 1.0
        assert "Poor" in result.iloc[0]["diversity_grade"]

    def test_low_correlation_high_diversity(self):
        """Test that low correlation gives high diversity grade."""
        np.random.seed(42)
        # Create uncorrelated predictions
        stacking_df = pd.DataFrame({
            "model1_pred": np.random.choice([-1, 0, 1], size=1000),
            "model2_pred": np.random.choice([-1, 0, 1], size=1000),
        })

        result = analyze_prediction_correlation(
            stacking_df,
            ["model1", "model2"]
        )

        # Uncorrelated random predictions should have low correlation
        assert abs(result.iloc[0]["correlation"]) < 0.3


# =============================================================================
# DIVERSITY GRADING TESTS
# =============================================================================

class TestDiversityGrading:
    """Tests for diversity grading function."""

    def test_excellent_diversity(self):
        """Test excellent diversity grade for low correlation."""
        assert "Excellent" in _grade_diversity(0.1)
        assert "Excellent" in _grade_diversity(0.29)

    def test_good_diversity(self):
        """Test good diversity grade for moderate-low correlation."""
        assert "Good" == _grade_diversity(0.35)
        assert "Good" == _grade_diversity(0.49)

    def test_moderate_diversity(self):
        """Test moderate diversity grade for moderate correlation."""
        assert "Moderate" == _grade_diversity(0.55)
        assert "Moderate" == _grade_diversity(0.69)

    def test_low_diversity(self):
        """Test low diversity grade for high correlation."""
        assert "Low" == _grade_diversity(0.75)
        assert "Low" == _grade_diversity(0.84)

    def test_poor_diversity(self):
        """Test poor diversity grade for very high correlation."""
        assert "Poor" in _grade_diversity(0.9)
        assert "Poor" in _grade_diversity(1.0)


# =============================================================================
# SAVE STACKING DATASET TESTS
# =============================================================================

class TestSaveStackingDataset:
    """Tests for saving stacking datasets."""

    def test_save_creates_parquet(self, small_cv, tmp_cv_output_dir):
        """Test saving creates parquet file."""
        df = pd.DataFrame({
            "model1_prob_short": np.random.rand(100),
            "model1_prob_neutral": np.random.rand(100),
            "model1_prob_long": np.random.rand(100),
            "y_true": np.random.choice([-1, 0, 1], size=100),
        })

        stacking_ds = StackingDataset(
            data=df,
            model_names=["model1"],
            horizon=20,
            metadata={"test": "value"},
        )

        oof_gen = OOFGenerator(small_cv)
        path = oof_gen.save_stacking_dataset(stacking_ds, tmp_cv_output_dir)

        assert path.exists()
        assert path.suffix == ".parquet"
        assert "h20" in path.name

    def test_save_creates_metadata_json(self, small_cv, tmp_cv_output_dir):
        """Test saving creates metadata JSON file."""
        df = pd.DataFrame({
            "model1_prob_short": np.random.rand(100),
            "y_true": np.random.choice([-1, 0, 1], size=100),
        })

        stacking_ds = StackingDataset(
            data=df,
            model_names=["model1"],
            horizon=20,
            metadata={"test": "value"},
        )

        oof_gen = OOFGenerator(small_cv)
        oof_gen.save_stacking_dataset(stacking_ds, tmp_cv_output_dir)

        metadata_path = tmp_cv_output_dir / "stacking_metadata_h20.json"
        assert metadata_path.exists()
