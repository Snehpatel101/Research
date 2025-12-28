"""
Smoke tests for inference package.

Tests the full serialize → load → predict cycle, plus batch and pipeline features.
"""
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.models.base import BaseModel, PredictionOutput, TrainingMetrics


# =============================================================================
# MOCK CALIBRATOR FOR TESTING (must be at module level for pickling)
# =============================================================================

class MockCalibrator:
    """Simple picklable calibrator for testing."""

    def calibrate(self, probs):
        """Return probabilities unchanged."""
        return probs


# =============================================================================
# MOCK MODEL FOR TESTING
# =============================================================================

class MockModel(BaseModel):
    """Simple mock model for testing inference pipeline."""

    def __init__(self, config=None):
        super().__init__(config)
        self._weights = None
        self._n_features = None

    @property
    def model_family(self) -> str:
        return "mock"

    @property
    def requires_scaling(self) -> bool:
        return True

    @property
    def requires_sequences(self) -> bool:
        return False

    def _get_model_type(self) -> str:
        return "mock_model"

    def get_default_config(self):
        return {"random_seed": 42}

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        self._n_features = X_train.shape[1]
        self._weights = np.random.randn(self._n_features)
        self._is_fitted = True
        return TrainingMetrics(
            train_loss=0.5,
            val_loss=0.6,
            train_accuracy=0.8,
            val_accuracy=0.75,
            train_f1=0.78,
            val_f1=0.72,
            epochs_trained=10,
            training_time_seconds=1.0,
            early_stopped=False,
            best_epoch=10,
        )

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        n_samples = X.shape[0]
        n_classes = 3

        # Generate deterministic probabilities based on input
        np.random.seed(42)
        probs = np.random.rand(n_samples, n_classes)
        probs = probs / probs.sum(axis=1, keepdims=True)

        class_preds = np.argmax(probs, axis=1) - 1  # Map to -1, 0, 1
        confidence = np.max(probs, axis=1)

        return PredictionOutput(
            class_predictions=class_preds,
            class_probabilities=probs,
            confidence=confidence,
            metadata={"model": "mock"},
        )

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "weights": self._weights,
                "n_features": self._n_features,
                "config": self._config,
            }, f)

    def load(self, path):
        path = Path(path)
        with open(path / "model.pkl", "rb") as f:
            data = pickle.load(f)
        self._weights = data["weights"]
        self._n_features = data["n_features"]
        self._config = data["config"]
        self._is_fitted = True


class MockSequenceModel(BaseModel):
    """Mock model that requires sequences."""

    def __init__(self, config=None):
        super().__init__(config)
        self._n_features = None
        self._seq_len = None

    @property
    def model_family(self) -> str:
        return "neural"

    @property
    def requires_scaling(self) -> bool:
        return True

    @property
    def requires_sequences(self) -> bool:
        return True

    def _get_model_type(self) -> str:
        return "mock_sequence_model"

    def get_default_config(self):
        return {"sequence_length": 30}

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        self._seq_len = X_train.shape[1]
        self._n_features = X_train.shape[2]
        self._is_fitted = True
        return TrainingMetrics(
            train_loss=0.5, val_loss=0.6, train_accuracy=0.8, val_accuracy=0.75,
            train_f1=0.78, val_f1=0.72, epochs_trained=5, training_time_seconds=2.0,
            early_stopped=True, best_epoch=3,
        )

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        n_samples = X.shape[0]
        n_classes = 3

        np.random.seed(42)
        probs = np.random.rand(n_samples, n_classes)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return PredictionOutput(
            class_predictions=np.argmax(probs, axis=1) - 1,
            class_probabilities=probs,
            confidence=np.max(probs, axis=1),
            metadata={"model": "mock_sequence"},
        )

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "n_features": self._n_features,
                "seq_len": self._seq_len,
                "config": self._config,
            }, f)

    def load(self, path):
        path = Path(path)
        with open(path / "model.pkl", "rb") as f:
            data = pickle.load(f)
        self._n_features = data["n_features"]
        self._seq_len = data["seq_len"]
        self._config = data["config"]
        self._is_fitted = True


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_features():
    """Generate sample feature data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    return X


@pytest.fixture
def sample_dataframe(sample_features):
    """Generate sample DataFrame with named columns."""
    columns = [f"feature_{i}" for i in range(sample_features.shape[1])]
    return pd.DataFrame(sample_features, columns=columns)


@pytest.fixture
def fitted_mock_model(sample_features):
    """Create a fitted mock model."""
    model = MockModel()
    y = np.random.randint(0, 3, sample_features.shape[0])
    model.fit(sample_features, y, sample_features, y)
    return model


@pytest.fixture
def fitted_scaler(sample_features):
    """Create a fitted scaler."""
    scaler = RobustScaler()
    scaler.fit(sample_features)
    return scaler


@pytest.fixture
def feature_columns():
    """Feature column names."""
    return [f"feature_{i}" for i in range(10)]


# =============================================================================
# MODEL BUNDLE TESTS
# =============================================================================

class TestModelBundle:
    """Test ModelBundle serialize/load/predict cycle."""

    def test_create_from_training(self, fitted_mock_model, fitted_scaler, feature_columns):
        """Test creating bundle from trained components."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        assert bundle.model is fitted_mock_model
        assert bundle.scaler is fitted_scaler
        assert bundle.feature_columns == feature_columns
        assert bundle.metadata.horizon == 20
        assert bundle.metadata.n_features == 10
        assert bundle.metadata.model_name == "mock_model"

    def test_save_and_load(self, fitted_mock_model, fitted_scaler, feature_columns):
        """Test saving and loading a bundle."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            # Mock the registry to return our mock model
            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                # Save
                saved_path = bundle.save(bundle_path)
                assert saved_path.exists()
                assert (saved_path / "manifest.json").exists()
                assert (saved_path / "metadata.json").exists()
                assert (saved_path / "features.json").exists()
                assert (saved_path / "scaler.pkl").exists()
                assert (saved_path / "model").exists()

                # Load
                loaded_bundle = ModelBundle.load(bundle_path)

                assert loaded_bundle.metadata.horizon == 20
                assert loaded_bundle.metadata.n_features == 10
                assert loaded_bundle.feature_columns == feature_columns
                assert loaded_bundle.model._is_fitted

    def test_predict_after_load(self, fitted_mock_model, fitted_scaler, feature_columns, sample_features):
        """Test predictions work after save/load cycle."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                loaded_bundle = ModelBundle.load(bundle_path)

                # Make predictions
                output = loaded_bundle.predict(sample_features)

                assert output.n_samples == 100
                assert output.n_classes == 3
                assert output.class_predictions.shape == (100,)
                assert output.class_probabilities.shape == (100, 3)
                assert output.confidence.shape == (100,)

    def test_predict_with_dataframe(self, fitted_mock_model, fitted_scaler, feature_columns, sample_dataframe):
        """Test predictions work with DataFrame input."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        output = bundle.predict(sample_dataframe)

        assert output.n_samples == 100
        assert output.n_classes == 3

    def test_predict_validates_features(self, fitted_mock_model, fitted_scaler, feature_columns):
        """Test prediction validates feature count."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Wrong number of features
        X_wrong = np.random.randn(10, 5).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 10 features"):
            bundle.predict(X_wrong)

    def test_bundle_with_calibrator(self, fitted_mock_model, fitted_scaler, feature_columns, sample_features):
        """Test bundle with calibrator."""
        from src.inference import ModelBundle

        # Use module-level calibrator for pickling
        calibrator = MockCalibrator()

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
            calibrator=calibrator,
        )

        assert bundle.metadata.has_calibrator is True
        assert bundle.calibrator is calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "calibrated_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                assert (bundle_path / "calibrator.pkl").exists()

                loaded = ModelBundle.load(bundle_path)
                assert loaded.calibrator is not None
                assert hasattr(loaded.calibrator, 'calibrate')

    def test_overwrite_existing_bundle(self, fitted_mock_model, fitted_scaler, feature_columns):
        """Test overwriting existing bundle."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"
            bundle_path.mkdir()

            # Should fail without overwrite
            with pytest.raises(FileExistsError):
                bundle.save(bundle_path, overwrite=False)

            # Should succeed with overwrite
            bundle.save(bundle_path, overwrite=True)
            assert (bundle_path / "manifest.json").exists()


# =============================================================================
# INFERENCE PIPELINE TESTS
# =============================================================================

class TestInferencePipeline:
    """Test InferencePipeline functionality."""

    def test_create_from_bundle(self, fitted_mock_model, fitted_scaler, feature_columns):
        """Test creating pipeline from bundle."""
        from src.inference import ModelBundle, InferencePipeline

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                pipeline = InferencePipeline.from_bundle(bundle_path)

                assert pipeline.n_models == 1
                assert pipeline.horizon == 20
                assert len(pipeline.feature_columns) == 10

    def test_predict(self, fitted_mock_model, fitted_scaler, feature_columns, sample_features):
        """Test basic prediction."""
        from src.inference import ModelBundle, InferencePipeline

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                pipeline = InferencePipeline.from_bundle(bundle_path)

                result = pipeline.predict(sample_features)

                assert result.n_samples == 100
                assert result.inference_time_ms > 0
                assert result.model_name == "mock_model"
                assert result.horizon == 20

    def test_predict_returns_dataframe(self, fitted_mock_model, fitted_scaler, feature_columns, sample_features):
        """Test inference result to_dataframe."""
        from src.inference import ModelBundle, InferencePipeline

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                pipeline = InferencePipeline.from_bundle(bundle_path)

                result = pipeline.predict(sample_features)
                df = result.to_dataframe()

                assert "prediction" in df.columns
                assert "prob_short" in df.columns
                assert "prob_neutral" in df.columns
                assert "prob_long" in df.columns
                assert "confidence" in df.columns
                assert len(df) == 100

    def test_ensemble_soft_vote(self, fitted_mock_model, fitted_scaler, feature_columns, sample_features):
        """Test ensemble soft voting."""
        from src.inference import ModelBundle, InferencePipeline

        # Create two bundles
        bundle1 = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        model2 = MockModel()
        model2.fit(sample_features, np.random.randint(0, 3, 100), sample_features, np.random.randint(0, 3, 100))
        bundle2 = ModelBundle.from_training(
            model=model2,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "bundle1"
            path2 = Path(tmpdir) / "bundle2"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle1.save(path1)
                bundle2.save(path2)

                pipeline = InferencePipeline.from_bundles([path1, path2])

                assert pipeline.n_models == 2

                result = pipeline.predict_ensemble(sample_features, method="soft_vote")

                assert result.voting_method == "soft_vote"
                assert len(result.individual_results) == 2
                assert result.predictions.n_samples == 100

    def test_ensemble_hard_vote(self, fitted_mock_model, fitted_scaler, feature_columns, sample_features):
        """Test ensemble hard voting."""
        from src.inference import ModelBundle, InferencePipeline

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "bundle1"
            path2 = Path(tmpdir) / "bundle2"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(path1)
                bundle.save(path2, overwrite=True)

                pipeline = InferencePipeline.from_bundles([path1, path2])
                result = pipeline.predict_ensemble(sample_features, method="hard_vote")

                assert result.voting_method == "hard_vote"

    def test_get_model_info(self, fitted_mock_model, fitted_scaler, feature_columns):
        """Test get_model_info returns correct structure."""
        from src.inference import ModelBundle, InferencePipeline

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                pipeline = InferencePipeline.from_bundle(bundle_path)

                info = pipeline.get_model_info()

                assert len(info) == 1
                assert info[0]["name"] == "mock_model"
                assert info[0]["horizon"] == 20
                assert info[0]["features"] == 10


# =============================================================================
# BATCH PREDICTOR TESTS
# =============================================================================

class TestBatchPredictor:
    """Test BatchPredictor functionality."""

    def test_batch_predict_from_array(self, fitted_mock_model, fitted_scaler, feature_columns):
        """Test batch prediction from numpy array."""
        from src.inference import ModelBundle, BatchPredictor

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                predictor = BatchPredictor.from_bundle(bundle_path, batch_size=30)

                # Create test data
                np.random.seed(42)
                X = np.random.randn(100, 10).astype(np.float32)
                df = pd.DataFrame(X, columns=feature_columns)

                result = predictor.predict_batch(df, batch_size=30)

                assert result.n_samples == 100
                assert result.n_batches == 4  # 100 / 30 = 3.33, rounds up to 4
                assert result.samples_per_second > 0
                assert len(result.errors) == 0

    def test_batch_predict_with_progress(self, fitted_mock_model, fitted_scaler, feature_columns):
        """Test batch prediction with progress callback."""
        from src.inference import ModelBundle, BatchPredictor

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        progress_updates = []

        def progress_cb(progress):
            progress_updates.append(progress.progress_pct)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                predictor = BatchPredictor.from_bundle(bundle_path)

                np.random.seed(42)
                df = pd.DataFrame(
                    np.random.randn(100, 10).astype(np.float32),
                    columns=feature_columns,
                )

                predictor.predict_batch(df, batch_size=25, progress_callback=progress_cb)

                assert len(progress_updates) == 4  # 100 / 25 = 4 batches
                assert progress_updates[-1] == 100.0

    def test_batch_predict_save_output(self, fitted_mock_model, fitted_scaler, feature_columns):
        """Test batch prediction saves to parquet."""
        from src.inference import ModelBundle, BatchPredictor

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"
            output_path = Path(tmpdir) / "predictions.parquet"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                predictor = BatchPredictor.from_bundle(bundle_path)

                np.random.seed(42)
                df = pd.DataFrame(
                    np.random.randn(100, 10).astype(np.float32),
                    columns=feature_columns,
                )

                result = predictor.predict_batch(df, output_path=output_path)

                assert output_path.exists()
                loaded = pd.read_parquet(output_path)
                assert len(loaded) == 100
                assert "prediction" in loaded.columns

    def test_batch_streaming(self, fitted_mock_model, fitted_scaler, feature_columns):
        """Test streaming batch predictions."""
        from src.inference import ModelBundle, BatchPredictor

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                predictor = BatchPredictor.from_bundle(bundle_path)

                np.random.seed(42)
                df = pd.DataFrame(
                    np.random.randn(100, 10).astype(np.float32),
                    columns=feature_columns,
                )

                batches = list(predictor.predict_streaming(df, batch_size=30))

                assert len(batches) == 4
                total_samples = sum(len(b) for b in batches)
                assert total_samples == 100


# =============================================================================
# SMOKE TESTS - FULL INTEGRATION
# =============================================================================

class TestSmokeIntegration:
    """Smoke tests for full serialize → load → predict cycle."""

    def test_full_cycle_non_sequential(self, sample_features):
        """Full integration test for non-sequential model."""
        from src.inference import ModelBundle, InferencePipeline, BatchPredictor

        # Create and train model
        model = MockModel()
        y = np.random.randint(0, 3, sample_features.shape[0])
        model.fit(sample_features, y, sample_features, y)

        # Create scaler
        scaler = RobustScaler()
        scaler.fit(sample_features)

        feature_columns = [f"feature_{i}" for i in range(10)]

        # Create bundle
        bundle = ModelBundle.from_training(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            horizon=20,
            training_metrics={"val_f1": 0.75},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "smoke_test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                # Save
                bundle.save(bundle_path)

                # Load and predict via pipeline
                pipeline = InferencePipeline.from_bundle(bundle_path)
                result = pipeline.predict(sample_features)

                assert result.n_samples == 100
                assert result.horizon == 20

                # Load and predict via batch
                predictor = BatchPredictor.from_bundle(bundle_path)
                df = pd.DataFrame(sample_features, columns=feature_columns)
                batch_result = predictor.predict_batch(df, batch_size=50)

                assert batch_result.n_samples == 100
                assert len(batch_result.errors) == 0

    def test_bundle_metadata_preserved(self):
        """Test that all metadata is preserved through save/load cycle."""
        from src.inference import ModelBundle

        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randint(0, 3, 50)

        model = MockModel()
        model.fit(X, y, X, y)

        scaler = RobustScaler()
        scaler.fit(X)

        feature_columns = ["a", "b", "c", "d", "e"]

        bundle = ModelBundle.from_training(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            horizon=15,
            training_metrics={"val_f1": 0.82, "train_f1": 0.85},
            extra_metadata={"symbol": "MES", "experiment": "test123"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "metadata_test"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                loaded = ModelBundle.load(bundle_path)

                assert loaded.metadata.horizon == 15
                assert loaded.metadata.n_features == 5
                assert loaded.metadata.training_metrics["val_f1"] == 0.82
                assert loaded.metadata.extra["symbol"] == "MES"
                assert loaded.feature_columns == feature_columns

    def test_validation_after_load(self):
        """Test bundle validation works after load."""
        from src.inference import ModelBundle

        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randint(0, 3, 50)

        model = MockModel()
        model.fit(X, y, X, y)

        scaler = RobustScaler()
        scaler.fit(X)

        bundle = ModelBundle.from_training(
            model=model,
            scaler=scaler,
            feature_columns=["a", "b", "c", "d", "e"],
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "validation_test"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockModel()

                bundle.save(bundle_path)
                loaded = ModelBundle.load(bundle_path)

                validation = loaded.validate()
                assert validation["valid"] is True
                assert len(validation["issues"]) == 0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_bundles_list_raises(self):
        """Test that empty bundles list raises error."""
        from src.inference import InferencePipeline

        with pytest.raises(ValueError, match="At least one bundle"):
            InferencePipeline([])

    def test_missing_features_raises(self, fitted_mock_model, fitted_scaler, feature_columns, sample_dataframe):
        """Test that missing features raises error."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_mock_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Remove a column
        df_missing = sample_dataframe.drop(columns=["feature_0"])

        with pytest.raises(ValueError, match="Missing features"):
            bundle.predict(df_missing)

    def test_bundle_not_found_raises(self):
        """Test loading non-existent bundle raises error."""
        from src.inference import ModelBundle

        with pytest.raises(FileNotFoundError):
            ModelBundle.load("/nonexistent/path")

    def test_invalid_bundle_raises(self):
        """Test loading invalid bundle raises error."""
        from src.inference import ModelBundle

        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "invalid_bundle"
            invalid_path.mkdir()

            with pytest.raises(ValueError, match="Invalid bundle"):
                ModelBundle.load(invalid_path)
