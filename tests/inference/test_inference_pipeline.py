"""
Tests for inference pipeline components.

These tests focus on:
1. Batch inference shape validation and output correctness
2. Missing features handling and error messages
3. Model bundle component verification
4. Scaler and model state persistence
5. Production-like inference scenarios

Complements test_inference.py with additional coverage for edge cases
and production scenarios.

Author: ML Pipeline
Created: 2025-12-29
"""
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.base import BaseModel, PredictionOutput, TrainingMetrics


# =============================================================================
# MOCK MODEL FOR TESTING
# =============================================================================

class MockInferenceModel(BaseModel):
    """Mock model with configurable behavior for inference testing."""

    def __init__(self, config=None, n_features=10):
        super().__init__(config)
        self._weights = None
        self._n_features = n_features
        self._expected_features = n_features

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
        return "mock_inference_model"

    def get_default_config(self):
        return {"random_seed": 42}

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        self._n_features = X_train.shape[1]
        self._expected_features = X_train.shape[1]
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

        # Generate deterministic probabilities
        np.random.seed(42)
        probs = np.random.rand(n_samples, n_classes)
        probs = probs / probs.sum(axis=1, keepdims=True)

        class_preds = np.argmax(probs, axis=1) - 1  # Map to -1, 0, 1
        confidence = np.max(probs, axis=1)

        return PredictionOutput(
            class_predictions=class_preds,
            class_probabilities=probs,
            confidence=confidence,
            metadata={"model": "mock_inference"},
        )

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "weights": self._weights,
                "n_features": self._n_features,
                "expected_features": self._expected_features,
                "config": self._config,
            }, f)

    def load(self, path):
        path = Path(path)
        with open(path / "model.pkl", "rb") as f:
            data = pickle.load(f)
        self._weights = data["weights"]
        self._n_features = data["n_features"]
        self._expected_features = data.get("expected_features", self._n_features)
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
    return np.random.randn(n_samples, n_features).astype(np.float32)


@pytest.fixture
def sample_dataframe(sample_features):
    """Generate sample DataFrame with named columns."""
    columns = [f"feature_{i}" for i in range(sample_features.shape[1])]
    return pd.DataFrame(sample_features, columns=columns)


@pytest.fixture
def feature_columns():
    """Feature column names."""
    return [f"feature_{i}" for i in range(10)]


@pytest.fixture
def fitted_model(sample_features):
    """Create a fitted mock model."""
    model = MockInferenceModel()
    y = np.random.randint(0, 3, sample_features.shape[0])
    model.fit(sample_features, y, sample_features, y)
    return model


@pytest.fixture
def fitted_scaler(sample_features):
    """Create a fitted scaler."""
    scaler = RobustScaler()
    scaler.fit(sample_features)
    return scaler


# =============================================================================
# BATCH INFERENCE TESTS
# =============================================================================

class TestBatchInference:
    """Test batch inference functionality."""

    def test_batch_inference_produces_correct_shape(
        self, fitted_model, fitted_scaler, feature_columns, sample_features
    ):
        """Batch inference output shape matches input."""
        from src.inference import ModelBundle, BatchPredictor

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockInferenceModel()

                bundle.save(bundle_path)
                predictor = BatchPredictor.from_bundle(bundle_path)

                df = pd.DataFrame(sample_features, columns=feature_columns)
                result = predictor.predict_batch(df, batch_size=25)

                # Verify output shape matches input
                assert result.n_samples == len(sample_features), \
                    f"Expected {len(sample_features)} samples, got {result.n_samples}"

                # Verify predictions DataFrame shape
                predictions_df = result.predictions_df
                assert len(predictions_df) == len(sample_features)
                assert "prediction" in predictions_df.columns
                assert "confidence" in predictions_df.columns

    def test_batch_inference_handles_varying_sizes(
        self, fitted_model, fitted_scaler, feature_columns
    ):
        """Batch inference handles different batch sizes correctly."""
        from src.inference import ModelBundle, BatchPredictor

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockInferenceModel()

                bundle.save(bundle_path)
                predictor = BatchPredictor.from_bundle(bundle_path)

                # Test with different input sizes
                for n_samples in [1, 10, 50, 100, 137]:  # 137 is prime, tests edge case
                    np.random.seed(42)
                    X = np.random.randn(n_samples, 10).astype(np.float32)
                    df = pd.DataFrame(X, columns=feature_columns)

                    result = predictor.predict_batch(df, batch_size=30)

                    assert result.n_samples == n_samples, \
                        f"For input size {n_samples}, expected {n_samples} outputs"

    def test_batch_inference_probability_constraints(
        self, fitted_model, fitted_scaler, feature_columns, sample_features
    ):
        """Batch inference probabilities should be valid (sum to 1, in [0,1])."""
        from src.inference import ModelBundle, BatchPredictor

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockInferenceModel()

                bundle.save(bundle_path)
                predictor = BatchPredictor.from_bundle(bundle_path)

                df = pd.DataFrame(sample_features, columns=feature_columns)
                result = predictor.predict_batch(df)

                predictions_df = result.predictions_df

                # Get probability columns
                prob_cols = [c for c in predictions_df.columns if c.startswith('prob_')]

                for col in prob_cols:
                    # All probabilities should be in [0, 1]
                    assert (predictions_df[col] >= 0).all(), \
                        f"Column {col} has negative probabilities"
                    assert (predictions_df[col] <= 1).all(), \
                        f"Column {col} has probabilities > 1"

                # Probabilities should sum to 1 (approximately)
                if len(prob_cols) == 3:  # short, neutral, long
                    prob_sums = predictions_df[prob_cols].sum(axis=1)
                    np.testing.assert_allclose(prob_sums, 1.0, atol=0.01)


# =============================================================================
# MISSING FEATURES HANDLING
# =============================================================================

class TestMissingFeaturesHandling:
    """Test handling of missing features in inference."""

    def test_handles_missing_features_error_message(
        self, fitted_model, fitted_scaler, feature_columns
    ):
        """Inference should raise clear error for missing features."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Create DataFrame with missing columns
        np.random.seed(42)
        X = np.random.randn(10, 8).astype(np.float32)
        df_missing = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])

        # Should raise with informative message
        with pytest.raises(ValueError, match="Missing features"):
            bundle.predict(df_missing)

    def test_handles_extra_features_gracefully(
        self, fitted_model, fitted_scaler, feature_columns, sample_features
    ):
        """Inference should handle extra features gracefully."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Add extra columns
        df = pd.DataFrame(sample_features, columns=feature_columns)
        df['extra_feature_1'] = np.random.randn(len(df))
        df['extra_feature_2'] = np.random.randn(len(df))

        # Should work (extra columns ignored)
        result = bundle.predict(df)
        assert result.n_samples == len(sample_features)

    def test_handles_reordered_features(
        self, fitted_model, fitted_scaler, feature_columns, sample_features
    ):
        """Inference should handle reordered feature columns."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Create DataFrame with shuffled columns
        df = pd.DataFrame(sample_features, columns=feature_columns)
        shuffled_cols = feature_columns.copy()
        np.random.shuffle(shuffled_cols)
        df_shuffled = df[shuffled_cols]

        # Should work with correct column ordering
        result = bundle.predict(df_shuffled)
        assert result.n_samples == len(sample_features)

    def test_wrong_feature_count_numpy_raises(
        self, fitted_model, fitted_scaler, feature_columns
    ):
        """Numpy array with wrong feature count should raise error."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Wrong number of features
        X_wrong = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 10 features"):
            bundle.predict(X_wrong)


# =============================================================================
# MODEL BUNDLE COMPONENT TESTS
# =============================================================================

class TestModelBundleComponents:
    """Test model bundling and loading components."""

    def test_bundle_saves_all_components(
        self, fitted_model, fitted_scaler, feature_columns
    ):
        """Bundle should include model, scaler, and config."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
            training_metrics={"val_f1": 0.75},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"
            bundle.save(bundle_path)

            # Check all components exist
            assert (bundle_path / "manifest.json").exists(), "Missing manifest.json"
            assert (bundle_path / "metadata.json").exists(), "Missing metadata.json"
            assert (bundle_path / "features.json").exists(), "Missing features.json"
            assert (bundle_path / "scaler.pkl").exists(), "Missing scaler.pkl"
            assert (bundle_path / "model").exists(), "Missing model directory"

    def test_bundle_loads_correctly(
        self, fitted_model, fitted_scaler, feature_columns, sample_features
    ):
        """Loaded bundle should produce same predictions as original."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Get original predictions
        original_output = bundle.predict(sample_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockInferenceModel()

                bundle.save(bundle_path)
                loaded_bundle = ModelBundle.load(bundle_path)

                # Get loaded predictions
                loaded_output = loaded_bundle.predict(sample_features)

                # Predictions should match
                np.testing.assert_array_equal(
                    original_output.class_predictions,
                    loaded_output.class_predictions,
                )

    def test_bundle_metadata_integrity(
        self, fitted_model, fitted_scaler, feature_columns
    ):
        """Bundle metadata should be preserved through save/load."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=15,
            training_metrics={"val_f1": 0.82, "train_f1": 0.85},
            extra_metadata={"symbol": "MES", "experiment": "test123"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockInferenceModel()

                bundle.save(bundle_path)
                loaded = ModelBundle.load(bundle_path)

                # Verify all metadata preserved
                assert loaded.metadata.horizon == 15
                assert loaded.metadata.n_features == 10
                assert loaded.metadata.training_metrics["val_f1"] == 0.82
                assert loaded.metadata.extra["symbol"] == "MES"
                assert loaded.feature_columns == feature_columns

    def test_bundle_manifest_structure(
        self, fitted_model, fitted_scaler, feature_columns
    ):
        """Manifest should have correct structure."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"
            bundle.save(bundle_path)

            # Read and verify manifest
            with open(bundle_path / "manifest.json") as f:
                manifest = json.load(f)

            assert "version" in manifest
            assert "files" in manifest
            assert "checksums" in manifest

            # Verify metadata has creation time
            with open(bundle_path / "metadata.json") as f:
                metadata = json.load(f)
            assert "created_at" in metadata
            assert "model_name" in metadata


# =============================================================================
# SCALER STATE PERSISTENCE
# =============================================================================

class TestScalerStatePersistence:
    """Test scaler state is correctly persisted."""

    def test_scaler_maintains_statistics_after_save_load(
        self, sample_features, feature_columns, fitted_model
    ):
        """Saved scaler produces identical transforms after loading."""
        from src.inference import ModelBundle

        # Fit scaler
        scaler = RobustScaler()
        scaler.fit(sample_features)

        # Create test data for transform comparison
        np.random.seed(99)
        X_test = np.random.randn(20, 10).astype(np.float32)
        original_transform = scaler.transform(X_test)

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockInferenceModel()

                bundle.save(bundle_path)

                # Load scaler directly
                with open(bundle_path / "scaler.pkl", "rb") as f:
                    loaded_scaler = pickle.load(f)

                # Transform with loaded scaler
                loaded_transform = loaded_scaler.transform(X_test)

                # Should be identical
                np.testing.assert_array_almost_equal(
                    original_transform, loaded_transform
                )

    def test_scaler_center_and_scale_preserved(
        self, sample_features, feature_columns, fitted_model
    ):
        """Scaler center and scale parameters are preserved."""
        from src.inference import ModelBundle

        scaler = RobustScaler()
        scaler.fit(sample_features)

        original_center = scaler.center_.copy()
        original_scale = scaler.scale_.copy()

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockInferenceModel()

                bundle.save(bundle_path)

                with open(bundle_path / "scaler.pkl", "rb") as f:
                    loaded_scaler = pickle.load(f)

                np.testing.assert_array_equal(loaded_scaler.center_, original_center)
                np.testing.assert_array_equal(loaded_scaler.scale_, original_scale)

    def test_standard_scaler_mean_var_preserved(
        self, sample_features, feature_columns, fitted_model
    ):
        """StandardScaler mean and variance are preserved."""
        from src.inference import ModelBundle

        scaler = StandardScaler()
        scaler.fit(sample_features)

        original_mean = scaler.mean_.copy()
        original_var = scaler.var_.copy()

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockInferenceModel()

                bundle.save(bundle_path)

                with open(bundle_path / "scaler.pkl", "rb") as f:
                    loaded_scaler = pickle.load(f)

                np.testing.assert_array_almost_equal(loaded_scaler.mean_, original_mean)
                np.testing.assert_array_almost_equal(loaded_scaler.var_, original_var)


# =============================================================================
# PRODUCTION-LIKE SCENARIOS
# =============================================================================

class TestProductionScenarios:
    """Test production-like inference scenarios."""

    def test_single_sample_inference(
        self, fitted_model, fitted_scaler, feature_columns
    ):
        """Inference on a single sample should work."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Single sample
        np.random.seed(42)
        X_single = np.random.randn(1, 10).astype(np.float32)

        result = bundle.predict(X_single)
        assert result.n_samples == 1
        assert len(result.class_predictions) == 1
        assert result.class_probabilities.shape == (1, 3)

    def test_inference_with_nan_values(
        self, fitted_model, fitted_scaler, feature_columns, sample_features
    ):
        """Inference should handle NaN values appropriately."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Add NaN values
        X_with_nan = sample_features.copy()
        X_with_nan[5, 3] = np.nan
        X_with_nan[10, 7] = np.nan

        # Should not crash (may produce NaN in output which is expected)
        try:
            result = bundle.predict(X_with_nan)
            assert result.n_samples == len(X_with_nan)
        except ValueError as e:
            # If it raises, ensure error message is informative
            assert "nan" in str(e).lower() or "NaN" in str(e)

    def test_inference_with_inf_values(
        self, fitted_model, fitted_scaler, feature_columns, sample_features
    ):
        """Inference should handle Inf values appropriately."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Add Inf values
        X_with_inf = sample_features.copy()
        X_with_inf[5, 3] = np.inf
        X_with_inf[10, 7] = -np.inf

        # Should handle (may clip or produce inf in output)
        try:
            result = bundle.predict(X_with_inf)
            assert result.n_samples == len(X_with_inf)
        except (ValueError, RuntimeError) as e:
            # If it raises, ensure error message is informative
            assert "inf" in str(e).lower() or "Inf" in str(e) or "overflow" in str(e).lower()

    def test_batch_streaming_produces_all_samples(
        self, fitted_model, fitted_scaler, feature_columns
    ):
        """Streaming batch predictions should produce all samples."""
        from src.inference import ModelBundle, BatchPredictor

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle"

            with patch("src.inference.bundle.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = MockInferenceModel()

                bundle.save(bundle_path)
                predictor = BatchPredictor.from_bundle(bundle_path)

                np.random.seed(42)
                n_samples = 137  # Prime number for edge case testing
                df = pd.DataFrame(
                    np.random.randn(n_samples, 10).astype(np.float32),
                    columns=feature_columns,
                )

                batches = list(predictor.predict_streaming(df, batch_size=30))

                # Total samples from all batches
                total_samples = sum(len(b) for b in batches)
                assert total_samples == n_samples

    def test_concurrent_predictions_same_result(
        self, fitted_model, fitted_scaler, feature_columns, sample_features
    ):
        """Multiple predictions on same data should be deterministic."""
        from src.inference import ModelBundle

        bundle = ModelBundle.from_training(
            model=fitted_model,
            scaler=fitted_scaler,
            feature_columns=feature_columns,
            horizon=20,
        )

        # Run prediction multiple times
        results = []
        for _ in range(3):
            result = bundle.predict(sample_features)
            results.append(result.class_predictions.copy())

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
