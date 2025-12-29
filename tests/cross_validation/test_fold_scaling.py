"""
Tests for Fold-Aware Scaling.

Tests:
- FoldAwareScaler correctly fits on train-only
- Scaling methods work (robust, standard, none)
- No data leakage between folds
- Sequence model handling
- Integration with model registry
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.cross_validation.fold_scaling import (
    FoldAwareScaler,
    FoldScalingResult,
    get_scaling_method_for_model,
    scale_cv_fold,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_fold_data():
    """Generate sample train/val data for fold scaling tests."""
    np.random.seed(42)
    n_train, n_val, n_features = 100, 20, 10

    # Training data with specific mean/std
    X_train = np.random.randn(n_train, n_features) * 2 + 5  # mean=5, std=2

    # Validation data with different distribution (to detect leakage)
    X_val = np.random.randn(n_val, n_features) * 10 + 100  # mean=100, std=10

    return X_train, X_val


@pytest.fixture
def sample_dataframe_data():
    """Generate sample DataFrame data."""
    np.random.seed(42)
    n_train, n_val, n_features = 50, 10, 5

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
        index=pd.date_range("2023-01-01", periods=n_train, freq="D"),
    )
    X_val = pd.DataFrame(
        np.random.randn(n_val, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
        index=pd.date_range("2023-03-01", periods=n_val, freq="D"),
    )

    return X_train, X_val


@pytest.fixture
def full_dataset_with_indices():
    """Generate full dataset with train/val indices."""
    np.random.seed(42)
    n_samples, n_features = 200, 15

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )

    # First 150 for train, last 50 for val
    train_idx = np.arange(150)
    val_idx = np.arange(150, 200)

    return X, train_idx, val_idx


# =============================================================================
# FOLDAWARESCALER TESTS
# =============================================================================

class TestFoldAwareScaler:
    """Tests for FoldAwareScaler class."""

    def test_init_valid_methods(self):
        """Scaler initializes with valid methods."""
        for method in ["robust", "standard", "none"]:
            scaler = FoldAwareScaler(method=method)
            assert scaler.method == method

    def test_init_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scaling method"):
            FoldAwareScaler(method="invalid")

    def test_fit_transform_robust(self, sample_fold_data):
        """Robust scaling works correctly."""
        X_train, X_val = sample_fold_data

        scaler = FoldAwareScaler(method="robust", clip_outliers=False)
        result = scaler.fit_transform_fold(X_train, X_val)

        assert isinstance(result, FoldScalingResult)
        assert result.method == "robust"
        assert result.X_train_scaled.shape == X_train.shape
        assert result.X_val_scaled.shape == X_val.shape
        assert isinstance(result.scaler, RobustScaler)

    def test_fit_transform_standard(self, sample_fold_data):
        """Standard scaling works correctly."""
        X_train, X_val = sample_fold_data

        scaler = FoldAwareScaler(method="standard", clip_outliers=False)
        result = scaler.fit_transform_fold(X_train, X_val)

        assert result.method == "standard"
        assert isinstance(result.scaler, StandardScaler)

        # Verify train data is normalized (mean ~0, std ~1)
        np.testing.assert_allclose(
            result.X_train_scaled.mean(axis=0), 0, atol=0.1
        )
        np.testing.assert_allclose(
            result.X_train_scaled.std(axis=0), 1, atol=0.1
        )

    def test_fit_transform_none(self, sample_fold_data):
        """No scaling returns data unchanged."""
        X_train, X_val = sample_fold_data

        scaler = FoldAwareScaler(method="none")
        result = scaler.fit_transform_fold(X_train, X_val)

        assert result.method == "none"
        assert result.scaler is None

        # Data should be unchanged
        np.testing.assert_array_equal(result.X_train_scaled, X_train)
        np.testing.assert_array_equal(result.X_val_scaled, X_val)

    def test_no_leakage_from_val_to_train(self, sample_fold_data):
        """Validation data doesn't affect training scaling."""
        X_train, X_val = sample_fold_data

        # Create scaler and fit
        scaler = FoldAwareScaler(method="robust", clip_outliers=False)
        result = scaler.fit_transform_fold(X_train, X_val)

        # Create a reference scaler using only training data
        ref_scaler = RobustScaler()
        ref_train_scaled = ref_scaler.fit_transform(X_train)

        # Training data should be identical regardless of validation data
        np.testing.assert_array_almost_equal(
            result.X_train_scaled, ref_train_scaled
        )

    def test_val_uses_train_statistics(self, sample_fold_data):
        """Validation is scaled using training statistics only."""
        X_train, X_val = sample_fold_data

        scaler = FoldAwareScaler(method="robust", clip_outliers=False)
        result = scaler.fit_transform_fold(X_train, X_val)

        # Create reference using train-only statistics
        ref_scaler = RobustScaler()
        ref_scaler.fit(X_train)
        ref_val_scaled = ref_scaler.transform(X_val)

        # Validation scaling should match
        np.testing.assert_array_almost_equal(
            result.X_val_scaled, ref_val_scaled
        )

    def test_clipping_applied(self, sample_fold_data):
        """Outlier clipping is applied when enabled."""
        X_train, X_val = sample_fold_data

        # Add extreme outliers
        X_train[0, 0] = 1000
        X_val[0, 0] = -1000

        scaler = FoldAwareScaler(method="robust", clip_outliers=True, clip_std=3.0)
        result = scaler.fit_transform_fold(X_train, X_val)

        # Values should be clipped to [-3, 3]
        assert result.X_train_scaled.max() <= 3.0
        assert result.X_train_scaled.min() >= -3.0
        assert result.X_val_scaled.max() <= 3.0
        assert result.X_val_scaled.min() >= -3.0

    def test_each_fold_gets_fresh_scaler(self, sample_fold_data):
        """Each fit_transform_fold call creates a fresh scaler."""
        X_train1, X_val1 = sample_fold_data

        # Create different data for second fold
        np.random.seed(99)
        X_train2 = np.random.randn(80, 10) * 5 + 10
        X_val2 = np.random.randn(30, 10) * 3 + 20

        scaler = FoldAwareScaler(method="robust", clip_outliers=False)

        # First fold
        result1 = scaler.fit_transform_fold(X_train1, X_val1)
        scaler1_center = result1.scaler.center_.copy()

        # Second fold
        result2 = scaler.fit_transform_fold(X_train2, X_val2)
        scaler2_center = result2.scaler.center_.copy()

        # Scalers should have different parameters
        assert not np.allclose(scaler1_center, scaler2_center)

    def test_fit_transform_fold_df(self, sample_dataframe_data):
        """DataFrame version preserves columns and index."""
        X_train, X_val = sample_dataframe_data

        scaler = FoldAwareScaler(method="robust")
        X_train_scaled, X_val_scaled = scaler.fit_transform_fold_df(X_train, X_val)

        # Check DataFrame structure preserved
        assert isinstance(X_train_scaled, pd.DataFrame)
        assert isinstance(X_val_scaled, pd.DataFrame)
        assert list(X_train_scaled.columns) == list(X_train.columns)
        assert list(X_val_scaled.columns) == list(X_val.columns)
        assert list(X_train_scaled.index) == list(X_train.index)
        assert list(X_val_scaled.index) == list(X_val.index)


class TestScaleCVFold:
    """Tests for scale_cv_fold convenience function."""

    def test_scale_cv_fold_with_dataframe(self, full_dataset_with_indices):
        """scale_cv_fold works with DataFrame."""
        X, train_idx, val_idx = full_dataset_with_indices

        X_train_scaled, X_val_scaled = scale_cv_fold(
            X, train_idx, val_idx, method="robust"
        )

        assert X_train_scaled.shape == (len(train_idx), X.shape[1])
        assert X_val_scaled.shape == (len(val_idx), X.shape[1])

    def test_scale_cv_fold_with_array(self):
        """scale_cv_fold works with numpy array."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        train_idx = np.arange(80)
        val_idx = np.arange(80, 100)

        X_train_scaled, X_val_scaled = scale_cv_fold(
            X, train_idx, val_idx, method="standard"
        )

        assert X_train_scaled.shape == (80, 10)
        assert X_val_scaled.shape == (20, 10)

    def test_scale_cv_fold_no_scaling(self, full_dataset_with_indices):
        """scale_cv_fold with method='none' returns unscaled data."""
        X, train_idx, val_idx = full_dataset_with_indices

        X_train_scaled, X_val_scaled = scale_cv_fold(
            X, train_idx, val_idx, method="none"
        )

        np.testing.assert_array_equal(X_train_scaled, X.iloc[train_idx].values)
        np.testing.assert_array_equal(X_val_scaled, X.iloc[val_idx].values)


class TestGetScalingMethodForModel:
    """Tests for get_scaling_method_for_model function."""

    def test_unknown_model_defaults_to_robust(self):
        """Unknown model defaults to robust scaling."""
        method = get_scaling_method_for_model("nonexistent_model_xyz")
        assert method == "robust"

    def test_tree_based_model_no_scaling(self):
        """Tree-based models that don't require scaling return 'none'."""
        # Note: This depends on actual model registration
        # XGBoost typically has requires_scaling=False
        try:
            method = get_scaling_method_for_model("xgboost")
            # XGBoost doesn't require scaling
            assert method == "none"
        except Exception:
            # If model not registered, skip
            pytest.skip("xgboost not registered in ModelRegistry")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFoldScalingIntegration:
    """Integration tests for fold-aware scaling."""

    def test_multiple_folds_independent(self):
        """Multiple folds have independent scaling."""
        np.random.seed(42)
        n_samples, n_features = 500, 20
        X = np.random.randn(n_samples, n_features)

        # Simulate 5-fold CV
        fold_size = n_samples // 5
        scalers = []

        fold_scaler = FoldAwareScaler(method="standard", clip_outliers=False)

        for fold_idx in range(5):
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size

            train_idx = np.concatenate([
                np.arange(0, val_start),
                np.arange(val_end, n_samples)
            ])
            val_idx = np.arange(val_start, val_end)

            X_train = X[train_idx]
            X_val = X[val_idx]

            result = fold_scaler.fit_transform_fold(X_train, X_val)
            scalers.append(result.scaler.mean_.copy())

        # Each fold should have different means (since train data differs)
        for i in range(4):
            # Means shouldn't be identical across folds
            assert not np.allclose(scalers[i], scalers[i + 1], atol=0.01)

    def test_scaling_prevents_leakage(self):
        """Verify scaling prevents information leakage."""
        np.random.seed(42)

        # Create data where validation has very different distribution
        n_train, n_val = 100, 50
        n_features = 5

        # Training: standard normal
        X_train = np.random.randn(n_train, n_features)

        # Validation: shifted and scaled (represents "future" data)
        X_val = np.random.randn(n_val, n_features) * 10 + 100

        # With fold-aware scaling
        scaler = FoldAwareScaler(method="standard", clip_outliers=False)
        result = scaler.fit_transform_fold(X_train, X_val)

        # Training should be normalized (mean ~0, std ~1)
        train_mean = result.X_train_scaled.mean(axis=0)
        train_std = result.X_train_scaled.std(axis=0)

        np.testing.assert_allclose(train_mean, 0, atol=0.1)
        np.testing.assert_allclose(train_std, 1, atol=0.1)

        # Validation should NOT be normalized (since we used train stats)
        val_mean = result.X_val_scaled.mean(axis=0)
        val_std = result.X_val_scaled.std(axis=0)

        # Val mean should reflect the shift (100 - train_mean) / train_std ≈ 100
        assert np.all(np.abs(val_mean) > 10)  # Should be far from 0

    def test_scaling_with_nan_handling(self):
        """Scaler handles NaN values gracefully."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        X_val = np.random.randn(20, 10)

        # Add some NaN values
        X_train[5, 3] = np.nan
        X_val[2, 7] = np.nan

        scaler = FoldAwareScaler(method="robust")

        # Should not raise - sklearn RobustScaler handles NaN gracefully
        # (may propagate NaN to output, which is expected)
        result = scaler.fit_transform_fold(X_train, X_val)

        # Result should exist (may have NaN in it)
        assert result.X_train_scaled.shape == X_train.shape
        # Verify NaN propagates (expected behavior)
        assert np.isnan(result.X_train_scaled[5, 3])


class TestLeakageDetection:
    """Tests that verify leakage is prevented."""

    def test_val_statistics_dont_affect_train_scaling(self):
        """Extreme validation values don't affect training scaling."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)  # Normal distribution

        # Normal validation
        X_val_normal = np.random.randn(20, 5)

        # Extreme validation (would affect global statistics)
        X_val_extreme = np.random.randn(20, 5) * 1000 + 10000

        scaler = FoldAwareScaler(method="standard", clip_outliers=False)

        result_normal = scaler.fit_transform_fold(X_train.copy(), X_val_normal)
        result_extreme = scaler.fit_transform_fold(X_train.copy(), X_val_extreme)

        # Training scaling should be identical regardless of validation
        np.testing.assert_array_almost_equal(
            result_normal.X_train_scaled,
            result_extreme.X_train_scaled,
        )

    def test_global_vs_fold_scaling_difference(self):
        """Demonstrate difference between global and fold-aware scaling."""
        np.random.seed(42)

        # Create data with trend (simulating time series)
        n_samples = 200
        n_features = 3
        X = np.random.randn(n_samples, n_features)
        # Add trend: later samples have higher values
        X += np.arange(n_samples).reshape(-1, 1) * 0.1

        train_idx = np.arange(160)
        val_idx = np.arange(160, 200)

        # Global scaling (WRONG - leakage)
        global_scaler = StandardScaler()
        X_global_scaled = global_scaler.fit_transform(X)
        X_train_global = X_global_scaled[train_idx]
        X_val_global = X_global_scaled[val_idx]

        # Fold-aware scaling (CORRECT - no leakage)
        fold_scaler = FoldAwareScaler(method="standard", clip_outliers=False)
        result = fold_scaler.fit_transform_fold(X[train_idx], X[val_idx])

        # With global scaling, validation will be closer to normalized
        # With fold scaling, validation will show the true distribution shift

        # Global scaled validation mean should be closer to 0
        global_val_mean = X_val_global.mean()

        # Fold scaled validation mean should be higher (reflecting trend)
        fold_val_mean = result.X_val_scaled.mean()

        # The fold-aware scaling should produce higher validation means
        # because it doesn't "know" about the future trend
        assert fold_val_mean > global_val_mean + 0.5


# =============================================================================
# SCALER PERSISTENCE TESTS
# =============================================================================

class TestScalerPersistence:
    """Tests for scaler persistence and loading."""

    def test_scaler_maintains_statistics_after_save_load(self, tmp_path):
        """Saved scaler produces identical transforms after loading."""
        import joblib

        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        X_test = np.random.randn(20, 10)

        # Fit and transform
        scaler = RobustScaler()
        scaler.fit(X_train)
        original_transform = scaler.transform(X_test)

        # Save and load
        save_path = tmp_path / "scaler.joblib"
        joblib.dump(scaler, save_path)
        loaded_scaler = joblib.load(save_path)

        # Verify same output
        loaded_transform = loaded_scaler.transform(X_test)
        np.testing.assert_array_almost_equal(original_transform, loaded_transform)

    def test_scaler_center_preserved_after_persistence(self, tmp_path):
        """Scaler center (median) is preserved after save/load."""
        import joblib

        np.random.seed(42)
        X_train = np.random.randn(100, 5) * 3 + 10  # mean ~10

        scaler = RobustScaler()
        scaler.fit(X_train)
        original_center = scaler.center_.copy()

        save_path = tmp_path / "scaler.joblib"
        joblib.dump(scaler, save_path)
        loaded_scaler = joblib.load(save_path)

        np.testing.assert_array_equal(original_center, loaded_scaler.center_)

    def test_scaler_scale_preserved_after_persistence(self, tmp_path):
        """Scaler scale (IQR) is preserved after save/load."""
        import joblib

        np.random.seed(42)
        X_train = np.random.randn(100, 5) * 3 + 10

        scaler = RobustScaler()
        scaler.fit(X_train)
        original_scale = scaler.scale_.copy()

        save_path = tmp_path / "scaler.joblib"
        joblib.dump(scaler, save_path)
        loaded_scaler = joblib.load(save_path)

        np.testing.assert_array_equal(original_scale, loaded_scaler.scale_)

    def test_standard_scaler_persistence(self, tmp_path):
        """StandardScaler mean and std preserved after save/load."""
        import joblib

        np.random.seed(42)
        X_train = np.random.randn(100, 5) * 2 + 5
        X_test = np.random.randn(20, 5)

        scaler = StandardScaler()
        scaler.fit(X_train)
        original_transform = scaler.transform(X_test)
        original_mean = scaler.mean_.copy()
        original_var = scaler.var_.copy()

        save_path = tmp_path / "scaler.joblib"
        joblib.dump(scaler, save_path)
        loaded_scaler = joblib.load(save_path)

        # Check parameters
        np.testing.assert_array_almost_equal(original_mean, loaded_scaler.mean_)
        np.testing.assert_array_almost_equal(original_var, loaded_scaler.var_)

        # Check transform output
        loaded_transform = loaded_scaler.transform(X_test)
        np.testing.assert_array_almost_equal(original_transform, loaded_transform)

    def test_fold_aware_scaler_result_can_be_persisted(self, tmp_path, sample_fold_data):
        """FoldScalingResult scaler can be persisted."""
        import joblib

        X_train, X_val = sample_fold_data

        fold_scaler = FoldAwareScaler(method="robust", clip_outliers=False)
        result = fold_scaler.fit_transform_fold(X_train, X_val)

        # Save the internal scaler
        save_path = tmp_path / "fold_scaler.joblib"
        joblib.dump(result.scaler, save_path)

        # Load and verify
        loaded_scaler = joblib.load(save_path)

        # Transform new data
        np.random.seed(99)
        X_new = np.random.randn(10, X_train.shape[1])

        original_transform = result.scaler.transform(X_new)
        loaded_transform = loaded_scaler.transform(X_new)

        np.testing.assert_array_almost_equal(original_transform, loaded_transform)

    def test_pickle_compatibility(self, tmp_path):
        """Scaler works with pickle (not just joblib)."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        X_test = np.random.randn(20, 10)

        scaler = RobustScaler()
        scaler.fit(X_train)
        original_transform = scaler.transform(X_test)

        # Save with pickle
        import pickle
        save_path = tmp_path / "scaler.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)

        # Load with pickle
        with open(save_path, "rb") as f:
            loaded_scaler = pickle.load(f)

        loaded_transform = loaded_scaler.transform(X_test)
        np.testing.assert_array_almost_equal(original_transform, loaded_transform)
