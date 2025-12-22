"""
Comprehensive tests for FeatureScaler module.
Tests train-only fitting, leakage prevention, and edge cases.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.feature_scaler import (
    FeatureScaler,
    ScalerConfig,
    scale_splits,
    validate_no_leakage
)


class TestFeatureScalerBasic:
    """Basic functionality tests."""

    @pytest.fixture
    def sample_data(self):
        """Create sample train/val/test data."""
        np.random.seed(42)
        n_train, n_val, n_test = 1000, 200, 200

        train_df = pd.DataFrame({
            'feature1': np.random.randn(n_train) * 10 + 100,
            'feature2': np.random.randn(n_train) * 5,
            'feature3': np.random.exponential(2, n_train),
            'label': np.random.choice([-1, 0, 1], n_train)
        })

        # Val/test have different distributions (simulating regime change)
        val_df = pd.DataFrame({
            'feature1': np.random.randn(n_val) * 15 + 120,
            'feature2': np.random.randn(n_val) * 8,
            'feature3': np.random.exponential(3, n_val),
            'label': np.random.choice([-1, 0, 1], n_val)
        })

        test_df = pd.DataFrame({
            'feature1': np.random.randn(n_test) * 12 + 110,
            'feature2': np.random.randn(n_test) * 6,
            'feature3': np.random.exponential(2.5, n_test),
            'label': np.random.choice([-1, 0, 1], n_test)
        })

        return train_df, val_df, test_df

    def test_fit_transform(self, sample_data):
        """Test basic fit and transform."""
        train_df, _, _ = sample_data
        feature_cols = ['feature1', 'feature2', 'feature3']

        scaler = FeatureScaler()
        train_scaled = scaler.fit_transform(train_df, feature_cols)

        assert scaler.is_fitted
        assert len(scaler.statistics) == 3
        assert train_scaled.shape == train_df.shape

    def test_transform_without_fit_raises(self, sample_data):
        """Test that transform without fit raises error."""
        train_df, _, _ = sample_data
        scaler = FeatureScaler()

        with pytest.raises(ValueError, match="has not been fitted"):
            scaler.transform(train_df)

    def test_train_only_fitting(self, sample_data):
        """Test that only training data affects statistics."""
        train_df, val_df, test_df = sample_data
        feature_cols = ['feature1', 'feature2', 'feature3']

        scaler = FeatureScaler()
        scaler.fit(train_df, feature_cols)

        # Get training statistics
        train_mean = scaler.statistics['feature1'].train_mean

        # Transform val/test - should NOT change statistics
        val_scaled = scaler.transform(val_df)
        test_scaled = scaler.transform(test_df)

        assert scaler.statistics['feature1'].train_mean == train_mean

    def test_scale_splits_function(self, sample_data):
        """Test the convenience function."""
        train_df, val_df, test_df = sample_data
        feature_cols = ['feature1', 'feature2', 'feature3']

        train_scaled, val_scaled, test_scaled, scaler = scale_splits(
            train_df, val_df, test_df, feature_cols
        )

        assert scaler.is_fitted
        assert train_scaled.shape == train_df.shape
        assert val_scaled.shape == val_df.shape
        assert test_scaled.shape == test_df.shape


class TestLeakagePrevention:
    """Tests for data leakage prevention."""

    def test_no_future_information(self):
        """Verify scaled val/test don't use their own statistics."""
        np.random.seed(42)

        # Training data: mean=0, std=1
        train_df = pd.DataFrame({'f1': np.random.randn(1000)})

        # Test data: mean=100, std=10 (very different)
        test_df = pd.DataFrame({'f1': np.random.randn(100) * 10 + 100})

        # Disable clipping to see the full effect of using train statistics
        scaler = FeatureScaler(clip_outliers=False)
        scaler.fit(train_df, ['f1'])
        test_scaled = scaler.transform(test_df)

        # If leakage occurred, test mean would be ~0
        # Without leakage, test mean should be far from 0
        # With train statistics (mean~0, std~1), test data (mean~100) should scale to ~100
        assert abs(test_scaled['f1'].mean()) > 10  # Should be ~100

    def test_validate_no_leakage_passes(self):
        """Test leakage validation function."""
        np.random.seed(42)

        train_df = pd.DataFrame({'f1': np.random.randn(1000)})
        val_df = pd.DataFrame({'f1': np.random.randn(200)})
        test_df = pd.DataFrame({'f1': np.random.randn(200)})

        scaler = FeatureScaler()
        train_scaled = scaler.fit_transform(train_df, ['f1'])
        val_scaled = scaler.transform(val_df)
        test_scaled = scaler.transform(test_df)

        # Should not raise - validate_no_leakage takes 4 args: train, val, test, scaler
        result = validate_no_leakage(train_df, val_df, test_df, scaler)
        assert not result['leakage_detected']


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises error."""
        scaler = FeatureScaler()
        empty_df = pd.DataFrame({'f1': []})

        with pytest.raises(ValueError, match="empty"):
            scaler.fit(empty_df, ['f1'])

    def test_missing_column_raises(self):
        """Test that missing column raises error."""
        df = pd.DataFrame({'f1': [1, 2, 3]})
        scaler = FeatureScaler()

        with pytest.raises(ValueError, match="not found"):
            scaler.fit(df, ['f1', 'f2'])  # f2 doesn't exist

    def test_all_nan_column(self):
        """Test handling of all-NaN column."""
        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'f2': [np.nan, np.nan, np.nan]
        })

        scaler = FeatureScaler()
        scaler.fit(df, ['f1', 'f2'])

        # Should handle gracefully - check that it's fitted
        assert scaler.is_fitted
        # All-NaN columns should have train_mean as NaN in statistics
        # but the scaler should still work
        assert 'f2' in scaler.statistics

    def test_constant_column(self):
        """Test handling of constant column (zero variance)."""
        df = pd.DataFrame({
            'f1': [5.0, 5.0, 5.0, 5.0, 5.0],
            'f2': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        scaler = FeatureScaler()
        scaler.fit(df, ['f1', 'f2'])

        # Should be fitted without error
        assert scaler.is_fitted
        # Constant column should have near-zero std
        assert scaler.statistics['f1'].train_std < 1e-6

    def test_outlier_clipping(self):
        """Test outlier clipping functionality."""
        df = pd.DataFrame({
            'f1': [0.0, 0.0, 0.0, 0.0, 100.0]  # One extreme outlier
        })

        config = ScalerConfig(clip_outliers=True, clip_range=(-3.0, 3.0))
        scaler = FeatureScaler(config=config)
        scaled = scaler.fit_transform(df, ['f1'])

        # Outlier should be clipped
        assert scaled['f1'].max() <= 3.0
        assert scaled['f1'].min() >= -3.0


class TestPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load(self):
        """Test scaler can be saved and loaded."""
        df = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100)
        })

        scaler = FeatureScaler()
        scaler.fit(df, ['f1', 'f2'])

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = Path(f.name)

        try:
            scaler.save(path)
            loaded = FeatureScaler.load(path)

            assert loaded.is_fitted
            assert loaded.feature_names == scaler.feature_names
            # Statistics are ScalingStatistics objects, compare key values
            for fname in scaler.feature_names:
                assert loaded.statistics[fname].train_mean == scaler.statistics[fname].train_mean
                assert loaded.statistics[fname].train_std == scaler.statistics[fname].train_std
        finally:
            path.unlink()
            # Also clean up the JSON file that gets created
            json_path = path.with_suffix('.json')
            if json_path.exists():
                json_path.unlink()

    def test_loaded_scaler_transforms_correctly(self):
        """Test that loaded scaler produces same results."""
        np.random.seed(42)
        df = pd.DataFrame({'f1': np.random.randn(100)})
        test_df = pd.DataFrame({'f1': np.random.randn(20)})

        scaler = FeatureScaler()
        scaler.fit(df, ['f1'])
        original_transformed = scaler.transform(test_df)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = Path(f.name)

        try:
            scaler.save(path)
            loaded = FeatureScaler.load(path)
            loaded_transformed = loaded.transform(test_df)

            pd.testing.assert_frame_equal(original_transformed, loaded_transformed)
        finally:
            path.unlink()
            # Also clean up the JSON file that gets created
            json_path = path.with_suffix('.json')
            if json_path.exists():
                json_path.unlink()


class TestScalerTypes:
    """Test different scaler types."""

    def test_standard_scaler(self):
        """Test StandardScaler normalization."""
        np.random.seed(42)
        df = pd.DataFrame({'f1': np.random.randn(1000) * 10 + 50})

        scaler = FeatureScaler(scaler_type='standard', clip_outliers=False)
        scaled = scaler.fit_transform(df, ['f1'])

        # Standard scaler should produce mean ~0, std ~1
        assert abs(scaled['f1'].mean()) < 0.1
        assert abs(scaled['f1'].std() - 1.0) < 0.1

    def test_robust_scaler(self):
        """Test RobustScaler normalization."""
        np.random.seed(42)
        df = pd.DataFrame({'f1': np.random.randn(1000) * 10 + 50})

        scaler = FeatureScaler(scaler_type='robust', clip_outliers=False)
        scaled = scaler.fit_transform(df, ['f1'])

        # Robust scaler should produce median ~0
        assert abs(scaled['f1'].median()) < 0.1

    def test_minmax_scaler(self):
        """Test MinMaxScaler normalization."""
        np.random.seed(42)
        df = pd.DataFrame({'f1': np.random.randn(1000) * 10 + 50})

        scaler = FeatureScaler(scaler_type='minmax', clip_outliers=False)
        scaled = scaler.fit_transform(df, ['f1'])

        # MinMax scaler should produce values in [0, 1]
        assert scaled['f1'].min() >= 0.0 - 1e-6
        assert scaled['f1'].max() <= 1.0 + 1e-6


class TestScalerConfig:
    """Test ScalerConfig class."""

    def test_config_creation(self):
        """Test ScalerConfig default values."""
        config = ScalerConfig()
        assert config.scaler_type == 'robust'
        assert config.clip_outliers is True
        assert config.clip_range == (-5.0, 5.0)

    def test_config_custom_values(self):
        """Test ScalerConfig with custom values."""
        config = ScalerConfig(
            scaler_type='standard',
            clip_outliers=False,
            clip_range=(-3.0, 3.0)
        )
        assert config.scaler_type == 'standard'
        assert config.clip_outliers is False
        assert config.clip_range == (-3.0, 3.0)

    def test_config_to_dict(self):
        """Test ScalerConfig serialization."""
        config = ScalerConfig(scaler_type='minmax', clip_outliers=True)
        d = config.to_dict()
        assert d['scaler_type'] == 'minmax'
        assert d['clip_outliers'] is True

    def test_config_from_dict(self):
        """Test ScalerConfig deserialization."""
        d = {'scaler_type': 'standard', 'clip_outliers': False, 'clip_range': (-2.0, 2.0)}
        config = ScalerConfig.from_dict(d)
        assert config.scaler_type == 'standard'
        assert config.clip_outliers is False
        assert config.clip_range == (-2.0, 2.0)


class TestInverseTransform:
    """Test inverse transform functionality."""

    def test_inverse_transform_recovers_original(self):
        """Test that inverse transform recovers original values."""
        np.random.seed(42)
        df = pd.DataFrame({'f1': np.random.randn(100) * 10 + 50})

        scaler = FeatureScaler(scaler_type='standard', clip_outliers=False)
        scaled = scaler.fit_transform(df, ['f1'])
        recovered = scaler.inverse_transform(scaled)

        # Should recover original values (with floating point tolerance)
        np.testing.assert_allclose(
            recovered['f1'].values,
            df['f1'].values,
            rtol=1e-5
        )

    def test_inverse_transform_robust(self):
        """Test inverse transform with robust scaler."""
        np.random.seed(42)
        df = pd.DataFrame({'f1': np.random.randn(100) * 10 + 50})

        scaler = FeatureScaler(scaler_type='robust', clip_outliers=False)
        scaled = scaler.fit_transform(df, ['f1'])
        recovered = scaler.inverse_transform(scaled)

        np.testing.assert_allclose(
            recovered['f1'].values,
            df['f1'].values,
            rtol=1e-5
        )


class TestMultipleFeatures:
    """Test scaling with multiple features."""

    def test_multiple_features_scaling(self):
        """Test that multiple features are scaled independently."""
        np.random.seed(42)
        df = pd.DataFrame({
            'f1': np.random.randn(100) * 10 + 50,
            'f2': np.random.randn(100) * 100 + 1000,
            'f3': np.random.exponential(5, 100)
        })

        scaler = FeatureScaler(scaler_type='standard', clip_outliers=False)
        scaled = scaler.fit_transform(df, ['f1', 'f2', 'f3'])

        # Each feature should be independently scaled
        for col in ['f1', 'f2', 'f3']:
            assert abs(scaled[col].mean()) < 0.2
            assert abs(scaled[col].std() - 1.0) < 0.2

    def test_non_feature_columns_preserved(self):
        """Test that non-feature columns are preserved."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature': np.random.randn(100),
            'label': np.random.choice([0, 1, 2], 100),
            'datetime': pd.date_range('2023-01-01', periods=100, freq='h')
        })

        scaler = FeatureScaler()
        scaled = scaler.fit_transform(df, ['feature'])

        # Label and datetime should be unchanged
        pd.testing.assert_series_equal(scaled['label'], df['label'])
        pd.testing.assert_series_equal(scaled['datetime'], df['datetime'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
