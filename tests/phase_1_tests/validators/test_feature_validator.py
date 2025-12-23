"""
Unit tests for Feature Quality Validation.

Tests the feature quality validation module which checks:
- Feature correlations
- Feature importance via Random Forest
- Stationarity tests (ADF)

Run with: pytest tests/phase_1_tests/validators/test_feature_validator.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.validators.features import (
    get_feature_columns,
    check_feature_correlations,
    compute_feature_importance,
    run_stationarity_tests,
    check_feature_quality,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def feature_df():
    """Create a DataFrame with features for testing."""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'symbol': 'MES',
        'open': 100 + np.random.randn(n) * 0.5,
        'high': 101 + np.random.randn(n) * 0.5,
        'low': 99 + np.random.randn(n) * 0.5,
        'close': 100 + np.random.randn(n) * 0.5,
        'volume': np.random.randint(100, 1000, n),
        # Features
        'rsi_14': 50 + np.random.randn(n) * 10,
        'macd_line': np.random.randn(n) * 0.5,
        'return_1': np.random.randn(n) * 0.01,
        'return_5': np.random.randn(n) * 0.02,
        'sma_20_ratio': 1 + np.random.randn(n) * 0.01,
        'bb_width': np.random.uniform(0.02, 0.05, n),
    })

    return df


@pytest.fixture
def labeled_feature_df(feature_df):
    """Create a DataFrame with features and labels."""
    df = feature_df.copy()

    # Add labels
    np.random.seed(42)
    n = len(df)
    df['label_h5'] = np.random.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2])
    df['label_h20'] = np.random.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25])

    return df


@pytest.fixture
def high_corr_feature_df():
    """Create a DataFrame with highly correlated features."""
    np.random.seed(42)
    n = 500

    base_feature = np.random.randn(n)

    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'symbol': 'MES',
        'open': 100 + np.random.randn(n) * 0.5,
        'high': 101 + np.random.randn(n) * 0.5,
        'low': 99 + np.random.randn(n) * 0.5,
        'close': 100 + np.random.randn(n) * 0.5,
        'volume': np.random.randint(100, 1000, n),
        # Highly correlated features
        'feature_a': base_feature,
        'feature_b': base_feature * 1.001 + np.random.randn(n) * 0.001,  # ~0.99 corr
        'feature_c': base_feature * 0.999 + np.random.randn(n) * 0.001,  # ~0.99 corr
        # Uncorrelated feature
        'feature_d': np.random.randn(n),
    })

    return df


# =============================================================================
# GET FEATURE COLUMNS TESTS
# =============================================================================

class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""

    def test_excludes_ohlcv_columns(self, feature_df):
        """Test that OHLCV columns are excluded."""
        feature_cols = get_feature_columns(feature_df)

        excluded = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in excluded:
            assert col not in feature_cols, f"{col} should be excluded"

    def test_includes_feature_columns(self, feature_df):
        """Test that feature columns are included."""
        feature_cols = get_feature_columns(feature_df)

        expected_features = ['rsi_14', 'macd_line', 'return_1', 'return_5',
                             'sma_20_ratio', 'bb_width']
        for col in expected_features:
            assert col in feature_cols, f"{col} should be included"

    def test_excludes_label_columns(self, labeled_feature_df):
        """Test that label columns are excluded."""
        feature_cols = get_feature_columns(labeled_feature_df)

        assert 'label_h5' not in feature_cols
        assert 'label_h20' not in feature_cols

    def test_excludes_meta_columns(self):
        """Test that meta columns (bars_to_hit, mae, quality) are excluded."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'close': [100] * 10,
            'feature_1': [1] * 10,
            'label_h5': [0] * 10,
            'bars_to_hit_h5': [5] * 10,
            'mae_h5': [0.01] * 10,
            'quality_h5': [0.8] * 10,
            'sample_weight_h5': [1.0] * 10,
        })

        feature_cols = get_feature_columns(df)

        assert 'feature_1' in feature_cols
        assert 'bars_to_hit_h5' not in feature_cols
        assert 'mae_h5' not in feature_cols
        assert 'quality_h5' not in feature_cols
        assert 'sample_weight_h5' not in feature_cols

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        feature_cols = get_feature_columns(df)
        assert feature_cols == []

    def test_only_ohlcv_columns(self):
        """Test DataFrame with only OHLCV columns."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10,
        })

        feature_cols = get_feature_columns(df)
        assert feature_cols == []


# =============================================================================
# FEATURE CORRELATION TESTS
# =============================================================================

class TestCheckFeatureCorrelations:
    """Tests for check_feature_correlations function."""

    def test_detects_high_correlations(self, high_corr_feature_df):
        """Test detection of highly correlated feature pairs."""
        feature_cols = get_feature_columns(high_corr_feature_df)
        feature_df = high_corr_feature_df[feature_cols]
        warnings_found = []

        high_corr_pairs = check_feature_correlations(
            feature_df, feature_cols, warnings_found, threshold=0.85
        )

        # Should find high correlations between feature_a, feature_b, feature_c
        assert len(high_corr_pairs) > 0
        assert len(warnings_found) > 0

        # Check structure of returned pairs
        pair = high_corr_pairs[0]
        assert 'feature1' in pair
        assert 'feature2' in pair
        assert 'correlation' in pair
        assert pair['correlation'] > 0.85

    def test_no_high_correlations_when_uncorrelated(self, feature_df):
        """Test no high correlations with uncorrelated features."""
        feature_cols = get_feature_columns(feature_df)
        feature_df_subset = feature_df[feature_cols]
        warnings_found = []

        high_corr_pairs = check_feature_correlations(
            feature_df_subset, feature_cols, warnings_found, threshold=0.99
        )

        # Random features should have few/no very high correlations
        assert len(high_corr_pairs) == 0

    def test_custom_threshold(self, high_corr_feature_df):
        """Test with custom correlation threshold."""
        feature_cols = get_feature_columns(high_corr_feature_df)
        feature_df = high_corr_feature_df[feature_cols]

        # Very low threshold should catch more pairs
        warnings_found_low = []
        pairs_low = check_feature_correlations(
            feature_df, feature_cols, warnings_found_low, threshold=0.5
        )

        # Very high threshold should catch fewer pairs
        warnings_found_high = []
        pairs_high = check_feature_correlations(
            feature_df, feature_cols, warnings_found_high, threshold=0.999
        )

        assert len(pairs_low) >= len(pairs_high)

    def test_single_feature(self):
        """Test with single feature - no correlation possible."""
        df = pd.DataFrame({
            'feature_1': np.random.randn(100)
        })

        warnings_found = []
        pairs = check_feature_correlations(
            df, ['feature_1'], warnings_found, threshold=0.85
        )

        assert len(pairs) == 0


# =============================================================================
# FEATURE IMPORTANCE TESTS
# =============================================================================

class TestComputeFeatureImportance:
    """Tests for compute_feature_importance function."""

    def test_computes_importance_successfully(self, labeled_feature_df):
        """Test successful feature importance computation."""
        feature_cols = get_feature_columns(labeled_feature_df)
        feature_df = labeled_feature_df[feature_cols].fillna(0)

        top_features, success = compute_feature_importance(
            labeled_feature_df,
            feature_df,
            feature_cols,
            label_col='label_h5',
            seed=42,
            sample_size=200
        )

        assert success is True
        assert len(top_features) > 0

        # Check structure
        feat = top_features[0]
        assert 'feature' in feat
        assert 'importance' in feat
        assert feat['importance'] >= 0

    def test_handles_missing_label_column(self, feature_df):
        """Test handling of missing label column."""
        feature_cols = get_feature_columns(feature_df)
        feature_df_subset = feature_df[feature_cols].fillna(0)

        top_features, success = compute_feature_importance(
            feature_df,
            feature_df_subset,
            feature_cols,
            label_col='nonexistent_label',
            seed=42
        )

        assert success is False
        assert top_features == []

    def test_respects_sample_size(self, labeled_feature_df):
        """Test that sample_size parameter is respected."""
        feature_cols = get_feature_columns(labeled_feature_df)
        feature_df = labeled_feature_df[feature_cols].fillna(0)

        # Should work with small sample
        top_features, success = compute_feature_importance(
            labeled_feature_df,
            feature_df,
            feature_cols,
            label_col='label_h5',
            seed=42,
            sample_size=200
        )

        assert success is True

    def test_handles_insufficient_samples(self):
        """Test handling of insufficient samples."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_1': np.random.randn(50),  # Too few samples
            'label_h5': np.random.choice([0, 1], size=50)
        })
        feature_df = df[['feature_1']]

        top_features, success = compute_feature_importance(
            df, feature_df, ['feature_1'], 'label_h5', seed=42
        )

        # Should fail gracefully with insufficient samples
        assert success is False

    def test_deterministic_with_seed(self, labeled_feature_df):
        """Test that results are deterministic with same seed."""
        feature_cols = get_feature_columns(labeled_feature_df)
        feature_df = labeled_feature_df[feature_cols].fillna(0)

        top_1, _ = compute_feature_importance(
            labeled_feature_df, feature_df, feature_cols,
            label_col='label_h5', seed=42, sample_size=200
        )

        top_2, _ = compute_feature_importance(
            labeled_feature_df, feature_df, feature_cols,
            label_col='label_h5', seed=42, sample_size=200
        )

        # Results should be identical with same seed
        assert len(top_1) == len(top_2)
        for f1, f2 in zip(top_1, top_2):
            assert f1['feature'] == f2['feature']


# =============================================================================
# STATIONARITY TESTS
# =============================================================================

class TestRunStationarityTests:
    """Tests for run_stationarity_tests function."""

    def test_runs_adf_on_return_features(self):
        """Test ADF tests run on return-type features."""
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            'return_1': np.random.randn(n) * 0.01,  # Stationary
            'return_5': np.random.randn(n) * 0.02,
            'rsi_14': 50 + np.random.randn(n) * 10,
            'price': 100 + np.cumsum(np.random.randn(n) * 0.1),  # Non-stationary
        })

        feature_cols = ['return_1', 'return_5', 'rsi_14', 'price']
        results = run_stationarity_tests(df, feature_cols)

        # Should test return and rsi features
        assert len(results) > 0

        # Check structure
        for result in results:
            assert 'feature' in result
            assert 'adf_statistic' in result
            assert 'p_value' in result
            assert 'is_stationary' in result

    def test_identifies_stationary_features(self):
        """Test correct identification of stationary features."""
        np.random.seed(42)
        n = 200

        # White noise is stationary
        df = pd.DataFrame({
            'return_1': np.random.randn(n) * 0.01,
        })

        results = run_stationarity_tests(df, ['return_1'])

        if len(results) > 0:
            # Random noise should typically be stationary
            result = results[0]
            assert result['p_value'] < 0.05
            assert result['is_stationary'] is True

    def test_handles_empty_feature_list(self):
        """Test handling of empty feature list."""
        df = pd.DataFrame({'price': [100] * 100})
        results = run_stationarity_tests(df, [])

        assert results == []

    def test_handles_short_series(self):
        """Test handling of series too short for ADF."""
        df = pd.DataFrame({
            'return_1': [0.01] * 10  # Too short
        })

        results = run_stationarity_tests(df, ['return_1'])
        # Should skip or handle gracefully
        assert isinstance(results, list)


# =============================================================================
# CHECK FEATURE QUALITY TESTS
# =============================================================================

class TestCheckFeatureQuality:
    """Tests for check_feature_quality function."""

    def test_returns_comprehensive_results(self, labeled_feature_df):
        """Test that all result keys are present."""
        warnings_found = []

        results = check_feature_quality(
            labeled_feature_df,
            horizons=[5, 20],
            warnings_found=warnings_found,
            seed=42,
            max_features=20
        )

        assert 'total_features' in results
        assert 'high_correlations' in results
        assert 'top_features' in results
        assert 'feature_importance_computed' in results
        assert 'stationarity_tests' in results

    def test_respects_max_features_limit(self, feature_df):
        """Test that max_features limits analysis."""
        # Add many features
        df = feature_df.copy()
        for i in range(100):
            df[f'feature_{i}'] = np.random.randn(len(df))

        warnings_found = []
        results = check_feature_quality(
            df, horizons=[5], warnings_found=warnings_found,
            seed=42, max_features=10
        )

        # Should analyze limited number of features
        assert results['total_features'] > 10  # Total is more
        # But correlation/importance analysis limited

    def test_populates_warnings_for_high_correlations(self, high_corr_feature_df):
        """Test that warnings are populated for high correlations."""
        # Add labels for importance calculation
        df = high_corr_feature_df.copy()
        df['label_h5'] = np.random.choice([0, 1, -1], size=len(df))

        warnings_found = []
        results = check_feature_quality(
            df, horizons=[5], warnings_found=warnings_found, seed=42
        )

        # Should have warnings about correlated features
        assert len(warnings_found) > 0
        corr_warnings = [w for w in warnings_found if 'correlated' in w.lower()]
        assert len(corr_warnings) > 0

    def test_handles_no_labels(self, feature_df):
        """Test handling when label column is missing."""
        warnings_found = []

        results = check_feature_quality(
            feature_df, horizons=[5], warnings_found=warnings_found, seed=42
        )

        # Should still return results, just skip importance
        assert 'top_features' in results
        assert results['feature_importance_computed'] is False


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestFeatureValidatorEdgeCases:
    """Edge case tests for feature validator."""

    def test_handles_nan_in_features(self):
        """Test handling of NaN values in features."""
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': 100 + np.random.randn(n),
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'label_h5': np.random.choice([0, 1], size=n),
        })

        # Inject NaN
        df.loc[0:10, 'feature_1'] = np.nan

        warnings_found = []
        # Should not raise
        results = check_feature_quality(
            df, horizons=[5], warnings_found=warnings_found, seed=42
        )

        assert results is not None

    def test_handles_constant_features(self):
        """Test handling of constant features."""
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': 100 + np.random.randn(n),
            'constant_feature': [1.0] * n,  # Constant
            'normal_feature': np.random.randn(n),
            'label_h5': np.random.choice([0, 1], size=n),
        })

        warnings_found = []
        results = check_feature_quality(
            df, horizons=[5], warnings_found=warnings_found, seed=42
        )

        # Should handle constant feature gracefully
        assert results is not None

    def test_handles_inf_in_features(self):
        """Test handling of infinite values in features."""
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': 100 + np.random.randn(n),
            'feature_1': np.random.randn(n),
            'label_h5': np.random.choice([0, 1], size=n),
        })

        # Inject inf
        df.loc[0, 'feature_1'] = np.inf
        df.loc[1, 'feature_1'] = -np.inf

        warnings_found = []
        results = check_feature_quality(
            df, horizons=[5], warnings_found=warnings_found, seed=42
        )

        assert results is not None
