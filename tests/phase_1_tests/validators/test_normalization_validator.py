"""
Unit tests for Feature Normalization Validation.

Tests the normalization validation module which checks:
- Feature distribution statistics
- Unnormalized features (large scale)
- Highly skewed features
- Z-score outlier detection
- Feature range issues
- Normalization recommendations

Run with: pytest tests/phase_1_tests/validators/test_normalization_validator.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.validators.normalization import (
    get_feature_columns,
    compute_feature_statistics,
    detect_outliers,
    analyze_feature_ranges,
    generate_recommendations,
    check_feature_normalization,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def normalized_df():
    """Create DataFrame with normalized features (mean~0, std~1)."""
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
        # Normalized features
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
    })

    return df


@pytest.fixture
def unnormalized_df():
    """Create DataFrame with unnormalized features."""
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
        # Unnormalized features
        'large_mean': 1000 + np.random.randn(n) * 50,  # Mean >> 0
        'large_std': np.random.randn(n) * 500,  # Std >> 1
        'price_feature': 4500 + np.random.randn(n) * 10,  # Price-like
    })

    return df


@pytest.fixture
def skewed_df():
    """Create DataFrame with highly skewed features."""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'close': 100 + np.random.randn(n) * 0.5,
        # Highly skewed features
        'right_skew': np.abs(np.random.randn(n) ** 3),  # Right skew
        'left_skew': -np.abs(np.random.randn(n) ** 3),  # Left skew
        # Normal feature
        'normal': np.random.randn(n),
    })

    return df


@pytest.fixture
def outlier_df():
    """Create DataFrame with outliers."""
    np.random.seed(42)
    n = 500

    # Create normal data with injected outliers
    feature_1 = np.random.randn(n)
    feature_1[0:10] = 20  # Extreme outliers (20 sigma)

    feature_2 = np.random.randn(n)
    feature_2[0:30] = 6  # 6 sigma outliers (1% of data > 5 sigma)

    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'close': 100 + np.random.randn(n) * 0.5,
        'extreme_outliers': feature_1,
        'moderate_outliers': feature_2,
        'clean': np.random.randn(n),
    })

    return df


@pytest.fixture
def range_issues_df():
    """Create DataFrame with range issues."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'close': 100 + np.random.randn(n) * 0.5,
        # Range issues
        'constant_feature': [5.0] * n,  # Zero variance
        'extreme_range': np.linspace(-1e7, 1e7, n),  # Range > 1M
        'normal_range': np.random.randn(n),
    })

    return df


# =============================================================================
# GET FEATURE COLUMNS TESTS
# =============================================================================

class TestGetFeatureColumnsNormalization:
    """Tests for get_feature_columns in normalization module."""

    def test_excludes_ohlcv(self, normalized_df):
        """Test OHLCV columns are excluded."""
        feature_cols = get_feature_columns(normalized_df)

        excluded = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in excluded:
            assert col not in feature_cols

    def test_includes_features(self, normalized_df):
        """Test feature columns are included."""
        feature_cols = get_feature_columns(normalized_df)

        assert 'feature_1' in feature_cols
        assert 'feature_2' in feature_cols
        assert 'feature_3' in feature_cols


# =============================================================================
# COMPUTE FEATURE STATISTICS TESTS
# =============================================================================

class TestComputeFeatureStatistics:
    """Tests for compute_feature_statistics function."""

    def test_returns_correct_structure(self, normalized_df):
        """Test that correct statistics are returned."""
        feature_cols = get_feature_columns(normalized_df)
        warnings_found = []

        stats, unnorm, skewed = compute_feature_statistics(
            normalized_df, feature_cols, warnings_found
        )

        assert len(stats) > 0

        # Check structure of stats
        stat = stats[0]
        assert 'feature' in stat
        assert 'mean' in stat
        assert 'std' in stat
        assert 'min' in stat
        assert 'max' in stat
        assert 'p1' in stat
        assert 'p99' in stat
        assert 'skewness' in stat
        assert 'kurtosis' in stat

    def test_identifies_unnormalized_features(self, unnormalized_df):
        """Test identification of unnormalized features."""
        feature_cols = get_feature_columns(unnormalized_df)
        warnings_found = []

        stats, unnorm, skewed = compute_feature_statistics(
            unnormalized_df, feature_cols, warnings_found
        )

        # Should identify features with large scale
        assert len(unnorm) > 0

        # Check structure
        for feat in unnorm:
            assert 'feature' in feat
            assert 'mean' in feat
            assert 'std' in feat
            assert 'issue' in feat
            assert feat['issue'] == 'large_scale'

    def test_identifies_high_skew_features(self, skewed_df):
        """Test identification of highly skewed features."""
        feature_cols = get_feature_columns(skewed_df)
        warnings_found = []

        stats, unnorm, skewed = compute_feature_statistics(
            skewed_df, feature_cols, warnings_found
        )

        # Should identify skewed features
        assert len(skewed) > 0

        skewed_names = [f['feature'] for f in skewed]
        # Right and left skew features should be identified
        assert 'right_skew' in skewed_names or 'left_skew' in skewed_names

    def test_populates_warnings(self, unnormalized_df):
        """Test that warnings are populated."""
        feature_cols = get_feature_columns(unnormalized_df)
        warnings_found = []

        compute_feature_statistics(unnormalized_df, feature_cols, warnings_found)

        # Should have warnings about normalization
        assert len(warnings_found) > 0

    def test_normalized_features_pass(self, normalized_df):
        """Test that normalized features don't generate warnings."""
        feature_cols = get_feature_columns(normalized_df)
        warnings_found = []

        stats, unnorm, skewed = compute_feature_statistics(
            normalized_df, feature_cols, warnings_found
        )

        # Should have few/no unnormalized features
        assert len(unnorm) == 0

    def test_handles_short_series(self):
        """Test handling of series with few samples."""
        df = pd.DataFrame({
            'feature_1': np.random.randn(50),  # Less than 100
        })

        warnings_found = []
        stats, unnorm, skewed = compute_feature_statistics(
            df, ['feature_1'], warnings_found
        )

        # Should skip short series
        assert len(stats) == 0


# =============================================================================
# DETECT OUTLIERS TESTS
# =============================================================================

class TestDetectOutliers:
    """Tests for detect_outliers function."""

    def test_detects_extreme_outliers(self, outlier_df):
        """Test detection of extreme outliers."""
        feature_cols = get_feature_columns(outlier_df)
        warnings_found = []

        summary, extreme = detect_outliers(
            outlier_df, feature_cols, warnings_found, z_threshold=3.0
        )

        # Should detect extreme outliers
        assert len(extreme) > 0

        extreme_names = [f['feature'] for f in extreme]
        assert 'extreme_outliers' in extreme_names or 'moderate_outliers' in extreme_names

    def test_outlier_statistics_structure(self, outlier_df):
        """Test structure of outlier statistics."""
        feature_cols = get_feature_columns(outlier_df)
        warnings_found = []

        summary, extreme = detect_outliers(
            outlier_df, feature_cols, warnings_found
        )

        # Check structure
        for info in summary:
            assert 'feature' in info
            assert 'outliers_3std' in info
            assert 'outliers_5std' in info
            assert 'outliers_10std' in info
            assert 'pct_beyond_3std' in info
            assert 'pct_beyond_5std' in info
            assert 'max_z_score' in info

    def test_clean_data_no_extreme_outliers(self, normalized_df):
        """Test that clean data has no extreme outliers."""
        feature_cols = get_feature_columns(normalized_df)
        warnings_found = []

        summary, extreme = detect_outliers(
            normalized_df, feature_cols, warnings_found
        )

        # Clean data should have no extreme outlier features
        assert len(extreme) == 0

    def test_custom_z_threshold(self, outlier_df):
        """Test with custom z-score threshold."""
        feature_cols = get_feature_columns(outlier_df)

        warnings_low = []
        summary_low, extreme_low = detect_outliers(
            outlier_df, feature_cols, warnings_low, z_threshold=2.0
        )

        warnings_high = []
        summary_high, extreme_high = detect_outliers(
            outlier_df, feature_cols, warnings_high, z_threshold=4.0
        )

        # Lower threshold should detect more
        total_outliers_low = sum(s['outliers_3std'] for s in summary_low)
        total_outliers_high = sum(s['outliers_3std'] for s in summary_high)
        # Both use 3std for counting, so should be equal
        assert total_outliers_low == total_outliers_high

    def test_handles_constant_feature(self):
        """Test handling of constant features (std=0)."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=200, freq='5min'),
            'constant': [5.0] * 200,
        })

        warnings_found = []
        summary, extreme = detect_outliers(
            df, ['constant'], warnings_found
        )

        # Should skip constant feature (std=0)
        assert len(summary) == 0


# =============================================================================
# ANALYZE FEATURE RANGES TESTS
# =============================================================================

class TestAnalyzeFeatureRanges:
    """Tests for analyze_feature_ranges function."""

    def test_detects_constant_features(self, range_issues_df):
        """Test detection of constant features."""
        feature_cols = get_feature_columns(range_issues_df)
        issues_found = []
        warnings_found = []

        result = analyze_feature_ranges(
            range_issues_df, feature_cols, issues_found, warnings_found
        )

        # Should detect constant feature
        constant_issues = [r for r in result if r['issue'] == 'constant_value']
        assert len(constant_issues) > 0

    def test_detects_extreme_range(self, range_issues_df):
        """Test detection of extreme range features."""
        feature_cols = get_feature_columns(range_issues_df)
        issues_found = []
        warnings_found = []

        result = analyze_feature_ranges(
            range_issues_df, feature_cols, issues_found, warnings_found
        )

        # Should detect extreme range
        extreme_issues = [r for r in result if r['issue'] == 'extreme_range']
        assert len(extreme_issues) > 0

        # Check structure
        extreme = extreme_issues[0]
        assert 'min' in extreme
        assert 'max' in extreme
        assert 'range' in extreme

    def test_populates_issues_for_constant(self, range_issues_df):
        """Test that issues are populated for constant features."""
        feature_cols = get_feature_columns(range_issues_df)
        issues_found = []
        warnings_found = []

        analyze_feature_ranges(
            range_issues_df, feature_cols, issues_found, warnings_found
        )

        # Should have issues about constant features
        constant_issues = [i for i in issues_found if 'constant' in i.lower()]
        assert len(constant_issues) > 0

    def test_binary_indicators_are_warnings_not_issues(self):
        """Test that binary indicator features are warnings, not issues."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=200, freq='5min'),
            'close': 100 + np.random.randn(200) * 0.5,
            'rsi_cross_up': [0] * 200,  # Binary indicator (constant)
            'regime_flag': [1] * 200,  # Binary indicator (constant)
        })

        feature_cols = get_feature_columns(df)
        issues_found = []
        warnings_found = []

        analyze_feature_ranges(df, feature_cols, issues_found, warnings_found)

        # Binary indicators should be warnings, not issues
        assert len(warnings_found) > 0
        # Regular constant features would be issues

    def test_normal_ranges_no_issues(self, normalized_df):
        """Test that normal ranges produce no issues."""
        feature_cols = get_feature_columns(normalized_df)
        issues_found = []
        warnings_found = []

        result = analyze_feature_ranges(
            normalized_df, feature_cols, issues_found, warnings_found
        )

        # Should have no range issues
        assert len(result) == 0


# =============================================================================
# GENERATE RECOMMENDATIONS TESTS
# =============================================================================

class TestGenerateRecommendations:
    """Tests for generate_recommendations function."""

    def test_recommends_standard_scaler(self):
        """Test StandardScaler recommendation for large std."""
        unnormalized = [
            {'feature': 'large_feature', 'std': 500.0, 'mean': 0.0}
        ]

        result = generate_recommendations(unnormalized, [], [])

        # Should recommend StandardScaler
        scaler_rec = [r for r in result if r['type'] == 'StandardScaler']
        assert len(scaler_rec) > 0
        assert 'large_feature' in scaler_rec[0]['features']

    def test_recommends_log_transform(self):
        """Test log transform recommendation for high skew."""
        high_skew = [
            {'feature': 'skewed_feature', 'skewness': 3.0, 'kurtosis': 10.0}
        ]

        result = generate_recommendations([], high_skew, [])

        # Should recommend log transform
        log_rec = [r for r in result if r['type'] == 'LogTransform']
        assert len(log_rec) > 0
        assert 'skewed_feature' in log_rec[0]['features']

    def test_recommends_robust_scaler(self):
        """Test RobustScaler recommendation for outliers."""
        extreme_outliers = [
            {'feature': 'outlier_feature', 'pct_beyond_5std': 2.0}
        ]

        result = generate_recommendations([], [], extreme_outliers)

        # Should recommend RobustScaler
        robust_rec = [r for r in result if r['type'] == 'RobustScaler']
        assert len(robust_rec) > 0
        assert 'outlier_feature' in robust_rec[0]['features']

    def test_no_recommendations_for_clean_data(self):
        """Test no recommendations for clean data."""
        result = generate_recommendations([], [], [])

        assert len(result) == 0

    def test_multiple_recommendations(self):
        """Test multiple recommendation types."""
        unnormalized = [{'feature': 'f1', 'std': 500.0, 'mean': 0.0}]
        high_skew = [{'feature': 'f2', 'skewness': 3.0, 'kurtosis': 10.0}]
        outliers = [{'feature': 'f3', 'pct_beyond_5std': 2.0}]

        result = generate_recommendations(unnormalized, high_skew, outliers)

        # Should have multiple recommendation types
        types = [r['type'] for r in result]
        assert 'StandardScaler' in types
        assert 'LogTransform' in types
        assert 'RobustScaler' in types


# =============================================================================
# CHECK FEATURE NORMALIZATION TESTS
# =============================================================================

class TestCheckFeatureNormalization:
    """Tests for check_feature_normalization function."""

    def test_returns_comprehensive_results(self, normalized_df):
        """Test that all result keys are present."""
        issues_found = []
        warnings_found = []

        result = check_feature_normalization(
            normalized_df, issues_found, warnings_found
        )

        assert 'total_features' in result
        assert 'feature_statistics' in result
        assert 'unnormalized_features' in result
        assert 'high_skew_features' in result
        assert 'outlier_analysis' in result
        assert 'extreme_outlier_features' in result
        assert 'range_issues' in result
        assert 'recommendations' in result
        assert 'summary' in result

    def test_summary_structure(self, normalized_df):
        """Test summary structure."""
        issues_found = []
        warnings_found = []

        result = check_feature_normalization(
            normalized_df, issues_found, warnings_found
        )

        summary = result['summary']
        assert 'features_analyzed' in summary
        assert 'unnormalized_count' in summary
        assert 'high_skew_count' in summary
        assert 'extreme_outlier_count' in summary
        assert 'constant_features' in summary
        assert 'needs_attention' in summary

    def test_clean_data_needs_no_attention(self, normalized_df):
        """Test that clean data needs no attention."""
        issues_found = []
        warnings_found = []

        result = check_feature_normalization(
            normalized_df, issues_found, warnings_found
        )

        # Clean data should not need attention
        assert result['summary']['needs_attention'] is False

    def test_problematic_data_needs_attention(self, unnormalized_df):
        """Test that problematic data needs attention."""
        issues_found = []
        warnings_found = []

        result = check_feature_normalization(
            unnormalized_df, issues_found, warnings_found
        )

        # Unnormalized data should need attention
        assert result['summary']['needs_attention'] is True

    def test_custom_thresholds(self, outlier_df):
        """Test with custom z-score thresholds."""
        issues_found = []
        warnings_found = []

        result = check_feature_normalization(
            outlier_df, issues_found, warnings_found,
            z_threshold=2.0,
            extreme_threshold=4.0
        )

        # Should still return results
        assert 'outlier_analysis' in result

    def test_populates_both_issues_and_warnings(self, range_issues_df):
        """Test that both issues and warnings are populated."""
        issues_found = []
        warnings_found = []

        check_feature_normalization(
            range_issues_df, issues_found, warnings_found
        )

        # Should have some issues or warnings
        total = len(issues_found) + len(warnings_found)
        assert total > 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestNormalizationValidatorEdgeCases:
    """Edge case tests for normalization validator."""

    def test_handles_nan_values(self):
        """Test handling of NaN values."""
        np.random.seed(42)
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=200, freq='5min'),
            'close': 100 + np.random.randn(200),
            'feature_with_nan': np.concatenate([
                np.random.randn(180),
                [np.nan] * 20
            ]),
        })

        issues_found = []
        warnings_found = []

        # Should not raise
        result = check_feature_normalization(
            df, issues_found, warnings_found
        )

        assert result is not None

    def test_handles_inf_values(self):
        """Test handling of infinite values."""
        np.random.seed(42)
        n = 200
        feature = np.random.randn(n)
        feature[0] = np.inf
        feature[1] = -np.inf

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': 100 + np.random.randn(n),
            'feature_with_inf': feature,
        })

        issues_found = []
        warnings_found = []

        # Should handle inf gracefully
        result = check_feature_normalization(
            df, issues_found, warnings_found
        )

        assert result is not None

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            'datetime': pd.Series([], dtype='datetime64[ns]'),
            'close': pd.Series([], dtype='float64'),
            'feature': pd.Series([], dtype='float64'),
        })

        issues_found = []
        warnings_found = []

        # Should not raise
        result = check_feature_normalization(
            df, issues_found, warnings_found
        )

        assert result is not None

    def test_only_ohlcv_columns(self):
        """Test DataFrame with only OHLCV columns."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': 100 + np.random.randn(100),
            'high': 101 + np.random.randn(100),
            'low': 99 + np.random.randn(100),
            'close': 100 + np.random.randn(100),
            'volume': np.random.randint(100, 1000, 100),
        })

        issues_found = []
        warnings_found = []

        result = check_feature_normalization(
            df, issues_found, warnings_found
        )

        # Should have 0 features analyzed
        assert result['total_features'] == 0

    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=200, freq='5min'),
            'close': 100 + np.random.randn(200),
            'single_feature': np.random.randn(200),
        })

        issues_found = []
        warnings_found = []

        result = check_feature_normalization(
            df, issues_found, warnings_found
        )

        assert result['total_features'] == 1

    def test_large_number_of_features(self):
        """Test with large number of features."""
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': 100 + np.random.randn(n),
        })

        # Add 100 features
        for i in range(100):
            df[f'feature_{i}'] = np.random.randn(n)

        issues_found = []
        warnings_found = []

        result = check_feature_normalization(
            df, issues_found, warnings_found
        )

        assert result['total_features'] == 100

    def test_deterministic_with_same_data(self, normalized_df):
        """Test that results are deterministic."""
        issues_1 = []
        warnings_1 = []
        result_1 = check_feature_normalization(
            normalized_df.copy(), issues_1, warnings_1
        )

        issues_2 = []
        warnings_2 = []
        result_2 = check_feature_normalization(
            normalized_df.copy(), issues_2, warnings_2
        )

        # Results should be identical
        assert result_1['summary'] == result_2['summary']
        assert len(result_1['feature_statistics']) == len(result_2['feature_statistics'])
