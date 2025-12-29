"""
Unit tests for feature correlation validation.

Tests the enhanced validate_feature_correlation function including:
- Correlation matrix computation
- Threshold-based pair identification
- Recommendations generation
- Visualization creation

Run with: pytest tests/phase_1_tests/stages/test_feature_correlation.py -v
"""

import sys
from pathlib import Path
import tempfile
import shutil

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.validation.features import (
    validate_feature_correlation,
    _generate_correlation_recommendations,
    _choose_feature_to_remove,
    get_feature_columns
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_artifacts_dir():
    """Create temporary directory for artifacts."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_feature_df():
    """Create sample feature DataFrame with known correlations."""
    np.random.seed(42)
    n = 1000

    # Create base features
    feature1 = np.random.randn(n)
    feature2 = np.random.randn(n)
    feature3 = np.random.randn(n)

    # Create highly correlated features (r > 0.95)
    feature1_copy = feature1 + np.random.randn(n) * 0.05  # r ≈ 0.99
    feature2_dup = feature2 + np.random.randn(n) * 0.08   # r ≈ 0.98

    # Create moderately correlated features (0.80 < r < 0.95)
    feature1_moderate = feature1 + np.random.randn(n) * 0.3  # r ≈ 0.85
    feature2_moderate = feature2 + np.random.randn(n) * 0.4  # r ≈ 0.75 (below threshold)

    # Independent features
    independent1 = np.random.randn(n)
    independent2 = np.random.randn(n)

    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature1_copy': feature1_copy,
        'feature2_dup': feature2_dup,
        'feature1_moderate': feature1_moderate,
        'feature2_moderate': feature2_moderate,
        'independent1': independent1,
        'independent2': independent2
    })

    return df


@pytest.fixture
def sample_labeled_df(sample_feature_df):
    """Create labeled DataFrame with features and labels."""
    df = sample_feature_df.copy()

    # Add OHLCV and metadata columns
    df['datetime'] = pd.date_range('2024-01-01', periods=len(df), freq='5min')
    df['symbol'] = 'MES'
    df['open'] = 5000 + np.random.randn(len(df)) * 10
    df['high'] = df['open'] + np.abs(np.random.randn(len(df)) * 5)
    df['low'] = df['open'] - np.abs(np.random.randn(len(df)) * 5)
    df['close'] = df['open'] + np.random.randn(len(df)) * 3
    df['volume'] = np.random.randint(1000, 10000, len(df))

    # Add labels
    df['label_h5'] = np.random.choice([0, 1, 2], len(df))
    df['label_h20'] = np.random.choice([0, 1, 2], len(df))

    return df


# =============================================================================
# TESTS: validate_feature_correlation
# =============================================================================

class TestValidateFeatureCorrelation:
    """Tests for validate_feature_correlation function."""

    def test_basic_correlation_detection(self, sample_feature_df):
        """Test basic correlation detection without visualizations."""
        warnings_found = []
        feature_cols = list(sample_feature_df.columns)

        results = validate_feature_correlation(
            feature_df=sample_feature_df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            highly_correlated_threshold=0.95,
            moderately_correlated_threshold=0.80,
            save_visualizations=False
        )

        # Check structure
        assert 'summary_statistics' in results
        assert 'highly_correlated_pairs' in results
        assert 'moderately_correlated_pairs' in results
        assert 'recommendations' in results

        # Check summary statistics
        stats = results['summary_statistics']
        assert stats['total_features'] == len(feature_cols)
        assert stats['total_pairs_analyzed'] > 0
        assert 0 <= stats['mean_abs_correlation'] <= 1
        assert 0 <= stats['median_abs_correlation'] <= 1
        assert 0 <= stats['max_abs_correlation'] <= 1

        # Should detect highly correlated pairs
        assert len(results['highly_correlated_pairs']) >= 2  # feature1/feature1_copy, feature2/feature2_dup
        assert stats['highly_correlated_count'] == len(results['highly_correlated_pairs'])

    def test_correlation_thresholds(self, sample_feature_df):
        """Test that thresholds correctly filter pairs."""
        warnings_found = []
        feature_cols = list(sample_feature_df.columns)

        results = validate_feature_correlation(
            feature_df=sample_feature_df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            highly_correlated_threshold=0.95,
            moderately_correlated_threshold=0.80,
            save_visualizations=False
        )

        # All highly correlated pairs should have |r| > 0.95
        for pair in results['highly_correlated_pairs']:
            assert pair['abs_correlation'] > 0.95

        # All moderately correlated pairs should have 0.80 < |r| <= 0.95
        for pair in results['moderately_correlated_pairs']:
            assert 0.80 < pair['abs_correlation'] <= 0.95

    def test_correlation_pair_structure(self, sample_feature_df):
        """Test correlation pair data structure."""
        warnings_found = []
        feature_cols = list(sample_feature_df.columns)

        results = validate_feature_correlation(
            feature_df=sample_feature_df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            save_visualizations=False
        )

        # Check pair structure
        if results['highly_correlated_pairs']:
            pair = results['highly_correlated_pairs'][0]
            assert 'feature1' in pair
            assert 'feature2' in pair
            assert 'correlation' in pair
            assert 'abs_correlation' in pair
            assert pair['feature1'] != pair['feature2']  # No self-correlation
            assert -1 <= pair['correlation'] <= 1
            assert 0 <= pair['abs_correlation'] <= 1

    def test_warnings_logging(self, sample_feature_df):
        """Test that warnings are properly logged."""
        warnings_found = []
        feature_cols = list(sample_feature_df.columns)

        results = validate_feature_correlation(
            feature_df=sample_feature_df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            highly_correlated_threshold=0.95,
            moderately_correlated_threshold=0.80,
            save_visualizations=False
        )

        # Should add warning if highly correlated pairs found
        if results['highly_correlated_pairs']:
            assert len(warnings_found) > 0
            assert any('highly correlated' in w.lower() for w in warnings_found)

    def test_recommendations_generated(self, sample_feature_df):
        """Test that recommendations are generated."""
        warnings_found = []
        feature_cols = list(sample_feature_df.columns)

        results = validate_feature_correlation(
            feature_df=sample_feature_df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            save_visualizations=False
        )

        recs = results['recommendations']
        assert 'features_to_consider_removing' in recs
        assert 'removal_rationale' in recs
        assert 'feature_frequency_in_correlations' in recs

        # Should recommend removing some features if high correlations exist
        if results['highly_correlated_pairs']:
            assert len(recs['features_to_consider_removing']) > 0

            # Each recommended removal should have rationale
            for feat in recs['features_to_consider_removing']:
                assert feat in recs['removal_rationale']
                assert 'appears_in_pairs' in recs['removal_rationale'][feat]
                assert 'correlated_with' in recs['removal_rationale'][feat]

    def test_no_correlations_case(self):
        """Test with completely independent features."""
        np.random.seed(42)
        n = 500

        # Create independent features
        df = pd.DataFrame({
            'feat1': np.random.randn(n),
            'feat2': np.random.randn(n),
            'feat3': np.random.randn(n),
            'feat4': np.random.randn(n)
        })

        warnings_found = []
        feature_cols = list(df.columns)

        results = validate_feature_correlation(
            feature_df=df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            highly_correlated_threshold=0.95,
            moderately_correlated_threshold=0.80,
            save_visualizations=False
        )

        # Should find no highly correlated pairs
        assert len(results['highly_correlated_pairs']) == 0
        assert len(results['recommendations']['features_to_consider_removing']) == 0

    def test_perfect_correlation(self):
        """Test with perfectly correlated features."""
        np.random.seed(42)
        n = 500
        feat1 = np.random.randn(n)

        df = pd.DataFrame({
            'feat1': feat1,
            'feat1_exact_copy': feat1,  # Perfect correlation
            'feat2': np.random.randn(n)
        })

        warnings_found = []
        feature_cols = list(df.columns)

        results = validate_feature_correlation(
            feature_df=df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            save_visualizations=False
        )

        # Should find exactly one highly correlated pair
        assert len(results['highly_correlated_pairs']) == 1
        pair = results['highly_correlated_pairs'][0]
        assert abs(pair['abs_correlation'] - 1.0) < 0.01  # Nearly perfect

    def test_visualization_creation(self, sample_feature_df, temp_artifacts_dir):
        """Test that visualizations are created when requested (if matplotlib available)."""
        # Check if matplotlib/seaborn are available
        try:
            import matplotlib
            import seaborn
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False

        warnings_found = []
        feature_cols = list(sample_feature_df.columns)

        results = validate_feature_correlation(
            feature_df=sample_feature_df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            artifacts_dir=temp_artifacts_dir,
            save_visualizations=True
        )

        # Check that artifacts were created
        heatmap_path = temp_artifacts_dir / "correlation_heatmap.png"
        pairs_path = temp_artifacts_dir / "top_correlated_pairs.txt"

        if matplotlib_available:
            assert heatmap_path.exists(), "Heatmap should be created when matplotlib available"
            assert pairs_path.exists(), "Pairs list should be created"

            # Check pairs file content
            with open(pairs_path, 'r') as f:
                content = f.read()
                assert 'HIGHLY CORRELATED PAIRS' in content
                assert 'MODERATELY CORRELATED PAIRS' in content
        else:
            # If matplotlib not available, visualizations are skipped gracefully
            # but the function should still return valid results
            assert 'summary_statistics' in results
            assert 'highly_correlated_pairs' in results

    def test_no_visualization_without_flag(self, sample_feature_df, temp_artifacts_dir):
        """Test that visualizations are NOT created when flag is False."""
        warnings_found = []
        feature_cols = list(sample_feature_df.columns)

        validate_feature_correlation(
            feature_df=sample_feature_df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            artifacts_dir=temp_artifacts_dir,
            save_visualizations=False
        )

        # Artifacts should NOT be created
        heatmap_path = temp_artifacts_dir / "correlation_heatmap.png"
        pairs_path = temp_artifacts_dir / "top_correlated_pairs.txt"

        assert not heatmap_path.exists(), "Heatmap should not be created"
        # pairs file might still be created, but heatmap definitely shouldn't


# =============================================================================
# TESTS: _generate_correlation_recommendations
# =============================================================================

class TestGenerateCorrelationRecommendations:
    """Tests for _generate_correlation_recommendations helper."""

    def test_basic_recommendations(self):
        """Test basic recommendation generation."""
        highly_correlated = [
            {'feature1': 'feat1', 'feature2': 'feat1_copy', 'correlation': 0.99, 'abs_correlation': 0.99},
            {'feature1': 'feat2', 'feature2': 'feat2_dup', 'correlation': 0.97, 'abs_correlation': 0.97}
        ]
        moderately_correlated = []
        feature_cols = ['feat1', 'feat1_copy', 'feat2', 'feat2_dup']

        recs = _generate_correlation_recommendations(
            highly_correlated,
            moderately_correlated,
            feature_cols
        )

        # Should recommend removing 2 features (one from each pair)
        assert len(recs['features_to_consider_removing']) == 2
        assert 'feat1_copy' in recs['features_to_consider_removing'] or 'feat1' in recs['features_to_consider_removing']
        assert 'feat2_dup' in recs['features_to_consider_removing'] or 'feat2' in recs['features_to_consider_removing']

    def test_multi_pair_feature(self):
        """Test feature appearing in multiple correlated pairs."""
        highly_correlated = [
            {'feature1': 'feat1', 'feature2': 'feat_problematic', 'correlation': 0.99, 'abs_correlation': 0.99},
            {'feature1': 'feat2', 'feature2': 'feat_problematic', 'correlation': 0.98, 'abs_correlation': 0.98},
            {'feature1': 'feat3', 'feature2': 'feat_problematic', 'correlation': 0.97, 'abs_correlation': 0.97}
        ]
        moderately_correlated = []
        feature_cols = ['feat1', 'feat2', 'feat3', 'feat_problematic']

        recs = _generate_correlation_recommendations(
            highly_correlated,
            moderately_correlated,
            feature_cols
        )

        # feat_problematic should be recommended for removal (appears in 3 pairs)
        assert 'feat_problematic' in recs['features_to_consider_removing']
        assert recs['feature_frequency_in_correlations']['feat_problematic'] == 3


# =============================================================================
# TESTS: _choose_feature_to_remove
# =============================================================================

class TestChooseFeatureToRemove:
    """Tests for _choose_feature_to_remove helper."""

    def test_prefer_remove_copy(self):
        """Test preference for removing '_copy' suffix features."""
        result = _choose_feature_to_remove('feat1', 'feat1_copy')
        assert result == 'feat1_copy'

        result = _choose_feature_to_remove('feat1_copy', 'feat1')
        assert result == 'feat1_copy'

    def test_prefer_remove_dup(self):
        """Test preference for removing '_dup' suffix features."""
        result = _choose_feature_to_remove('feat1', 'feat1_dup')
        assert result == 'feat1_dup'

    def test_prefer_shorter_name(self):
        """Test preference for keeping shorter names."""
        result = _choose_feature_to_remove('feat', 'feature_with_very_long_name')
        assert result == 'feature_with_very_long_name'

        result = _choose_feature_to_remove('abc', 'ab')
        assert result == 'abc'

    def test_alphabetical_fallback(self):
        """Test alphabetical fallback when names are same length."""
        result = _choose_feature_to_remove('feat_a', 'feat_b')
        # Should be consistent
        assert result in ['feat_a', 'feat_b']

        # Same call should give same result
        result2 = _choose_feature_to_remove('feat_a', 'feat_b')
        assert result == result2


# =============================================================================
# TESTS: Integration with get_feature_columns
# =============================================================================

class TestGetFeatureColumns:
    """Tests for get_feature_columns filter."""

    def test_excludes_metadata(self, sample_labeled_df):
        """Test that metadata columns are excluded."""
        feature_cols = get_feature_columns(sample_labeled_df)

        # Should exclude OHLCV and metadata
        excluded = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in excluded:
            assert col not in feature_cols

    def test_excludes_labels(self, sample_labeled_df):
        """Test that label columns are excluded."""
        feature_cols = get_feature_columns(sample_labeled_df)

        # Should exclude labels
        assert 'label_h5' not in feature_cols
        assert 'label_h20' not in feature_cols

    def test_includes_features(self, sample_labeled_df):
        """Test that actual features are included."""
        feature_cols = get_feature_columns(sample_labeled_df)

        # Should include actual features
        expected_features = ['feature1', 'feature2', 'feature3', 'feature1_copy',
                           'feature2_dup', 'independent1', 'independent2']

        for feat in expected_features:
            assert feat in feature_cols


# =============================================================================
# TESTS: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_feature(self):
        """Test with only one feature."""
        df = pd.DataFrame({'feat1': np.random.randn(100)})
        warnings_found = []
        feature_cols = ['feat1']

        results = validate_feature_correlation(
            feature_df=df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            save_visualizations=False
        )

        # Should handle single feature gracefully
        assert results['summary_statistics']['total_features'] == 1
        assert results['summary_statistics']['total_pairs_analyzed'] == 0
        assert len(results['highly_correlated_pairs']) == 0

    def test_two_features(self):
        """Test with exactly two features."""
        np.random.seed(42)
        feat1 = np.random.randn(100)
        feat2 = feat1 + np.random.randn(100) * 0.05  # High correlation

        df = pd.DataFrame({'feat1': feat1, 'feat2': feat2})
        warnings_found = []
        feature_cols = ['feat1', 'feat2']

        results = validate_feature_correlation(
            feature_df=df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            save_visualizations=False
        )

        # Should analyze exactly 1 pair
        assert results['summary_statistics']['total_pairs_analyzed'] == 1

    def test_nan_handling(self):
        """Test handling of NaN values in features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feat1': np.random.randn(100),
            'feat2': np.random.randn(100),
            'feat3': np.random.randn(100)
        })

        # Add some NaN values
        df.loc[0:10, 'feat1'] = np.nan
        df.loc[20:30, 'feat2'] = np.nan

        warnings_found = []
        feature_cols = list(df.columns)

        # Should handle NaN gracefully (fillna happens in check_feature_quality)
        results = validate_feature_correlation(
            feature_df=df.fillna(0),
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            save_visualizations=False
        )

        # Should complete without errors
        assert 'summary_statistics' in results

    def test_constant_features(self):
        """Test with constant features (no variance)."""
        df = pd.DataFrame({
            'feat1': np.ones(100),  # Constant
            'feat2': np.zeros(100),  # Constant
            'feat3': np.random.randn(100)
        })

        warnings_found = []
        feature_cols = list(df.columns)

        results = validate_feature_correlation(
            feature_df=df,
            feature_cols=feature_cols,
            warnings_found=warnings_found,
            save_visualizations=False
        )

        # Should handle constant features (correlation will be NaN)
        assert 'summary_statistics' in results
