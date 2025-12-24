"""
Tests for DirectionalLabeler strategy.

Tests cover:
- Basic direction labeling
- Threshold-based filtering
- Log returns option
- Quality metrics
- Input validation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.labeling import DirectionalLabeler, LabelingType


@pytest.fixture
def sample_df():
    """Create sample DataFrame with close prices."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)

    return pd.DataFrame({
        'close': close,
        'volume': np.random.randint(100, 1000, n)
    })


@pytest.fixture
def trending_up_df():
    """Create DataFrame with clear uptrend."""
    n = 50
    # Clear upward trend
    close = 100.0 + np.arange(n) * 0.5

    return pd.DataFrame({'close': close})


@pytest.fixture
def trending_down_df():
    """Create DataFrame with clear downtrend."""
    n = 50
    # Clear downward trend
    close = 100.0 - np.arange(n) * 0.5

    return pd.DataFrame({'close': close})


class TestDirectionalLabelerInit:
    """Tests for DirectionalLabeler initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        labeler = DirectionalLabeler()
        assert labeler.labeling_type == LabelingType.DIRECTIONAL
        assert labeler._threshold == 0.0
        assert labeler._use_log_returns is False

    def test_custom_threshold(self):
        """Test initialization with custom threshold."""
        labeler = DirectionalLabeler(threshold=0.001)
        assert labeler._threshold == 0.001

    def test_log_returns_option(self):
        """Test initialization with log returns."""
        labeler = DirectionalLabeler(use_log_returns=True)
        assert labeler._use_log_returns is True

    def test_negative_threshold_raises(self):
        """Test that negative threshold raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            DirectionalLabeler(threshold=-0.001)

    def test_required_columns(self):
        """Test required columns."""
        labeler = DirectionalLabeler()
        assert labeler.required_columns == ['close']


class TestDirectionalLabelComputation:
    """Tests for label computation."""

    def test_compute_labels_basic(self, sample_df):
        """Test basic label computation."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        assert result.horizon == 5
        assert len(result.labels) == len(sample_df)
        assert 'forward_return' in result.metadata

    def test_uptrend_produces_positive_labels(self, trending_up_df):
        """Test that uptrend produces positive labels."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(trending_up_df, horizon=5)

        # Exclude last 5 (invalid) samples
        valid_labels = result.labels[:-5]

        # Most should be +1 in uptrend
        positive_ratio = (valid_labels == 1).sum() / len(valid_labels)
        assert positive_ratio > 0.8

    def test_downtrend_produces_negative_labels(self, trending_down_df):
        """Test that downtrend produces negative labels."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(trending_down_df, horizon=5)

        # Exclude last 5 (invalid) samples
        valid_labels = result.labels[:-5]

        # Most should be -1 in downtrend
        negative_ratio = (valid_labels == -1).sum() / len(valid_labels)
        assert negative_ratio > 0.8

    def test_threshold_creates_neutrals(self, sample_df):
        """Test that threshold creates neutral labels."""
        # With zero threshold, few neutrals
        labeler_no_threshold = DirectionalLabeler(threshold=0.0)
        result_no_threshold = labeler_no_threshold.compute_labels(sample_df, horizon=5)

        # With high threshold, more neutrals
        labeler_threshold = DirectionalLabeler(threshold=0.05)  # 5% threshold
        result_threshold = labeler_threshold.compute_labels(sample_df, horizon=5)

        valid_no_threshold = result_no_threshold.labels[:-5]
        valid_threshold = result_threshold.labels[:-5]

        neutrals_no_threshold = (valid_no_threshold == 0).sum()
        neutrals_threshold = (valid_threshold == 0).sum()

        # Higher threshold should have more neutrals
        assert neutrals_threshold >= neutrals_no_threshold

    def test_log_returns(self, sample_df):
        """Test log returns computation."""
        labeler = DirectionalLabeler(use_log_returns=True)
        result = labeler.compute_labels(sample_df, horizon=5)

        # Forward returns should be present
        returns = result.metadata['forward_return']
        valid_returns = returns[:-5]

        # Log returns should be reasonable values
        assert np.all(np.abs(valid_returns) < 1.0)  # Log returns bounded

    def test_last_horizon_samples_invalid(self, sample_df):
        """Test that last horizon samples are marked invalid."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(sample_df, horizon=10)

        # Last 10 samples should be -99
        assert np.all(result.labels[-10:] == -99)

    def test_horizon_validation(self, sample_df):
        """Test horizon validation."""
        labeler = DirectionalLabeler()

        with pytest.raises(ValueError, match="positive integer"):
            labeler.compute_labels(sample_df, horizon=0)

    def test_threshold_override(self, sample_df):
        """Test threshold override in compute_labels."""
        labeler = DirectionalLabeler(threshold=0.0)

        # Override with high threshold
        result = labeler.compute_labels(sample_df, horizon=5, threshold=0.1)

        valid_labels = result.labels[:-5]
        neutrals = (valid_labels == 0).sum()

        # High threshold should create neutrals
        assert neutrals > 0


class TestDirectionalReturnMetadata:
    """Tests for forward return metadata."""

    def test_forward_return_values(self, trending_up_df):
        """Test forward return values."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(trending_up_df, horizon=5)

        returns = result.metadata['forward_return']
        valid_returns = returns[:-5]

        # In uptrend, forward returns should be positive
        assert np.mean(valid_returns) > 0

    def test_forward_return_length(self, sample_df):
        """Test forward return length matches labels."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        assert len(result.metadata['forward_return']) == len(result.labels)


class TestDirectionalQualityMetrics:
    """Tests for quality metrics."""

    def test_quality_metrics_present(self, sample_df):
        """Test that quality metrics are computed."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        metrics = result.quality_metrics

        assert 'total_samples' in metrics
        assert 'valid_samples' in metrics
        assert 'long_count' in metrics
        assert 'short_count' in metrics

    def test_return_statistics(self, sample_df):
        """Test return statistics in metrics."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        metrics = result.quality_metrics

        assert 'avg_return' in metrics
        assert 'std_return' in metrics

    def test_return_by_label(self, trending_up_df):
        """Test return statistics by label."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(trending_up_df, horizon=5)

        metrics = result.quality_metrics

        # In uptrend, up labels should have positive return
        if 'avg_return_up' in metrics:
            assert metrics['avg_return_up'] > 0


class TestDirectionalInputValidation:
    """Tests for input validation."""

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises error."""
        labeler = DirectionalLabeler()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            labeler.compute_labels(empty_df, horizon=5)

    def test_missing_close_column(self):
        """Test that missing close column raises error."""
        labeler = DirectionalLabeler()
        df = pd.DataFrame({'volume': [100, 200, 300]})

        with pytest.raises(KeyError):
            labeler.compute_labels(df, horizon=5)

    def test_negative_threshold_in_compute(self, sample_df):
        """Test that negative threshold in compute raises error."""
        labeler = DirectionalLabeler()

        with pytest.raises(ValueError, match="non-negative"):
            labeler.compute_labels(sample_df, horizon=5, threshold=-0.01)


class TestDirectionalAddToDataframe:
    """Tests for adding labels to DataFrame."""

    def test_adds_expected_columns(self, sample_df):
        """Test that expected columns are added."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(sample_df, result)

        assert 'label_h5' in df_labeled.columns
        assert 'forward_return_h5' in df_labeled.columns

    def test_preserves_original_columns(self, sample_df):
        """Test that original columns are preserved."""
        labeler = DirectionalLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(sample_df, result)

        assert 'close' in df_labeled.columns
        assert 'volume' in df_labeled.columns
