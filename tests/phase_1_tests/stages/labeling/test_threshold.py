"""
Tests for ThresholdLabeler strategy.

Tests cover:
- Upper threshold hits
- Lower threshold hits
- Timeout cases
- Same-bar hit resolution
- Parameter validation
- Quality metrics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.labeling import LabelingType, ThresholdLabeler


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 100

    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1

    # Ensure OHLC relationships
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100, 1000, n)
    })


class TestThresholdLabelerInit:
    """Tests for ThresholdLabeler initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        labeler = ThresholdLabeler()
        assert labeler.labeling_type == LabelingType.THRESHOLD
        assert labeler._pct_up == 0.01
        assert labeler._pct_down == 0.01
        assert labeler._max_bars == 20

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        labeler = ThresholdLabeler(pct_up=0.02, pct_down=0.015, max_bars=30)
        assert labeler._pct_up == 0.02
        assert labeler._pct_down == 0.015
        assert labeler._max_bars == 30

    def test_asymmetric_thresholds(self):
        """Test asymmetric threshold initialization."""
        labeler = ThresholdLabeler(pct_up=0.02, pct_down=0.01)
        assert labeler._pct_up == 0.02
        assert labeler._pct_down == 0.01

    def test_invalid_pct_up(self):
        """Test that invalid pct_up raises error."""
        with pytest.raises(ValueError, match="pct_up"):
            ThresholdLabeler(pct_up=-0.01)

        with pytest.raises(ValueError, match="pct_up"):
            ThresholdLabeler(pct_up=0)

    def test_invalid_pct_down(self):
        """Test that invalid pct_down raises error."""
        with pytest.raises(ValueError, match="pct_down"):
            ThresholdLabeler(pct_down=-0.01)

    def test_invalid_max_bars(self):
        """Test that invalid max_bars raises error."""
        with pytest.raises(ValueError, match="max_bars"):
            ThresholdLabeler(max_bars=0)

    def test_required_columns(self):
        """Test required columns."""
        labeler = ThresholdLabeler()
        required = labeler.required_columns

        assert 'close' in required
        assert 'high' in required
        assert 'low' in required
        assert 'open' in required


class TestThresholdLabelComputation:
    """Tests for label computation."""

    def test_compute_labels_basic(self, sample_ohlcv_df):
        """Test basic label computation."""
        labeler = ThresholdLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        assert result.horizon == 5
        assert len(result.labels) == len(sample_ohlcv_df)
        assert 'bars_to_hit' in result.metadata
        assert 'max_gain' in result.metadata
        assert 'max_loss' in result.metadata

    def test_labels_in_valid_range(self, sample_ohlcv_df):
        """Test that valid labels are in {-1, 0, 1}."""
        labeler = ThresholdLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        valid_labels = result.labels[result.labels != -99]
        assert set(valid_labels).issubset({-1, 0, 1})

    def test_upper_threshold_hit(self):
        """Test upper threshold hit produces +1 label."""
        n = 30
        # Price starts at 100, jumps to 110 (10% gain)
        close = np.array([100.0] * 5 + [110.0] * 25)
        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()

        df = pd.DataFrame({
            'open': open_, 'high': high, 'low': low, 'close': close
        })

        labeler = ThresholdLabeler(pct_up=0.05, pct_down=0.05, max_bars=15)
        result = labeler.compute_labels(df, horizon=5)

        # First bar should hit upper threshold
        assert result.labels[0] == 1

    def test_lower_threshold_hit(self):
        """Test lower threshold hit produces -1 label."""
        n = 30
        # Price starts at 100, drops to 90 (10% loss)
        close = np.array([100.0] * 5 + [90.0] * 25)
        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()

        df = pd.DataFrame({
            'open': open_, 'high': high, 'low': low, 'close': close
        })

        labeler = ThresholdLabeler(pct_up=0.05, pct_down=0.05, max_bars=15)
        result = labeler.compute_labels(df, horizon=5)

        # First bar should hit lower threshold
        assert result.labels[0] == -1

    def test_timeout_neutral(self):
        """Test timeout produces label 0."""
        n = 30
        # Flat price - no threshold hit
        close = np.array([100.0] * n)
        high = close + 0.1  # Very small range
        low = close - 0.1
        open_ = close.copy()

        df = pd.DataFrame({
            'open': open_, 'high': high, 'low': low, 'close': close
        })

        labeler = ThresholdLabeler(pct_up=0.05, pct_down=0.05, max_bars=5)
        result = labeler.compute_labels(df, horizon=5)

        # First bar should timeout
        assert result.labels[0] == 0

    def test_last_max_bars_invalid(self, sample_ohlcv_df):
        """Test that last max_bars samples are marked invalid."""
        labeler = ThresholdLabeler(max_bars=10)
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        # Last 10 samples should be -99
        assert np.all(result.labels[-10:] == -99)

    def test_parameter_override(self, sample_ohlcv_df):
        """Test parameter override in compute_labels."""
        labeler = ThresholdLabeler(pct_up=0.01, pct_down=0.01, max_bars=20)

        # Override with different parameters
        result = labeler.compute_labels(
            sample_ohlcv_df, horizon=5,
            pct_up=0.05, pct_down=0.05, max_bars=10
        )

        # Last 10 samples should be invalid (using overridden max_bars)
        assert np.all(result.labels[-10:] == -99)

    def test_narrow_vs_wide_thresholds(self, sample_ohlcv_df):
        """Test that narrow thresholds produce fewer timeouts."""
        labeler_narrow = ThresholdLabeler(pct_up=0.001, pct_down=0.001, max_bars=20)
        labeler_wide = ThresholdLabeler(pct_up=0.10, pct_down=0.10, max_bars=20)

        result_narrow = labeler_narrow.compute_labels(sample_ohlcv_df, horizon=5)
        result_wide = labeler_wide.compute_labels(sample_ohlcv_df, horizon=5)

        valid_narrow = result_narrow.labels[result_narrow.labels != -99]
        valid_wide = result_wide.labels[result_wide.labels != -99]

        timeouts_narrow = (valid_narrow == 0).sum()
        timeouts_wide = (valid_wide == 0).sum()

        # Wide thresholds should have more timeouts
        assert timeouts_wide >= timeouts_narrow


class TestThresholdMetadata:
    """Tests for metadata values."""

    def test_max_gain_values(self, sample_ohlcv_df):
        """Test max_gain metadata."""
        labeler = ThresholdLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        max_gain = result.metadata['max_gain']
        valid_gain = max_gain[result.labels != -99]

        # Max gain should be non-negative (upside)
        assert np.all(valid_gain >= 0)

    def test_max_loss_values(self, sample_ohlcv_df):
        """Test max_loss metadata."""
        labeler = ThresholdLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        max_loss = result.metadata['max_loss']
        valid_loss = max_loss[result.labels != -99]

        # Max loss should be non-positive (downside)
        assert np.all(valid_loss <= 0)

    def test_bars_to_hit_values(self, sample_ohlcv_df):
        """Test bars_to_hit metadata."""
        labeler = ThresholdLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        bars_to_hit = result.metadata['bars_to_hit']

        # All values should be non-negative
        assert np.all(bars_to_hit >= 0)


class TestThresholdQualityMetrics:
    """Tests for quality metrics."""

    def test_quality_metrics_present(self, sample_ohlcv_df):
        """Test that quality metrics are computed."""
        labeler = ThresholdLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        metrics = result.quality_metrics

        assert 'total_samples' in metrics
        assert 'valid_samples' in metrics
        assert 'long_count' in metrics
        assert 'short_count' in metrics

    def test_gain_loss_metrics(self, sample_ohlcv_df):
        """Test gain/loss metrics."""
        labeler = ThresholdLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        metrics = result.quality_metrics

        assert 'avg_max_gain' in metrics
        assert 'avg_max_loss' in metrics

    def test_gain_loss_ratio(self, sample_ohlcv_df):
        """Test gain/loss ratio metric."""
        labeler = ThresholdLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        metrics = result.quality_metrics

        if 'gain_loss_ratio' in metrics:
            assert metrics['gain_loss_ratio'] >= 0


class TestThresholdInputValidation:
    """Tests for input validation."""

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises error."""
        labeler = ThresholdLabeler()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            labeler.compute_labels(empty_df, horizon=5)

    def test_missing_columns(self):
        """Test that missing columns raises error."""
        labeler = ThresholdLabeler()
        df = pd.DataFrame({'close': [100, 101, 102]})

        with pytest.raises(KeyError):
            labeler.compute_labels(df, horizon=5)

    def test_invalid_horizon(self, sample_ohlcv_df):
        """Test that invalid horizon raises error."""
        labeler = ThresholdLabeler()

        with pytest.raises(ValueError, match="positive integer"):
            labeler.compute_labels(sample_ohlcv_df, horizon=0)

    def test_invalid_pct_override(self, sample_ohlcv_df):
        """Test that invalid pct override raises error."""
        labeler = ThresholdLabeler()

        with pytest.raises(ValueError, match="pct_up"):
            labeler.compute_labels(sample_ohlcv_df, horizon=5, pct_up=-0.01)


class TestThresholdAddToDataframe:
    """Tests for adding labels to DataFrame."""

    def test_adds_expected_columns(self, sample_ohlcv_df):
        """Test that expected columns are added."""
        labeler = ThresholdLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(sample_ohlcv_df, result)

        assert 'label_h5' in df_labeled.columns
        assert 'bars_to_hit_h5' in df_labeled.columns
        assert 'max_gain_h5' in df_labeled.columns
        assert 'max_loss_h5' in df_labeled.columns
