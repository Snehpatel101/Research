"""
Tests for RegressionLabeler strategy.

Tests cover:
- Basic regression target computation
- Log returns option
- Winsorization
- Scaling factor
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

from stages.labeling import LabelingType, RegressionLabeler


@pytest.fixture
def sample_df():
    """Create sample DataFrame with close prices."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)

    return pd.DataFrame({'close': close})


@pytest.fixture
def trending_up_df():
    """Create DataFrame with clear uptrend."""
    n = 50
    close = 100.0 + np.arange(n) * 1.0  # 1% per bar

    return pd.DataFrame({'close': close})


class TestRegressionLabelerInit:
    """Tests for RegressionLabeler initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        labeler = RegressionLabeler()
        assert labeler.labeling_type == LabelingType.REGRESSION
        assert labeler._use_log_returns is False
        assert labeler._winsorize_pct == 0.0
        assert labeler._scale_factor == 1.0

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        labeler = RegressionLabeler(
            use_log_returns=True,
            winsorize_pct=0.01,
            scale_factor=100.0
        )
        assert labeler._use_log_returns is True
        assert labeler._winsorize_pct == 0.01
        assert labeler._scale_factor == 100.0

    def test_invalid_winsorize_pct(self):
        """Test that invalid winsorize_pct raises error."""
        with pytest.raises(ValueError, match="winsorize_pct"):
            RegressionLabeler(winsorize_pct=-0.01)

        with pytest.raises(ValueError, match="winsorize_pct"):
            RegressionLabeler(winsorize_pct=0.5)

    def test_invalid_scale_factor(self):
        """Test that invalid scale_factor raises error."""
        with pytest.raises(ValueError, match="scale_factor"):
            RegressionLabeler(scale_factor=0)

        with pytest.raises(ValueError, match="scale_factor"):
            RegressionLabeler(scale_factor=-1.0)

    def test_required_columns(self):
        """Test required columns."""
        labeler = RegressionLabeler()
        assert labeler.required_columns == ['close']


class TestRegressionLabelComputation:
    """Tests for label computation."""

    def test_compute_labels_basic(self, sample_df):
        """Test basic label computation."""
        labeler = RegressionLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        assert result.horizon == 5
        assert len(result.labels) == len(sample_df)
        assert 'regression_target' in result.metadata
        assert 'raw_return' in result.metadata

    def test_regression_target_type(self, sample_df):
        """Test that regression target is float array."""
        labeler = RegressionLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        target = result.metadata['regression_target']
        assert target.dtype == np.float32

    def test_uptrend_positive_returns(self, trending_up_df):
        """Test that uptrend produces positive returns."""
        labeler = RegressionLabeler()
        result = labeler.compute_labels(trending_up_df, horizon=5)

        target = result.metadata['regression_target']
        valid_target = target[~np.isnan(target)]

        # Uptrend should have positive returns
        assert np.mean(valid_target) > 0

    def test_last_horizon_samples_nan(self, sample_df):
        """Test that last horizon samples are NaN."""
        labeler = RegressionLabeler()
        result = labeler.compute_labels(sample_df, horizon=10)

        target = result.metadata['regression_target']

        # Last 10 samples should be NaN
        assert np.all(np.isnan(target[-10:]))

    def test_log_returns(self, sample_df):
        """Test log returns computation."""
        labeler_simple = RegressionLabeler(use_log_returns=False)
        labeler_log = RegressionLabeler(use_log_returns=True)

        result_simple = labeler_simple.compute_labels(sample_df, horizon=5)
        result_log = labeler_log.compute_labels(sample_df, horizon=5)

        simple_returns = result_simple.metadata['regression_target']
        log_returns = result_log.metadata['regression_target']

        # Returns should be different
        valid_simple = simple_returns[~np.isnan(simple_returns)]
        valid_log = log_returns[~np.isnan(log_returns)]

        assert not np.allclose(valid_simple, valid_log)

    def test_scale_factor(self, sample_df):
        """Test scale factor application."""
        labeler_default = RegressionLabeler(scale_factor=1.0)
        labeler_scaled = RegressionLabeler(scale_factor=100.0)

        result_default = labeler_default.compute_labels(sample_df, horizon=5)
        result_scaled = labeler_scaled.compute_labels(sample_df, horizon=5)

        default_target = result_default.metadata['regression_target']
        scaled_target = result_scaled.metadata['regression_target']

        valid_default = default_target[~np.isnan(default_target)]
        valid_scaled = scaled_target[~np.isnan(scaled_target)]

        # Scaled should be 100x larger
        np.testing.assert_allclose(valid_scaled, valid_default * 100, rtol=1e-5)

    def test_winsorization(self):
        """Test winsorization of extreme returns."""
        # Create data with extreme outliers
        n = 100
        close = np.ones(n) * 100.0
        close[50] = 200.0  # 100% jump
        close[51:] = 100.0  # Back to normal

        df = pd.DataFrame({'close': close})

        labeler_no_winsor = RegressionLabeler(winsorize_pct=0.0)
        labeler_winsor = RegressionLabeler(winsorize_pct=0.05)

        result_no_winsor = labeler_no_winsor.compute_labels(df, horizon=1)
        result_winsor = labeler_winsor.compute_labels(df, horizon=1)

        no_winsor_target = result_no_winsor.metadata['regression_target']
        winsor_target = result_winsor.metadata['regression_target']

        # Winsorized should have smaller max
        valid_no_winsor = no_winsor_target[~np.isnan(no_winsor_target)]
        valid_winsor = winsor_target[~np.isnan(winsor_target)]

        assert np.max(np.abs(valid_winsor)) <= np.max(np.abs(valid_no_winsor))

    def test_horizon_validation(self, sample_df):
        """Test horizon validation."""
        labeler = RegressionLabeler()

        with pytest.raises(ValueError, match="positive integer"):
            labeler.compute_labels(sample_df, horizon=0)


class TestRegressionQualityMetrics:
    """Tests for quality metrics."""

    def test_quality_metrics_present(self, sample_df):
        """Test that quality metrics are computed."""
        labeler = RegressionLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        metrics = result.quality_metrics

        assert 'total_samples' in metrics
        assert 'valid_samples' in metrics
        assert 'mean' in metrics
        assert 'std' in metrics

    def test_percentile_metrics(self, sample_df):
        """Test percentile metrics."""
        labeler = RegressionLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        metrics = result.quality_metrics

        assert 'percentile_5' in metrics
        assert 'percentile_25' in metrics
        assert 'percentile_75' in metrics
        assert 'percentile_95' in metrics

    def test_skewness_kurtosis(self, sample_df):
        """Test skewness and kurtosis metrics."""
        labeler = RegressionLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        metrics = result.quality_metrics

        assert 'skewness' in metrics
        assert 'kurtosis' in metrics

    def test_positive_negative_split(self, sample_df):
        """Test positive/negative split metrics."""
        labeler = RegressionLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        metrics = result.quality_metrics

        assert 'positive_pct' in metrics
        assert 'negative_pct' in metrics

        # Should sum to approximately 100
        # (excluding neutral which is rare for continuous targets)
        total_pct = metrics['positive_pct'] + metrics['negative_pct']
        assert 95 <= total_pct <= 100


class TestRegressionInputValidation:
    """Tests for input validation."""

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises error."""
        labeler = RegressionLabeler()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            labeler.compute_labels(empty_df, horizon=5)

    def test_missing_close_column(self):
        """Test that missing close column raises error."""
        labeler = RegressionLabeler()
        df = pd.DataFrame({'volume': [100, 200, 300]})

        with pytest.raises(KeyError):
            labeler.compute_labels(df, horizon=5)


class TestRegressionAddToDataframe:
    """Tests for adding labels to DataFrame."""

    def test_adds_expected_columns(self, sample_df):
        """Test that expected columns are added."""
        labeler = RegressionLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(sample_df, result)

        # Note: RegressionLabeler uses 'target' prefix
        assert 'target_h5' in df_labeled.columns
        assert 'return_h5' in df_labeled.columns

    def test_preserves_original_columns(self, sample_df):
        """Test that original columns are preserved."""
        labeler = RegressionLabeler()
        result = labeler.compute_labels(sample_df, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(sample_df, result)

        assert 'close' in df_labeled.columns
