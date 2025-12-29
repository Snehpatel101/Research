"""
Tests for TripleBarrierLabeler strategy.

Tests cover:
- Upper barrier hits
- Lower barrier hits
- Timeout cases
- Same-bar hit resolution
- ATR-based barrier calculation
- Invalid sample marking
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

from src.phase1.stages.labeling import LabelingType, TripleBarrierLabeler, triple_barrier_numba


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame with ATR."""
    np.random.seed(42)
    n = 100

    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1

    # Ensure OHLC relationships
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100, 1000, n),
        'atr_14': np.ones(n) * 2.0
    })

    return df


class TestTripleBarrierLabelerInit:
    """Tests for TripleBarrierLabeler initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        labeler = TripleBarrierLabeler()
        assert labeler.labeling_type == LabelingType.TRIPLE_BARRIER
        assert 'close' in labeler.required_columns
        assert 'atr_14' in labeler.required_columns

    def test_custom_atr_column(self):
        """Test initialization with custom ATR column."""
        labeler = TripleBarrierLabeler(atr_column='atr_20')
        assert 'atr_20' in labeler.required_columns

    def test_with_preset_parameters(self):
        """Test initialization with preset k values."""
        labeler = TripleBarrierLabeler(k_up=1.5, k_down=1.0, max_bars=15)
        assert labeler._k_up == 1.5
        assert labeler._k_down == 1.0
        assert labeler._max_bars == 15


class TestTripleBarrierLabelComputation:
    """Tests for label computation."""

    def test_compute_labels_basic(self, sample_ohlcv_df):
        """Test basic label computation."""
        labeler = TripleBarrierLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        assert result.horizon == 5
        assert len(result.labels) == len(sample_ohlcv_df)
        assert 'bars_to_hit' in result.metadata
        assert 'mae' in result.metadata
        assert 'mfe' in result.metadata

    def test_labels_in_valid_range(self, sample_ohlcv_df):
        """Test that valid labels are in {-1, 0, 1}."""
        labeler = TripleBarrierLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        valid_labels = result.labels[result.labels != -99]
        assert set(valid_labels).issubset({-1, 0, 1})

    def test_invalid_labels_at_end(self, sample_ohlcv_df):
        """Test that last max_bars samples are marked invalid."""
        labeler = TripleBarrierLabeler()
        result = labeler.compute_labels(
            sample_ohlcv_df, horizon=5, max_bars=10
        )

        # Last 10 samples should be -99
        assert np.all(result.labels[-10:] == -99)

    def test_custom_k_values(self, sample_ohlcv_df):
        """Test label computation with custom k values."""
        labeler = TripleBarrierLabeler()

        # Wide barriers = more timeouts
        result_wide = labeler.compute_labels(
            sample_ohlcv_df, horizon=5, k_up=5.0, k_down=5.0, max_bars=10
        )

        # Narrow barriers = fewer timeouts
        result_narrow = labeler.compute_labels(
            sample_ohlcv_df, horizon=5, k_up=0.5, k_down=0.5, max_bars=10
        )

        wide_timeouts = (result_wide.labels == 0).sum()
        narrow_timeouts = (result_narrow.labels == 0).sum()

        # Wide barriers should have more timeouts
        assert wide_timeouts > narrow_timeouts

    def test_horizon_validation(self, sample_ohlcv_df):
        """Test that invalid horizon raises error."""
        labeler = TripleBarrierLabeler()

        with pytest.raises(ValueError, match="positive integer"):
            labeler.compute_labels(sample_ohlcv_df, horizon=0)

        with pytest.raises(ValueError, match="positive integer"):
            labeler.compute_labels(sample_ohlcv_df, horizon=-5)


class TestTripleBarrierNumba:
    """Direct tests for numba-optimized function."""

    def test_upper_barrier_hit(self):
        """Test that upper barrier hit produces label +1."""
        n = 30
        close = np.array([100.0] * n)
        close[5:] = 110.0  # Price jumps up

        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0

        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up=2.0, k_down=2.0, max_bars=10
        )

        # First bar should hit upper barrier
        assert labels[0] == 1
        assert touch_type[0] == 1
        assert bars_to_hit[0] > 0

    def test_lower_barrier_hit(self):
        """Test that lower barrier hit produces label -1."""
        n = 30
        close = np.array([100.0] * n)
        close[5:] = 90.0  # Price drops

        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0

        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up=2.0, k_down=2.0, max_bars=10
        )

        assert labels[0] == -1
        assert touch_type[0] == -1

    def test_timeout_neutral(self):
        """Test that timeout produces label 0."""
        n = 30
        close = np.array([100.0] * n)  # Flat price
        high = close + 0.5
        low = close - 0.5
        open_ = close.copy()
        atr = np.ones(n) * 10.0  # Large ATR = wide barriers

        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up=2.0, k_down=2.0, max_bars=5
        )

        # First bar should timeout (barriers too far)
        assert labels[0] == 0
        assert touch_type[0] == 0
        assert bars_to_hit[0] == 5  # max_bars

    def test_same_bar_hit_resolution(self):
        """Test resolution when both barriers hit on same bar."""
        n = 30
        close = np.full(n, 100.0)
        high = close.copy()
        low = close.copy()
        open_ = close.copy()

        # Bar 1 crosses both barriers
        high[1] = 108.0   # Hits upper (100 + 2*2 = 104)
        low[1] = 92.0     # Hits lower (100 - 2*2 = 96)
        open_[1] = 102.0  # Open closer to upper

        atr = np.ones(n) * 2.0

        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up=2.0, k_down=2.0, max_bars=15
        )

        # Open closer to upper barrier, so upper hit first
        assert labels[0] == 1
        assert touch_type[0] == 1

    def test_invalid_atr_handling(self):
        """Test that invalid ATR values are handled."""
        n = 30
        close = np.full(n, 100.0)
        high = close + 5.0
        low = close - 5.0
        open_ = close.copy()

        atr = np.ones(n) * 2.0
        atr[0] = np.nan   # Invalid ATR
        atr[3] = 0.0      # Zero ATR

        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up=2.0, k_down=2.0, max_bars=15
        )

        # Invalid ATR should result in timeout
        assert labels[0] == 0
        assert bars_to_hit[0] == 15

    def test_last_max_bars_invalid(self):
        """Test that last max_bars samples are marked -99."""
        n = 30
        max_bars = 10
        close = np.full(n, 100.0)
        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0

        labels, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_, atr, k_up=2.0, k_down=2.0, max_bars=max_bars
        )

        # Last max_bars should be -99
        assert np.all(labels[-max_bars:] == -99)
        # Earlier samples should not be -99
        assert not np.any(labels[:-max_bars] == -99)


class TestTripleBarrierQualityMetrics:
    """Tests for quality metrics."""

    def test_quality_metrics_present(self, sample_ohlcv_df):
        """Test that quality metrics are computed."""
        labeler = TripleBarrierLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        metrics = result.quality_metrics

        assert 'total_samples' in metrics
        assert 'valid_samples' in metrics
        assert 'long_count' in metrics
        assert 'short_count' in metrics

    def test_avg_mae_mfe_metrics(self, sample_ohlcv_df):
        """Test MAE/MFE quality metrics."""
        labeler = TripleBarrierLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        metrics = result.quality_metrics

        # Should have MAE/MFE metrics
        assert 'avg_mae' in metrics
        assert 'avg_mfe' in metrics

    def test_avg_bars_to_hit_metric(self, sample_ohlcv_df):
        """Test average bars to hit metric."""
        labeler = TripleBarrierLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        metrics = result.quality_metrics

        if 'avg_bars_to_hit' in metrics:
            assert metrics['avg_bars_to_hit'] > 0


class TestTripleBarrierInputValidation:
    """Tests for input validation."""

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises error."""
        labeler = TripleBarrierLabeler()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            labeler.compute_labels(empty_df, horizon=5)

    def test_missing_columns(self):
        """Test that missing columns raises error."""
        labeler = TripleBarrierLabeler()
        df = pd.DataFrame({'close': [100, 101, 102]})

        with pytest.raises(KeyError):
            labeler.compute_labels(df, horizon=5)

    def test_invalid_k_up(self, sample_ohlcv_df):
        """Test that invalid k_up raises error."""
        labeler = TripleBarrierLabeler()

        with pytest.raises(ValueError, match="k_up"):
            labeler.compute_labels(sample_ohlcv_df, horizon=5, k_up=-1.0)

    def test_invalid_k_down(self, sample_ohlcv_df):
        """Test that invalid k_down raises error."""
        labeler = TripleBarrierLabeler()

        # Note: k_down=0 uses default due to `or` operator in implementation
        # Use negative value to trigger validation error
        with pytest.raises(ValueError, match="k_down"):
            labeler.compute_labels(sample_ohlcv_df, horizon=5, k_down=-1.0)

    def test_invalid_max_bars(self, sample_ohlcv_df):
        """Test that invalid max_bars raises error."""
        labeler = TripleBarrierLabeler()

        with pytest.raises(ValueError, match="max_bars"):
            labeler.compute_labels(sample_ohlcv_df, horizon=5, max_bars=-10)


class TestTripleBarrierAddToDataframe:
    """Tests for adding labels to DataFrame."""

    def test_adds_expected_columns(self, sample_ohlcv_df):
        """Test that expected columns are added."""
        labeler = TripleBarrierLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(sample_ohlcv_df, result)

        assert 'label_h5' in df_labeled.columns
        assert 'bars_to_hit_h5' in df_labeled.columns
        assert 'mae_h5' in df_labeled.columns
        assert 'mfe_h5' in df_labeled.columns
        assert 'touch_type_h5' in df_labeled.columns

    def test_preserves_original_columns(self, sample_ohlcv_df):
        """Test that original columns are preserved."""
        labeler = TripleBarrierLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(sample_ohlcv_df, result)

        assert 'close' in df_labeled.columns
        assert 'high' in df_labeled.columns
        assert 'low' in df_labeled.columns
        assert 'open' in df_labeled.columns


class TestTripleBarrierTransactionCosts:
    """Tests for transaction cost adjustment in triple-barrier labeling."""

    def test_transaction_costs_applied_by_default(self, sample_ohlcv_df):
        """Test that transaction costs are applied by default."""
        labeler = TripleBarrierLabeler()
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        # Metadata should contain cost information
        assert result.metadata.get('transaction_cost_applied', False) is True
        assert 'cost_in_atr' in result.metadata
        assert result.metadata['cost_in_atr'] > 0

    def test_transaction_costs_disabled(self, sample_ohlcv_df):
        """Test that transaction costs can be disabled."""
        labeler = TripleBarrierLabeler(apply_transaction_costs=False)
        result = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        # Metadata should NOT contain cost information
        assert result.metadata.get('transaction_cost_applied', False) is False
        assert 'cost_in_atr' not in result.metadata

    def test_transaction_costs_override_in_compute(self, sample_ohlcv_df):
        """Test that transaction costs can be overridden in compute_labels."""
        labeler = TripleBarrierLabeler(apply_transaction_costs=True)

        # Override to disable costs
        result = labeler.compute_labels(
            sample_ohlcv_df, horizon=5, apply_transaction_costs=False
        )

        assert result.metadata.get('transaction_cost_applied', False) is False

    def test_numba_with_costs_makes_upper_barrier_harder(self):
        """Test that cost adjustment makes upper barrier harder to hit."""
        from src.phase1.stages.labeling import triple_barrier_numba, triple_barrier_numba_with_costs

        n = 30
        close = np.full(n, 100.0)
        high = close.copy()
        low = close.copy()
        open_ = close.copy()
        atr = np.ones(n) * 2.0

        # Create a price move that BARELY hits the upper barrier without costs
        # Upper barrier at 100 + 2.0 * 2 = 104
        # Price moves to exactly 104 on bar 1
        high[1] = 104.0

        # Without costs: should hit upper barrier (WIN)
        labels_no_cost, _, _, _, touch_type_no_cost = triple_barrier_numba(
            close, high, low, open_, atr, k_up=2.0, k_down=2.0, max_bars=15
        )

        # With costs (0.15 ATR): upper barrier at 100 + (2.0 + 0.15) * 2 = 104.3
        # Price at 104 does NOT hit the adjusted barrier
        labels_with_cost, _, _, _, touch_type_with_cost = triple_barrier_numba_with_costs(
            close, high, low, open_, atr, k_up=2.0, k_down=2.0, max_bars=15, cost_in_atr=0.15
        )

        # Without costs: bar 0 should be labeled as WIN (+1)
        assert labels_no_cost[0] == 1, f"Expected WIN without costs, got {labels_no_cost[0]}"
        assert touch_type_no_cost[0] == 1

        # With costs: bar 0 should NOT be WIN (timeout or different label)
        assert labels_with_cost[0] != 1, f"Expected NOT WIN with costs, got {labels_with_cost[0]}"

    def test_cost_adjustment_requires_more_profit_for_win(self):
        """Verify that cost adjustment requires higher gross profit for WIN."""
        from src.phase1.stages.labeling import triple_barrier_numba_with_costs

        n = 30
        close = np.full(n, 100.0)
        high = close.copy()
        low = close.copy()
        open_ = close.copy()
        atr = np.ones(n) * 2.0

        k_up = 2.0
        cost_in_atr = 0.25

        # With costs, effective upper barrier = 100 + (2.0 + 0.25) * 2 = 104.5
        # Price at 104.5 should just barely hit the upper barrier
        high[1] = 104.5

        labels, bars_to_hit, _, _, touch_type = triple_barrier_numba_with_costs(
            close, high, low, open_, atr, k_up=k_up, k_down=2.0, max_bars=15, cost_in_atr=cost_in_atr
        )

        # Should now hit upper barrier with the adjusted higher price
        assert labels[0] == 1, f"Expected WIN with sufficient profit, got {labels[0]}"
        assert touch_type[0] == 1
        assert bars_to_hit[0] == 1

    def test_cost_in_atr_calculation(self, sample_ohlcv_df):
        """Test that cost_in_atr is calculated correctly for different symbols."""
        labeler = TripleBarrierLabeler(symbol='MES', volatility_regime='low_vol')
        result_mes = labeler.compute_labels(sample_ohlcv_df, horizon=5)

        labeler_mgc = TripleBarrierLabeler(symbol='MGC', volatility_regime='low_vol')
        result_mgc = labeler_mgc.compute_labels(sample_ohlcv_df, horizon=5)

        # Both should have cost_in_atr, values may differ based on tick values
        assert result_mes.metadata.get('cost_in_atr', 0) > 0
        assert result_mgc.metadata.get('cost_in_atr', 0) > 0

    def test_high_volatility_regime_higher_costs(self, sample_ohlcv_df):
        """Test that high volatility regime results in higher cost adjustment."""
        labeler_low = TripleBarrierLabeler(
            symbol='MES', volatility_regime='low_vol'
        )
        labeler_high = TripleBarrierLabeler(
            symbol='MES', volatility_regime='high_vol'
        )

        result_low = labeler_low.compute_labels(sample_ohlcv_df, horizon=5)
        result_high = labeler_high.compute_labels(sample_ohlcv_df, horizon=5)

        cost_low = result_low.metadata.get('cost_in_atr', 0)
        cost_high = result_high.metadata.get('cost_in_atr', 0)

        # High volatility should have higher costs (more slippage)
        assert cost_high > cost_low, (
            f"High vol cost ({cost_high}) should be > low vol cost ({cost_low})"
        )

    def test_symbol_override_in_compute_labels(self, sample_ohlcv_df):
        """Test that symbol can be overridden in compute_labels."""
        labeler = TripleBarrierLabeler(symbol='MES')

        result = labeler.compute_labels(sample_ohlcv_df, horizon=5, symbol='MGC')

        assert result.metadata.get('symbol') == 'MGC'

    def test_backward_compatibility_with_disabled_costs(self, sample_ohlcv_df):
        """Test that disabling costs produces identical results to old behavior."""
        from src.phase1.stages.labeling import triple_barrier_numba

        labeler = TripleBarrierLabeler(apply_transaction_costs=False)
        result = labeler.compute_labels(
            sample_ohlcv_df, horizon=5, k_up=2.0, k_down=2.0, max_bars=10
        )

        # Compare with direct numba call (old behavior)
        close = sample_ohlcv_df['close'].values
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        open_prices = sample_ohlcv_df['open'].values
        atr = sample_ohlcv_df['atr_14'].values

        labels_direct, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_prices, atr, k_up=2.0, k_down=2.0, max_bars=10
        )

        np.testing.assert_array_equal(result.labels, labels_direct)

    def test_lower_barrier_not_affected_by_costs(self):
        """Test that the lower barrier is NOT adjusted by transaction costs."""
        from src.phase1.stages.labeling import triple_barrier_numba, triple_barrier_numba_with_costs

        n = 30
        close = np.full(n, 100.0)
        high = close.copy()
        low = close.copy()
        open_ = close.copy()
        atr = np.ones(n) * 2.0

        # Lower barrier at 100 - 2.0 * 2 = 96
        # Price drops to exactly 96 on bar 1
        low[1] = 96.0

        # Without costs
        labels_no_cost, _, _, _, touch_type_no_cost = triple_barrier_numba(
            close, high, low, open_, atr, k_up=2.0, k_down=2.0, max_bars=15
        )

        # With costs (should NOT affect lower barrier)
        labels_with_cost, _, _, _, touch_type_with_cost = triple_barrier_numba_with_costs(
            close, high, low, open_, atr, k_up=2.0, k_down=2.0, max_bars=15, cost_in_atr=0.5
        )

        # Both should hit lower barrier (LOSS)
        assert labels_no_cost[0] == -1
        assert labels_with_cost[0] == -1
        assert touch_type_no_cost[0] == -1
        assert touch_type_with_cost[0] == -1
