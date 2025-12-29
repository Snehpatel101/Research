"""
Tests for Final Labels Core Logic.

Tests cover:
- Gap-aware label_end_time computation
- Quality score calculation
- Sample weight assignment
"""

import numpy as np
import pandas as pd
import pytest

from src.phase1.stages.final_labels.core import apply_optimized_labels


@pytest.fixture
def ohlcv_with_gaps():
    """
    Create OHLCV DataFrame with overnight gaps.

    Simulates trading sessions: 9:30-16:00 with 5-minute bars.
    Day 1: 9:30-16:00 (78 bars)
    Gap: 16:00 -> 9:30 next day (17.5 hours)
    Day 2: 9:30-16:00 (78 bars)
    """
    # Day 1 bars
    day1_start = pd.Timestamp("2024-01-15 09:30:00")
    day1_times = pd.date_range(start=day1_start, periods=78, freq="5min")

    # Day 2 bars (next day, starting at 9:30)
    day2_start = pd.Timestamp("2024-01-16 09:30:00")
    day2_times = pd.date_range(start=day2_start, periods=78, freq="5min")

    all_times = day1_times.append(day2_times)
    n = len(all_times)

    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.1

    # Ensure OHLC relationships
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    df = pd.DataFrame({
        'datetime': all_times,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100, 1000, n),
        'atr_14': np.ones(n) * 1.0,
    })

    return df


@pytest.fixture
def ohlcv_no_gaps():
    """Create continuous OHLCV DataFrame with no gaps."""
    n = 100
    times = pd.date_range(start="2024-01-15 09:30:00", periods=n, freq="5min")

    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.1

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    df = pd.DataFrame({
        'datetime': times,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100, 1000, n),
        'atr_14': np.ones(n) * 1.0,
    })

    return df


class TestLabelEndTimeGapAware:
    """Tests for gap-aware label_end_time computation."""

    def test_label_end_time_with_overnight_gap(self, ohlcv_with_gaps):
        """
        Test that label_end_time correctly uses actual bar times, not arithmetic.

        If a bar at 15:55 on Day 1 has bars_to_hit=3, the label_end_time should
        be at bar index + 3, which crosses into Day 2 (09:40 next day), NOT
        15:55 + 15min = 16:10 (which is during market close).
        """
        df = ohlcv_with_gaps
        horizon = 5
        best_params = {'k_up': 1.0, 'k_down': 1.0, 'max_bars': 20}

        result = apply_optimized_labels(df, horizon, best_params)

        # Find a bar near end of Day 1 that has bars_to_hit crossing overnight
        # Day 1 ends at index 77 (16:00). Check bar at index 75 (15:50)
        bar_idx = 75
        bars_to_hit = int(result[f'bars_to_hit_h{horizon}'].iloc[bar_idx])
        label_end_time = result[f'label_end_time_h{horizon}'].iloc[bar_idx]
        entry_time = df['datetime'].iloc[bar_idx]

        if bars_to_hit > 0 and bar_idx + bars_to_hit < len(df):
            # The label_end_time should be the actual datetime at bar_idx + bars_to_hit
            expected_end_time = df['datetime'].iloc[bar_idx + bars_to_hit]

            assert pd.Timestamp(label_end_time) == pd.Timestamp(expected_end_time), (
                f"label_end_time should be datetime at forward index, not arithmetic. "
                f"Got {label_end_time}, expected {expected_end_time}"
            )

            # Verify that naive arithmetic would give wrong result if gap exists
            naive_duration = pd.Timedelta(minutes=5)  # Assumed uniform bar duration
            naive_end_time = entry_time + bars_to_hit * naive_duration

            # If the label crosses the overnight gap, naive end time will be wrong
            if bar_idx + bars_to_hit >= 78:  # Crosses into Day 2
                assert naive_end_time != expected_end_time, (
                    "This test only makes sense if gap causes difference"
                )

    def test_label_end_time_matches_forward_index(self, ohlcv_with_gaps):
        """
        Verify that label_end_time equals datetime[i + bars_to_hit[i]] for all bars.
        """
        df = ohlcv_with_gaps
        horizon = 5
        best_params = {'k_up': 1.0, 'k_down': 1.0, 'max_bars': 15}

        result = apply_optimized_labels(df, horizon, best_params)

        datetime_col = df['datetime'].values
        bars_to_hit = result[f'bars_to_hit_h{horizon}'].values
        label_end_times = result[f'label_end_time_h{horizon}'].values
        n = len(df)

        # Check all bars (except those at end with clipped indices)
        for i in range(n):
            forward_idx = min(i + int(bars_to_hit[i]), n - 1)
            expected = datetime_col[forward_idx]
            actual = label_end_times[i]

            assert pd.Timestamp(actual) == pd.Timestamp(expected), (
                f"Bar {i}: label_end_time mismatch. "
                f"Got {actual}, expected {expected} (forward_idx={forward_idx})"
            )

    def test_label_end_time_continuous_data(self, ohlcv_no_gaps):
        """
        Test label_end_time on continuous data (no gaps).

        With no gaps, forward-lookup and arithmetic should give same result.
        """
        df = ohlcv_no_gaps
        horizon = 5
        best_params = {'k_up': 1.0, 'k_down': 1.0, 'max_bars': 10}

        result = apply_optimized_labels(df, horizon, best_params)

        datetime_col = df['datetime'].values
        bars_to_hit = result[f'bars_to_hit_h{horizon}'].values
        label_end_times = result[f'label_end_time_h{horizon}'].values
        n = len(df)

        # Verify forward-lookup matches actual bar times
        for i in range(n):
            forward_idx = min(i + int(bars_to_hit[i]), n - 1)
            expected = datetime_col[forward_idx]
            actual = label_end_times[i]

            assert pd.Timestamp(actual) == pd.Timestamp(expected)

    def test_label_end_time_clipped_at_end(self, ohlcv_no_gaps):
        """
        Test that label_end_time is clipped to last bar for samples near end.
        """
        df = ohlcv_no_gaps
        horizon = 5
        best_params = {'k_up': 10.0, 'k_down': 10.0, 'max_bars': 50}  # Large barriers -> timeouts

        result = apply_optimized_labels(df, horizon, best_params)

        n = len(df)
        max_datetime = df['datetime'].iloc[-1]

        # Last few bars should have label_end_time clipped to max_datetime
        for i in range(n - 10, n - best_params['max_bars']):  # Before invalid zone
            label_end_time = result[f'label_end_time_h{horizon}'].iloc[i]
            bars_to_hit = result[f'bars_to_hit_h{horizon}'].iloc[i]

            if i + bars_to_hit >= n:
                assert pd.Timestamp(label_end_time) == pd.Timestamp(max_datetime), (
                    f"Bar {i} with bars_to_hit={bars_to_hit} should clip to max_datetime"
                )


class TestQualityScoreIntegration:
    """Tests for quality score computation in apply_optimized_labels."""

    def test_quality_columns_added(self, ohlcv_no_gaps):
        """Test that quality-related columns are added."""
        df = ohlcv_no_gaps
        horizon = 5
        best_params = {'k_up': 1.0, 'k_down': 1.0, 'max_bars': 15}

        result = apply_optimized_labels(df, horizon, best_params)

        assert f'quality_h{horizon}' in result.columns
        assert f'sample_weight_h{horizon}' in result.columns
        assert f'pain_to_gain_h{horizon}' in result.columns
        assert f'time_weighted_dd_h{horizon}' in result.columns

    def test_sample_weights_in_range(self, ohlcv_no_gaps):
        """Test that sample weights are in expected tiers."""
        df = ohlcv_no_gaps
        horizon = 5
        best_params = {'k_up': 1.0, 'k_down': 1.0, 'max_bars': 15}

        result = apply_optimized_labels(df, horizon, best_params)

        weights = result[f'sample_weight_h{horizon}'].values
        unique_weights = set(weights)

        # Should only have weights 0.5, 1.0, 1.5
        assert unique_weights.issubset({0.5, 1.0, 1.5})
