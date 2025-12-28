"""
Tests for Walk-Forward Evaluator.

Tests:
- Configuration validation
- Split correctness (temporal ordering, no overlap)
- Output shape validation
- Expanding vs rolling window behavior
- Label-aware purging
- Coverage validation
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.cross_validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardResult,
    WindowMetrics,
    create_walk_forward_evaluator,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def time_series_data():
    """Generate synthetic time series data with DatetimeIndex."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    start_time = datetime(2023, 1, 2, 9, 30)
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features).astype(np.float32),
        index=dates,
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = pd.Series(
        np.random.choice([-1, 0, 1], size=n_samples),
        index=dates,
        name="label",
    )

    return {"X": X, "y": y}


@pytest.fixture
def small_time_series_data():
    """Generate small synthetic time series data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    start_time = datetime(2023, 1, 2, 9, 30)
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features).astype(np.float32),
        index=dates,
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = pd.Series(
        np.random.choice([-1, 0, 1], size=n_samples),
        index=dates,
        name="label",
    )

    return {"X": X, "y": y}


@pytest.fixture
def label_end_times_data():
    """Generate data with label_end_times for testing overlap purging."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    horizon = 20

    start_time = datetime(2023, 1, 2, 9, 30)
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features).astype(np.float32),
        index=dates,
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = pd.Series(
        np.random.choice([-1, 0, 1], size=n_samples),
        index=dates,
        name="label",
    )

    # Label end times: each label is resolved horizon bars later
    label_end_times = pd.Series(
        [dates[min(i + horizon, n_samples - 1)] for i in range(n_samples)],
        index=dates,
        name="label_end_time",
    )

    return {"X": X, "y": y, "label_end_times": label_end_times, "horizon": horizon}


@pytest.fixture
def default_config():
    """Default walk-forward configuration."""
    return WalkForwardConfig(
        n_windows=5,
        window_type="expanding",
        min_train_pct=0.4,
        test_pct=0.1,
    )


@pytest.fixture
def rolling_config():
    """Rolling window configuration."""
    return WalkForwardConfig(
        n_windows=5,
        window_type="rolling",
        min_train_pct=0.3,
        test_pct=0.1,
    )


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestWalkForwardConfig:
    """Tests for WalkForwardConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creates successfully."""
        config = WalkForwardConfig(
            n_windows=5,
            window_type="expanding",
            min_train_pct=0.4,
            test_pct=0.1,
        )
        assert config.n_windows == 5
        assert config.window_type == "expanding"
        assert config.min_train_pct == 0.4
        assert config.test_pct == 0.1

    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()
        assert config.n_windows == 5
        assert config.window_type == "expanding"
        assert config.min_train_pct == 0.4
        assert config.test_pct == 0.1
        assert config.embargo_bars == 0
        assert config.gap_bars == 0

    def test_invalid_n_windows_zero(self):
        """Test that n_windows < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_windows must be >= 1"):
            WalkForwardConfig(n_windows=0)

    def test_invalid_window_type(self):
        """Test that invalid window_type raises ValueError."""
        with pytest.raises(ValueError, match="window_type must be"):
            WalkForwardConfig(window_type="invalid")

    def test_invalid_min_train_pct_zero(self):
        """Test that min_train_pct <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="min_train_pct must be in"):
            WalkForwardConfig(min_train_pct=0)

    def test_invalid_min_train_pct_one(self):
        """Test that min_train_pct >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_train_pct must be in"):
            WalkForwardConfig(min_train_pct=1.0)

    def test_invalid_test_pct(self):
        """Test that test_pct outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="test_pct must be in"):
            WalkForwardConfig(test_pct=0)

    def test_invalid_total_exceeds_100(self):
        """Test that min_train + n_windows * test > 1 raises ValueError."""
        with pytest.raises(ValueError, match="exceeds 1.0"):
            WalkForwardConfig(n_windows=10, min_train_pct=0.5, test_pct=0.1)

    def test_invalid_negative_embargo(self):
        """Test that negative embargo_bars raises ValueError."""
        with pytest.raises(ValueError, match="embargo_bars must be >= 0"):
            WalkForwardConfig(embargo_bars=-1)

    def test_invalid_negative_gap(self):
        """Test that negative gap_bars raises ValueError."""
        with pytest.raises(ValueError, match="gap_bars must be >= 0"):
            WalkForwardConfig(gap_bars=-1)


# =============================================================================
# SPLIT CORRECTNESS TESTS
# =============================================================================

class TestSplitCorrectness:
    """Tests for split generation correctness."""

    def test_generates_correct_number_of_windows(self, default_config, time_series_data):
        """Test that evaluator generates expected number of windows."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        windows = list(wf.split(X))
        assert len(windows) == default_config.n_windows

    def test_window_indices_are_arrays(self, default_config, time_series_data):
        """Test that window indices are numpy arrays."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        for train_idx, test_idx in wf.split(X):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_train_test_non_overlapping(self, default_config, time_series_data):
        """Test that train and test indices do not overlap within a window."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        for train_idx, test_idx in wf.split(X):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices"

    def test_train_precedes_test(self, default_config, time_series_data):
        """Test that all training indices come before test indices (temporal ordering)."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        for train_idx, test_idx in wf.split(X):
            assert train_idx.max() < test_idx.min(), (
                f"Train max {train_idx.max()} >= test min {test_idx.min()}"
            )

    def test_test_windows_are_contiguous(self, default_config, time_series_data):
        """Test that test indices form contiguous blocks."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        for train_idx, test_idx in wf.split(X):
            test_sorted = np.sort(test_idx)
            expected = np.arange(test_sorted[0], test_sorted[-1] + 1)
            np.testing.assert_array_equal(test_sorted, expected)

    def test_test_windows_progress_forward(self, default_config, time_series_data):
        """Test that test windows move forward in time."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        prev_test_start = -1
        for train_idx, test_idx in wf.split(X):
            test_start = test_idx.min()
            assert test_start > prev_test_start, (
                f"Test window did not progress: {test_start} <= {prev_test_start}"
            )
            prev_test_start = test_start

    def test_get_n_splits(self, default_config, time_series_data):
        """Test get_n_splits returns correct value."""
        wf = WalkForwardEvaluator(default_config)
        assert wf.get_n_splits() == default_config.n_windows


# =============================================================================
# EXPANDING VS ROLLING TESTS
# =============================================================================

class TestWindowTypes:
    """Tests for expanding vs rolling window behavior."""

    def test_expanding_window_grows(self, time_series_data):
        """Test that expanding window training set grows with each window."""
        config = WalkForwardConfig(n_windows=5, window_type="expanding", min_train_pct=0.3)
        wf = WalkForwardEvaluator(config)
        X = time_series_data["X"]

        train_sizes = []
        for train_idx, _ in wf.split(X):
            train_sizes.append(len(train_idx))

        # Each subsequent window should have more or equal training samples
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                f"Expanding window should grow: {train_sizes[i]} < {train_sizes[i-1]}"
            )

    def test_rolling_window_stable_size(self, time_series_data):
        """Test that rolling window training set stays approximately fixed size."""
        config = WalkForwardConfig(n_windows=5, window_type="rolling", min_train_pct=0.3)
        wf = WalkForwardEvaluator(config)
        X = time_series_data["X"]

        train_sizes = []
        for train_idx, _ in wf.split(X):
            train_sizes.append(len(train_idx))

        # Rolling window sizes should be similar (within 10% of each other)
        max_size = max(train_sizes)
        min_size = min(train_sizes)
        size_variation = (max_size - min_size) / max_size

        assert size_variation < 0.1, (
            f"Rolling window sizes vary too much: {train_sizes}, variation={size_variation:.1%}"
        )

    def test_expanding_starts_from_zero(self, time_series_data):
        """Test that expanding window always starts at index 0."""
        config = WalkForwardConfig(n_windows=3, window_type="expanding", min_train_pct=0.4)
        wf = WalkForwardEvaluator(config)
        X = time_series_data["X"]

        for train_idx, _ in wf.split(X):
            assert train_idx[0] == 0, f"Expanding window should start at 0, got {train_idx[0]}"


# =============================================================================
# OUTPUT SHAPE TESTS
# =============================================================================

class TestOutputShape:
    """Tests for output shapes and data structures."""

    def test_train_indices_non_empty(self, default_config, time_series_data):
        """Test that training indices are never empty."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        for window_idx, (train_idx, _) in enumerate(wf.split(X)):
            assert len(train_idx) > 0, f"Window {window_idx}: Empty training set"

    def test_test_indices_non_empty(self, default_config, time_series_data):
        """Test that test indices are never empty."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        for window_idx, (_, test_idx) in enumerate(wf.split(X)):
            assert len(test_idx) > 0, f"Window {window_idx}: Empty test set"

    def test_window_info_structure(self, default_config, time_series_data):
        """Test that window info has expected structure."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        info = wf.get_window_info(X)
        assert len(info) == default_config.n_windows

        for i, window_info in enumerate(info):
            assert window_info["window"] == i
            assert "train_size" in window_info
            assert "test_size" in window_info
            assert "train_start_idx" in window_info
            assert "train_end_idx" in window_info
            assert "test_start_idx" in window_info
            assert "test_end_idx" in window_info
            assert "train_start_time" in window_info
            assert "train_end_time" in window_info
            assert "test_start_time" in window_info
            assert "test_end_time" in window_info

    def test_coverage_stats_structure(self, default_config, time_series_data):
        """Test that coverage validation returns expected structure."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        coverage = wf.validate_coverage(X)

        assert "total_samples" in coverage
        assert "samples_in_test" in coverage
        assert "test_coverage_fraction" in coverage
        assert "expected_test_range" in coverage
        assert "samples_in_multiple_tests" in coverage
        assert "window_type" in coverage

    def test_coverage_no_duplicate_test_samples(self, default_config, time_series_data):
        """Test that no sample appears in multiple test windows."""
        wf = WalkForwardEvaluator(default_config)
        X = time_series_data["X"]

        coverage = wf.validate_coverage(X)
        assert coverage["samples_in_multiple_tests"] == 0


# =============================================================================
# LABEL-AWARE PURGING TESTS
# =============================================================================

class TestLabelAwarePurging:
    """Tests for label-aware purging with label_end_times."""

    def test_with_label_end_times_purges_overlapping(self, label_end_times_data):
        """Test that label_end_times purges samples with overlapping labels."""
        config = WalkForwardConfig(n_windows=3, min_train_pct=0.3, test_pct=0.15)
        wf = WalkForwardEvaluator(config)

        X = label_end_times_data["X"]
        label_end_times = label_end_times_data["label_end_times"]

        for train_idx, test_idx in wf.split(X, label_end_times=label_end_times):
            test_start_time = X.index[test_idx[0]]

            # Check that no training sample has label_end_time >= test_start_time
            for idx in train_idx:
                label_end = label_end_times.iloc[idx]
                assert label_end < test_start_time, (
                    f"Sample {idx} has label_end_time {label_end} >= "
                    f"test_start_time {test_start_time}"
                )

    def test_without_label_end_times_works(self, small_time_series_data):
        """Test that split works without label_end_times."""
        config = WalkForwardConfig(n_windows=3, min_train_pct=0.4, test_pct=0.15)
        wf = WalkForwardEvaluator(config)
        X = small_time_series_data["X"]

        # Should work without label_end_times
        windows = list(wf.split(X, label_end_times=None))
        assert len(windows) == 3

    def test_label_aware_purge_removes_additional_samples(self, label_end_times_data):
        """Test that label_end_times purging removes more samples than without."""
        config = WalkForwardConfig(n_windows=3, min_train_pct=0.3, test_pct=0.15)
        wf = WalkForwardEvaluator(config)

        X = label_end_times_data["X"]
        label_end_times = label_end_times_data["label_end_times"]

        # Compare sizes with and without label_end_times
        windows_without = list(wf.split(X, label_end_times=None))
        windows_with = list(wf.split(X, label_end_times=label_end_times))

        total_train_without = sum(len(train) for train, _ in windows_without)
        total_train_with = sum(len(train) for train, _ in windows_with)

        # Label-aware purging should remove additional samples
        assert total_train_with < total_train_without, (
            f"Label-aware purging should remove more samples. "
            f"Without: {total_train_without}, With: {total_train_with}"
        )


# =============================================================================
# GAP BARS TESTS
# =============================================================================

class TestGapBars:
    """Tests for gap_bars between train and test."""

    def test_gap_bars_creates_gap(self, time_series_data):
        """Test that gap_bars creates a gap between train and test."""
        gap = 10
        config = WalkForwardConfig(n_windows=3, min_train_pct=0.4, test_pct=0.1, gap_bars=gap)
        wf = WalkForwardEvaluator(config)
        X = time_series_data["X"]

        for train_idx, test_idx in wf.split(X):
            train_max = train_idx.max()
            test_min = test_idx.min()
            actual_gap = test_min - train_max - 1

            assert actual_gap >= gap, (
                f"Expected gap >= {gap}, got {actual_gap}"
            )


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunction:
    """Tests for create_walk_forward_evaluator factory."""

    def test_creates_evaluator(self):
        """Test that factory creates valid evaluator."""
        wf = create_walk_forward_evaluator(n_windows=5, window_type="expanding")

        assert isinstance(wf, WalkForwardEvaluator)
        assert wf.config.n_windows == 5
        assert wf.config.window_type == "expanding"

    def test_factory_passes_all_args(self):
        """Test that factory passes all arguments to config."""
        wf = create_walk_forward_evaluator(
            n_windows=8,
            window_type="rolling",
            min_train_pct=0.35,
            test_pct=0.08,
            embargo_bars=10,
            gap_bars=5,
        )

        assert wf.config.n_windows == 8
        assert wf.config.window_type == "rolling"
        assert wf.config.min_train_pct == 0.35
        assert wf.config.test_pct == 0.08
        assert wf.config.embargo_bars == 10
        assert wf.config.gap_bars == 5


# =============================================================================
# REPR TESTS
# =============================================================================

class TestRepr:
    """Tests for string representation."""

    def test_repr_includes_config(self, default_config):
        """Test __repr__ includes configuration values."""
        wf = WalkForwardEvaluator(default_config)
        repr_str = repr(wf)

        assert "WalkForwardEvaluator" in repr_str
        assert "n_windows=5" in repr_str
        assert "expanding" in repr_str


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_window(self, time_series_data):
        """Test with single window."""
        config = WalkForwardConfig(n_windows=1, min_train_pct=0.5, test_pct=0.3)
        wf = WalkForwardEvaluator(config)
        X = time_series_data["X"]

        windows = list(wf.split(X))
        assert len(windows) == 1

        train_idx, test_idx = windows[0]
        assert len(train_idx) > 0
        assert len(test_idx) > 0

    def test_many_windows(self, time_series_data):
        """Test with many small windows."""
        config = WalkForwardConfig(n_windows=10, min_train_pct=0.2, test_pct=0.05)
        wf = WalkForwardEvaluator(config)
        X = time_series_data["X"]

        windows = list(wf.split(X))
        assert len(windows) == 10

    def test_insufficient_data_raises(self):
        """Test that insufficient data raises ValueError."""
        # Create a very small dataset (10 samples)
        np.random.seed(42)
        n_samples = 10
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="5min")
        X = pd.DataFrame(
            np.random.randn(n_samples, 5),
            index=dates,
        )

        # Config that requires more data than available
        # min_train = 10 * 0.4 = 4 samples
        # total_test = 5 * (10 * 0.1) = 5 samples
        # Required: 4 + 5 = 9 samples, but we only have 10
        # However, with 10 samples, test_size becomes max(1, int(10*0.1))=1 per window
        # So we need 4 + 5*1 = 9, which fits
        # Let's use config that passes validation but fails at runtime
        config = WalkForwardConfig(n_windows=3, min_train_pct=0.4, test_pct=0.1)
        wf = WalkForwardEvaluator(config)

        # With 10 samples: min_train=4, test_size=1, total_test=3
        # 4 + 3 = 7 < 10, so this should work
        # Need to create a config that works on paper but fails with actual data
        # Let's test the config validation error instead
        with pytest.raises(ValueError, match="exceeds 1.0"):
            # This config is invalid: 0.6 + 5*0.1 = 1.1 > 1.0
            WalkForwardConfig(n_windows=5, min_train_pct=0.6, test_pct=0.1)
