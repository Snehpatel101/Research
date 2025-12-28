"""
Tests for Combinatorially Purged Cross-Validation (CPCV).

Tests:
- Configuration validation
- Combination limiting
- Split correctness (non-overlapping, purging)
- Coverage validation
- Label-aware purging
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from math import comb

from src.cross_validation.cpcv import (
    CPCVConfig,
    CombinatorialPurgedCV,
    CPCVResult,
    CPCVPathResult,
    create_cpcv,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def time_series_data():
    """Generate synthetic time series data with DatetimeIndex."""
    np.random.seed(42)
    n_samples = 600
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
def label_end_times_data():
    """Generate data with label_end_times."""
    np.random.seed(42)
    n_samples = 600
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

    label_end_times = pd.Series(
        [dates[min(i + horizon, n_samples - 1)] for i in range(n_samples)],
        index=dates,
        name="label_end_time",
    )

    return {"X": X, "y": y, "label_end_times": label_end_times}


@pytest.fixture
def default_config():
    """Default CPCV configuration."""
    return CPCVConfig(
        n_groups=6,
        n_test_groups=2,
        max_combinations=15,
    )


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestCPCVConfig:
    """Tests for CPCVConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creates successfully."""
        config = CPCVConfig(n_groups=6, n_test_groups=2, max_combinations=15)
        assert config.n_groups == 6
        assert config.n_test_groups == 2
        assert config.max_combinations == 15
        assert config.n_train_groups == 4

    def test_default_config(self):
        """Test default configuration values."""
        config = CPCVConfig()
        assert config.n_groups == 6
        assert config.n_test_groups == 2
        assert config.max_combinations == 20

    def test_total_combinations_property(self):
        """Test total_combinations property."""
        config = CPCVConfig(n_groups=6, n_test_groups=2)
        assert config.total_combinations == comb(6, 2)  # C(6,2) = 15

        config = CPCVConfig(n_groups=10, n_test_groups=2)
        assert config.total_combinations == comb(10, 2)  # C(10,2) = 45

    def test_invalid_n_groups_too_low(self):
        """Test that n_groups < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_groups must be >= 2"):
            CPCVConfig(n_groups=1)

    def test_invalid_n_test_groups_zero(self):
        """Test that n_test_groups < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_test_groups must be >= 1"):
            CPCVConfig(n_test_groups=0)

    def test_invalid_n_test_groups_exceeds_n_groups(self):
        """Test that n_test_groups >= n_groups raises ValueError."""
        with pytest.raises(ValueError, match="n_test_groups.*must be < n_groups"):
            CPCVConfig(n_groups=5, n_test_groups=5)

    def test_invalid_max_combinations(self):
        """Test that max_combinations < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_combinations must be >= 1"):
            CPCVConfig(max_combinations=0)

    def test_invalid_purge_pct(self):
        """Test that purge_pct outside [0, 0.5) raises ValueError."""
        with pytest.raises(ValueError, match="purge_pct must be in"):
            CPCVConfig(purge_pct=0.5)

    def test_invalid_embargo_pct(self):
        """Test that embargo_pct outside [0, 0.5) raises ValueError."""
        with pytest.raises(ValueError, match="embargo_pct must be in"):
            CPCVConfig(embargo_pct=-0.1)


# =============================================================================
# COMBINATION LIMITING TESTS
# =============================================================================

class TestCombinationLimiting:
    """Tests for combination limiting functionality."""

    def test_respects_max_combinations(self, time_series_data):
        """Test that CPCV respects max_combinations limit."""
        X = time_series_data["X"]

        # C(10, 2) = 45 combinations, but limit to 10
        config = CPCVConfig(n_groups=10, n_test_groups=2, max_combinations=10)
        cpcv = CombinatorialPurgedCV(config)

        paths = list(cpcv.split(X))
        assert len(paths) == 10  # Should be limited to 10

    def test_uses_all_combinations_when_under_limit(self, time_series_data):
        """Test that all combinations are used when under limit."""
        X = time_series_data["X"]

        # C(6, 2) = 15 combinations, limit is 20
        config = CPCVConfig(n_groups=6, n_test_groups=2, max_combinations=20)
        cpcv = CombinatorialPurgedCV(config)

        paths = list(cpcv.split(X))
        assert len(paths) == 15  # C(6,2) = 15

    def test_get_n_splits_respects_limit(self):
        """Test get_n_splits respects max_combinations."""
        config = CPCVConfig(n_groups=10, n_test_groups=2, max_combinations=10)
        cpcv = CombinatorialPurgedCV(config)

        assert cpcv.get_n_splits() == 10

    def test_combinations_are_different(self, time_series_data):
        """Test that different paths have different test indices."""
        X = time_series_data["X"]

        config = CPCVConfig(n_groups=6, n_test_groups=2)
        cpcv = CombinatorialPurgedCV(config)

        test_sets = []
        for _, test_idx, _ in cpcv.split(X):
            test_sets.append(frozenset(test_idx))

        # All test sets should be unique
        assert len(test_sets) == len(set(test_sets))


# =============================================================================
# SPLIT CORRECTNESS TESTS
# =============================================================================

class TestSplitCorrectness:
    """Tests for split generation correctness."""

    def test_train_test_non_overlapping(self, default_config, time_series_data):
        """Test that train and test indices do not overlap."""
        X = time_series_data["X"]
        cpcv = CombinatorialPurgedCV(default_config)

        for train_idx, test_idx, path_id in cpcv.split(X):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Path {path_id}: Found {len(overlap)} overlapping indices"

    def test_indices_are_valid(self, default_config, time_series_data):
        """Test that all indices are within valid range."""
        X = time_series_data["X"]
        n_samples = len(X)
        cpcv = CombinatorialPurgedCV(default_config)

        for train_idx, test_idx, path_id in cpcv.split(X):
            assert train_idx.min() >= 0
            assert train_idx.max() < n_samples
            assert test_idx.min() >= 0
            assert test_idx.max() < n_samples

    def test_returns_numpy_arrays(self, default_config, time_series_data):
        """Test that split returns numpy arrays."""
        X = time_series_data["X"]
        cpcv = CombinatorialPurgedCV(default_config)

        for train_idx, test_idx, path_id in cpcv.split(X):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert isinstance(path_id, int)

    def test_train_and_test_non_empty(self, default_config, time_series_data):
        """Test that train and test sets are non-empty."""
        X = time_series_data["X"]
        cpcv = CombinatorialPurgedCV(default_config)

        for train_idx, test_idx, path_id in cpcv.split(X):
            assert len(train_idx) > 0, f"Path {path_id}: Empty training set"
            assert len(test_idx) > 0, f"Path {path_id}: Empty test set"

    def test_purging_removes_samples_near_test(self, time_series_data):
        """Test that purging removes samples near test boundaries."""
        X = time_series_data["X"]

        config = CPCVConfig(n_groups=4, n_test_groups=1, purge_pct=0.05)
        cpcv = CombinatorialPurgedCV(config)

        for train_idx, test_idx, path_id in cpcv.split(X):
            test_start = test_idx.min()

            # Check that samples immediately before test are not in train
            purge_size = int(len(X) * 0.05)
            purge_zone = range(max(0, test_start - purge_size), test_start)

            for idx in purge_zone:
                assert idx not in train_idx, (
                    f"Path {path_id}: Index {idx} in purge zone but in training"
                )


# =============================================================================
# LABEL-AWARE PURGING TESTS
# =============================================================================

class TestLabelAwarePurging:
    """Tests for label-aware purging."""

    def test_with_label_end_times(self, label_end_times_data):
        """Test that label_end_times enables additional purging."""
        X = label_end_times_data["X"]
        label_end_times = label_end_times_data["label_end_times"]

        config = CPCVConfig(n_groups=4, n_test_groups=1)
        cpcv = CombinatorialPurgedCV(config)

        for train_idx, test_idx, path_id in cpcv.split(X, label_end_times=label_end_times):
            test_start_time = X.index[test_idx.min()]

            # Check no training sample has label_end_time >= test_start_time
            for idx in train_idx:
                if idx < test_idx.min():
                    label_end = label_end_times.iloc[idx]
                    assert label_end < test_start_time, (
                        f"Path {path_id}: Sample {idx} has overlapping label"
                    )

    def test_without_label_end_times_works(self, time_series_data):
        """Test that split works without label_end_times."""
        X = time_series_data["X"]

        config = CPCVConfig(n_groups=4, n_test_groups=1)
        cpcv = CombinatorialPurgedCV(config)

        paths = list(cpcv.split(X, label_end_times=None))
        assert len(paths) == 4


# =============================================================================
# COVERAGE TESTS
# =============================================================================

class TestCoverage:
    """Tests for coverage validation."""

    def test_validate_coverage_returns_stats(self, default_config, time_series_data):
        """Test validate_coverage returns expected structure."""
        X = time_series_data["X"]
        cpcv = CombinatorialPurgedCV(default_config)

        coverage = cpcv.validate_coverage(X)

        assert "total_samples" in coverage
        assert "n_paths" in coverage
        assert "n_groups" in coverage
        assert "samples_never_in_test" in coverage
        assert "avg_test_appearances" in coverage

    def test_samples_appear_in_test_multiple_times(self, time_series_data):
        """Test that samples appear in test across multiple paths."""
        X = time_series_data["X"]

        config = CPCVConfig(n_groups=6, n_test_groups=2)
        cpcv = CombinatorialPurgedCV(config)

        coverage = cpcv.validate_coverage(X)

        # With C(6,2)=15 paths and 2/6 groups per test, avg appearances should be > 1
        assert coverage["avg_test_appearances"] > 1


# =============================================================================
# PATH INFO TESTS
# =============================================================================

class TestPathInfo:
    """Tests for get_path_info method."""

    def test_path_info_structure(self, default_config, time_series_data):
        """Test that path info has expected structure."""
        X = time_series_data["X"]
        cpcv = CombinatorialPurgedCV(default_config)

        info = cpcv.get_path_info(X)

        assert len(info) > 0
        for path_info in info:
            assert "path_id" in path_info
            assert "train_size" in path_info
            assert "test_size" in path_info
            assert "train_start_idx" in path_info
            assert "test_start_time" in path_info  # Should have datetime

    def test_path_info_has_correct_count(self, default_config, time_series_data):
        """Test that path_info returns info for all paths."""
        X = time_series_data["X"]
        cpcv = CombinatorialPurgedCV(default_config)

        info = cpcv.get_path_info(X)
        n_splits = cpcv.get_n_splits()

        assert len(info) == n_splits


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunction:
    """Tests for create_cpcv factory."""

    def test_creates_evaluator(self):
        """Test that factory creates valid evaluator."""
        cpcv = create_cpcv(n_groups=6, n_test_groups=2)

        assert isinstance(cpcv, CombinatorialPurgedCV)
        assert cpcv.config.n_groups == 6
        assert cpcv.config.n_test_groups == 2

    def test_factory_passes_all_args(self):
        """Test that factory passes all arguments to config."""
        cpcv = create_cpcv(
            n_groups=8,
            n_test_groups=3,
            max_combinations=25,
            purge_pct=0.02,
            embargo_pct=0.01,
        )

        assert cpcv.config.n_groups == 8
        assert cpcv.config.n_test_groups == 3
        assert cpcv.config.max_combinations == 25
        assert cpcv.config.purge_pct == 0.02
        assert cpcv.config.embargo_pct == 0.01


# =============================================================================
# REPR TESTS
# =============================================================================

class TestRepr:
    """Tests for string representation."""

    def test_repr_includes_config(self, default_config):
        """Test __repr__ includes configuration values."""
        cpcv = CombinatorialPurgedCV(default_config)
        repr_str = repr(cpcv)

        assert "CombinatorialPurgedCV" in repr_str
        assert "n_groups=6" in repr_str
        assert "n_test_groups=2" in repr_str
