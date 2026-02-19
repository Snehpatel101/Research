"""
Tests for PurgedKFold cross-validation.

Verifies no data leakage via purge/embargo gaps, correct fold counts,
full index coverage, and config validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def large_df() -> pd.DataFrame:
    """DataFrame large enough for 5-fold CV with generous purge/embargo."""
    rng = np.random.default_rng(42)
    n = 10_000
    return pd.DataFrame(
        {"feature_1": rng.standard_normal(n), "feature_2": rng.standard_normal(n)},
        index=pd.date_range("2020-01-01", periods=n, freq="5min"),
    )


@pytest.fixture
def small_config() -> PurgedKFoldConfig:
    """Config with small purge/embargo for easier testing."""
    return PurgedKFoldConfig(
        n_splits=5,
        purge_bars=10,
        embargo_bars=20,
        min_train_size=0.3,
    )


# ---------------------------------------------------------------------------
# Core CV correctness
# ---------------------------------------------------------------------------


class TestNoTrainTestOverlap:
    """Train and test indices must never overlap in any fold."""

    def test_no_overlap(self, large_df: pd.DataFrame, small_config: PurgedKFoldConfig) -> None:
        cv = PurgedKFold(small_config)
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(large_df)):
            overlap = np.intersect1d(train_idx, test_idx)
            assert (
                len(overlap) == 0
            ), f"Fold {fold_idx}: train/test overlap at indices {overlap[:5]}"


class TestPurgeGap:
    """A gap of at least purge_bars must exist between training data and the
    start of the test set (for samples before the test set)."""

    def test_purge_gap_exists(
        self, large_df: pd.DataFrame, small_config: PurgedKFoldConfig
    ) -> None:
        cv = PurgedKFold(small_config)
        purge = small_config.purge_bars

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(large_df)):
            test_start = test_idx[0]
            # Training indices before the test set
            train_before = train_idx[train_idx < test_start]
            if len(train_before) == 0:
                continue
            last_train_before = train_before[-1]
            gap = test_start - last_train_before - 1
            assert gap >= purge - 1, f"Fold {fold_idx}: purge gap is {gap}, expected >= {purge - 1}"


class TestEmbargoGap:
    """A gap of at least embargo_bars must exist after the test set before
    training data resumes."""

    def test_embargo_gap_exists(
        self, large_df: pd.DataFrame, small_config: PurgedKFoldConfig
    ) -> None:
        cv = PurgedKFold(small_config)
        embargo = small_config.embargo_bars

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(large_df)):
            test_end = test_idx[-1]
            # Training indices after the test set
            train_after = train_idx[train_idx > test_end]
            if len(train_after) == 0:
                continue
            first_train_after = train_after[0]
            gap = first_train_after - test_end - 1
            assert (
                gap >= embargo - 1
            ), f"Fold {fold_idx}: embargo gap is {gap}, expected >= {embargo - 1}"


class TestNSplits:
    """Number of yielded folds must match config.n_splits."""

    @pytest.mark.parametrize("n_splits", [2, 3, 5])
    def test_fold_count(self, large_df: pd.DataFrame, n_splits: int) -> None:
        config = PurgedKFoldConfig(n_splits=n_splits, purge_bars=10, embargo_bars=20)
        cv = PurgedKFold(config)
        folds = list(cv.split(large_df))
        assert len(folds) == n_splits

    def test_get_n_splits(self, large_df: pd.DataFrame, small_config: PurgedKFoldConfig) -> None:
        cv = PurgedKFold(small_config)
        assert cv.get_n_splits() == small_config.n_splits


class TestCoverage:
    """Every index must appear in at least one test fold."""

    def test_all_indices_covered(
        self, large_df: pd.DataFrame, small_config: PurgedKFoldConfig
    ) -> None:
        cv = PurgedKFold(small_config)
        coverage = cv.validate_coverage(large_df)
        assert coverage["coverage_fraction"] == pytest.approx(1.0, abs=0.01)
        assert coverage["uncovered_samples"] <= 1  # at most rounding residual


class TestConfigValidation:
    """Invalid configuration parameters must raise ValueError."""

    def test_n_splits_below_2(self) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            PurgedKFoldConfig(n_splits=1)

    def test_negative_purge(self) -> None:
        with pytest.raises(ValueError, match="purge_bars"):
            PurgedKFoldConfig(purge_bars=-1)

    def test_negative_embargo(self) -> None:
        with pytest.raises(ValueError, match="embargo_bars"):
            PurgedKFoldConfig(embargo_bars=-5)

    def test_min_train_size_zero(self) -> None:
        with pytest.raises(ValueError, match="min_train_size"):
            PurgedKFoldConfig(min_train_size=0)

    def test_min_train_size_one(self) -> None:
        with pytest.raises(ValueError, match="min_train_size"):
            PurgedKFoldConfig(min_train_size=1)

    def test_too_many_splits_for_data(self) -> None:
        """n_splits too large relative to data size should raise."""
        small_df = pd.DataFrame(
            {"a": range(50)},
            index=pd.date_range("2020-01-01", periods=50, freq="5min"),
        )
        config = PurgedKFoldConfig(n_splits=5, purge_bars=10, embargo_bars=20, min_train_size=0.3)
        cv = PurgedKFold(config)
        with pytest.raises(ValueError, match="too large"):
            list(cv.split(small_df))


class TestFromHorizons:
    """Factory method from_horizons computes purge_bars correctly."""

    def test_default_multiplier(self) -> None:
        config = PurgedKFoldConfig.from_horizons([5, 10, 20, 60, 120])
        assert config.purge_bars == 360  # 120 * 3

    def test_custom_multiplier(self) -> None:
        config = PurgedKFoldConfig.from_horizons([5, 10, 20], purge_multiplier=2)
        assert config.purge_bars == 40  # 20 * 2

    def test_empty_horizons_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            PurgedKFoldConfig.from_horizons([])

    def test_negative_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PurgedKFoldConfig.from_horizons([5, -1])
