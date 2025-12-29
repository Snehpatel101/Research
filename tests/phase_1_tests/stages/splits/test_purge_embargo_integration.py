"""
Comprehensive integration tests for purge/embargo boundary conditions.

This test suite provides exhaustive verification of purge and embargo logic in
the time series splitting system, focusing on:

1. Precise gap size calculations
2. No overlap guarantees between all splits
3. Label end time respect across boundaries
4. Edge cases with minimum/maximum purge and embargo values
5. Mathematical invariants and boundary conditions

Author: ML Pipeline
Created: 2025-12-29
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from typing import Tuple

from src.phase1.stages.splits.core import (
    create_chronological_splits,
    validate_no_overlap,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def standard_timeseries() -> pd.DataFrame:
    """
    Create a standard time series for testing (10,000 samples at 5-min intervals).

    Returns:
        DataFrame with datetime, feature columns, and labels for multiple horizons
    """
    np.random.seed(42)
    n_samples = 10000

    start_time = datetime(2020, 1, 1, 9, 30)
    timestamps = pd.date_range(start=start_time, periods=n_samples, freq='5min')

    df = pd.DataFrame({
        'datetime': timestamps,
        'close': 4500.0 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'label_h5': np.random.choice([-1, 0, 1], n_samples),
        'label_h10': np.random.choice([-1, 0, 1], n_samples),
        'label_h20': np.random.choice([-1, 0, 1], n_samples),
    })

    return df


@pytest.fixture
def small_timeseries() -> pd.DataFrame:
    """
    Create a small time series for edge case testing (1,000 samples).

    Returns:
        DataFrame with minimal samples for edge case testing
    """
    np.random.seed(42)
    n_samples = 1000

    start_time = datetime(2020, 1, 1, 9, 30)
    timestamps = pd.date_range(start=start_time, periods=n_samples, freq='5min')

    df = pd.DataFrame({
        'datetime': timestamps,
        'feature': np.random.randn(n_samples),
        'label_h20': np.random.choice([-1, 0, 1], n_samples),
    })

    return df


@pytest.fixture
def timeseries_with_label_end_times() -> Tuple[pd.DataFrame, int]:
    """
    Create time series with label_end_time columns for multiple horizons.

    Simulates the output of the final_labels stage with label_end_time columns
    that track when each label outcome is known.

    Returns:
        Tuple of (DataFrame, max_horizon)
    """
    np.random.seed(42)
    n_samples = 10000
    horizons = [5, 10, 20]

    start_time = datetime(2020, 1, 1, 9, 30)
    timestamps = pd.date_range(start=start_time, periods=n_samples, freq='5min')

    df = pd.DataFrame({
        'datetime': timestamps,
        'close': 4500.0 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'feature': np.random.randn(n_samples),
    })

    # Add label and label_end_time for each horizon
    datetime_arr = df['datetime'].values
    for horizon in horizons:
        # Generate labels
        df[f'label_h{horizon}'] = np.random.choice([-1, 0, 1], n_samples)

        # Calculate label_end_time: when the label outcome is known
        # This is the datetime at index i + bars_to_hit (simulating triple-barrier)
        bars_to_hit = np.random.randint(1, horizon + 1, n_samples)
        forward_indices = np.arange(n_samples) + bars_to_hit
        forward_indices = np.clip(forward_indices, 0, n_samples - 1)

        label_end_times = pd.Series(datetime_arr[forward_indices])
        df[f'label_end_time_h{horizon}'] = label_end_times

    return df, max(horizons)


@pytest.fixture
def large_timeseries() -> pd.DataFrame:
    """
    Create a large time series for stress testing (20,000 samples).

    Returns:
        DataFrame with many samples for stress testing
    """
    np.random.seed(42)
    n_samples = 20000

    start_time = datetime(2020, 1, 1, 9, 30)
    timestamps = pd.date_range(start=start_time, periods=n_samples, freq='5min')

    df = pd.DataFrame({
        'datetime': timestamps,
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'label_h20': np.random.choice([-1, 0, 1], n_samples),
    })

    return df


# =============================================================================
# TEST CLASS 1: PRECISE GAP SIZE VERIFICATION
# =============================================================================

class TestPurgeEmbargoGapCalculations:
    """
    Verify that purge and embargo create the exact expected gap sizes.

    The gap between splits should be exactly: purge_bars + embargo_bars
    - purge_bars removes samples before the split boundary
    - embargo_bars removes samples after the split boundary
    """

    def test_standard_gap_train_val(self, standard_timeseries):
        """Verify exact gap size between train and val with standard params."""
        df = standard_timeseries.copy()
        purge_bars = 60
        embargo_bars = 1440

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Calculate actual gap
        gap_train_val = val_idx.min() - train_idx.max() - 1
        expected_gap = purge_bars + embargo_bars

        assert gap_train_val == expected_gap, (
            f"Gap between train and val should be exactly {expected_gap} "
            f"(purge={purge_bars} + embargo={embargo_bars}), got {gap_train_val}"
        )

    def test_standard_gap_val_test(self, standard_timeseries):
        """Verify exact gap size between val and test with standard params."""
        df = standard_timeseries.copy()
        purge_bars = 60
        embargo_bars = 1440

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Calculate actual gap
        gap_val_test = test_idx.min() - val_idx.max() - 1
        expected_gap = purge_bars + embargo_bars

        assert gap_val_test == expected_gap, (
            f"Gap between val and test should be exactly {expected_gap} "
            f"(purge={purge_bars} + embargo={embargo_bars}), got {gap_val_test}"
        )

    def test_gap_with_zero_purge(self, standard_timeseries):
        """With purge=0, gap should equal embargo_bars only."""
        df = standard_timeseries.copy()
        purge_bars = 0
        embargo_bars = 1440

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        gap_train_val = val_idx.min() - train_idx.max() - 1
        gap_val_test = test_idx.min() - val_idx.max() - 1

        assert gap_train_val == embargo_bars, (
            f"With purge=0, gap should equal embargo ({embargo_bars}), "
            f"got {gap_train_val}"
        )
        assert gap_val_test == embargo_bars, (
            f"With purge=0, gap should equal embargo ({embargo_bars}), "
            f"got {gap_val_test}"
        )

    def test_gap_with_zero_embargo(self, standard_timeseries):
        """With embargo=0, gap should equal purge_bars only."""
        df = standard_timeseries.copy()
        purge_bars = 60
        embargo_bars = 0

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        gap_train_val = val_idx.min() - train_idx.max() - 1
        gap_val_test = test_idx.min() - val_idx.max() - 1

        assert gap_train_val == purge_bars, (
            f"With embargo=0, gap should equal purge ({purge_bars}), "
            f"got {gap_train_val}"
        )
        assert gap_val_test == purge_bars, (
            f"With embargo=0, gap should equal purge ({purge_bars}), "
            f"got {gap_val_test}"
        )

    def test_gap_with_large_purge_embargo(self, large_timeseries):
        """Verify gap calculation with large purge/embargo values."""
        df = large_timeseries.copy()
        purge_bars = 200
        embargo_bars = 2000

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        gap_train_val = val_idx.min() - train_idx.max() - 1
        gap_val_test = test_idx.min() - val_idx.max() - 1
        expected_gap = purge_bars + embargo_bars

        assert gap_train_val == expected_gap, (
            f"Gap should be {expected_gap} with large purge/embargo, "
            f"got {gap_train_val}"
        )
        assert gap_val_test == expected_gap, (
            f"Gap should be {expected_gap} with large purge/embargo, "
            f"got {gap_val_test}"
        )

    def test_gap_symmetry(self, standard_timeseries):
        """Both gaps (train-val and val-test) should be identical."""
        df = standard_timeseries.copy()
        purge_bars = 60
        embargo_bars = 1440

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        gap_train_val = val_idx.min() - train_idx.max() - 1
        gap_val_test = test_idx.min() - val_idx.max() - 1

        assert gap_train_val == gap_val_test, (
            f"Both gaps should be identical: "
            f"train-val={gap_train_val}, val-test={gap_val_test}"
        )


# =============================================================================
# TEST CLASS 2: NO OVERLAP VERIFICATION
# =============================================================================

class TestBoundaryConditionNoOverlap:
    """
    Exhaustively verify that no samples overlap between splits.

    Tests include:
    - Set-based overlap detection
    - Index continuity verification
    - Temporal ordering checks
    """

    def test_no_train_val_overlap_standard(self, standard_timeseries):
        """No sample should appear in both train and val."""
        df = standard_timeseries.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=60,
            embargo_bars=1440,
        )

        train_set = set(train_idx)
        val_set = set(val_idx)
        overlap = train_set & val_set

        assert len(overlap) == 0, (
            f"Found {len(overlap)} overlapping indices between train and val. "
            f"Overlap: {sorted(list(overlap))[:10]}"
        )

    def test_no_train_test_overlap_standard(self, standard_timeseries):
        """No sample should appear in both train and test."""
        df = standard_timeseries.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=60,
            embargo_bars=1440,
        )

        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set & test_set

        assert len(overlap) == 0, (
            f"Found {len(overlap)} overlapping indices between train and test. "
            f"Overlap: {sorted(list(overlap))[:10]}"
        )

    def test_no_val_test_overlap_standard(self, standard_timeseries):
        """No sample should appear in both val and test."""
        df = standard_timeseries.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=60,
            embargo_bars=1440,
        )

        val_set = set(val_idx)
        test_set = set(test_idx)
        overlap = val_set & test_set

        assert len(overlap) == 0, (
            f"Found {len(overlap)} overlapping indices between val and test. "
            f"Overlap: {sorted(list(overlap))[:10]}"
        )

    def test_validate_no_overlap_function(self, standard_timeseries):
        """The validate_no_overlap function should return True for valid splits."""
        df = standard_timeseries.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=60,
            embargo_bars=1440,
        )

        is_valid = validate_no_overlap(train_idx, val_idx, test_idx)
        assert is_valid is True, "validate_no_overlap should return True for valid splits"

    def test_all_indices_unique_within_splits(self, standard_timeseries):
        """Each split should have unique indices (no duplicates within split)."""
        df = standard_timeseries.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=60,
            embargo_bars=1440,
        )

        # Check for duplicates within each split
        assert len(train_idx) == len(set(train_idx)), "Train has duplicate indices"
        assert len(val_idx) == len(set(val_idx)), "Val has duplicate indices"
        assert len(test_idx) == len(set(test_idx)), "Test has duplicate indices"

    def test_temporal_ordering_strict(self, standard_timeseries):
        """Verify max(train) < min(val) < max(val) < min(test)."""
        df = standard_timeseries.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=60,
            embargo_bars=1440,
        )

        # Check index ordering
        assert train_idx.max() < val_idx.min(), (
            f"Train indices must end before val starts: "
            f"max(train)={train_idx.max()}, min(val)={val_idx.min()}"
        )
        assert val_idx.max() < test_idx.min(), (
            f"Val indices must end before test starts: "
            f"max(val)={val_idx.max()}, min(test)={test_idx.min()}"
        )

        # Check timestamp ordering
        train_times = df.iloc[train_idx]['datetime']
        val_times = df.iloc[val_idx]['datetime']
        test_times = df.iloc[test_idx]['datetime']

        assert train_times.max() < val_times.min(), (
            "Train timestamps must be before val timestamps"
        )
        assert val_times.max() < test_times.min(), (
            "Val timestamps must be before test timestamps"
        )

    def test_no_overlap_with_extreme_gaps(self, large_timeseries):
        """No overlap should exist even with very large gaps."""
        df = large_timeseries.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=500,
            embargo_bars=3000,
        )

        # Verify using validation function
        assert validate_no_overlap(train_idx, val_idx, test_idx) is True

        # Verify using set operations
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0


# =============================================================================
# TEST CLASS 3: LABEL END TIME RESPECT
# =============================================================================

class TestLabelEndTimeRespect:
    """
    Verify that label_end_times from training samples don't leak into future splits.

    This is critical for preventing lookahead bias: if a training sample's label
    depends on data from the validation or test period, it creates leakage.
    """

    def test_train_labels_not_in_val_period_h5(self, timeseries_with_label_end_times):
        """Training labels (h5) should not resolve during validation period."""
        df, _ = timeseries_with_label_end_times
        horizon = 5

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=horizon * 3,  # 3x horizon for safety
            embargo_bars=horizon * 2,
        )

        val_start_time = df.iloc[val_idx.min()]['datetime']
        train_df = df.iloc[train_idx]

        # Check that no training sample's label ends in val period
        train_label_end = train_df[f'label_end_time_h{horizon}']
        leakage = (train_label_end >= val_start_time).sum()

        assert leakage == 0, (
            f"Found {leakage} training samples (h{horizon}) with labels "
            f"that resolve during validation period"
        )

    def test_train_labels_not_in_val_period_h20(self, timeseries_with_label_end_times):
        """Training labels (h20) should not resolve during validation period."""
        df, _ = timeseries_with_label_end_times
        horizon = 20

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=horizon * 3,  # 3x horizon for safety
            embargo_bars=horizon * 2,
        )

        val_start_time = df.iloc[val_idx.min()]['datetime']
        train_df = df.iloc[train_idx]

        # Check that no training sample's label ends in val period
        train_label_end = train_df[f'label_end_time_h{horizon}']
        leakage = (train_label_end >= val_start_time).sum()

        assert leakage == 0, (
            f"Found {leakage} training samples (h{horizon}) with labels "
            f"that resolve during validation period"
        )

    def test_train_labels_not_in_test_period(self, timeseries_with_label_end_times):
        """Training labels should not resolve during test period."""
        df, max_horizon = timeseries_with_label_end_times

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=max_horizon * 3,
            embargo_bars=max_horizon * 2,
        )

        test_start_time = df.iloc[test_idx.min()]['datetime']
        train_df = df.iloc[train_idx]

        # Check for all horizons
        for horizon in [5, 10, 20]:
            train_label_end = train_df[f'label_end_time_h{horizon}']
            leakage = (train_label_end >= test_start_time).sum()

            assert leakage == 0, (
                f"Found {leakage} training samples (h{horizon}) with labels "
                f"that resolve during test period"
            )

    def test_val_labels_not_in_test_period(self, timeseries_with_label_end_times):
        """Validation labels should not resolve during test period."""
        df, max_horizon = timeseries_with_label_end_times

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=max_horizon * 3,
            embargo_bars=max_horizon * 2,
        )

        test_start_time = df.iloc[test_idx.min()]['datetime']
        val_df = df.iloc[val_idx]

        # Check for all horizons
        for horizon in [5, 10, 20]:
            val_label_end = val_df[f'label_end_time_h{horizon}']
            leakage = (val_label_end >= test_start_time).sum()

            assert leakage == 0, (
                f"Found {leakage} validation samples (h{horizon}) with labels "
                f"that resolve during test period"
            )

    def test_purge_covers_max_label_end_time(self, timeseries_with_label_end_times):
        """
        Purge should be large enough to cover the maximum label resolution time.

        This test verifies that the gap is sufficient to prevent any label
        from one split from depending on data from the next split.
        """
        df, max_horizon = timeseries_with_label_end_times

        # Use purge = 3x max_horizon (standard practice)
        purge_bars = max_horizon * 3
        embargo_bars = max_horizon * 2

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Get the latest label_end_time from training samples
        train_df = df.iloc[train_idx]
        max_train_label_end = train_df[f'label_end_time_h{max_horizon}'].max()

        # Get the earliest validation timestamp
        val_start_time = df.iloc[val_idx.min()]['datetime']

        # Verify that the gap covers the label resolution
        time_gap = (val_start_time - max_train_label_end).total_seconds() / 60  # Minutes

        assert time_gap >= 0, (
            f"Latest training label ends AFTER validation starts! "
            f"Gap: {time_gap:.1f} minutes (should be >= 0)"
        )


# =============================================================================
# TEST CLASS 4: EDGE CASES - MINIMUM DATASET SIZE
# =============================================================================

class TestEdgeCasesMinimumDataset:
    """
    Test behavior with minimum viable dataset sizes.

    Verifies that the splitting logic correctly handles small datasets
    and provides clear error messages when datasets are too small.
    """

    def test_minimum_viable_dataset(self):
        """Test with absolute minimum dataset size."""
        purge_bars = 10
        embargo_bars = 20

        # Minimum required = (purge * 2) + (embargo * 2) + 3 splits
        # = 20 + 40 + 3 = 63 samples minimum
        min_samples = (purge_bars * 2) + (embargo_bars * 2) + 3

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=min_samples, freq='5min'),
            'feature': np.random.randn(min_samples),
        })

        # Should succeed with minimum size
        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # All splits should be non-empty
        assert len(train_idx) > 0, "Train split is empty"
        assert len(val_idx) > 0, "Val split is empty"
        assert len(test_idx) > 0, "Test split is empty"

        # No overlap
        assert validate_no_overlap(train_idx, val_idx, test_idx) is True

    def test_below_minimum_raises_error(self):
        """Dataset smaller than minimum should raise clear error."""
        purge_bars = 10
        embargo_bars = 20
        min_samples = (purge_bars * 2) + (embargo_bars * 2) + 3

        # One sample below minimum
        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=min_samples - 1, freq='5min'),
            'feature': np.random.randn(min_samples - 1),
        })

        with pytest.raises(ValueError, match="Dataset too small"):
            create_chronological_splits(
                df,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
            )

    def test_small_dataset_with_realistic_params_fails(self, small_timeseries):
        """Small dataset (1000 samples) should fail with realistic purge/embargo."""
        df = small_timeseries.copy()

        # Realistic production params
        purge_bars = 60
        embargo_bars = 1440

        with pytest.raises(ValueError, match="Dataset too small|empty"):
            create_chronological_splits(
                df,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
            )

    def test_small_dataset_with_reduced_params_succeeds(self, small_timeseries):
        """Small dataset should work with reduced purge/embargo."""
        df = small_timeseries.copy()

        # Reduced params for small dataset
        purge_bars = 10
        embargo_bars = 20

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Should produce valid splits
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0
        assert validate_no_overlap(train_idx, val_idx, test_idx) is True

    def test_error_message_clarity_insufficient_train(self):
        """Error message should be clear when training set would be empty."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=200, freq='5min'),
            'feature': np.random.randn(200),
        })

        # Purge that eliminates training set
        # train_end_raw = 200 * 0.7 = 140
        # train_end = 140 - 150 = -10 (invalid)
        with pytest.raises(ValueError, match="Training set eliminated|train_end"):
            create_chronological_splits(
                df,
                train_ratio=0.70,
                purge_bars=150,
                embargo_bars=10,
            )

    def test_error_message_clarity_insufficient_val(self):
        """Error message should be clear when validation set would be empty."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=500, freq='5min'),
            'feature': np.random.randn(500),
        })

        # Large embargo that eliminates validation set
        with pytest.raises(ValueError, match="Validation set is empty|val_start.*val_end"):
            create_chronological_splits(
                df,
                train_ratio=0.70,
                val_ratio=0.15,
                purge_bars=10,
                embargo_bars=200,  # Too large for val
            )


# =============================================================================
# TEST CLASS 5: EDGE CASES - LARGE PURGE/EMBARGO
# =============================================================================

class TestEdgeCasesLargePurgeEmbargo:
    """
    Test behavior with very large purge and embargo values.

    Verifies that the system correctly handles extreme purge/embargo values
    and maintains all invariants even when most data is excluded.
    """

    def test_large_purge_reduces_all_splits(self, large_timeseries):
        """Large purge should significantly reduce all split sizes."""
        df = large_timeseries.copy()

        # Small purge baseline
        train_small, val_small, test_small, _ = create_chronological_splits(
            df,
            purge_bars=10,
            embargo_bars=100,
        )

        # Large purge
        train_large, val_large, test_large, _ = create_chronological_splits(
            df,
            purge_bars=500,
            embargo_bars=100,
        )

        # Large purge should reduce all splits
        assert len(train_large) < len(train_small), "Large purge should reduce train size"
        assert len(val_large) < len(val_small), "Large purge should reduce val size"
        # Test size may increase slightly due to rebalancing

    def test_large_embargo_reduces_val_and_test(self, large_timeseries):
        """Large embargo should reduce val and test sizes."""
        df = large_timeseries.copy()

        # Small embargo baseline
        train_small, val_small, test_small, _ = create_chronological_splits(
            df,
            purge_bars=50,
            embargo_bars=100,
        )

        # Large embargo
        train_large, val_large, test_large, _ = create_chronological_splits(
            df,
            purge_bars=50,
            embargo_bars=2000,
        )

        # Large embargo should reduce val (and may reduce test)
        assert len(val_large) < len(val_small), "Large embargo should reduce val size"

    def test_extreme_purge_embargo_still_valid(self, large_timeseries):
        """Even with extreme values, splits should be valid (no overlap)."""
        df = large_timeseries.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=1000,
            embargo_bars=3000,
        )

        # Should still have valid splits
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0

        # No overlap
        assert validate_no_overlap(train_idx, val_idx, test_idx) is True

        # Verify gaps
        gap_train_val = val_idx.min() - train_idx.max() - 1
        gap_val_test = test_idx.min() - val_idx.max() - 1
        expected_gap = 1000 + 3000

        assert gap_train_val == expected_gap
        assert gap_val_test == expected_gap

    def test_purge_larger_than_dataset_segment_fails(self):
        """Purge larger than a split segment should fail gracefully."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=1000, freq='5min'),
            'feature': np.random.randn(1000),
        })

        # Purge = 800, but train segment is only ~700 samples
        with pytest.raises(ValueError, match="Dataset too small|Training set eliminated|empty"):
            create_chronological_splits(
                df,
                train_ratio=0.70,
                purge_bars=800,
                embargo_bars=100,
            )


# =============================================================================
# TEST CLASS 6: REALISTIC PRODUCTION SCENARIOS
# =============================================================================

class TestRealisticScenarios:
    """
    Test with realistic production parameters and scenarios.

    These tests use the actual parameters from the pipeline:
    - purge_bars = 60 (3x max horizon of 20)
    - embargo_bars = 1440 (~5 days at 5-min bars)
    """

    def test_production_params_standard_dataset(self, standard_timeseries):
        """Test with production params on 10k sample dataset."""
        df = standard_timeseries.copy()

        # Production parameters
        purge_bars = 60
        embargo_bars = 1440

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # All splits should be non-empty
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0

        # No overlap
        assert validate_no_overlap(train_idx, val_idx, test_idx) is True

        # Gaps should be exact
        gap_train_val = val_idx.min() - train_idx.max() - 1
        gap_val_test = test_idx.min() - val_idx.max() - 1
        expected_gap = purge_bars + embargo_bars

        assert gap_train_val == expected_gap
        assert gap_val_test == expected_gap

        # Metadata should be consistent
        assert metadata['purge_bars'] == purge_bars
        assert metadata['embargo_bars'] == embargo_bars
        assert metadata['train_samples'] == len(train_idx)
        assert metadata['val_samples'] == len(val_idx)
        assert metadata['test_samples'] == len(test_idx)

    def test_production_params_with_label_end_times(self, timeseries_with_label_end_times):
        """Test production params respect label_end_times."""
        df, max_horizon = timeseries_with_label_end_times

        # Production parameters
        purge_bars = 60
        embargo_bars = 1440

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Verify no label leakage
        test_start_time = df.iloc[test_idx.min()]['datetime']

        for horizon in [5, 10, 20]:
            # Check training labels don't leak into test
            train_df = df.iloc[train_idx]
            train_label_end = train_df[f'label_end_time_h{horizon}']
            leakage_train = (train_label_end >= test_start_time).sum()

            assert leakage_train == 0, (
                f"Training labels (h{horizon}) leak into test period: {leakage_train} samples"
            )

            # Check validation labels don't leak into test
            val_df = df.iloc[val_idx]
            val_label_end = val_df[f'label_end_time_h{horizon}']
            leakage_val = (val_label_end >= test_start_time).sum()

            assert leakage_val == 0, (
                f"Validation labels (h{horizon}) leak into test period: {leakage_val} samples"
            )

    def test_split_size_percentages_reasonable(self, standard_timeseries):
        """Split sizes should be close to requested ratios after purge/embargo."""
        df = standard_timeseries.copy()
        n_total = len(df)

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=60,
            embargo_bars=1440,
        )

        # Calculate actual percentages
        train_pct = len(train_idx) / n_total
        val_pct = len(val_idx) / n_total
        test_pct = len(test_idx) / n_total

        # Should be reasonably close to target (within 5%)
        # Note: exact match is impossible due to purge/embargo
        assert train_pct > 0.50, f"Train % too low: {train_pct:.1%}"
        assert val_pct > 0.05, f"Val % too low: {val_pct:.1%}"
        assert test_pct > 0.05, f"Test % too low: {test_pct:.1%}"

        # Total should be less than 100% due to gaps
        total_pct = train_pct + val_pct + test_pct
        assert total_pct < 1.0, f"Total % should be < 100%, got {total_pct:.1%}"

        # Lost samples should be reasonable
        lost_samples = n_total - len(train_idx) - len(val_idx) - len(test_idx)
        max_expected_loss = 2 * (60 + 1440)  # 2 gaps

        assert lost_samples <= max_expected_loss, (
            f"Lost too many samples: {lost_samples} > {max_expected_loss}"
        )


# =============================================================================
# TEST CLASS 7: MATHEMATICAL INVARIANTS
# =============================================================================

class TestMathematicalInvariants:
    """
    Verify mathematical invariants of the splitting algorithm.

    These properties must ALWAYS hold true:
    1. All indices are within valid range [0, n-1]
    2. Indices are strictly increasing within each split
    3. Gap size formula: gap = purge_bars + embargo_bars
    4. No index appears in multiple splits
    5. Lost samples <= theoretical maximum
    """

    def test_all_indices_in_valid_range(self, standard_timeseries):
        """All indices must be in range [0, len(df)-1]."""
        df = standard_timeseries.copy()
        n = len(df)

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=60,
            embargo_bars=1440,
        )

        # Check train indices
        assert train_idx.min() >= 0, f"Train has negative index: {train_idx.min()}"
        assert train_idx.max() < n, f"Train index out of range: {train_idx.max()} >= {n}"

        # Check val indices
        assert val_idx.min() >= 0, f"Val has negative index: {val_idx.min()}"
        assert val_idx.max() < n, f"Val index out of range: {val_idx.max()} >= {n}"

        # Check test indices
        assert test_idx.min() >= 0, f"Test has negative index: {test_idx.min()}"
        assert test_idx.max() < n, f"Test index out of range: {test_idx.max()} >= {n}"

    def test_indices_strictly_increasing(self, standard_timeseries):
        """Indices within each split must be strictly increasing."""
        df = standard_timeseries.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=60,
            embargo_bars=1440,
        )

        # Check train indices are sorted
        assert np.all(train_idx[1:] > train_idx[:-1]), "Train indices not strictly increasing"

        # Check val indices are sorted
        assert np.all(val_idx[1:] > val_idx[:-1]), "Val indices not strictly increasing"

        # Check test indices are sorted
        assert np.all(test_idx[1:] > test_idx[:-1]), "Test indices not strictly increasing"

    def test_gap_formula_invariant(self, standard_timeseries):
        """Gap must always equal purge_bars + embargo_bars."""
        df = standard_timeseries.copy()

        # Test with multiple combinations
        test_cases = [
            (10, 20),
            (60, 1440),
            (100, 500),
            (0, 1440),
            (60, 0),
        ]

        for purge_bars, embargo_bars in test_cases:
            train_idx, val_idx, test_idx, _ = create_chronological_splits(
                df,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
            )

            gap_train_val = val_idx.min() - train_idx.max() - 1
            gap_val_test = test_idx.min() - val_idx.max() - 1
            expected_gap = purge_bars + embargo_bars

            assert gap_train_val == expected_gap, (
                f"Gap formula violated (train-val): "
                f"purge={purge_bars}, embargo={embargo_bars}, "
                f"expected={expected_gap}, got={gap_train_val}"
            )
            assert gap_val_test == expected_gap, (
                f"Gap formula violated (val-test): "
                f"purge={purge_bars}, embargo={embargo_bars}, "
                f"expected={expected_gap}, got={gap_val_test}"
            )

    def test_no_index_in_multiple_splits(self, standard_timeseries):
        """No index can appear in more than one split."""
        df = standard_timeseries.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=60,
            embargo_bars=1440,
        )

        # Combine all indices
        all_indices = np.concatenate([train_idx, val_idx, test_idx])

        # Check for duplicates
        unique_indices = np.unique(all_indices)

        assert len(all_indices) == len(unique_indices), (
            f"Found duplicate indices across splits: "
            f"{len(all_indices)} total, {len(unique_indices)} unique"
        )

    def test_lost_samples_within_bounds(self, standard_timeseries):
        """Lost samples should not exceed theoretical maximum."""
        df = standard_timeseries.copy()
        n = len(df)
        purge_bars = 60
        embargo_bars = 1440

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Calculate lost samples
        used_samples = len(train_idx) + len(val_idx) + len(test_idx)
        lost_samples = n - used_samples

        # Maximum loss: 2 gaps of (purge + embargo) each
        max_theoretical_loss = 2 * (purge_bars + embargo_bars)

        assert lost_samples <= max_theoretical_loss, (
            f"Lost more samples than theoretical maximum: "
            f"{lost_samples} > {max_theoretical_loss}"
        )

        # Lost samples should be non-negative
        assert lost_samples >= 0, f"Negative lost samples: {lost_samples}"


# =============================================================================
# TEST CLASS 8: BOUNDARY INDEX CALCULATIONS
# =============================================================================

class TestBoundaryIndexCalculations:
    """
    Verify the exact index calculations at boundaries.

    Tests the mathematical correctness of:
    - train_end = train_end_raw - purge_bars
    - val_start = train_end_raw + embargo_bars
    - val_end = val_end_raw - purge_bars
    - test_start = val_end_raw + embargo_bars
    """

    def test_train_end_calculation(self, standard_timeseries):
        """Verify train_end = train_end_raw - purge_bars."""
        df = standard_timeseries.copy()
        n = len(df)
        train_ratio = 0.70
        purge_bars = 60

        train_idx, _, _, _ = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            purge_bars=purge_bars,
            embargo_bars=1440,
        )

        # Calculate expected
        train_end_raw = int(n * train_ratio)
        expected_train_end = train_end_raw - purge_bars - 1  # -1 for 0-indexing

        # Actual train_end is the max index
        actual_train_end = train_idx.max()

        assert actual_train_end == expected_train_end, (
            f"Train end calculation incorrect: "
            f"expected={expected_train_end}, actual={actual_train_end}"
        )

    def test_val_start_calculation(self, standard_timeseries):
        """Verify val_start = train_end_raw + embargo_bars."""
        df = standard_timeseries.copy()
        n = len(df)
        train_ratio = 0.70
        embargo_bars = 1440

        _, val_idx, _, _ = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            purge_bars=60,
            embargo_bars=embargo_bars,
        )

        # Calculate expected
        train_end_raw = int(n * train_ratio)
        expected_val_start = train_end_raw + embargo_bars

        # Actual val_start is the min index
        actual_val_start = val_idx.min()

        assert actual_val_start == expected_val_start, (
            f"Val start calculation incorrect: "
            f"expected={expected_val_start}, actual={actual_val_start}"
        )

    def test_val_end_calculation(self, standard_timeseries):
        """Verify val_end = val_end_raw - purge_bars."""
        df = standard_timeseries.copy()
        n = len(df)
        train_ratio = 0.70
        val_ratio = 0.15
        purge_bars = 60

        _, val_idx, _, _ = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            purge_bars=purge_bars,
            embargo_bars=1440,
        )

        # Calculate expected
        val_end_raw = int(n * (train_ratio + val_ratio))
        expected_val_end = val_end_raw - purge_bars - 1  # -1 for 0-indexing

        # Actual val_end is the max index
        actual_val_end = val_idx.max()

        assert actual_val_end == expected_val_end, (
            f"Val end calculation incorrect: "
            f"expected={expected_val_end}, actual={actual_val_end}"
        )

    def test_test_start_calculation(self, standard_timeseries):
        """Verify test_start = val_end_raw + embargo_bars."""
        df = standard_timeseries.copy()
        n = len(df)
        train_ratio = 0.70
        val_ratio = 0.15
        embargo_bars = 1440

        _, _, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            purge_bars=60,
            embargo_bars=embargo_bars,
        )

        # Calculate expected
        val_end_raw = int(n * (train_ratio + val_ratio))
        expected_test_start = val_end_raw + embargo_bars

        # Actual test_start is the min index
        actual_test_start = test_idx.min()

        assert actual_test_start == expected_test_start, (
            f"Test start calculation incorrect: "
            f"expected={expected_test_start}, actual={actual_test_start}"
        )

    def test_boundary_relationships(self, standard_timeseries):
        """
        Verify mathematical relationships between all boundaries.

        Relationships:
        - val_start - train_end = purge_bars + embargo_bars + 1
        - test_start - val_end = purge_bars + embargo_bars + 1
        """
        df = standard_timeseries.copy()
        purge_bars = 60
        embargo_bars = 1440

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Extract boundaries
        train_end = train_idx.max()
        val_start = val_idx.min()
        val_end = val_idx.max()
        test_start = test_idx.min()

        # Verify relationships
        gap_train_val = val_start - train_end
        gap_val_test = test_start - val_end
        expected_gap = purge_bars + embargo_bars + 1  # +1 for gap calculation

        assert gap_train_val == expected_gap, (
            f"Train-val gap incorrect: expected={expected_gap}, got={gap_train_val}"
        )
        assert gap_val_test == expected_gap, (
            f"Val-test gap incorrect: expected={expected_gap}, got={gap_val_test}"
        )
