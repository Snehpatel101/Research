"""
Tests for temporal ordering and leakage prevention in train/val/test splits.

These tests verify:
1. Strict temporal ordering: train < val < test
2. No index overlap between splits
3. Purge removes boundary samples correctly
4. Label end times don't leak across splits
5. Embargo creates proper buffer zones

Author: ML Pipeline
Created: 2025-12-29
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.phase1.stages.splits.core import (
    create_chronological_splits,
    validate_no_overlap,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_timeseries_df():
    """
    Create a sample DataFrame with timestamps for split testing.

    Returns DataFrame with:
    - 1000 samples at 5-minute intervals
    - datetime column for temporal ordering
    - feature and label columns for split validation
    """
    np.random.seed(42)
    n_samples = 1000

    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = pd.date_range(start=start_time, periods=n_samples, freq='5min')

    df = pd.DataFrame({
        'datetime': timestamps,
        'feature': np.random.randn(n_samples),
        'label_h5': np.random.choice([-1, 0, 1], n_samples),
        'label_h20': np.random.choice([-1, 0, 1], n_samples),
    })

    return df


@pytest.fixture
def large_timeseries_df():
    """
    Create a larger DataFrame for stress testing splits.

    Returns DataFrame with 5000 samples for testing edge cases.
    """
    np.random.seed(42)
    n_samples = 5000

    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = pd.date_range(start=start_time, periods=n_samples, freq='5min')

    df = pd.DataFrame({
        'datetime': timestamps,
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'label_h5': np.random.choice([-1, 0, 1], n_samples),
        'label_h20': np.random.choice([-1, 0, 1], n_samples),
    })

    return df


@pytest.fixture
def df_with_label_end_times():
    """
    Create DataFrame with label_end_time for overlap testing.

    Each sample has a label that resolves 20 bars in the future.
    """
    np.random.seed(42)
    n_samples = 1000
    horizon = 20

    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = pd.date_range(start=start_time, periods=n_samples, freq='5min')

    # Label end times: each label is resolved horizon bars later
    label_end_times = pd.Series([
        timestamps[min(i + horizon, n_samples - 1)]
        for i in range(n_samples)
    ])

    df = pd.DataFrame({
        'datetime': timestamps,
        'feature': np.random.randn(n_samples),
        'label_h20': np.random.choice([-1, 0, 1], n_samples),
        'label_end_time': label_end_times,
    })

    return df, horizon


# =============================================================================
# TEST: SPLIT TEMPORAL ORDERING
# =============================================================================

class TestSplitTemporalOrdering:
    """Verify splits maintain strict temporal ordering."""

    def test_train_before_val_before_test(self, sample_timeseries_df):
        """Train indices must be strictly before val, val before test."""
        df = sample_timeseries_df.copy()

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=10,
            embargo_bars=20,
        )

        # Verify ordering: max(train) < min(val) < max(val) < min(test)
        assert train_idx.max() < val_idx.min(), \
            f"Train indices must end before val indices start. " \
            f"train_max={train_idx.max()}, val_min={val_idx.min()}"

        assert val_idx.max() < test_idx.min(), \
            f"Val indices must end before test indices start. " \
            f"val_max={val_idx.max()}, test_min={test_idx.min()}"

    def test_timestamps_strictly_ordered(self, sample_timeseries_df):
        """Timestamps in each split must be strictly ordered."""
        df = sample_timeseries_df.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=10,
            embargo_bars=20,
        )

        train_times = df.iloc[train_idx]['datetime']
        val_times = df.iloc[val_idx]['datetime']
        test_times = df.iloc[test_idx]['datetime']

        # Latest train timestamp must be before earliest val timestamp
        assert train_times.max() < val_times.min(), \
            "Train time range must be before val time range"

        # Latest val timestamp must be before earliest test timestamp
        assert val_times.max() < test_times.min(), \
            "Val time range must be before test time range"

    def test_no_index_overlap(self, sample_timeseries_df):
        """No sample should appear in multiple splits."""
        df = sample_timeseries_df.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=10,
            embargo_bars=20,
        )

        # Convert to sets for overlap checking
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        # Check no overlaps
        train_val_overlap = train_set & val_set
        assert len(train_val_overlap) == 0, \
            f"Found {len(train_val_overlap)} overlapping samples between train and val"

        train_test_overlap = train_set & test_set
        assert len(train_test_overlap) == 0, \
            f"Found {len(train_test_overlap)} overlapping samples between train and test"

        val_test_overlap = val_set & test_set
        assert len(val_test_overlap) == 0, \
            f"Found {len(val_test_overlap)} overlapping samples between val and test"

    def test_validate_no_overlap_utility(self, sample_timeseries_df):
        """Test the validate_no_overlap utility function."""
        df = sample_timeseries_df.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=10,
            embargo_bars=20,
        )

        # Should return True for valid splits
        assert validate_no_overlap(train_idx, val_idx, test_idx) is True

        # Create overlapping indices
        bad_val_idx = np.concatenate([val_idx, train_idx[-5:]])
        assert validate_no_overlap(train_idx, bad_val_idx, test_idx) is False


# =============================================================================
# TEST: PURGE REMOVES BOUNDARY SAMPLES
# =============================================================================

class TestPurgeRemovesBoundarySamples:
    """Verify purge removes samples at split boundaries."""

    def test_purge_creates_gap_before_val(self, sample_timeseries_df):
        """Purge should create a gap between train and val."""
        df = sample_timeseries_df.copy()
        purge_bars = 20
        embargo_bars = 30

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Gap between train end and val start should be at least purge + embargo
        gap_train_val = val_idx.min() - train_idx.max() - 1
        assert gap_train_val >= purge_bars, \
            f"Gap between train and val ({gap_train_val}) should be >= purge_bars ({purge_bars})"

    def test_purge_creates_gap_before_test(self, sample_timeseries_df):
        """Purge should create a gap between val and test."""
        df = sample_timeseries_df.copy()
        purge_bars = 20
        embargo_bars = 30

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Gap between val end and test start should be at least purge + embargo
        gap_val_test = test_idx.min() - val_idx.max() - 1
        assert gap_val_test >= purge_bars, \
            f"Gap between val and test ({gap_val_test}) should be >= purge_bars ({purge_bars})"

    def test_purge_zero_allows_consecutive(self, sample_timeseries_df):
        """With purge=0, splits can still have embargo gap."""
        df = sample_timeseries_df.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=0,
            embargo_bars=50,
        )

        # Should still work with zero purge
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0

        # Embargo gap should still exist
        gap_train_val = val_idx.min() - train_idx.max() - 1
        assert gap_train_val >= 0

    def test_large_purge_reduces_train_size(self, sample_timeseries_df):
        """Larger purge should reduce effective training size."""
        df = sample_timeseries_df.copy()

        # Small purge
        train_small, _, _, _ = create_chronological_splits(
            df, purge_bars=10, embargo_bars=20
        )

        # Large purge
        train_large, _, _, _ = create_chronological_splits(
            df, purge_bars=50, embargo_bars=20
        )

        # Larger purge should result in smaller train set
        assert len(train_large) < len(train_small), \
            "Larger purge should reduce training set size"


# =============================================================================
# TEST: EMBARGO BUFFER ZONES
# =============================================================================

class TestEmbargoBufferZones:
    """Verify embargo creates proper buffer zones between splits."""

    def test_embargo_creates_minimum_gap(self, sample_timeseries_df):
        """Embargo should ensure minimum gap between splits."""
        df = sample_timeseries_df.copy()
        embargo_bars = 100

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=10,
            embargo_bars=embargo_bars,
        )

        # Gap should include embargo
        gap_train_val = val_idx.min() - train_idx.max() - 1
        assert gap_train_val >= embargo_bars, \
            f"Gap ({gap_train_val}) should include embargo ({embargo_bars})"

    def test_embargo_zero_uses_purge_only(self, sample_timeseries_df):
        """With embargo=0, only purge creates gap."""
        df = sample_timeseries_df.copy()
        purge_bars = 20

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=0,
        )

        # Gap should be exactly purge_bars (val starts after purge)
        gap_train_val = val_idx.min() - train_idx.max() - 1
        assert gap_train_val >= purge_bars - 1, \
            f"Gap ({gap_train_val}) should be at least purge_bars-1 ({purge_bars - 1})"

    def test_embargo_metadata_recorded(self, sample_timeseries_df):
        """Metadata should record embargo value."""
        df = sample_timeseries_df.copy()
        embargo_bars = 75

        _, _, _, metadata = create_chronological_splits(
            df,
            purge_bars=10,
            embargo_bars=embargo_bars,
        )

        assert 'embargo_bars' in metadata
        assert metadata['embargo_bars'] == embargo_bars


# =============================================================================
# TEST: LABEL LEAKAGE PREVENTION
# =============================================================================

class TestLabelLeakagePrevention:
    """Verify labels don't leak across splits."""

    def test_samples_with_future_labels_not_in_train(self, df_with_label_end_times):
        """
        Samples whose labels depend on test period data should not be in train.

        This is the key test for label leakage prevention: if a sample's label
        is determined by price action that occurs during the test period, that
        sample must be excluded from training.
        """
        df, horizon = df_with_label_end_times

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=horizon * 2,  # Purge should cover horizon
            embargo_bars=horizon,
        )

        # Get test period start time
        test_start_time = df.iloc[test_idx.min()]['datetime']

        # Get all training samples
        train_df = df.iloc[train_idx]

        # Check that no training sample's label end time is in test period
        train_label_end_times = train_df['label_end_time']
        labels_in_test = train_label_end_times >= test_start_time

        # With proper purge, no training labels should end in test period
        assert not labels_in_test.any(), \
            f"Found {labels_in_test.sum()} training samples with labels " \
            f"that depend on test period data"

    def test_val_samples_not_using_test_data(self, df_with_label_end_times):
        """Validation sample labels should not depend on test period data."""
        df, horizon = df_with_label_end_times

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=horizon * 2,
            embargo_bars=horizon,
        )

        # Get test period start time
        test_start_time = df.iloc[test_idx.min()]['datetime']

        # Get validation samples
        val_df = df.iloc[val_idx]

        # Check that no val sample's label end time is in test period
        val_label_end_times = val_df['label_end_time']
        labels_in_test = val_label_end_times >= test_start_time

        assert not labels_in_test.any(), \
            f"Found {labels_in_test.sum()} validation samples with labels " \
            f"that depend on test period data"

    def test_purge_scales_with_horizon(self):
        """Larger horizons require larger purge to prevent leakage."""
        np.random.seed(42)
        n_samples = 2000

        start_time = datetime(2024, 1, 1, 9, 30)
        timestamps = pd.date_range(start=start_time, periods=n_samples, freq='5min')

        df = pd.DataFrame({
            'datetime': timestamps,
            'feature': np.random.randn(n_samples),
            'label': np.random.choice([-1, 0, 1], n_samples),
        })

        # Test with different purge sizes
        for purge_mult in [1, 2, 3]:
            horizon = 20
            purge_bars = horizon * purge_mult

            train_idx, val_idx, test_idx, _ = create_chronological_splits(
                df,
                purge_bars=purge_bars,
                embargo_bars=50,
            )

            # Verify gap is at least purge_bars
            gap = val_idx.min() - train_idx.max() - 1
            assert gap >= purge_bars, \
                f"Gap ({gap}) should be >= purge_bars ({purge_bars})"


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestSplitEdgeCases:
    """Test edge cases in split creation."""

    def test_small_dataset_raises_on_large_purge(self):
        """Small datasets should raise error with large purge/embargo."""
        np.random.seed(42)
        n_samples = 100  # Small dataset

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
            'feature': np.random.randn(n_samples),
        })

        # Large purge/embargo should fail
        with pytest.raises(ValueError, match="Dataset too small"):
            create_chronological_splits(
                df,
                purge_bars=30,
                embargo_bars=50,  # Total gap > dataset size
            )

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise error."""
        df = pd.DataFrame({
            'datetime': pd.Series(dtype='datetime64[ns]'),
            'feature': pd.Series(dtype=float),
        })

        with pytest.raises(ValueError, match="empty"):
            create_chronological_splits(df)

    def test_unsorted_data_gets_sorted(self):
        """Unsorted data should be sorted before splitting."""
        np.random.seed(42)
        n_samples = 500

        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='5min')

        # Shuffle the data
        shuffle_idx = np.random.permutation(n_samples)

        df = pd.DataFrame({
            'datetime': timestamps[shuffle_idx],
            'feature': np.random.randn(n_samples),
        })

        # Should not raise and should produce valid splits
        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            purge_bars=10,
            embargo_bars=20,
        )

        # Verify temporal ordering is maintained
        assert train_idx.max() < val_idx.min()
        assert val_idx.max() < test_idx.min()

    def test_split_ratios_must_sum_to_one(self):
        """Split ratios not summing to 1.0 should raise error."""
        np.random.seed(42)
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=500, freq='5min'),
            'feature': np.random.randn(500),
        })

        with pytest.raises(ValueError, match="sum to 1"):
            create_chronological_splits(
                df,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum = 1.1
            )

    def test_negative_purge_raises(self):
        """Negative purge should raise error."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=500, freq='5min'),
            'feature': np.random.randn(500),
        })

        with pytest.raises(ValueError, match="non-negative"):
            create_chronological_splits(df, purge_bars=-10)

    def test_metadata_contains_all_fields(self, sample_timeseries_df):
        """Metadata should contain all expected fields."""
        df = sample_timeseries_df.copy()

        _, _, _, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=20,
            embargo_bars=30,
        )

        expected_fields = [
            'total_samples',
            'train_samples',
            'val_samples',
            'test_samples',
            'train_ratio',
            'val_ratio',
            'test_ratio',
            'purge_bars',
            'embargo_bars',
            'train_date_start',
            'train_date_end',
            'val_date_start',
            'val_date_end',
            'test_date_start',
            'test_date_end',
        ]

        for field in expected_fields:
            assert field in metadata, f"Missing metadata field: {field}"


# =============================================================================
# TEST: INTEGRATION WITH LARGE DATASETS
# =============================================================================

class TestSplitIntegration:
    """Integration tests with larger datasets."""

    def test_large_dataset_split_consistency(self, large_timeseries_df):
        """Splits should be consistent across multiple calls with same params."""
        df = large_timeseries_df.copy()

        results = []
        for _ in range(3):
            train_idx, val_idx, test_idx, _ = create_chronological_splits(
                df,
                purge_bars=60,
                embargo_bars=100,
            )
            results.append((train_idx, val_idx, test_idx))

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0][0], results[i][0])
            np.testing.assert_array_equal(results[0][1], results[i][1])
            np.testing.assert_array_equal(results[0][2], results[i][2])

    def test_split_covers_all_data_except_gaps(self, large_timeseries_df):
        """Total samples in splits plus gaps should equal dataset size."""
        df = large_timeseries_df.copy()
        n_total = len(df)

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            purge_bars=30,
            embargo_bars=50,
        )

        n_in_splits = len(train_idx) + len(val_idx) + len(test_idx)
        n_gaps = n_total - n_in_splits

        # Verify metadata is consistent
        assert metadata['train_samples'] == len(train_idx)
        assert metadata['val_samples'] == len(val_idx)
        assert metadata['test_samples'] == len(test_idx)

        # Gaps should be non-negative and reasonable
        assert n_gaps >= 0
        assert n_gaps <= (metadata['purge_bars'] * 2 + metadata['embargo_bars'] * 2)

    def test_split_dates_are_valid(self, large_timeseries_df):
        """Date ranges in metadata should be valid datetime strings."""
        df = large_timeseries_df.copy()

        _, _, _, metadata = create_chronological_splits(
            df,
            purge_bars=30,
            embargo_bars=50,
        )

        # Verify date fields are parseable
        date_fields = [
            'train_date_start', 'train_date_end',
            'val_date_start', 'val_date_end',
            'test_date_start', 'test_date_end',
        ]

        for field in date_fields:
            try:
                pd.Timestamp(metadata[field])
            except Exception as e:
                pytest.fail(f"Could not parse {field}: {metadata[field]}, error: {e}")
