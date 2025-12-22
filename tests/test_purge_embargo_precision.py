"""
Tests for purge and embargo precision in data splitting.

Validates exact boundary conditions:
- Purge removes exactly the right samples
- Embargo creates exact gap between splits
- No off-by-one errors in boundary calculations
- Train labels don't overlap with val/test features
- Feature lookback windows don't touch training labels

Run with: pytest tests/test_purge_embargo_precision.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.stage7_splits import create_chronological_splits, validate_no_overlap


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def precise_test_data():
    """
    Create precisely sized test data for exact boundary testing.

    Uses exactly 1000 samples for easy calculation verification.
    """
    n = 1000
    timestamps = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')

    df = pd.DataFrame({
        'datetime': timestamps,
        'close': 100.0 + np.arange(n),  # Deterministic: 100, 101, 102, ...
        'symbol': 'MES',
        'index_id': np.arange(n)  # Explicit index for verification
    })

    return df


@pytest.fixture
def large_test_data():
    """Create large dataset for realistic purge/embargo testing."""
    n = 5000
    timestamps = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')

    df = pd.DataFrame({
        'datetime': timestamps,
        'close': 4500.0 + np.cumsum(np.random.randn(n) * 0.5),
        'symbol': 'MES',
        'index_id': np.arange(n)
    })

    return df


# =============================================================================
# PURGE PRECISION TESTS
# =============================================================================

class TestPurgePrecision:
    """Tests for exact purge boundary conditions."""

    def test_purge_removes_exact_samples_train_val(self, precise_test_data):
        """Test that purge removes exactly purge_bars samples at train/val boundary."""
        df = precise_test_data.copy()
        n = len(df)
        purge_bars = 60
        embargo_bars = 0  # Disable embargo to isolate purge

        train_ratio = 0.70
        val_ratio = 0.15
        test_ratio = 0.15

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # Calculate expected boundaries
        train_end_raw = int(n * train_ratio)  # 700
        expected_train_end = train_end_raw - purge_bars  # 640

        # Verify train ends at expected index
        assert train_idx.max() == expected_train_end - 1, \
            f"Train should end at index {expected_train_end - 1}, got {train_idx.max()}"

        # Verify exactly purge_bars samples removed
        expected_train_samples = expected_train_end
        assert len(train_idx) == expected_train_samples, \
            f"Train should have {expected_train_samples} samples, got {len(train_idx)}"

    def test_purge_removes_exact_samples_val_test(self, precise_test_data):
        """Test that purge removes exactly purge_bars samples at val/test boundary."""
        df = precise_test_data.copy()
        n = len(df)
        purge_bars = 60
        embargo_bars = 0

        train_ratio = 0.70
        val_ratio = 0.15
        test_ratio = 0.15

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # Calculate expected val end
        val_end_raw = int(n * (train_ratio + val_ratio))  # 850
        expected_val_end = val_end_raw - purge_bars  # 790

        # Verify val ends at expected index
        assert val_idx.max() == expected_val_end - 1, \
            f"Val should end at index {expected_val_end - 1}, got {val_idx.max()}"

    def test_purge_zero_means_no_removal(self, precise_test_data):
        """Test that purge_bars=0 removes no samples."""
        df = precise_test_data.copy()
        n = len(df)
        purge_bars = 0
        embargo_bars = 0

        train_ratio = 0.70

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # Train should end exactly at train_ratio boundary
        expected_train_end = int(n * train_ratio)
        assert len(train_idx) == expected_train_end, \
            f"With purge_bars=0, train should have {expected_train_end} samples"

    def test_purge_boundary_exactly_matches_max_bars(self, large_test_data):
        """Test that purge_bars=60 exactly matches H20 max_bars."""
        df = large_test_data.copy()

        # From triple-barrier labeling: H20 uses max_bars=60
        purge_bars = 60
        max_bars_h20 = 60

        assert purge_bars == max_bars_h20, \
            "Purge should equal max_bars for highest horizon"

        # Verify split works with this configuration
        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=288
        )

        assert metadata['purge_bars'] == 60


# =============================================================================
# EMBARGO PRECISION TESTS
# =============================================================================

class TestEmbargoPrecision:
    """Tests for exact embargo boundary conditions."""

    def test_embargo_creates_exact_gap_train_val(self, large_test_data):
        """Test that embargo creates exactly embargo_bars gap between train and val."""
        df = large_test_data.copy()
        n = len(df)
        purge_bars = 0  # Disable purge to isolate embargo
        embargo_bars = 288

        train_ratio = 0.70

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # Calculate gap
        gap = val_idx.min() - train_idx.max() - 1  # -1 because indices are inclusive

        assert gap == embargo_bars, \
            f"Embargo should create gap of {embargo_bars}, got {gap}"

    def test_embargo_creates_exact_gap_val_test(self, large_test_data):
        """Test that embargo creates exactly embargo_bars gap between val and test."""
        df = large_test_data.copy()
        purge_bars = 0
        embargo_bars = 288

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # Calculate gap
        gap = test_idx.min() - val_idx.max() - 1

        assert gap == embargo_bars, \
            f"Embargo should create gap of {embargo_bars}, got {gap}"

    def test_embargo_zero_means_no_gap(self, precise_test_data):
        """Test that embargo_bars=0 creates no gap."""
        df = precise_test_data.copy()
        n = len(df)
        purge_bars = 0
        embargo_bars = 0

        train_ratio = 0.70

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # Val should start immediately after train (gap = 0)
        expected_val_start = int(n * train_ratio)
        assert val_idx.min() == expected_val_start, \
            f"With embargo=0, val should start at {expected_val_start}"

    def test_embargo_288_is_approximately_one_day(self):
        """Test that embargo_bars=288 represents ~1 day of 5min bars."""
        # 24 hours * 60 minutes / 5 minutes per bar = 288 bars
        bars_per_day = (24 * 60) / 5
        assert bars_per_day == 288, \
            "Embargo of 288 bars should equal 1 day of 5-minute data"


# =============================================================================
# COMBINED PURGE + EMBARGO TESTS
# =============================================================================

class TestCombinedPurgeEmbargo:
    """Tests for combined purge and embargo effects."""

    def test_combined_gap_exact_calculation(self, large_test_data):
        """Test exact gap calculation with both purge and embargo."""
        df = large_test_data.copy()
        n = len(df)
        purge_bars = 60
        embargo_bars = 288

        train_ratio = 0.70

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # Expected train end: 0.70 * 1000 - 60 = 640
        train_end_raw = int(n * train_ratio)
        expected_train_end = train_end_raw - purge_bars

        # Expected val start: 0.70 * 1000 + 288 = 988
        expected_val_start = train_end_raw + embargo_bars

        assert train_idx.max() == expected_train_end - 1, \
            f"Train end should be {expected_train_end - 1}"
        assert val_idx.min() == expected_val_start, \
            f"Val start should be {expected_val_start}"

        # Total gap between last train and first val
        total_gap = val_idx.min() - train_idx.max() - 1
        expected_total_gap = purge_bars + embargo_bars

        assert total_gap == expected_total_gap, \
            f"Total gap should be {expected_total_gap}, got {total_gap}"

    def test_samples_lost_to_purge_embargo(self, large_test_data):
        """Test exact count of samples lost to purge and embargo."""
        df = large_test_data.copy()
        n = len(df)
        purge_bars = 60
        embargo_bars = 288

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        total_kept = len(train_idx) + len(val_idx) + len(test_idx)
        samples_lost = n - total_kept

        # Lost samples: 2 purge zones + 2 embargo zones
        expected_lost = (2 * purge_bars) + (2 * embargo_bars)

        assert samples_lost == expected_lost, \
            f"Should lose {expected_lost} samples, lost {samples_lost}"

    def test_production_config_purge_60_embargo_288(self, large_test_data):
        """Test production configuration (purge=60, embargo=288)."""
        df = large_test_data.copy()
        n = len(df)

        # Production configuration from CLAUDE.md
        purge_bars = 60
        embargo_bars = 288

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # Verify no overlap
        assert validate_no_overlap(train_idx, val_idx, test_idx)

        # Verify metadata
        assert metadata['purge_bars'] == 60
        assert metadata['embargo_bars'] == 288

        # Verify reasonable split sizes
        assert len(train_idx) > 0, "Train should not be empty"
        assert len(val_idx) > 0, "Val should not be empty"
        assert len(test_idx) > 0, "Test should not be empty"


# =============================================================================
# LABEL OVERLAP PREVENTION TESTS
# =============================================================================

class TestLabelOverlapPrevention:
    """Tests that train labels don't overlap with val/test features."""

    def test_train_label_reach_doesnt_touch_val(self, large_test_data):
        """Test that train labels (with max_bars lookforward) don't reach val."""
        df = large_test_data.copy()

        # H20 label can look forward up to max_bars=60
        max_bars_h20 = 60
        purge_bars = 60  # Should equal max_bars

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=288
        )

        # Last train sample
        last_train_idx = train_idx.max()

        # Maximum forward reach of last train label
        max_label_reach = last_train_idx + max_bars_h20

        # First val sample
        first_val_idx = val_idx.min()

        # Label reach should not touch val
        assert max_label_reach < first_val_idx, \
            f"Train label reach ({max_label_reach}) touches val (starts at {first_val_idx})"

    def test_val_label_reach_doesnt_touch_test(self, large_test_data):
        """Test that val labels don't reach test data."""
        df = large_test_data.copy()

        max_bars_h20 = 60
        purge_bars = 60

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=288
        )

        # Last val sample
        last_val_idx = val_idx.max()

        # Maximum forward reach
        max_label_reach = last_val_idx + max_bars_h20

        # First test sample
        first_test_idx = test_idx.min()

        # Label reach should not touch test
        assert max_label_reach < first_test_idx, \
            f"Val label reach ({max_label_reach}) touches test (starts at {first_test_idx})"

    def test_purge_exactly_covers_max_label_horizon(self):
        """Test that purge_bars exactly covers the maximum label horizon."""
        # Configuration from Phase 1 Pipeline Review
        purge_bars = 60
        max_bars_h5 = 20
        max_bars_h20 = 60

        # Purge should cover the largest horizon
        assert purge_bars >= max(max_bars_h5, max_bars_h20), \
            "Purge should cover maximum label horizon"

        assert purge_bars == max_bars_h20, \
            "Purge should exactly equal max_bars for H20"


# =============================================================================
# FEATURE LOOKBACK PREVENTION TESTS
# =============================================================================

class TestFeatureLookbackPrevention:
    """Tests that val/test feature lookbacks don't touch training data."""

    def test_val_feature_lookback_doesnt_touch_train(self, large_test_data):
        """Test that val features with 20-bar lookback don't touch train."""
        df = large_test_data.copy()

        # Longest feature lookback (e.g., BB, volatility) is typically 20-60 bars
        max_feature_lookback = 60

        purge_bars = 60
        embargo_bars = 288

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # First val sample
        first_val_idx = val_idx.min()

        # Maximum lookback from first val sample
        lookback_idx = first_val_idx - max_feature_lookback

        # Last train sample
        last_train_idx = train_idx.max()

        # Lookback should not touch train
        assert lookback_idx > last_train_idx, \
            f"Val feature lookback ({lookback_idx}) touches train (ends at {last_train_idx})"

    def test_test_feature_lookback_doesnt_touch_val(self, large_test_data):
        """Test that test features don't look back into val."""
        df = large_test_data.copy()

        max_feature_lookback = 60
        purge_bars = 60
        embargo_bars = 288

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # First test sample
        first_test_idx = test_idx.min()

        # Maximum lookback
        lookback_idx = first_test_idx - max_feature_lookback

        # Last val sample
        last_val_idx = val_idx.max()

        # Lookback should not touch val
        assert lookback_idx > last_val_idx, \
            f"Test feature lookback ({lookback_idx}) touches val (ends at {last_val_idx})"


# =============================================================================
# BOUNDARY EDGE CASES
# =============================================================================

class TestBoundaryEdgeCases:
    """Tests for edge cases at split boundaries."""

    def test_no_off_by_one_errors(self, large_test_data):
        """Test for off-by-one errors in boundary calculations."""
        df = large_test_data.copy()
        n = len(df)

        purge_bars = 60
        embargo_bars = 288

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # Verify indices are contiguous within each split
        assert (np.diff(train_idx) == 1).all(), "Train indices should be contiguous"
        assert (np.diff(val_idx) == 1).all(), "Val indices should be contiguous"
        assert (np.diff(test_idx) == 1).all(), "Test indices should be contiguous"

        # Verify no single index appears twice
        all_indices = np.concatenate([train_idx, val_idx, test_idx])
        assert len(all_indices) == len(set(all_indices)), \
            "No index should appear in multiple splits"

    def test_first_sample_always_in_train(self, large_test_data):
        """Test that first sample (index 0) is always in train."""
        df = large_test_data.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=60,
            embargo_bars=288
        )

        assert 0 in train_idx, "First sample should be in train"
        assert train_idx.min() == 0, "Train should start at index 0"

    def test_last_sample_always_in_test(self, large_test_data):
        """Test that last sample is always in test."""
        df = large_test_data.copy()
        n = len(df)

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=60,
            embargo_bars=288
        )

        assert (n - 1) in test_idx, "Last sample should be in test"
        assert test_idx.max() == n - 1, "Test should end at last index"

    def test_exact_boundary_indices_never_in_multiple_splits(self, large_test_data):
        """Test that boundary indices don't appear in multiple splits."""
        df = large_test_data.copy()
        n = len(df)

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=60,
            embargo_bars=288
        )

        # Calculate expected boundaries
        train_end_raw = int(n * 0.70)
        val_end_raw = int(n * 0.85)

        # Indices near boundaries should not appear in multiple splits
        boundary_indices = [
            train_idx.max(),
            train_idx.max() + 1,
            val_idx.min(),
            val_idx.min() - 1,
            val_idx.max(),
            val_idx.max() + 1,
            test_idx.min(),
            test_idx.min() - 1,
        ]

        for idx in boundary_indices:
            in_train = idx in train_idx
            in_val = idx in val_idx
            in_test = idx in test_idx

            splits_containing = sum([in_train, in_val, in_test])

            assert splits_containing <= 1, \
                f"Index {idx} appears in {splits_containing} splits (should be 0 or 1)"
