"""
Unit tests for Stage 7: Data Splitting.

Time-based splits with purge and embargo

Run with: pytest tests/phase_1_tests/stages/test_stage7_*.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.stage7_splits import create_chronological_splits, validate_no_overlap, create_splits


# =============================================================================
# TESTS
# =============================================================================

class TestStage7DataSplitter:
    """Tests for Stage 7: Time-Based Splitting."""

    def test_split_ratios_sum_to_one(self, sample_labeled_data):
        """Test that split ratios must sum to 1.0."""
        df = sample_labeled_data

        # Valid ratios with small purge/embargo for 1000 row dataset
        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20  # Use smaller values for test data size
        )

        assert metadata is not None, "Should succeed with valid ratios"

    def test_split_ratios_validation_error(self, sample_labeled_data):
        """Test that invalid ratios raise error."""
        df = sample_labeled_data

        with pytest.raises(ValueError, match="sum to 1.0"):
            create_chronological_splits(
                df, train_ratio=0.50, val_ratio=0.20, test_ratio=0.20
            )

    def test_splits_chronological_order(self, sample_labeled_data):
        """Test that splits maintain chronological order."""
        df = sample_labeled_data

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # Train should come before val
        assert train_idx.max() < val_idx.min(), "Train should precede validation"
        # Val should come before test
        assert val_idx.max() < test_idx.min(), "Validation should precede test"

    def test_no_overlap_between_splits(self, sample_labeled_data):
        """Test that there is no overlap between splits."""
        df = sample_labeled_data

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # Verify no overlap
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        assert len(train_set & val_set) == 0, "Train/val overlap"
        assert len(train_set & test_set) == 0, "Train/test overlap"
        assert len(val_set & test_set) == 0, "Val/test overlap"

    def test_purge_removes_correct_samples(self, sample_labeled_data):
        """Test that purging removes samples at boundaries."""
        df = sample_labeled_data
        n = len(df)
        purge_bars = 60
        embargo_bars = 0  # Disable embargo for this test

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=purge_bars, embargo_bars=embargo_bars
        )

        # Calculate expected train end
        expected_train_end_raw = int(n * 0.70)
        expected_train_end = expected_train_end_raw - purge_bars

        # Train should end before the raw split point
        assert train_idx.max() < expected_train_end_raw, \
            "Purging should remove samples before split"

    def test_embargo_creates_gap(self, sample_labeled_data):
        """Test that embargo creates gap between splits."""
        df = sample_labeled_data
        purge_bars = 0  # Disable purge for this test
        embargo_bars = 50

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=purge_bars, embargo_bars=embargo_bars
        )

        # Gap between train and val should be at least embargo_bars
        gap_train_val = val_idx.min() - train_idx.max()
        assert gap_train_val >= embargo_bars, \
            f"Gap should be >= {embargo_bars}, got {gap_train_val}"

    def test_purge_value_matches_max_horizon(self):
        """Test that purge with max_bars=60 (H20) prevents leakage."""
        # Create a larger dataset for this test (5000 rows)
        np.random.seed(42)
        n = 5000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        })

        # PURGE_BARS should equal max(max_bars) = 60 for H20
        purge_bars = 60
        embargo_bars = 288

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=purge_bars, embargo_bars=embargo_bars
        )

        assert metadata['purge_bars'] == 60, "Purge should match H20 max_bars"

    def test_split_indices_valid(self, sample_labeled_data):
        """Test that all indices are valid for the dataframe."""
        df = sample_labeled_data
        n = len(df)

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # All indices should be in valid range
        all_indices = np.concatenate([train_idx, val_idx, test_idx])
        assert all_indices.min() >= 0, "Negative indices"
        assert all_indices.max() < n, "Indices exceed dataframe length"

    def test_per_symbol_splitting(self, sample_labeled_data):
        """Test splitting preserves symbol column."""
        df = sample_labeled_data

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # Verify symbol is preserved in splits
        train_symbols = df.iloc[train_idx]['symbol'].unique()
        val_symbols = df.iloc[val_idx]['symbol'].unique()
        test_symbols = df.iloc[test_idx]['symbol'].unique()

        # All splits should have same symbols as original
        original_symbols = df['symbol'].unique()
        np.testing.assert_array_equal(
            sorted(train_symbols), sorted(original_symbols)
        )

    def test_validate_no_overlap_function(self):
        """Test the validate_no_overlap utility function."""
        # No overlap case
        train = np.array([0, 1, 2, 3, 4])
        val = np.array([10, 11, 12, 13])
        test = np.array([20, 21, 22, 23, 24])

        assert validate_no_overlap(train, val, test) is True

        # With overlap
        train_overlap = np.array([0, 1, 2, 3, 4])
        val_overlap = np.array([4, 5, 6, 7])  # Overlaps at 4
        test_no_overlap = np.array([10, 11, 12])

        assert validate_no_overlap(train_overlap, val_overlap, test_no_overlap) is False

    def test_create_splits_saves_files(self, sample_labeled_data, temp_directory):
        """Test that create_splits saves all required files."""
        df = sample_labeled_data

        # Save test data
        data_path = temp_directory / "test_data.parquet"
        df.to_parquet(data_path)

        output_dir = temp_directory / "splits"

        metadata = create_splits(
            data_path=data_path,
            output_dir=output_dir,
            run_id="test_run",
            purge_bars=10,
            embargo_bars=20
        )

        # Check files exist
        split_dir = output_dir / "test_run"
        assert (split_dir / "train_indices.npy").exists(), "train_indices.npy missing"
        assert (split_dir / "val_indices.npy").exists(), "val_indices.npy missing"
        assert (split_dir / "test_indices.npy").exists(), "test_indices.npy missing"
        assert (split_dir / "split_config.json").exists(), "config missing"


# =============================================================================
# STAGE 8 TESTS: DataValidator
# =============================================================================
