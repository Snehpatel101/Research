"""
Tests for Sequence-aware Cross-Validation utilities.

Tests:
- SequenceCVBuilder construction and validation
- Symbol boundary handling
- Fold sequence generation
- Coverage validation
- Integration with PurgedKFold
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.cross_validation.sequence_cv import (
    SequenceCVBuilder,
    SequenceFoldResult,
    build_sequences_for_cv_fold,
    validate_sequence_cv_coverage,
)
from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_data():
    """Generate simple time series data without symbols."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    dates = pd.date_range("2023-01-01", periods=n_samples, freq="5min")

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

    weights = pd.Series(
        np.random.uniform(0.5, 1.5, size=n_samples).astype(np.float32),
        index=dates,
        name="weight",
    )

    return {"X": X, "y": y, "weights": weights}


@pytest.fixture
def multi_symbol_data():
    """Generate data with multiple symbols (simulating multiple contracts)."""
    np.random.seed(42)
    n_per_symbol = 100
    n_features = 5

    dfs = []
    for symbol in ["MES", "MGC", "ES"]:
        dates = pd.date_range("2023-01-01", periods=n_per_symbol, freq="5min")
        df = pd.DataFrame({
            "symbol": symbol,
            **{f"feature_{i}": np.random.randn(n_per_symbol).astype(np.float32)
               for i in range(n_features)},
            "label": np.random.choice([-1, 0, 1], size=n_per_symbol),
            "weight": np.random.uniform(0.5, 1.5, size=n_per_symbol).astype(np.float32),
        }, index=dates)
        dfs.append(df)

    # Concatenate but keep sorted by symbol (creates boundaries)
    combined = pd.concat(dfs, ignore_index=True)

    feature_cols = [f"feature_{i}" for i in range(n_features)]
    X = combined[["symbol"] + feature_cols]
    y = pd.Series(combined["label"].values, name="label")
    weights = pd.Series(combined["weight"].values, name="weight")

    return {"X": X, "y": y, "weights": weights}


@pytest.fixture
def purged_cv():
    """Standard PurgedKFold for testing."""
    config = PurgedKFoldConfig(n_splits=3, purge_bars=10, embargo_bars=5)
    return PurgedKFold(config)


# =============================================================================
# SEQUENCECVBUILDER CONSTRUCTION TESTS
# =============================================================================

class TestSequenceCVBuilderConstruction:
    """Tests for SequenceCVBuilder initialization."""

    def test_valid_construction(self, simple_data):
        """Test valid construction."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
            weights=simple_data["weights"],
        )
        assert builder.seq_len == 20
        assert builder.n_samples == 200
        assert builder.n_features == 5

    def test_invalid_seq_len_zero(self, simple_data):
        """Test that seq_len=0 raises ValueError."""
        with pytest.raises(ValueError, match="seq_len must be positive"):
            SequenceCVBuilder(
                X=simple_data["X"],
                y=simple_data["y"],
                seq_len=0,
            )

    def test_invalid_seq_len_negative(self, simple_data):
        """Test that negative seq_len raises ValueError."""
        with pytest.raises(ValueError, match="seq_len must be positive"):
            SequenceCVBuilder(
                X=simple_data["X"],
                y=simple_data["y"],
                seq_len=-5,
            )

    def test_empty_data_raises(self):
        """Test that empty X raises ValueError."""
        with pytest.raises(ValueError, match="X cannot be empty"):
            SequenceCVBuilder(
                X=pd.DataFrame(),
                y=pd.Series([]),
                seq_len=10,
            )

    def test_mismatched_lengths_raises(self, simple_data):
        """Test that mismatched X and y lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            SequenceCVBuilder(
                X=simple_data["X"],
                y=simple_data["y"].iloc[:100],  # Truncated
                seq_len=10,
            )

    def test_weights_optional(self, simple_data):
        """Test that weights are optional."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
            weights=None,
        )
        # Should have uniform weights
        result = builder.build_fold_sequences(np.arange(50, 100))
        assert np.allclose(result.weights, 1.0)


# =============================================================================
# SEQUENCE BUILDING TESTS
# =============================================================================

class TestSequenceBuilding:
    """Tests for sequence building functionality."""

    def test_builds_correct_shape(self, simple_data):
        """Test that sequences have correct 3D shape."""
        seq_len = 20
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=seq_len,
        )

        fold_indices = np.arange(50, 150)  # 100 samples
        result = builder.build_fold_sequences(fold_indices)

        # Shape: (n_sequences, seq_len, n_features)
        assert result.X_sequences.ndim == 3
        assert result.X_sequences.shape[1] == seq_len
        assert result.X_sequences.shape[2] == 5  # n_features

    def test_sequence_length_matches(self, simple_data):
        """Test that each sequence has correct length."""
        seq_len = 30
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=seq_len,
        )

        result = builder.build_fold_sequences(np.arange(100))
        for i in range(result.n_sequences):
            assert result.X_sequences[i].shape[0] == seq_len

    def test_target_indices_correct(self, simple_data):
        """Test that target indices map correctly to original positions."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        fold_indices = np.arange(50, 100)
        result = builder.build_fold_sequences(fold_indices)

        # All target indices should be in the fold
        for target_idx in result.target_indices:
            assert target_idx in fold_indices

    def test_labels_match_targets(self, simple_data):
        """Test that labels correspond to target indices."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        fold_indices = np.arange(50, 100)
        result = builder.build_fold_sequences(fold_indices)

        # Each label should match the y value at target_idx
        for seq_idx, target_idx in enumerate(result.target_indices):
            expected_label = simple_data["y"].iloc[target_idx]
            assert result.y[seq_idx] == expected_label

    def test_drops_insufficient_history(self, simple_data):
        """Test that samples without sufficient lookback are dropped."""
        seq_len = 30
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=seq_len,
        )

        # Fold starts at 0 - first seq_len-1 samples can't form sequences
        fold_indices = np.arange(0, 50)
        result = builder.build_fold_sequences(fold_indices, allow_lookback_outside=False)

        # Should drop first seq_len-1 samples
        assert result.n_dropped == seq_len - 1
        assert result.n_sequences == 50 - (seq_len - 1)

    def test_allows_lookback_outside_fold(self, simple_data):
        """Test that lookback can extend outside fold when allowed."""
        seq_len = 20
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=seq_len,
        )

        # Fold starts at 50, but lookback can use 0-49
        fold_indices = np.arange(50, 100)
        result = builder.build_fold_sequences(fold_indices, allow_lookback_outside=True)

        # All fold samples should have sequences (since they can look back outside)
        assert result.n_sequences == len(fold_indices)
        assert result.n_dropped == 0

    def test_empty_fold_returns_empty_result(self, simple_data):
        """Test that empty fold indices return empty result."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        result = builder.build_fold_sequences(np.array([], dtype=np.int64))

        assert result.n_sequences == 0
        assert len(result.y) == 0
        assert len(result.target_indices) == 0


# =============================================================================
# SYMBOL BOUNDARY TESTS
# =============================================================================

class TestSymbolBoundaries:
    """Tests for symbol boundary handling."""

    def test_detects_symbol_boundaries(self, multi_symbol_data):
        """Test that symbol boundaries are detected."""
        builder = SequenceCVBuilder(
            X=multi_symbol_data["X"],
            y=multi_symbol_data["y"],
            seq_len=20,
            symbol_column="symbol",
        )

        # Should have 2 boundaries (MES->MGC, MGC->ES)
        assert len(builder._symbol_boundaries) == 2

    def test_sequences_dont_cross_boundaries(self, multi_symbol_data):
        """Test that sequences don't cross symbol boundaries."""
        builder = SequenceCVBuilder(
            X=multi_symbol_data["X"],
            y=multi_symbol_data["y"],
            seq_len=20,
            symbol_column="symbol",
        )

        # Fold that spans boundary
        fold_indices = np.arange(80, 120)  # Crosses MES/MGC boundary at 100
        result = builder.build_fold_sequences(fold_indices, allow_lookback_outside=True)

        # Verify no sequence crosses boundary
        # The boundary is at index 100 (symbols[99]=MES, symbols[100]=MGC)
        for seq_idx, target_idx in enumerate(result.target_indices):
            start_idx = target_idx - builder.seq_len + 1
            # Check that start and end have same symbol
            start_symbol = multi_symbol_data["X"]["symbol"].iloc[start_idx]
            end_symbol = multi_symbol_data["X"]["symbol"].iloc[target_idx]
            assert start_symbol == end_symbol, (
                f"Sequence {seq_idx} crosses boundary: "
                f"start={start_symbol} at {start_idx}, end={end_symbol} at {target_idx}"
            )

    def test_drops_boundary_crossing_sequences(self, multi_symbol_data):
        """Test that boundary-crossing sequences are dropped."""
        builder = SequenceCVBuilder(
            X=multi_symbol_data["X"],
            y=multi_symbol_data["y"],
            seq_len=10,
            symbol_column="symbol",
        )

        # Fold at boundary (around index 100)
        fold_indices = np.arange(95, 110)
        result = builder.build_fold_sequences(fold_indices, allow_lookback_outside=True)

        # Some sequences should be dropped due to boundary
        assert result.n_dropped > 0

    def test_no_symbol_column_disables_isolation(self, simple_data):
        """Test that missing symbol column disables isolation."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
            symbol_column="nonexistent",
        )

        assert builder._symbol_boundaries is None


# =============================================================================
# FOLD COVERAGE TESTS
# =============================================================================

class TestFoldCoverage:
    """Tests for coverage calculation."""

    def test_coverage_calculation(self, simple_data):
        """Test that coverage is calculated correctly."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        fold_indices = np.arange(50, 100)
        coverage = builder.get_fold_coverage(fold_indices, allow_lookback_outside=True)

        # With lookback outside allowed, coverage should be 100%
        assert coverage == 1.0

    def test_coverage_with_boundary_drops(self, multi_symbol_data):
        """Test that coverage reflects boundary drops."""
        builder = SequenceCVBuilder(
            X=multi_symbol_data["X"],
            y=multi_symbol_data["y"],
            seq_len=20,
            symbol_column="symbol",
        )

        # Fold spanning boundary
        fold_indices = np.arange(90, 110)
        coverage = builder.get_fold_coverage(fold_indices, allow_lookback_outside=True)

        # Coverage should be < 100% due to boundary
        assert coverage < 1.0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_build_sequences_for_cv_fold(self, simple_data):
        """Test the convenience function."""
        fold_indices = np.arange(50, 100)

        result = build_sequences_for_cv_fold(
            X=simple_data["X"],
            y=simple_data["y"],
            fold_indices=fold_indices,
            seq_len=20,
            weights=simple_data["weights"],
        )

        assert isinstance(result, SequenceFoldResult)
        assert result.n_sequences == 50

    def test_validate_sequence_cv_coverage(self, simple_data, purged_cv):
        """Test CV coverage validation."""
        coverage_stats = validate_sequence_cv_coverage(
            X=simple_data["X"],
            y=simple_data["y"],
            cv=purged_cv,
            seq_len=20,
        )

        assert "n_folds" in coverage_stats
        assert "fold_coverages" in coverage_stats
        assert "mean_coverage" in coverage_stats
        assert coverage_stats["n_folds"] == 3


# =============================================================================
# INTEGRATION WITH PURGEDKFOLD
# =============================================================================

class TestPurgedKFoldIntegration:
    """Tests for integration with PurgedKFold."""

    def test_works_with_purged_kfold(self, simple_data, purged_cv):
        """Test that SequenceCVBuilder works with PurgedKFold splits."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        total_sequences = 0
        for train_idx, val_idx in purged_cv.split(simple_data["X"], simple_data["y"]):
            train_result = builder.build_fold_sequences(train_idx, allow_lookback_outside=True)
            val_result = builder.build_fold_sequences(val_idx, allow_lookback_outside=True)

            # Both should produce sequences
            assert train_result.n_sequences > 0
            assert val_result.n_sequences > 0

            total_sequences += val_result.n_sequences

        # Total val sequences across folds should be close to n_samples
        assert total_sequences > 0.8 * len(simple_data["X"])

    def test_non_overlapping_val_sequences(self, simple_data, purged_cv):
        """Test that validation sequences don't overlap across folds."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        all_val_targets = []
        for train_idx, val_idx in purged_cv.split(simple_data["X"], simple_data["y"]):
            val_result = builder.build_fold_sequences(val_idx, allow_lookback_outside=True)
            all_val_targets.extend(val_result.target_indices.tolist())

        # No duplicates in validation targets
        assert len(all_val_targets) == len(set(all_val_targets))


# =============================================================================
# SEQUENCE RESULT TESTS
# =============================================================================

class TestSequenceFoldResult:
    """Tests for SequenceFoldResult dataclass."""

    def test_properties(self, simple_data):
        """Test SequenceFoldResult properties."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        result = builder.build_fold_sequences(np.arange(50, 100))

        assert result.n_sequences == 50
        assert result.seq_len == 20
        assert result.n_features == 5

    def test_repr(self, simple_data):
        """Test string representation."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        result = builder.build_fold_sequences(np.arange(50, 100))
        repr_str = repr(result)

        assert "SequenceFoldResult" in repr_str
        assert "n_sequences=50" in repr_str


# =============================================================================
# STRIDE TESTS
# =============================================================================

class TestStride:
    """Tests for stride parameter."""

    def test_stride_reduces_sequences(self, simple_data):
        """Test that stride>1 reduces number of sequences."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        fold_indices = np.arange(50, 100)

        result_stride1 = builder.build_fold_sequences(fold_indices, stride=1)
        result_stride2 = builder.build_fold_sequences(fold_indices, stride=2)

        # Stride 2 should have roughly half the sequences
        assert result_stride2.n_sequences < result_stride1.n_sequences
        assert result_stride2.n_sequences == (result_stride1.n_sequences + 1) // 2

    def test_stride_targets_correct(self, simple_data):
        """Test that stride targets are correctly spaced."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        fold_indices = np.arange(50, 100)
        result = builder.build_fold_sequences(fold_indices, stride=3)

        # Targets should be spaced by stride
        sorted_targets = np.sort(result.target_indices)
        if len(sorted_targets) > 1:
            diffs = np.diff(sorted_targets)
            # Most differences should be stride (except at boundaries)
            assert np.median(diffs) == 3


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_seq_len_equals_fold_size(self, simple_data):
        """Test when seq_len equals fold size."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        fold_indices = np.arange(100, 120)  # Exactly 20 samples
        result = builder.build_fold_sequences(fold_indices, allow_lookback_outside=True)

        # Should still produce sequences (with lookback outside)
        assert result.n_sequences > 0

    def test_seq_len_larger_than_fold(self, simple_data):
        """Test when seq_len is larger than fold."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=30,
        )

        fold_indices = np.arange(100, 120)  # Only 20 samples
        result = builder.build_fold_sequences(fold_indices, allow_lookback_outside=True)

        # Should still work with lookback outside
        assert result.n_sequences == 20

    def test_single_sample_fold(self, simple_data):
        """Test with single sample in fold."""
        builder = SequenceCVBuilder(
            X=simple_data["X"],
            y=simple_data["y"],
            seq_len=20,
        )

        fold_indices = np.array([100])
        result = builder.build_fold_sequences(fold_indices, allow_lookback_outside=True)

        # Single sample should produce one sequence
        assert result.n_sequences == 1

    def test_handles_float32_data(self, simple_data):
        """Test that float32 data is handled correctly."""
        builder = SequenceCVBuilder(
            X=simple_data["X"].astype(np.float32),
            y=simple_data["y"].astype(np.float32),
            seq_len=20,
        )

        result = builder.build_fold_sequences(np.arange(50, 100))
        assert result.X_sequences.dtype == np.float32
