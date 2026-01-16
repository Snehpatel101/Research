"""
Tests for Phase 4 OOF Integrity - Index-Based Alignment for Heterogeneous Stacking.

Tests the OOF alignment system that enables heterogeneous stacking of tabular
and sequence models with different valid index ranges. Tabular models have
offset=0 (all samples valid), while sequence models have offset=seq_len-1
(first seq_len-1 samples cannot be predicted due to lookback requirements).

Test Structure:
    - TestOOFAlignmentResult: Result dataclass tests
    - TestOOFAlignmentValidator: Core validator tests
    - TestComputeOOFCoverage: Coverage computation tests
    - TestValidateOOFForStacking: Stacking validation tests
    - TestHeterogeneousStackingBuilder: Builder tests
    - TestPhase4Integration: End-to-end integration tests
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.cross_validation.oof_alignment import (
    OOFAlignmentResult,
    OOFAlignmentValidator,
    compute_oof_coverage,
    validate_oof_for_stacking,
)
from src.cross_validation.oof_core import OOFPrediction
from src.cross_validation.oof_stacking import HeterogeneousStackingBuilder


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_datetimes() -> pd.DatetimeIndex:
    """Generate 1000 sample datetimes at 5-min intervals."""
    start_time = datetime(2023, 1, 2, 9, 30)
    return pd.date_range(start=start_time, periods=1000, freq="5min")


@pytest.fixture
def sample_tabular_oof(sample_datetimes) -> OOFPrediction:
    """
    OOFPrediction from a tabular model with full 1000-sample coverage.

    Tabular models (XGBoost, CatBoost, etc.) can predict all samples
    because they don't require lookback windows. offset=0.
    """
    np.random.seed(42)
    n_samples = 1000

    # Generate random probabilities that sum to 1
    probs = np.random.dirichlet([1, 1, 1], size=n_samples)

    # All predictions valid (no NaN)
    predictions = np.argmax(probs, axis=1) - 1  # Convert to -1, 0, 1

    df = pd.DataFrame(
        {
            "datetime": sample_datetimes,
            "xgboost_prob_short": probs[:, 0],
            "xgboost_prob_neutral": probs[:, 1],
            "xgboost_prob_long": probs[:, 2],
            "xgboost_pred": predictions,
            "xgboost_confidence": probs.max(axis=1),
        }
    )

    return OOFPrediction(
        model_name="xgboost",
        predictions=df,
        fold_info=[
            {"fold": 0, "train_size": 600, "val_size": 200, "val_accuracy": 0.55, "val_f1": 0.52},
        ],
        coverage=1.0,
        original_indices=np.arange(n_samples),
        sequence_length=None,
        n_total_samples=n_samples,
    )


@pytest.fixture
def sample_sequence_oof(sample_datetimes) -> OOFPrediction:
    """
    OOFPrediction from a sequence model with seq_len=60.

    Sequence models (LSTM, GRU, TCN, etc.) cannot predict the first seq_len-1
    samples because they need lookback windows. offset=59 for seq_len=60.
    """
    np.random.seed(43)
    n_total = 1000
    seq_len = 60
    n_valid = n_total - seq_len + 1  # 941 samples

    # Generate random probabilities that sum to 1
    probs = np.random.dirichlet([1, 1, 1], size=n_total)

    # Set first seq_len-1 samples to NaN (cannot predict due to lookback)
    probs[: seq_len - 1] = np.nan

    predictions = np.full(n_total, np.nan)
    predictions[seq_len - 1 :] = np.argmax(probs[seq_len - 1 :], axis=1) - 1

    df = pd.DataFrame(
        {
            "datetime": sample_datetimes,
            "lstm_prob_short": probs[:, 0],
            "lstm_prob_neutral": probs[:, 1],
            "lstm_prob_long": probs[:, 2],
            "lstm_pred": predictions,
            "lstm_confidence": np.where(np.isnan(probs[:, 0]), np.nan, np.nanmax(probs, axis=1)),
        }
    )

    # Valid indices are from seq_len-1 to n_total-1
    valid_indices = np.arange(seq_len - 1, n_total)

    return OOFPrediction(
        model_name="lstm",
        predictions=df,
        fold_info=[
            {"fold": 0, "train_size": 600, "val_size": 200, "val_accuracy": 0.52, "val_f1": 0.49},
        ],
        coverage=n_valid / n_total,
        original_indices=valid_indices,
        sequence_length=seq_len,
        n_total_samples=n_total,
    )


@pytest.fixture
def sample_labels() -> pd.Series:
    """Generate 1000 sample labels (-1, 0, 1)."""
    np.random.seed(44)
    return pd.Series(np.random.choice([-1, 0, 1], size=1000))


@pytest.fixture
def sample_weights() -> pd.Series:
    """Generate 1000 sample weights."""
    np.random.seed(45)
    return pd.Series(np.random.uniform(0.5, 1.5, size=1000))


# =============================================================================
# TEST CLASS: OOFAlignmentResult
# =============================================================================


class TestOOFAlignmentResult:
    """Tests for OOFAlignmentResult dataclass."""

    def test_alignment_result_creation(self):
        """Test OOFAlignmentResult can be created with required fields."""
        result = OOFAlignmentResult(
            is_aligned=True,
            issues=[],
            sample_counts={"xgboost": 1000, "lstm": 941},
            common_start_idx=59,
            common_end_idx=1000,
            n_common_samples=941,
            offsets={"xgboost": 59, "lstm": 0},
        )

        assert result.is_aligned is True
        assert result.common_start_idx == 59
        assert result.common_end_idx == 1000
        assert result.n_common_samples == 941
        assert result.offsets["xgboost"] == 59
        assert result.offsets["lstm"] == 0
        assert len(result.issues) == 0

    def test_alignment_result_repr_aligned(self):
        """Test repr for aligned result shows success."""
        result = OOFAlignmentResult(
            is_aligned=True,
            common_start_idx=59,
            common_end_idx=1000,
            n_common_samples=941,
        )

        repr_str = repr(result)
        assert "ALIGNED" in repr_str
        assert "941" in repr_str

    def test_alignment_result_repr_misaligned(self):
        """Test repr for misaligned result shows issues."""
        result = OOFAlignmentResult(
            is_aligned=False,
            common_start_idx=0,
            common_end_idx=0,
            n_common_samples=0,
            issues=["No common samples found"],
        )

        repr_str = repr(result)
        assert "MISALIGNED" in repr_str

    def test_alignment_result_defaults(self):
        """Test OOFAlignmentResult has sensible defaults."""
        # Minimal creation - should have defaults for optional fields
        result = OOFAlignmentResult(is_aligned=True)

        assert result.issues == []
        assert result.sample_counts == {}
        assert result.common_start_idx == 0
        assert result.common_end_idx == 0
        assert result.n_common_samples == 0
        assert result.offsets == {}


# =============================================================================
# TEST CLASS: OOFAlignmentValidator
# =============================================================================


class TestOOFAlignmentValidator:
    """Tests for OOFAlignmentValidator class."""

    def test_validator_init(self):
        """Test OOFAlignmentValidator initializes correctly."""
        validator = OOFAlignmentValidator()
        assert validator.model_coverages == {}

    def test_register_tabular_model(self):
        """Test registering a tabular model with full coverage."""
        validator = OOFAlignmentValidator()
        validator.register_oof(
            model_name="xgboost",
            oof_indices=None,
            n_total_samples=1000,
            sequence_length=None,
        )

        assert "xgboost" in validator.model_coverages
        coverage = validator.model_coverages["xgboost"]
        assert coverage.start_idx == 0
        assert coverage.end_idx == 1000
        assert coverage.n_samples == 1000

    def test_register_sequence_model(self):
        """Test registering a sequence model with partial coverage."""
        validator = OOFAlignmentValidator()
        validator.register_oof(
            model_name="lstm",
            oof_indices=None,
            n_total_samples=1000,
            sequence_length=60,
        )

        assert "lstm" in validator.model_coverages
        coverage = validator.model_coverages["lstm"]
        assert coverage.start_idx == 59  # seq_len - 1
        assert coverage.end_idx == 1000
        assert coverage.n_samples == 941

    def test_register_with_explicit_indices(self):
        """Test registering with explicit valid indices."""
        validator = OOFAlignmentValidator()
        valid_indices = np.array([100, 101, 102, 103, 104])
        validator.register_oof(
            model_name="custom",
            oof_indices=valid_indices,
            n_total_samples=1000,
        )

        coverage = validator.model_coverages["custom"]
        assert coverage.start_idx == 100
        assert coverage.end_idx == 105  # max + 1
        assert coverage.n_samples == 5

    def test_validate_single_tabular(self):
        """Test validation with single tabular model."""
        validator = OOFAlignmentValidator()
        validator.register_oof("xgboost", None, 1000, None)

        result = validator.validate()
        assert result.is_aligned is True
        assert result.common_start_idx == 0
        assert result.common_end_idx == 1000
        assert result.n_common_samples == 1000

    def test_validate_tabular_and_sequence(self):
        """Test validation with tabular + sequence models."""
        validator = OOFAlignmentValidator()
        validator.register_oof("xgboost", None, 1000, None)
        validator.register_oof("lstm", None, 1000, 60)

        result = validator.validate()
        # Common range should be [59, 1000)
        assert result.common_start_idx == 59
        assert result.common_end_idx == 1000
        assert result.n_common_samples == 941
        assert result.offsets["xgboost"] == 59  # offset into xgboost's array
        assert result.offsets["lstm"] == 0  # lstm starts at common_start

    def test_validate_multiple_sequence_models(self):
        """Test validation with multiple sequence models."""
        validator = OOFAlignmentValidator()
        validator.register_oof("lstm", None, 1000, 60)  # starts at 59
        validator.register_oof("gru", None, 1000, 30)  # starts at 29
        validator.register_oof("tcn", None, 1000, 120)  # starts at 119

        result = validator.validate()
        # Common range should start at max(59, 29, 119) = 119
        assert result.common_start_idx == 119
        assert result.common_end_idx == 1000
        assert result.n_common_samples == 881

    def test_validate_no_models(self):
        """Test validation with no models registered."""
        validator = OOFAlignmentValidator()
        result = validator.validate()

        assert result.is_aligned is False
        assert len(result.issues) > 0
        assert "No models" in result.issues[0]

    def test_validate_low_coverage_warning(self):
        """Test validation warns on low common coverage."""
        validator = OOFAlignmentValidator()
        validator.register_oof("xgboost", None, 1000, None)
        # Very long sequence that leaves few common samples
        validator.register_oof("lstm", None, 1000, 250)

        result = validator.validate()
        # Common range = [249, 1000) = 751 samples = 75.1% coverage
        # This should trigger the low coverage warning (< 80%)
        assert result.n_common_samples == 751
        assert any("Low common coverage" in issue for issue in result.issues)

    def test_validate_invalid_n_total(self):
        """Test registration rejects invalid n_total_samples."""
        validator = OOFAlignmentValidator()
        with pytest.raises(ValueError, match="positive"):
            validator.register_oof("model", None, 0, None)

    def test_align_oof_predictions_basic(self):
        """Test aligning OOF prediction arrays."""
        validator = OOFAlignmentValidator()
        validator.register_oof("xgboost", None, 100, None)  # Full 0-100
        validator.register_oof("lstm", None, 100, 20)  # 19-100

        alignment = validator.validate()

        # Create mock OOF arrays
        xgb_oof = np.random.rand(100)  # Full range
        lstm_oof = np.random.rand(81)  # Only 81 samples (100 - 19)

        aligned = validator.align_oof_predictions(
            {"xgboost": xgb_oof, "lstm": lstm_oof}, alignment
        )

        # Both should now have 81 samples
        assert len(aligned["xgboost"]) == 81
        assert len(aligned["lstm"]) == 81

        # xgboost should have samples from index 19 onwards
        np.testing.assert_array_equal(aligned["xgboost"], xgb_oof[19:])

        # lstm should have all its samples
        np.testing.assert_array_equal(aligned["lstm"], lstm_oof)

    def test_align_oof_predictions_unregistered_model(self):
        """Test alignment fails for unregistered model."""
        validator = OOFAlignmentValidator()
        validator.register_oof("xgboost", None, 100, None)

        alignment = validator.validate()

        with pytest.raises(ValueError, match="not registered"):
            validator.align_oof_predictions(
                {"unknown": np.random.rand(100)}, alignment
            )


# =============================================================================
# TEST CLASS: ComputeOOFCoverage
# =============================================================================


class TestComputeOOFCoverage:
    """Tests for compute_oof_coverage function."""

    def test_full_coverage(self):
        """Test coverage computation for full coverage."""
        oof_preds = np.random.rand(1000)
        coverage, n_missing = compute_oof_coverage("model", oof_preds, 1000)

        assert coverage == 1.0
        assert n_missing == 0

    def test_partial_coverage(self):
        """Test coverage computation for partial coverage."""
        oof_preds = np.random.rand(941)  # 941/1000
        coverage, n_missing = compute_oof_coverage("model", oof_preds, 1000)

        assert coverage == 0.941
        assert n_missing == 59

    def test_zero_total(self):
        """Test coverage computation with zero total samples."""
        oof_preds = np.array([])
        coverage, n_missing = compute_oof_coverage("model", oof_preds, 0)

        assert coverage == 0.0
        assert n_missing == 0


# =============================================================================
# TEST CLASS: ValidateOOFForStacking
# =============================================================================


class TestValidateOOFForStacking:
    """Tests for validate_oof_for_stacking function."""

    def test_valid_tabular_only(self, sample_tabular_oof):
        """Test validation passes for tabular models with full coverage."""
        oof_results = {"xgboost": sample_tabular_oof}

        is_valid, issues = validate_oof_for_stacking(oof_results)

        assert is_valid is True
        assert len(issues) == 0

    def test_empty_oof_results(self):
        """Test validation fails for empty results."""
        is_valid, issues = validate_oof_for_stacking({})

        assert is_valid is False
        assert any("No OOF" in issue for issue in issues)

    def test_low_coverage_warning(self, sample_datetimes):
        """Test validation warns on low coverage."""
        np.random.seed(46)
        n_samples = 1000

        # Create OOF with 80% coverage
        probs = np.random.dirichlet([1, 1, 1], size=n_samples)
        predictions = np.argmax(probs, axis=1) - 1

        df = pd.DataFrame(
            {
                "datetime": sample_datetimes,
                "model_prob_short": probs[:, 0],
                "model_prob_neutral": probs[:, 1],
                "model_prob_long": probs[:, 2],
                "model_pred": predictions,
                "model_confidence": probs.max(axis=1),
            }
        )

        oof = OOFPrediction(
            model_name="model",
            predictions=df,
            fold_info=[],
            coverage=0.80,  # Below 0.95 threshold
        )

        is_valid, issues = validate_oof_for_stacking({"model": oof})

        assert is_valid is False
        assert any("coverage" in issue.lower() for issue in issues)


# =============================================================================
# TEST CLASS: HeterogeneousStackingBuilder
# =============================================================================


class TestHeterogeneousStackingBuilder:
    """Tests for HeterogeneousStackingBuilder class."""

    def test_builder_init(self):
        """Test HeterogeneousStackingBuilder initializes correctly."""
        builder = HeterogeneousStackingBuilder(purge_bars=60, embargo_bars=1440)

        assert builder.purge_bars == 60
        assert builder.embargo_bars == 1440
        assert builder.validator is not None

    def test_build_tabular_only(self, sample_tabular_oof, sample_labels, sample_weights):
        """Test building stacking dataset with only tabular models."""
        builder = HeterogeneousStackingBuilder()

        X_stack, y_aligned, w_aligned = builder.build_stacking_dataset(
            oof_results={"xgboost": sample_tabular_oof},
            y=sample_labels,
            sample_weights=sample_weights,
        )

        assert X_stack.shape == (1000, 3)  # 1 model * 3 classes
        assert len(y_aligned) == 1000
        assert len(w_aligned) == 1000

    def test_build_heterogeneous(
        self, sample_tabular_oof, sample_sequence_oof, sample_labels, sample_weights
    ):
        """Test building stacking dataset with tabular + sequence models."""
        builder = HeterogeneousStackingBuilder()

        X_stack, y_aligned, w_aligned = builder.build_stacking_dataset(
            oof_results={"xgboost": sample_tabular_oof, "lstm": sample_sequence_oof},
            y=sample_labels,
            sample_weights=sample_weights,
        )

        # Common range is [59, 1000) = 941 samples
        assert X_stack.shape == (941, 6)  # 2 models * 3 classes
        assert len(y_aligned) == 941
        assert len(w_aligned) == 941

    def test_build_without_weights(self, sample_tabular_oof, sample_labels):
        """Test building stacking dataset without sample weights."""
        builder = HeterogeneousStackingBuilder()

        X_stack, y_aligned, w_aligned = builder.build_stacking_dataset(
            oof_results={"xgboost": sample_tabular_oof},
            y=sample_labels,
            sample_weights=None,
        )

        assert X_stack.shape == (1000, 3)
        assert len(y_aligned) == 1000
        assert w_aligned is None

    def test_build_empty_results(self, sample_labels):
        """Test building with empty results raises error."""
        builder = HeterogeneousStackingBuilder()

        with pytest.raises(ValueError, match="cannot be empty"):
            builder.build_stacking_dataset(
                oof_results={},
                y=sample_labels,
            )

    def test_get_alignment_summary_before_build(self):
        """Test get_alignment_summary before building."""
        builder = HeterogeneousStackingBuilder()
        summary = builder.get_alignment_summary()

        assert "error" in summary

    def test_get_alignment_summary_after_build(
        self, sample_tabular_oof, sample_sequence_oof, sample_labels
    ):
        """Test get_alignment_summary after building."""
        builder = HeterogeneousStackingBuilder()

        builder.build_stacking_dataset(
            oof_results={"xgboost": sample_tabular_oof, "lstm": sample_sequence_oof},
            y=sample_labels,
        )

        summary = builder.get_alignment_summary()

        assert "common_start" in summary
        assert "common_end" in summary
        assert "n_common" in summary
        assert summary["common_start"] == 59
        assert summary["n_common"] == 941


# =============================================================================
# TEST CLASS: Phase 4 Integration Tests
# =============================================================================


class TestPhase4Integration:
    """End-to-end integration tests for Phase 4 OOF alignment."""

    def test_full_workflow_heterogeneous(
        self, sample_tabular_oof, sample_sequence_oof, sample_labels, sample_weights
    ):
        """Test complete workflow: validate -> align -> build stacking dataset."""
        # Step 1: Validate alignment
        is_valid, issues = validate_oof_for_stacking(
            {"xgboost": sample_tabular_oof, "lstm": sample_sequence_oof},
            min_coverage=0.90,  # Lower threshold for sequence models
        )

        # Note: may have issues due to different sample counts, but should still work
        # since the HeterogeneousStackingBuilder handles alignment

        # Step 2: Build stacking dataset using heterogeneous builder
        builder = HeterogeneousStackingBuilder()
        X_stack, y_aligned, w_aligned = builder.build_stacking_dataset(
            oof_results={"xgboost": sample_tabular_oof, "lstm": sample_sequence_oof},
            y=sample_labels,
            sample_weights=sample_weights,
        )

        # Verify results
        assert X_stack.shape[0] == 941  # Common samples
        assert X_stack.shape[1] == 6  # 2 models * 3 classes
        assert len(y_aligned) == 941
        assert len(w_aligned) == 941

        # Verify alignment summary
        summary = builder.get_alignment_summary()
        assert summary["common_start"] == 59
        assert summary["common_end"] == 1000
        assert summary["n_common"] == 941

    def test_three_model_heterogeneous(self, sample_datetimes, sample_labels, sample_weights):
        """Test heterogeneous stacking with 3 models."""
        np.random.seed(47)
        n_total = 1000

        # Create tabular OOF (full coverage)
        probs_xgb = np.random.dirichlet([1, 1, 1], size=n_total)
        df_xgb = pd.DataFrame(
            {
                "datetime": sample_datetimes,
                "xgboost_prob_short": probs_xgb[:, 0],
                "xgboost_prob_neutral": probs_xgb[:, 1],
                "xgboost_prob_long": probs_xgb[:, 2],
                "xgboost_pred": np.argmax(probs_xgb, axis=1) - 1,
                "xgboost_confidence": probs_xgb.max(axis=1),
            }
        )
        xgb_oof = OOFPrediction(
            model_name="xgboost",
            predictions=df_xgb,
            fold_info=[],
            coverage=1.0,
            original_indices=np.arange(n_total),
            n_total_samples=n_total,
        )

        # Create LSTM OOF (seq_len=60)
        seq_len_lstm = 60
        probs_lstm = np.random.dirichlet([1, 1, 1], size=n_total)
        probs_lstm[: seq_len_lstm - 1] = np.nan
        df_lstm = pd.DataFrame(
            {
                "datetime": sample_datetimes,
                "lstm_prob_short": probs_lstm[:, 0],
                "lstm_prob_neutral": probs_lstm[:, 1],
                "lstm_prob_long": probs_lstm[:, 2],
                "lstm_pred": np.where(
                    np.isnan(probs_lstm[:, 0]),
                    np.nan,
                    np.argmax(probs_lstm, axis=1) - 1,
                ),
                "lstm_confidence": np.where(
                    np.isnan(probs_lstm[:, 0]), np.nan, np.nanmax(probs_lstm, axis=1)
                ),
            }
        )
        lstm_oof = OOFPrediction(
            model_name="lstm",
            predictions=df_lstm,
            fold_info=[],
            coverage=(n_total - seq_len_lstm + 1) / n_total,
            original_indices=np.arange(seq_len_lstm - 1, n_total),
            sequence_length=seq_len_lstm,
            n_total_samples=n_total,
        )

        # Create TCN OOF (seq_len=120)
        seq_len_tcn = 120
        probs_tcn = np.random.dirichlet([1, 1, 1], size=n_total)
        probs_tcn[: seq_len_tcn - 1] = np.nan
        df_tcn = pd.DataFrame(
            {
                "datetime": sample_datetimes,
                "tcn_prob_short": probs_tcn[:, 0],
                "tcn_prob_neutral": probs_tcn[:, 1],
                "tcn_prob_long": probs_tcn[:, 2],
                "tcn_pred": np.where(
                    np.isnan(probs_tcn[:, 0]),
                    np.nan,
                    np.argmax(probs_tcn, axis=1) - 1,
                ),
                "tcn_confidence": np.where(
                    np.isnan(probs_tcn[:, 0]), np.nan, np.nanmax(probs_tcn, axis=1)
                ),
            }
        )
        tcn_oof = OOFPrediction(
            model_name="tcn",
            predictions=df_tcn,
            fold_info=[],
            coverage=(n_total - seq_len_tcn + 1) / n_total,
            original_indices=np.arange(seq_len_tcn - 1, n_total),
            sequence_length=seq_len_tcn,
            n_total_samples=n_total,
        )

        # Build stacking dataset
        builder = HeterogeneousStackingBuilder()
        X_stack, y_aligned, w_aligned = builder.build_stacking_dataset(
            oof_results={"xgboost": xgb_oof, "lstm": lstm_oof, "tcn": tcn_oof},
            y=sample_labels,
            sample_weights=sample_weights,
        )

        # Common range should be [119, 1000) = 881 samples
        # (max of 0, 59, 119 = 119)
        assert X_stack.shape == (881, 9)  # 3 models * 3 classes
        assert len(y_aligned) == 881
        assert len(w_aligned) == 881

        summary = builder.get_alignment_summary()
        assert summary["common_start"] == 119
        assert summary["n_common"] == 881


# =============================================================================
# TEST CLASS: OOFPrediction Alignment Methods
# =============================================================================


class TestOOFPredictionAlignmentMethods:
    """Tests for OOFPrediction alignment-related methods."""

    def test_n_valid_tabular(self, sample_tabular_oof):
        """Test n_valid property for tabular model."""
        assert sample_tabular_oof.n_valid == 1000

    def test_n_valid_sequence(self, sample_sequence_oof):
        """Test n_valid property for sequence model."""
        assert sample_sequence_oof.n_valid == 941

    def test_alignment_offset_tabular(self, sample_tabular_oof):
        """Test alignment_offset property for tabular model."""
        assert sample_tabular_oof.alignment_offset == 0

    def test_alignment_offset_sequence(self, sample_sequence_oof):
        """Test alignment_offset property for sequence model."""
        assert sample_sequence_oof.alignment_offset == 59  # seq_len - 1

    def test_get_aligned_probabilities(self, sample_tabular_oof):
        """Test get_aligned_probabilities extracts correct range."""
        # Get probabilities from index 50 to 150 (100 samples)
        aligned_probs = sample_tabular_oof.get_aligned_probabilities(50, 100)

        assert aligned_probs.shape == (100, 3)

        # Verify values match original
        full_probs = sample_tabular_oof.get_probabilities()
        np.testing.assert_array_equal(aligned_probs, full_probs[50:150])

    def test_get_aligned_probabilities_out_of_range(self, sample_tabular_oof):
        """Test get_aligned_probabilities raises for out of range request."""
        with pytest.raises(ValueError, match="exceeds valid range"):
            sample_tabular_oof.get_aligned_probabilities(900, 200)  # 900+200 > 1000


__all__ = [
    "TestOOFAlignmentResult",
    "TestOOFAlignmentValidator",
    "TestComputeOOFCoverage",
    "TestValidateOOFForStacking",
    "TestHeterogeneousStackingBuilder",
    "TestPhase4Integration",
    "TestOOFPredictionAlignmentMethods",
]
