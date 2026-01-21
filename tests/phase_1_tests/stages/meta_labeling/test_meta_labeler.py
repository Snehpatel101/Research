"""
Tests for MetaLabelGenerator, MetaLabelingConfig, and MetaLabelResult.

Tests cover:
- MetaLabelingConfig validation
- MetaLabelResult properties
- MetaLabelGenerator label generation
- Quality metrics computation
- Edge cases and validation
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline._phase1_impl.stages.meta_labeling.meta_labeler import (
    MetaLabelGenerator,
    MetaLabelingConfig,
    MetaLabelResult,
    INVALID_LABEL,
    META_LABEL_CORRECT,
    META_LABEL_INCORRECT,
    META_LABEL_INVALID,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_true_labels():
    """Create sample true direction labels."""
    np.random.seed(42)
    # Mix of -1 (short), 0 (neutral), 1 (long)
    return np.random.choice([-1, 0, 1], size=100, p=[0.3, 0.1, 0.6])


@pytest.fixture
def sample_primary_signals():
    """Create sample primary prediction signals."""
    np.random.seed(43)
    # Mix of -1, 0, 1
    return np.random.choice([-1, 0, 1], size=100, p=[0.3, 0.1, 0.6])


@pytest.fixture
def sample_returns():
    """Create sample forward returns."""
    np.random.seed(42)
    return np.random.normal(0, 0.02, 100)  # 2% volatility


@pytest.fixture
def perfect_predictions():
    """Create data where all predictions are correct."""
    n = 50

    # True labels: all +1
    y_true = np.ones(n)
    # Primary predictions: all +1 (matching)
    y_primary = np.ones(n)

    return y_true, y_primary


@pytest.fixture
def all_wrong_predictions():
    """Create data where all predictions are wrong."""
    n = 50

    # True labels: all +1
    y_true = np.ones(n)
    # Primary predictions: all -1 (opposite)
    y_primary = -np.ones(n)

    return y_true, y_primary


@pytest.fixture
def mixed_signals_df():
    """Create DataFrame with explicit scenarios for testing."""
    data = {
        "label_h5": [1, 1, -1, -1, 0, 1, -1, 1, -1, 0],  # True labels
        "primary_pred_h5": [1, -1, -1, 1, 1, 1, -1, -1, 1, 0],  # Primary predictions
        # Expected meta-labels:
        # 0: true=1, pred=1 -> correct (1)
        # 1: true=1, pred=-1 -> incorrect (0)
        # 2: true=-1, pred=-1 -> correct (1)
        # 3: true=-1, pred=1 -> incorrect (0)
        # 4: true=0, pred=1 -> pred signal when true neutral = incorrect (0)
        # 5: true=1, pred=1 -> correct (1)
        # 6: true=-1, pred=-1 -> correct (1)
        # 7: true=1, pred=-1 -> incorrect (0)
        # 8: true=-1, pred=1 -> incorrect (0)
        # 9: true=0, pred=0 -> neutral prediction = invalid (-99)
    }
    return pd.DataFrame(data)


# =============================================================================
# META LABELING CONFIG TESTS
# =============================================================================


class TestMetaLabelingConfig:
    """Tests for MetaLabelingConfig validation."""

    def test_default_config(self):
        """Test default configuration with required horizon."""
        config = MetaLabelingConfig(horizon=5)

        assert config.horizon == 5
        assert config.include_neutrals is False
        assert config.correctness_margin_type == "binary"

    def test_default_column_names(self):
        """Test default column name generation based on horizon."""
        config = MetaLabelingConfig(horizon=10)

        assert config.primary_signal_column == "primary_pred_h10"
        assert config.return_column == "fwd_return_h10"

    def test_custom_column_names(self):
        """Test custom column names."""
        config = MetaLabelingConfig(
            horizon=5,
            primary_signal_column="my_signal",
            return_column="my_return",
        )

        assert config.primary_signal_column == "my_signal"
        assert config.return_column == "my_return"

    def test_invalid_horizon_raises(self):
        """Test that invalid horizon raises ValueError."""
        with pytest.raises(ValueError, match="horizon must be positive"):
            MetaLabelingConfig(horizon=0)

        with pytest.raises(ValueError, match="horizon must be positive"):
            MetaLabelingConfig(horizon=-5)

    def test_invalid_correctness_margin_type_raises(self):
        """Test that invalid correctness_margin_type raises ValueError."""
        with pytest.raises(ValueError, match="correctness_margin_type must be"):
            MetaLabelingConfig(horizon=5, correctness_margin_type="invalid")

    def test_valid_correctness_margin_types(self):
        """Test valid correctness margin types."""
        config_binary = MetaLabelingConfig(horizon=5, correctness_margin_type="binary")
        assert config_binary.correctness_margin_type == "binary"

        config_magnitude = MetaLabelingConfig(horizon=5, correctness_margin_type="magnitude")
        assert config_magnitude.correctness_margin_type == "magnitude"


# =============================================================================
# META LABEL RESULT TESTS
# =============================================================================


class TestMetaLabelResult:
    """Tests for MetaLabelResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = MetaLabelResult(
            meta_labels=np.array([1, 0, 1, -99, 1]),
            primary_signals=np.array([1, -1, 1, 0, -1]),
            correctness_margin=np.array([1.0, -1.0, 1.0, 0.0, 1.0]),
        )

        assert result.n_samples == 5
        assert result.n_valid == 4  # One -99
        assert result.n_correct == 3  # Three 1s
        assert result.n_incorrect == 1  # One 0

    def test_n_samples_property(self):
        """Test n_samples property."""
        labels = np.array([1, 0, 1, 1, 0, -99, -99])
        result = MetaLabelResult(
            meta_labels=labels,
            primary_signals=np.ones(len(labels)),
            correctness_margin=np.zeros(len(labels)),
        )

        assert result.n_samples == 7

    def test_n_valid_property(self):
        """Test n_valid property."""
        labels = np.array([1, 0, 1, 1, 0, -99, -99, 1])
        result = MetaLabelResult(
            meta_labels=labels,
            primary_signals=np.ones(len(labels)),
            correctness_margin=np.zeros(len(labels)),
        )

        assert result.n_valid == 6  # 8 - 2 invalid

    def test_accuracy_property(self):
        """Test accuracy property."""
        # 4 correct, 2 incorrect, 2 invalid
        labels = np.array([1, 1, 0, 1, 0, 1, -99, -99])
        result = MetaLabelResult(
            meta_labels=labels,
            primary_signals=np.ones(len(labels)),
            correctness_margin=np.zeros(len(labels)),
        )

        # Accuracy = 4 correct / 6 valid = 0.6667
        assert abs(result.accuracy - (4 / 6)) < 0.001

    def test_accuracy_all_invalid(self):
        """Test accuracy with all invalid labels."""
        labels = np.array([-99, -99, -99])
        result = MetaLabelResult(
            meta_labels=labels,
            primary_signals=np.zeros(len(labels)),
            correctness_margin=np.zeros(len(labels)),
        )

        assert result.accuracy == 0.0

    def test_quality_metrics(self):
        """Test quality_metrics storage."""
        result = MetaLabelResult(
            meta_labels=np.array([1, 0]),
            primary_signals=np.array([1, 1]),
            correctness_margin=np.array([1.0, -1.0]),
            quality_metrics={"precision": 0.8, "recall": 0.9},
        )

        assert result.quality_metrics["precision"] == 0.8
        assert result.quality_metrics["recall"] == 0.9

    def test_metadata(self):
        """Test metadata storage."""
        result = MetaLabelResult(
            meta_labels=np.array([1]),
            primary_signals=np.array([1]),
            correctness_margin=np.array([1.0]),
            metadata={"horizon": 5, "model": "test"},
        )

        assert result.metadata["horizon"] == 5
        assert result.metadata["model"] == "test"


# =============================================================================
# META LABEL GENERATOR TESTS
# =============================================================================


class TestMetaLabelGenerator:
    """Tests for MetaLabelGenerator."""

    def test_initialization_with_horizon(self):
        """Test initialization with horizon."""
        generator = MetaLabelGenerator(horizon=5)

        assert generator.config.horizon == 5

    def test_initialization_with_kwargs(self):
        """Test initialization with keyword arguments."""
        generator = MetaLabelGenerator(
            horizon=10,
            include_neutrals=True,
            correctness_margin_type="magnitude",
        )

        assert generator.config.horizon == 10
        assert generator.config.include_neutrals is True
        assert generator.config.correctness_margin_type == "magnitude"

    def test_generate_basic(self, sample_true_labels, sample_primary_signals):
        """Test basic meta-label generation."""
        generator = MetaLabelGenerator(horizon=5)

        result = generator.generate(
            y_true=sample_true_labels,
            y_primary=sample_primary_signals,
        )

        assert isinstance(result, MetaLabelResult)
        assert result.n_samples == len(sample_true_labels)

    def test_generate_perfect_predictions(self, perfect_predictions):
        """Test meta-labels for perfect predictions."""
        y_true, y_primary = perfect_predictions
        generator = MetaLabelGenerator(horizon=5)

        result = generator.generate(y_true, y_primary)

        # All valid predictions should be correct
        valid_mask = result.meta_labels != META_LABEL_INVALID
        assert np.all(result.meta_labels[valid_mask] == META_LABEL_CORRECT)

    def test_generate_all_wrong_predictions(self, all_wrong_predictions):
        """Test meta-labels for all wrong predictions."""
        y_true, y_primary = all_wrong_predictions
        generator = MetaLabelGenerator(horizon=5)

        result = generator.generate(y_true, y_primary)

        # All valid predictions should be incorrect
        valid_mask = result.meta_labels != META_LABEL_INVALID
        assert np.all(result.meta_labels[valid_mask] == META_LABEL_INCORRECT)

    def test_generate_explicit_scenarios(self):
        """Test meta-labels with explicit scenarios."""
        generator = MetaLabelGenerator(horizon=5)

        # Explicit test cases
        y_true = np.array([1, 1, -1, -1, 1])
        y_primary = np.array([1, -1, -1, 1, 1])
        # Expected:
        # 0: match -> correct (1)
        # 1: mismatch -> incorrect (0)
        # 2: match -> correct (1)
        # 3: mismatch -> incorrect (0)
        # 4: match -> correct (1)

        result = generator.generate(y_true, y_primary)

        assert result.meta_labels[0] == META_LABEL_CORRECT
        assert result.meta_labels[1] == META_LABEL_INCORRECT
        assert result.meta_labels[2] == META_LABEL_CORRECT
        assert result.meta_labels[3] == META_LABEL_INCORRECT
        assert result.meta_labels[4] == META_LABEL_CORRECT

    def test_neutral_primary_signals_invalid(self):
        """Test that neutral primary predictions produce invalid meta-labels."""
        generator = MetaLabelGenerator(horizon=5, include_neutrals=False)

        # All neutral primary signals
        y_true = np.array([1, -1, 1, -1, 1])
        y_primary = np.array([0, 0, 0, 0, 0])

        result = generator.generate(y_true, y_primary)

        # All should be invalid (neutral predictions are skipped)
        assert np.all(result.meta_labels == META_LABEL_INVALID)

    def test_neutral_true_labels_incorrect(self):
        """Test that neutral true labels with non-neutral prediction are incorrect."""
        generator = MetaLabelGenerator(horizon=5)

        # Neutral true labels, non-neutral predictions
        y_true = np.array([0, 0, 0])
        y_primary = np.array([1, -1, 1])

        result = generator.generate(y_true, y_primary)

        # Predictions of direction when true is neutral = incorrect
        assert np.all(result.meta_labels == META_LABEL_INCORRECT)

    def test_correctness_margin_binary(self):
        """Test correctness margin in binary mode."""
        generator = MetaLabelGenerator(horizon=5, correctness_margin_type="binary")

        y_true = np.array([1, 1, -1])
        y_primary = np.array([1, -1, -1])

        result = generator.generate(y_true, y_primary)

        # Binary mode: +1 for correct, -1 for incorrect
        assert result.correctness_margin[0] == 1.0  # Correct
        assert result.correctness_margin[1] == -1.0  # Incorrect
        assert result.correctness_margin[2] == 1.0  # Correct

    def test_correctness_margin_with_returns(self):
        """Test correctness margin with returns provided."""
        generator = MetaLabelGenerator(horizon=5, correctness_margin_type="binary")

        y_true = np.array([1, 1, -1])
        y_primary = np.array([1, -1, -1])
        returns = np.array([0.02, -0.01, -0.015])

        result = generator.generate(y_true, y_primary, returns=returns)

        # With returns: margin = return * sign(prediction)
        # Long prediction: margin = return
        # Short prediction: margin = -return
        assert result.correctness_margin[0] == pytest.approx(0.02, abs=0.001)  # Long, positive return
        assert result.correctness_margin[1] == pytest.approx(0.01, abs=0.001)  # Short, negative return * -1 = positive
        assert result.correctness_margin[2] == pytest.approx(0.015, abs=0.001)  # Short, negative * -1 = positive

    def test_quality_metrics_computed(self, sample_true_labels, sample_primary_signals):
        """Test that quality metrics are computed."""
        generator = MetaLabelGenerator(horizon=5)

        result = generator.generate(sample_true_labels, sample_primary_signals)

        # Should have quality metrics
        assert result.quality_metrics is not None
        assert len(result.quality_metrics) > 0

    def test_generate_empty_signals_raises(self):
        """Test that empty signals raise error."""
        generator = MetaLabelGenerator(horizon=5)

        with pytest.raises(ValueError, match="empty"):
            generator.generate(np.array([]), np.array([]))

    def test_generate_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise error."""
        generator = MetaLabelGenerator(horizon=5)

        with pytest.raises(ValueError, match="same length"):
            generator.generate(
                y_true=np.array([1, -1, 1]),
                y_primary=np.array([1, -1]),  # Different length
            )


# =============================================================================
# DATAFRAME GENERATION TESTS
# =============================================================================


class TestMetaLabelGeneratorDataFrame:
    """Tests for generating meta-labels from DataFrame."""

    def test_generate_from_dataframe(self, mixed_signals_df):
        """Test generating from DataFrame."""
        generator = MetaLabelGenerator(horizon=5)

        result, df_out = generator.generate_from_dataframe(
            mixed_signals_df,
            primary_signal_column="primary_pred_h5",
            true_label_column="label_h5",
        )

        assert isinstance(result, MetaLabelResult)
        assert result.n_samples == len(mixed_signals_df)
        assert isinstance(df_out, pd.DataFrame)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestMetaLabelGeneratorEdgeCases:
    """Tests for edge cases in meta-label generation."""

    def test_nan_in_returns_ignored(self):
        """Test that NaN in returns doesn't crash."""
        generator = MetaLabelGenerator(horizon=5)

        y_true = np.array([1, 1, 1])
        y_primary = np.array([1, 1, 1])
        returns = np.array([0.01, np.nan, 0.02])

        result = generator.generate(y_true, y_primary, returns=returns)

        # Should still produce results
        assert result.n_samples == 3
        # All predictions match, so all should be correct
        assert result.meta_labels[0] == META_LABEL_CORRECT
        assert result.meta_labels[2] == META_LABEL_CORRECT

    def test_invalid_true_labels_skipped(self):
        """Test that invalid true labels are skipped."""
        generator = MetaLabelGenerator(horizon=5)

        y_true = np.array([1, -99, 1])  # Middle one is invalid
        y_primary = np.array([1, 1, 1])

        result = generator.generate(y_true, y_primary)

        # Invalid true label should produce invalid meta-label
        assert result.meta_labels[0] == META_LABEL_CORRECT
        assert result.meta_labels[1] == META_LABEL_INVALID
        assert result.meta_labels[2] == META_LABEL_CORRECT

    def test_include_neutrals_option(self):
        """Test include_neutrals configuration option."""
        generator = MetaLabelGenerator(horizon=5, include_neutrals=True)

        y_true = np.array([0, 0, 1])  # Some neutral true labels
        y_primary = np.array([0, 1, 1])  # Some neutral predictions

        result = generator.generate(y_true, y_primary)

        # With include_neutrals=True:
        # 0: neutral pred with neutral true -> correct (both agree on "no signal")
        assert result.meta_labels[0] == META_LABEL_CORRECT
        # 1: non-neutral pred with neutral true -> incorrect
        assert result.meta_labels[1] == META_LABEL_INCORRECT
        # 2: correct match
        assert result.meta_labels[2] == META_LABEL_CORRECT


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestMetaLabelGeneratorIntegration:
    """Integration tests for meta-label generation workflow."""

    def test_full_workflow(self, sample_true_labels, sample_primary_signals):
        """Test full meta-labeling workflow."""
        # 1. Create generator
        generator = MetaLabelGenerator(
            horizon=5,
            include_neutrals=False,
            correctness_margin_type="binary",
        )

        # 2. Generate meta-labels
        result = generator.generate(sample_true_labels, sample_primary_signals)

        # 3. Verify result
        assert isinstance(result, MetaLabelResult)
        assert result.n_samples == len(sample_true_labels)
        assert result.n_valid <= result.n_samples
        assert result.n_correct + result.n_incorrect == result.n_valid


# =============================================================================
# VALIDATION MODULE INTEGRATION
# =============================================================================


class TestValidationModuleIntegration:
    """Test that classes are importable from validation module."""

    def test_import_from_validation(self):
        """Test importing from src.validation."""
        from src.validation import (
            MetaLabelGenerator,
            MetaLabelingConfig,
            MetaLabelResult,
        )

        assert MetaLabelGenerator is not None
        assert MetaLabelingConfig is not None
        assert MetaLabelResult is not None

    def test_create_from_validation_import(self, sample_true_labels, sample_primary_signals):
        """Test creating instances from validation imports."""
        from src.validation import MetaLabelGenerator

        generator = MetaLabelGenerator(horizon=5)
        result = generator.generate(sample_true_labels, sample_primary_signals)

        assert result.n_samples == len(sample_true_labels)


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestMetaLabelConstants:
    """Tests for meta-labeling constants."""

    def test_invalid_label_value(self):
        """Test invalid label constant."""
        assert INVALID_LABEL == -99
        assert META_LABEL_INVALID == -99

    def test_correct_label_value(self):
        """Test correct label constant."""
        assert META_LABEL_CORRECT == 1

    def test_incorrect_label_value(self):
        """Test incorrect label constant."""
        assert META_LABEL_INCORRECT == 0
