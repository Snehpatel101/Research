"""
Tests for MetaLabeler strategy.

Tests cover:
- Basic meta-labeling computation
- Primary signal correctness evaluation
- Bet size methods (probability vs fixed)
- Quality metrics (accuracy, precision, recall)
- Input validation
- Edge cases (neutrals, invalid signals)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.labeling import BetSizeMethod, LabelingType, MetaLabeler


@pytest.fixture
def sample_df_with_labels():
    """Create sample DataFrame with primary labels and close prices."""
    np.random.seed(42)
    n = 100

    # Generate prices with trend
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)

    # Generate primary labels (-1, 0, 1)
    # Mix of correct and incorrect predictions
    primary_labels = np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.2, 0.5])

    return pd.DataFrame({
        'close': close,
        'label_h5': primary_labels,
        'volume': np.random.randint(100, 1000, n)
    })


@pytest.fixture
def perfect_signals_df():
    """Create DataFrame where all primary signals are correct."""
    n = 50

    # Clear uptrend
    close = 100.0 + np.arange(n) * 1.0

    # All +1 signals in uptrend = all correct
    primary_labels = np.ones(n, dtype=int)

    return pd.DataFrame({
        'close': close,
        'label_h5': primary_labels
    })


@pytest.fixture
def all_wrong_signals_df():
    """Create DataFrame where all primary signals are incorrect."""
    n = 50

    # Clear uptrend
    close = 100.0 + np.arange(n) * 1.0

    # All -1 signals in uptrend = all incorrect
    primary_labels = -np.ones(n, dtype=int)

    return pd.DataFrame({
        'close': close,
        'label_h5': primary_labels
    })


@pytest.fixture
def mixed_signals_df():
    """Create DataFrame with explicit return column for precise testing."""
    n = 20

    close = np.array([100.0] * n)

    # Explicit forward returns
    fwd_returns = np.array([
        0.02,  # +2%
        -0.01,  # -1%
        0.03,  # +3%
        -0.02,  # -2%
        0.00,  # 0%
        0.01,  # +1%
        -0.015,  # -1.5%
        0.025,  # +2.5%
        -0.005,  # -0.5%
        0.015,  # +1.5%
        0.02, -0.01, 0.03, -0.02, 0.00,
        0.01, -0.015, 0.025, -0.005, 0.015
    ])

    # Primary signals
    # 1=long, -1=short, 0=neutral
    primary_labels = np.array([
        1,   # return +2%, long signal -> correct (1)
        1,   # return -1%, long signal -> incorrect (0)
        -1,  # return +3%, short signal -> incorrect (0)
        -1,  # return -2%, short signal -> correct (1)
        0,   # return 0%, neutral -> stays -99
        1,   # return +1%, long signal -> correct (1)
        -1,  # return -1.5%, short signal -> correct (1)
        1,   # return +2.5%, long signal -> correct (1)
        -1,  # return -0.5%, short signal -> correct (1)
        1,   # return +1.5%, long signal -> correct (1)
        1, 1, -1, -1, 0,
        1, -1, 1, -1, 1
    ])

    return pd.DataFrame({
        'close': close,
        'label_h5': primary_labels,
        'fwd_return_h5': fwd_returns
    })


class TestMetaLabelerInit:
    """Tests for MetaLabeler initialization."""

    def test_default_initialization(self):
        """Test default initialization with required params."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return'
        )
        assert labeler.labeling_type == LabelingType.META
        assert labeler.primary_signal_column == 'label_h5'
        assert labeler.bet_size_method == BetSizeMethod.PROBABILITY

    def test_fixed_bet_size_method(self):
        """Test initialization with fixed bet size method."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return',
            bet_size_method='fixed'
        )
        assert labeler.bet_size_method == BetSizeMethod.FIXED

    def test_enum_bet_size_method(self):
        """Test initialization with enum bet size method."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return',
            bet_size_method=BetSizeMethod.PROBABILITY
        )
        assert labeler.bet_size_method == BetSizeMethod.PROBABILITY

    def test_horizon_without_return_column(self):
        """Test initialization with horizon instead of return column."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=10
        )
        assert labeler._horizon == 10
        assert labeler._return_column is None

    def test_missing_both_return_and_horizon_raises(self):
        """Test that missing both return_column and horizon raises error."""
        with pytest.raises(ValueError, match="Either return_column or horizon"):
            MetaLabeler(primary_signal_column='label_h5')

    def test_invalid_primary_signal_column_raises(self):
        """Test that invalid primary_signal_column raises error."""
        with pytest.raises(ValueError, match="non-empty string"):
            MetaLabeler(primary_signal_column='', return_column='fwd_return')

        with pytest.raises(ValueError, match="non-empty string"):
            MetaLabeler(primary_signal_column=None, return_column='fwd_return')

    def test_invalid_bet_size_method_raises(self):
        """Test that invalid bet_size_method raises error."""
        with pytest.raises(ValueError, match="Invalid bet_size_method"):
            MetaLabeler(
                primary_signal_column='label_h5',
                return_column='fwd_return',
                bet_size_method='invalid_method'
            )

    def test_required_columns_with_return_column(self):
        """Test required columns when return_column is provided."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return'
        )
        assert 'label_h5' in labeler.required_columns
        assert 'fwd_return' in labeler.required_columns

    def test_required_columns_without_return_column(self):
        """Test required columns when using horizon."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        assert 'label_h5' in labeler.required_columns
        assert 'close' in labeler.required_columns


class TestMetaLabelComputation:
    """Tests for meta-label computation."""

    def test_compute_labels_basic(self, sample_df_with_labels):
        """Test basic meta-label computation."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(sample_df_with_labels, horizon=5)

        assert result.horizon == 5
        assert len(result.labels) == len(sample_df_with_labels)
        assert 'bet_size' in result.metadata
        assert 'correctness_margin' in result.metadata
        assert 'primary_signal' in result.metadata

    def test_perfect_signals_all_correct(self, perfect_signals_df):
        """Test that perfect signals produce all correct meta-labels."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(perfect_signals_df, horizon=5)

        # Exclude last 5 samples (invalid) and any potential neutrals
        valid_mask = result.labels != -99
        valid_labels = result.labels[valid_mask]

        # All should be correct (1)
        assert np.all(valid_labels == 1)

    def test_all_wrong_signals(self, all_wrong_signals_df):
        """Test that wrong signals produce incorrect meta-labels."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(all_wrong_signals_df, horizon=5)

        # Exclude last 5 samples (invalid)
        valid_mask = result.labels != -99
        valid_labels = result.labels[valid_mask]

        # All should be incorrect (0)
        assert np.all(valid_labels == 0)

    def test_explicit_return_column(self, mixed_signals_df):
        """Test meta-labeling with explicit return column."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return_h5'
        )
        result = labeler.compute_labels(mixed_signals_df, horizon=5)

        labels = result.labels

        # Check specific cases (first 10)
        # Index 0: +2% return, long signal -> correct (1)
        assert labels[0] == 1
        # Index 1: -1% return, long signal -> incorrect (0)
        assert labels[1] == 0
        # Index 2: +3% return, short signal -> incorrect (0)
        assert labels[2] == 0
        # Index 3: -2% return, short signal -> correct (1)
        assert labels[3] == 1
        # Index 4: 0% return, neutral signal -> stays -99
        assert labels[4] == -99
        # Index 5: +1% return, long signal -> correct (1)
        assert labels[5] == 1

    def test_neutrals_remain_invalid(self, mixed_signals_df):
        """Test that neutral primary signals produce invalid meta-labels."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return_h5'
        )
        result = labeler.compute_labels(mixed_signals_df, horizon=5)

        # Find neutral signals in primary
        primary = mixed_signals_df['label_h5'].values
        neutral_mask = primary == 0

        # Meta-labels for neutrals should be -99
        assert np.all(result.labels[neutral_mask] == -99)

    def test_last_horizon_samples_invalid(self, sample_df_with_labels):
        """Test that last horizon samples are marked invalid."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=10
        )
        result = labeler.compute_labels(sample_df_with_labels, horizon=10)

        # Last 10 samples should be -99
        assert np.all(result.labels[-10:] == -99)

    def test_horizon_validation(self, sample_df_with_labels):
        """Test horizon validation."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )

        with pytest.raises(ValueError, match="positive integer"):
            labeler.compute_labels(sample_df_with_labels, horizon=0)

        with pytest.raises(ValueError, match="positive integer"):
            labeler.compute_labels(sample_df_with_labels, horizon=-5)


class TestBetSizeComputation:
    """Tests for bet size computation."""

    def test_fixed_bet_size(self, mixed_signals_df):
        """Test fixed bet size method."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return_h5',
            bet_size_method='fixed'
        )
        result = labeler.compute_labels(mixed_signals_df, horizon=5)

        bet_sizes = result.metadata['bet_size']

        # For correct labels (1), bet size should be 1.0
        correct_mask = result.labels == 1
        assert np.all(bet_sizes[correct_mask] == 1.0)

        # For incorrect labels (0), bet size should be 0.0
        incorrect_mask = result.labels == 0
        assert np.all(bet_sizes[incorrect_mask] == 0.0)

    def test_probability_bet_size(self, mixed_signals_df):
        """Test probability bet size method."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return_h5',
            bet_size_method='probability'
        )
        result = labeler.compute_labels(mixed_signals_df, horizon=5)

        bet_sizes = result.metadata['bet_size']

        # Bet sizes should be proportional to return magnitude
        valid_mask = result.labels != -99
        valid_bet_sizes = bet_sizes[valid_mask]

        # All bet sizes should be non-negative
        assert np.all(valid_bet_sizes >= 0)


class TestCorrectnessMargin:
    """Tests for correctness margin computation."""

    def test_correctness_margin_values(self, mixed_signals_df):
        """Test correctness margin values."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return_h5'
        )
        result = labeler.compute_labels(mixed_signals_df, horizon=5)

        margins = result.metadata['correctness_margin']

        # For long signals, margin = return
        # Index 0: +2% return, long signal
        assert abs(margins[0] - 0.02) < 1e-6

        # For short signals, margin = -return (flipped)
        # Index 3: -2% return, short signal -> margin = 0.02
        assert abs(margins[3] - 0.02) < 1e-6


class TestMetaQualityMetrics:
    """Tests for quality metrics."""

    def test_quality_metrics_present(self, sample_df_with_labels):
        """Test that quality metrics are computed."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(sample_df_with_labels, horizon=5)

        metrics = result.quality_metrics

        assert 'total_samples' in metrics
        assert 'valid_samples' in metrics
        assert 'invalid_samples' in metrics
        assert 'correct_count' in metrics
        assert 'incorrect_count' in metrics
        assert 'accuracy' in metrics

    def test_accuracy_calculation(self, perfect_signals_df):
        """Test accuracy calculation."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(perfect_signals_df, horizon=5)

        metrics = result.quality_metrics

        # Perfect signals should have 100% accuracy
        assert metrics['accuracy'] == 1.0

    def test_accuracy_for_wrong_signals(self, all_wrong_signals_df):
        """Test accuracy for all wrong signals."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(all_wrong_signals_df, horizon=5)

        metrics = result.quality_metrics

        # All wrong signals should have 0% accuracy
        assert metrics['accuracy'] == 0.0

    def test_precision_recall_present(self, sample_df_with_labels):
        """Test that precision and recall are computed."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(sample_df_with_labels, horizon=5)

        metrics = result.quality_metrics

        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

    def test_correct_incorrect_ratio(self, sample_df_with_labels):
        """Test correct/incorrect ratio calculation."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(sample_df_with_labels, horizon=5)

        metrics = result.quality_metrics

        if metrics['incorrect_count'] > 0:
            expected_ratio = metrics['correct_count'] / metrics['incorrect_count']
            assert abs(metrics['correct_incorrect_ratio'] - expected_ratio) < 1e-6

    def test_margin_statistics(self, sample_df_with_labels):
        """Test margin statistics in metrics."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(sample_df_with_labels, horizon=5)

        metrics = result.quality_metrics

        # Check if margin statistics are present when applicable
        if metrics.get('correct_count', 0) > 0:
            assert 'avg_correct_margin' in metrics


class TestMetaInputValidation:
    """Tests for input validation."""

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises error."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            labeler.compute_labels(empty_df, horizon=5)

    def test_missing_primary_signal_column(self):
        """Test that missing primary signal column raises error."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        df = pd.DataFrame({'close': [100, 101, 102]})

        with pytest.raises(KeyError):
            labeler.compute_labels(df, horizon=5)

    def test_missing_close_for_return_computation(self):
        """Test that missing close column raises error when needed."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        df = pd.DataFrame({'label_h5': [1, -1, 0]})

        with pytest.raises(KeyError):
            labeler.compute_labels(df, horizon=5)

    def test_missing_return_column(self):
        """Test that missing return column raises error when specified."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return'
        )
        df = pd.DataFrame({
            'label_h5': [1, -1, 0],
            'close': [100, 101, 102]
        })

        with pytest.raises(KeyError):
            labeler.compute_labels(df, horizon=5)


class TestMetaAddToDataframe:
    """Tests for adding labels to DataFrame."""

    def test_adds_expected_columns(self, sample_df_with_labels):
        """Test that expected columns are added."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(sample_df_with_labels, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(sample_df_with_labels, result)

        assert 'meta_label_h5' in df_labeled.columns
        assert 'bet_size_h5' in df_labeled.columns
        assert 'correctness_margin_h5' in df_labeled.columns

    def test_custom_prefix(self, sample_df_with_labels):
        """Test adding labels with custom prefix."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(sample_df_with_labels, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(
            sample_df_with_labels, result, prefix='meta'
        )

        assert 'meta_h5' in df_labeled.columns

    def test_preserves_original_columns(self, sample_df_with_labels):
        """Test that original columns are preserved."""
        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(sample_df_with_labels, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(sample_df_with_labels, result)

        assert 'close' in df_labeled.columns
        assert 'label_h5' in df_labeled.columns
        assert 'volume' in df_labeled.columns


class TestMetaEdgeCases:
    """Tests for edge cases."""

    def test_all_neutral_signals(self):
        """Test handling of all neutral primary signals."""
        n = 20
        df = pd.DataFrame({
            'close': [100.0] * n,
            'label_h5': [0] * n  # All neutral
        })

        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(df, horizon=5)

        # All should be invalid (-99)
        assert np.all(result.labels == -99)

    def test_mixed_invalid_signals(self):
        """Test handling of mixed invalid signal values."""
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0] * 10,
            'label_h5': [1, -1, 0, -99, 2] * 10  # Includes -99 and invalid 2
        })

        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            horizon=5
        )
        result = labeler.compute_labels(df, horizon=5)

        # -99 and 2 in primary should remain -99 in meta
        primary = df['label_h5'].values
        invalid_primary_mask = (primary == 0) | (primary == -99) | (primary == 2)

        # First n-5 samples with invalid primary should have -99 meta-label
        for i in range(len(df) - 5):
            if invalid_primary_mask[i]:
                assert result.labels[i] == -99

    def test_zero_return_is_incorrect(self):
        """Test that zero return is treated as incorrect for both directions."""
        df = pd.DataFrame({
            'close': [100.0] * 20,
            'label_h5': [1, -1] * 10,
            'fwd_return': [0.0] * 20  # All zero returns
        })

        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return'
        )
        result = labeler.compute_labels(df, horizon=5)

        # Zero return means neither long nor short wins
        valid_mask = result.labels != -99
        valid_labels = result.labels[valid_mask]

        # All should be incorrect (0) since return is exactly 0
        assert np.all(valid_labels == 0)

    def test_nan_returns_handled(self):
        """Test handling of NaN returns."""
        n = 20
        returns = np.array([0.01, np.nan, 0.02, np.nan, 0.03] * 4)
        df = pd.DataFrame({
            'close': [100.0] * n,
            'label_h5': [1] * n,
            'fwd_return': returns
        })

        labeler = MetaLabeler(
            primary_signal_column='label_h5',
            return_column='fwd_return'
        )
        result = labeler.compute_labels(df, horizon=5)

        # NaN returns should produce invalid labels
        nan_mask = np.isnan(returns)
        assert np.all(result.labels[nan_mask] == -99)

        # Non-NaN returns with positive should be correct
        valid_positive_mask = (~nan_mask) & (returns > 0)
        # Exclude last horizon samples
        valid_positive_mask[-5:] = False
        if valid_positive_mask.sum() > 0:
            assert np.all(result.labels[valid_positive_mask] == 1)


class TestMetaFactoryIntegration:
    """Tests for factory integration."""

    def test_create_via_factory(self, sample_df_with_labels):
        """Test creating MetaLabeler via factory."""
        from src.phase1.stages.labeling import get_labeler, LabelingType

        labeler = get_labeler(
            LabelingType.META,
            primary_signal_column='label_h5',
            horizon=5
        )

        assert isinstance(labeler, MetaLabeler)

        result = labeler.compute_labels(sample_df_with_labels, horizon=5)
        assert result.horizon == 5

    def test_create_via_factory_string(self, sample_df_with_labels):
        """Test creating MetaLabeler via factory with string type."""
        from src.phase1.stages.labeling import get_labeler

        labeler = get_labeler(
            'meta',
            primary_signal_column='label_h5',
            horizon=5
        )

        assert isinstance(labeler, MetaLabeler)

    def test_available_strategies_includes_meta(self):
        """Test that META is in available strategies."""
        from src.phase1.stages.labeling import get_available_strategies, LabelingType

        strategies = get_available_strategies()
        assert LabelingType.META in strategies
