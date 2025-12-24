"""
Unit tests for Label Sanity Validation.

Tests the label sanity validation module which checks:
- Label distribution
- Label balance
- Per-symbol distribution
- Bars-to-hit statistics
- Quality score statistics

Run with: pytest tests/phase_1_tests/validators/test_label_validator.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.validation.labels import (
    check_label_distribution,
    check_label_balance,
    check_per_symbol_distribution,
    check_bars_to_hit,
    check_quality_scores,
    check_label_sanity,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def balanced_labels_df():
    """Create DataFrame with balanced labels (33% each class)."""
    np.random.seed(42)
    n = 300

    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'symbol': 'MES',
        'close': 100 + np.random.randn(n) * 0.5,
        'label_h5': np.array([-1, 0, 1] * 100),  # Perfectly balanced
        'label_h20': np.array([-1, 0, 1] * 100),
        'bars_to_hit_h5': np.random.randint(1, 5, n),
        'bars_to_hit_h20': np.random.randint(1, 20, n),
        'quality_h5': np.random.uniform(0.5, 1.0, n),
        'quality_h20': np.random.uniform(0.5, 1.0, n),
    })

    return df


@pytest.fixture
def imbalanced_labels_df():
    """Create DataFrame with imbalanced labels."""
    np.random.seed(42)
    n = 100

    # 80% neutral, 10% long, 10% short
    labels = np.random.choice([-1, 0, 1], size=n, p=[0.1, 0.8, 0.1])

    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'symbol': 'MES',
        'close': 100 + np.random.randn(n) * 0.5,
        'label_h5': labels,
    })

    return df


@pytest.fixture
def multi_symbol_labels_df():
    """Create multi-symbol DataFrame with labels."""
    np.random.seed(42)
    n = 100

    dfs = []
    for symbol in ['MES', 'MGC']:
        if symbol == 'MES':
            # More longs for MES
            labels = np.random.choice([-1, 0, 1], size=n, p=[0.2, 0.3, 0.5])
        else:
            # More shorts for MGC
            labels = np.random.choice([-1, 0, 1], size=n, p=[0.5, 0.3, 0.2])

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': symbol,
            'close': 100 + np.random.randn(n) * 0.5,
            'label_h5': labels,
            'bars_to_hit_h5': np.random.randint(1, 5, n),
            'quality_h5': np.random.uniform(0.5, 1.0, n),
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# CHECK LABEL DISTRIBUTION TESTS
# =============================================================================

class TestCheckLabelDistribution:
    """Tests for check_label_distribution function."""

    def test_returns_all_classes(self, balanced_labels_df):
        """Test that all label classes are included."""
        result = check_label_distribution(balanced_labels_df, 'label_h5')

        assert 'long' in result
        assert 'short' in result
        assert 'neutral' in result

    def test_correct_counts_balanced(self, balanced_labels_df):
        """Test correct count for balanced labels."""
        result = check_label_distribution(balanced_labels_df, 'label_h5')

        # 300 rows, 100 each
        assert result['long']['count'] == 100
        assert result['short']['count'] == 100
        assert result['neutral']['count'] == 100

    def test_correct_percentages(self, balanced_labels_df):
        """Test correct percentage calculation."""
        result = check_label_distribution(balanced_labels_df, 'label_h5')

        # Each should be ~33.33%
        for label_name in ['long', 'short', 'neutral']:
            pct = result[label_name]['percentage']
            assert abs(pct - 33.33) < 1  # Within 1%

    def test_imbalanced_distribution(self, imbalanced_labels_df):
        """Test distribution with imbalanced labels."""
        result = check_label_distribution(imbalanced_labels_df, 'label_h5')

        # Neutral should dominate
        assert result['neutral']['percentage'] > 60

    def test_binary_labels(self):
        """Test with binary labels (only 0 and 1)."""
        df = pd.DataFrame({
            'label_h5': [0, 0, 0, 1, 1],
        })

        result = check_label_distribution(df, 'label_h5')

        assert 'neutral' in result
        assert 'long' in result
        assert result['neutral']['count'] == 3
        assert result['long']['count'] == 2


# =============================================================================
# CHECK LABEL BALANCE TESTS
# =============================================================================

class TestCheckLabelBalance:
    """Tests for check_label_balance function."""

    def test_no_warning_for_balanced(self):
        """Test no warnings for balanced distribution."""
        label_dist = {
            'long': {'count': 100, 'percentage': 33.33},
            'short': {'count': 100, 'percentage': 33.33},
            'neutral': {'count': 100, 'percentage': 33.34},
        }

        warnings_found = []
        check_label_balance(label_dist, horizon=5, warnings_found=warnings_found)

        assert len(warnings_found) == 0

    def test_warning_for_low_representation(self):
        """Test warning for class with <20% representation."""
        label_dist = {
            'long': {'count': 10, 'percentage': 10.0},  # Low
            'short': {'count': 20, 'percentage': 20.0},
            'neutral': {'count': 70, 'percentage': 70.0},  # High
        }

        warnings_found = []
        check_label_balance(label_dist, horizon=5, warnings_found=warnings_found)

        # Should warn about low long and high neutral
        assert len(warnings_found) >= 2
        low_warnings = [w for w in warnings_found if 'low' in w.lower()]
        high_warnings = [w for w in warnings_found if 'high' in w.lower()]
        assert len(low_warnings) >= 1
        assert len(high_warnings) >= 1

    def test_warning_for_high_representation(self):
        """Test warning for class with >60% representation."""
        label_dist = {
            'long': {'count': 20, 'percentage': 20.0},
            'short': {'count': 15, 'percentage': 15.0},  # Low
            'neutral': {'count': 65, 'percentage': 65.0},  # High
        }

        warnings_found = []
        check_label_balance(label_dist, horizon=20, warnings_found=warnings_found)

        high_warnings = [w for w in warnings_found if 'high' in w.lower()]
        assert len(high_warnings) >= 1
        assert 'h20' in high_warnings[0]  # Should include horizon

    def test_horizon_in_warning_message(self):
        """Test that horizon is included in warning message."""
        label_dist = {
            'long': {'count': 5, 'percentage': 5.0},
            'neutral': {'count': 95, 'percentage': 95.0},
        }

        warnings_found = []
        check_label_balance(label_dist, horizon=42, warnings_found=warnings_found)

        # Horizon should be in the warning
        assert any('h42' in w for w in warnings_found)


# =============================================================================
# CHECK PER SYMBOL DISTRIBUTION TESTS
# =============================================================================

class TestCheckPerSymbolDistribution:
    """Tests for check_per_symbol_distribution function."""

    def test_returns_stats_per_symbol(self, multi_symbol_labels_df):
        """Test that stats are returned per symbol."""
        result = check_per_symbol_distribution(
            multi_symbol_labels_df, 'label_h5'
        )

        assert 'MES' in result
        assert 'MGC' in result

    def test_correct_structure(self, multi_symbol_labels_df):
        """Test correct structure of per-symbol stats."""
        result = check_per_symbol_distribution(
            multi_symbol_labels_df, 'label_h5'
        )

        for symbol, stats in result.items():
            assert 'total' in stats
            assert 'long_count' in stats
            assert 'short_count' in stats
            assert 'neutral_count' in stats
            assert 'long_pct' in stats
            assert 'short_pct' in stats
            assert 'neutral_pct' in stats

    def test_counts_sum_to_total(self, multi_symbol_labels_df):
        """Test that counts sum to total."""
        result = check_per_symbol_distribution(
            multi_symbol_labels_df, 'label_h5'
        )

        for symbol, stats in result.items():
            total = stats['total']
            count_sum = (
                stats['long_count'] +
                stats['short_count'] +
                stats['neutral_count']
            )
            assert count_sum == total

    def test_percentages_sum_to_100(self, multi_symbol_labels_df):
        """Test that percentages sum to 100."""
        result = check_per_symbol_distribution(
            multi_symbol_labels_df, 'label_h5'
        )

        for symbol, stats in result.items():
            pct_sum = (
                stats['long_pct'] +
                stats['short_pct'] +
                stats['neutral_pct']
            )
            assert abs(pct_sum - 100.0) < 0.1

    def test_different_distributions_per_symbol(self, multi_symbol_labels_df):
        """Test that different symbols have different distributions."""
        result = check_per_symbol_distribution(
            multi_symbol_labels_df, 'label_h5'
        )

        # MES should have more longs, MGC more shorts
        assert result['MES']['long_pct'] > result['MGC']['long_pct']
        assert result['MGC']['short_pct'] > result['MES']['short_pct']


# =============================================================================
# CHECK BARS TO HIT TESTS
# =============================================================================

class TestCheckBarsToHit:
    """Tests for check_bars_to_hit function."""

    def test_returns_correct_structure(self, balanced_labels_df):
        """Test that correct stats are returned."""
        result = check_bars_to_hit(
            balanced_labels_df, 'label_h5', 'bars_to_hit_h5'
        )

        assert 'mean_all' in result
        assert 'median_all' in result
        assert 'mean_hit' in result
        assert 'median_hit' in result

    def test_mean_hit_excludes_neutral(self):
        """Test that mean_hit excludes neutral labels."""
        df = pd.DataFrame({
            'label_h5': [1, -1, 0, 0, 0],  # 2 hits, 3 neutral
            'bars_to_hit_h5': [2, 4, 5, 5, 5],  # hits: 2,4 avg=3
        })

        result = check_bars_to_hit(df, 'label_h5', 'bars_to_hit_h5')

        assert result['mean_hit'] == 3.0  # Average of 2 and 4

    def test_all_neutral_labels(self):
        """Test handling when all labels are neutral."""
        df = pd.DataFrame({
            'label_h5': [0, 0, 0, 0, 0],
            'bars_to_hit_h5': [1, 2, 3, 4, 5],
        })

        result = check_bars_to_hit(df, 'label_h5', 'bars_to_hit_h5')

        # When no hits, mean_hit falls back to mean_all
        assert result['mean_hit'] == result['mean_all']

    def test_float_values(self):
        """Test with float values for bars_to_hit."""
        df = pd.DataFrame({
            'label_h5': [1, 1, -1],
            'bars_to_hit_h5': [2.5, 3.5, 4.0],
        })

        result = check_bars_to_hit(df, 'label_h5', 'bars_to_hit_h5')

        assert isinstance(result['mean_all'], float)
        assert isinstance(result['median_all'], float)


# =============================================================================
# CHECK QUALITY SCORES TESTS
# =============================================================================

class TestCheckQualityScores:
    """Tests for check_quality_scores function."""

    def test_returns_correct_structure(self, balanced_labels_df):
        """Test that correct stats are returned."""
        result = check_quality_scores(balanced_labels_df, 'quality_h5')

        assert 'mean' in result
        assert 'median' in result
        assert 'std' in result
        assert 'min' in result
        assert 'max' in result

    def test_correct_calculations(self):
        """Test correct statistical calculations."""
        df = pd.DataFrame({
            'quality_h5': [0.5, 0.6, 0.7, 0.8, 0.9],
        })

        result = check_quality_scores(df, 'quality_h5')

        assert result['mean'] == pytest.approx(0.7, abs=0.001)
        assert result['median'] == pytest.approx(0.7, abs=0.001)
        assert result['min'] == pytest.approx(0.5, abs=0.001)
        assert result['max'] == pytest.approx(0.9, abs=0.001)

    def test_std_calculation(self):
        """Test standard deviation calculation."""
        df = pd.DataFrame({
            'quality_h5': [0.5, 0.5, 0.5, 0.5, 0.5],  # Constant
        })

        result = check_quality_scores(df, 'quality_h5')

        assert result['std'] == 0.0

    def test_handles_negative_values(self):
        """Test handling of negative quality values."""
        df = pd.DataFrame({
            'quality_h5': [-0.5, 0.0, 0.5, 1.0],
        })

        result = check_quality_scores(df, 'quality_h5')

        assert result['min'] == pytest.approx(-0.5, abs=0.001)
        assert result['max'] == pytest.approx(1.0, abs=0.001)


# =============================================================================
# CHECK LABEL SANITY TESTS
# =============================================================================

class TestCheckLabelSanity:
    """Tests for check_label_sanity function."""

    def test_returns_results_per_horizon(self, balanced_labels_df):
        """Test that results are returned per horizon."""
        warnings_found = []

        result = check_label_sanity(
            balanced_labels_df,
            horizons=[5, 20],
            warnings_found=warnings_found
        )

        assert 'horizon_5' in result
        assert 'horizon_20' in result

    def test_includes_distribution(self, balanced_labels_df):
        """Test that distribution is included."""
        warnings_found = []

        result = check_label_sanity(
            balanced_labels_df, horizons=[5], warnings_found=warnings_found
        )

        assert 'distribution' in result['horizon_5']

    def test_includes_per_symbol_with_symbol_column(self, multi_symbol_labels_df):
        """Test per-symbol stats when symbol column exists."""
        warnings_found = []

        result = check_label_sanity(
            multi_symbol_labels_df, horizons=[5], warnings_found=warnings_found
        )

        assert 'per_symbol' in result['horizon_5']

    def test_includes_bars_to_hit_when_column_exists(self, balanced_labels_df):
        """Test bars_to_hit stats when column exists."""
        warnings_found = []

        result = check_label_sanity(
            balanced_labels_df, horizons=[5], warnings_found=warnings_found
        )

        assert 'bars_to_hit' in result['horizon_5']

    def test_includes_quality_when_column_exists(self, balanced_labels_df):
        """Test quality stats when column exists."""
        warnings_found = []

        result = check_label_sanity(
            balanced_labels_df, horizons=[5], warnings_found=warnings_found
        )

        assert 'quality' in result['horizon_5']

    def test_skips_missing_label_column(self):
        """Test skipping when label column is missing."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'close': [100] * 10,
        })

        warnings_found = []
        result = check_label_sanity(df, horizons=[5], warnings_found=warnings_found)

        # Should not have horizon_5 since label_h5 is missing
        assert 'horizon_5' not in result

    def test_populates_warnings_for_imbalance(self, imbalanced_labels_df):
        """Test that warnings are populated for imbalanced labels."""
        warnings_found = []

        check_label_sanity(
            imbalanced_labels_df, horizons=[5], warnings_found=warnings_found
        )

        # Should have warnings about imbalance
        assert len(warnings_found) > 0

    def test_no_per_symbol_without_symbol_column(self):
        """Test no per-symbol stats without symbol column."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'close': [100] * 10,
            'label_h5': [0, 1, -1] * 3 + [0],
        })

        warnings_found = []
        result = check_label_sanity(df, horizons=[5], warnings_found=warnings_found)

        assert 'per_symbol' not in result.get('horizon_5', {})


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestLabelValidatorEdgeCases:
    """Edge case tests for label validator."""

    def test_single_label_value(self):
        """Test with only one label value."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'symbol': 'MES',
            'label_h5': [0] * 100,  # All neutral
        })

        warnings_found = []
        result = check_label_sanity(df, horizons=[5], warnings_found=warnings_found)

        assert result['horizon_5']['distribution']['neutral']['percentage'] == 100.0
        # Should warn about 100% representation
        assert len(warnings_found) > 0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({
            'datetime': pd.Series([], dtype='datetime64[ns]'),
            'label_h5': pd.Series([], dtype='int64'),
        })

        warnings_found = []
        # Should handle empty gracefully
        result = check_label_sanity(df, horizons=[5], warnings_found=warnings_found)

    def test_non_standard_label_values(self):
        """Test with non-standard label values (not -1, 0, 1)."""
        df = pd.DataFrame({
            'label_h5': [0, 1, 2, 3],  # Non-standard
        })

        result = check_label_distribution(df, 'label_h5')

        # Should still work, just use the values as keys
        assert '0' in result or 0 in result or 'neutral' in result

    def test_nan_in_labels(self):
        """Test handling of NaN values in labels."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'symbol': 'MES',
            'label_h5': [0, 1, -1, np.nan, 0, 1, -1, 0, 0, np.nan],
        })

        warnings_found = []
        # Should not raise
        result = check_label_sanity(df, horizons=[5], warnings_found=warnings_found)

    def test_multiple_horizons(self, balanced_labels_df):
        """Test with multiple horizons."""
        warnings_found = []

        result = check_label_sanity(
            balanced_labels_df,
            horizons=[5, 20],
            warnings_found=warnings_found
        )

        assert 'horizon_5' in result
        assert 'horizon_20' in result

        # Both should have distributions
        assert 'distribution' in result['horizon_5']
        assert 'distribution' in result['horizon_20']

    def test_float_labels(self):
        """Test handling of float labels (should be cast to int)."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'symbol': 'MES',
            'label_h5': [0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        })

        warnings_found = []
        result = check_label_sanity(df, horizons=[5], warnings_found=warnings_found)

        # Should work with float labels
        assert 'distribution' in result['horizon_5']
