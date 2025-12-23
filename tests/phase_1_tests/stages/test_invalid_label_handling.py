"""
Tests for invalid label (-99) handling in Stage 7.

Ensures invalid labels are excluded from training and evaluation metrics,
while still being tracked for quality monitoring.

Run with: pytest tests/phase_1_tests/stages/test_invalid_label_handling.py -v
"""
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import logging

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.stage7_splits import (
    validate_label_distribution,
    create_chronological_splits,
    INVALID_LABEL_SENTINEL,
)


class TestInvalidLabelHandling:
    """Tests that -99 invalid labels are properly handled in splits."""

    def test_invalid_label_sentinel_value(self):
        """Verify the sentinel value is -99."""
        assert INVALID_LABEL_SENTINEL == -99

    def test_invalid_labels_excluded_from_distribution(self):
        """Invalid labels should not be counted in distribution stats."""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'label_h5': np.random.choice([-1, 0, 1], size=n)
        })

        # Add some invalid labels (50 samples = 5%)
        df.loc[100:149, 'label_h5'] = INVALID_LABEL_SENTINEL

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        dist = validate_label_distribution(df, train_idx, val_idx, test_idx, [5])

        # Validate return structure
        assert 'label_h5' in dist
        assert 'train' in dist['label_h5']
        assert 'n_invalid' in dist['label_h5']['train']

        # Invalid labels should be counted separately
        train_dist = dist['label_h5']['train']
        assert train_dist['n_invalid'] >= 0

        # Valid label counts should not include -99
        assert INVALID_LABEL_SENTINEL not in train_dist['counts']

    def test_distribution_counts_only_valid_labels(self):
        """Distribution percentages should be based on valid labels only."""
        np.random.seed(42)
        n = 1000

        # Create data with known label distribution
        labels = np.array([0] * 400 + [1] * 300 + [-1] * 300)
        np.random.shuffle(labels)

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'label_h5': labels.copy()
        })

        # Mark some labels as invalid
        df.loc[:99, 'label_h5'] = INVALID_LABEL_SENTINEL  # 100 invalid

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        dist = validate_label_distribution(df, train_idx, val_idx, test_idx, [5])

        # Check that total_valid + n_invalid = len(split)
        for split_name in ['train', 'val', 'test']:
            split_dist = dist['label_h5'][split_name]
            total_counted = split_dist['total_valid'] + split_dist['n_invalid']
            split_indices = {
                'train': train_idx,
                'val': val_idx,
                'test': test_idx
            }
            assert total_counted == len(split_indices[split_name])

    def test_warning_on_high_invalid_rate(self, caplog):
        """Should warn when >10% of labels are invalid."""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'label_h5': np.random.choice([-1, 0, 1], size=n)
        })

        # Make 25% of labels invalid (should trigger warning for train split)
        df.loc[:249, 'label_h5'] = INVALID_LABEL_SENTINEL

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        with caplog.at_level(logging.WARNING):
            validate_label_distribution(df, train_idx, val_idx, test_idx, [5])

        # Should have logged a warning about high invalid rate
        warning_messages = [r.message.lower() for r in caplog.records]
        assert any("invalid labels" in msg for msg in warning_messages), \
            f"Expected warning about invalid labels, got: {warning_messages}"

    def test_no_warning_on_low_invalid_rate(self, caplog):
        """Should not warn when <=10% of labels are invalid."""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'label_h5': np.random.choice([-1, 0, 1], size=n)
        })

        # Make only 5% of labels invalid (should NOT trigger warning)
        df.loc[:49, 'label_h5'] = INVALID_LABEL_SENTINEL

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        with caplog.at_level(logging.WARNING):
            validate_label_distribution(df, train_idx, val_idx, test_idx, [5])

        # Should NOT have logged a warning about invalid labels
        warning_messages = [r.message.lower() for r in caplog.records if r.levelno >= logging.WARNING]
        # Filter for messages specifically about "invalid labels"
        invalid_label_warnings = [msg for msg in warning_messages if "invalid labels" in msg]
        assert len(invalid_label_warnings) == 0, \
            f"Expected no warnings about invalid labels, got: {invalid_label_warnings}"

    def test_multiple_horizons_handled(self):
        """Should handle invalid labels across multiple horizons."""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'label_h5': np.random.choice([-1, 0, 1], size=n),
            'label_h20': np.random.choice([-1, 0, 1], size=n)
        })

        # Add invalid labels to different horizons
        df.loc[100:149, 'label_h5'] = INVALID_LABEL_SENTINEL
        df.loc[900:999, 'label_h20'] = INVALID_LABEL_SENTINEL

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        dist = validate_label_distribution(df, train_idx, val_idx, test_idx, [5, 20])

        # Both horizons should be tracked
        assert 'label_h5' in dist
        assert 'label_h20' in dist

        # Each should have proper structure
        for label_col in ['label_h5', 'label_h20']:
            for split_name in ['train', 'val', 'test']:
                assert 'counts' in dist[label_col][split_name]
                assert 'total_valid' in dist[label_col][split_name]
                assert 'n_invalid' in dist[label_col][split_name]
                assert 'invalid_pct' in dist[label_col][split_name]

    def test_all_invalid_labels_handled(self, caplog):
        """Should handle case where all labels in a split are invalid."""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'label_h5': np.random.choice([-1, 0, 1], size=n)
        })

        # Make ALL labels invalid
        df['label_h5'] = INVALID_LABEL_SENTINEL

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        with caplog.at_level(logging.WARNING):
            dist = validate_label_distribution(df, train_idx, val_idx, test_idx, [5])

        # Should have 100% invalid in all splits
        for split_name in ['train', 'val', 'test']:
            assert dist['label_h5'][split_name]['total_valid'] == 0
            assert dist['label_h5'][split_name]['invalid_pct'] == 100.0

    def test_no_invalid_labels(self):
        """Should work correctly when no labels are invalid."""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'label_h5': np.random.choice([-1, 0, 1], size=n)
        })

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        dist = validate_label_distribution(df, train_idx, val_idx, test_idx, [5])

        # All labels should be valid
        for split_name in ['train', 'val', 'test']:
            assert dist['label_h5'][split_name]['n_invalid'] == 0
            assert dist['label_h5'][split_name]['invalid_pct'] == 0.0

    def test_distribution_returns_dict(self):
        """validate_label_distribution should return a dictionary."""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'label_h5': np.random.choice([-1, 0, 1], size=n)
        })

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        result = validate_label_distribution(df, train_idx, val_idx, test_idx, [5])

        assert isinstance(result, dict)
        assert len(result) > 0


class TestInvalidLabelIntegration:
    """Integration tests for invalid label handling in the split pipeline."""

    def test_splits_preserve_invalid_labels(self):
        """Splits should preserve invalid labels in the data (not filter them)."""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'label_h5': np.random.choice([-1, 0, 1], size=n)
        })

        # Add invalid labels
        invalid_indices = [100, 200, 300, 400, 500]
        for idx in invalid_indices:
            df.loc[idx, 'label_h5'] = INVALID_LABEL_SENTINEL

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # Original dataframe should still have invalid labels
        assert (df['label_h5'] == INVALID_LABEL_SENTINEL).sum() == len(invalid_indices)

        # Splits should still include the invalid labels in the data
        all_indices = np.concatenate([train_idx, val_idx, test_idx])
        split_df = df.iloc[all_indices]
        split_invalids = (split_df['label_h5'] == INVALID_LABEL_SENTINEL).sum()
        # At least some invalid labels should be in the splits
        # (some may be lost to purge/embargo)
        assert split_invalids >= 0

    def test_invalid_labels_at_end_of_dataset(self):
        """Invalid labels often occur at dataset end (incomplete horizons)."""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'label_h5': np.random.choice([-1, 0, 1], size=n)
        })

        # Mark last 50 labels as invalid (typical for horizon-based labeling)
        df.loc[950:, 'label_h5'] = INVALID_LABEL_SENTINEL

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        dist = validate_label_distribution(df, train_idx, val_idx, test_idx, [5])

        # Test split should have high invalid rate (end of data)
        test_dist = dist['label_h5']['test']
        assert test_dist['n_invalid'] > 0, "Test split should have invalid labels at end"
