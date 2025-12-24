"""
Tests for Critical Data Leakage Fixes (2025-12-21).

This module tests two critical fixes to prevent data leakage:

1. Critical Issue #7: Cross-asset rolling statistics using full arrays
   - Tests that cross-asset features validate array lengths
   - Tests that proper warnings are logged for mismatched arrays
   - Documents proper usage patterns

2. Critical Issue #8: Last bars handling in triple barrier labeling
   - Tests that last max_bars samples are marked as invalid (-99)
   - Tests that invalid samples are excluded from statistics
   - Tests that edge case leakage is prevented

Run with: pytest tests/phase_1_tests/test_critical_leakage_fixes.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.labeling import triple_barrier_numba, apply_triple_barrier
from src.phase1.stages.features.cross_asset import add_cross_asset_features


# =============================================================================
# CRITICAL ISSUE #7: Cross-Asset Rolling Statistics Tests
# =============================================================================

class TestCrossAssetLeakagePrevention:
    """Tests for cross-asset feature leakage prevention."""

    def test_cross_asset_requires_matching_lengths(self):
        """Test that cross-asset features require matching array lengths."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'close': np.random.randn(100) * 10 + 100
        })

        # Mismatched lengths - this should be rejected
        mes_close = np.random.randn(100) * 10 + 4500
        mgc_close = np.random.randn(80) * 10 + 2000  # Wrong length!

        feature_metadata = {}

        # Act
        result_df = add_cross_asset_features(
            df,
            feature_metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Assert - Should skip cross-asset features and set to NaN
        assert 'mes_mgc_correlation_20' in result_df.columns
        assert result_df['mes_mgc_correlation_20'].isna().all()
        assert result_df['mes_mgc_spread_zscore'].isna().all()
        assert result_df['mes_mgc_beta'].isna().all()
        assert result_df['relative_strength'].isna().all()

    def test_cross_asset_rejects_empty_arrays(self):
        """Test that cross-asset features reject empty arrays."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'close': np.random.randn(100) * 10 + 100
        })

        # Empty arrays
        mes_close = np.array([])
        mgc_close = np.array([])

        feature_metadata = {}

        # Act
        result_df = add_cross_asset_features(
            df,
            feature_metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Assert - Should skip and set to NaN
        assert result_df['mes_mgc_correlation_20'].isna().all()

    def test_cross_asset_correct_length_validation_message(self, caplog):
        """Test that cross-asset features log validation messages."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'close': np.random.randn(100) * 10 + 100
        })

        mes_close = np.random.randn(100) * 10 + 4500
        mgc_close = np.random.randn(100) * 10 + 2000

        feature_metadata = {}

        # Act
        with caplog.at_level('INFO'):
            result_df = add_cross_asset_features(
                df,
                feature_metadata,
                mes_close=mes_close,
                mgc_close=mgc_close,
                current_symbol='MES'
            )

        # Assert - Should log validation
        log_messages = [record.message for record in caplog.records]
        assert any('Array lengths validated' in msg for msg in log_messages)

    def test_cross_asset_subset_usage_documentation(self):
        """
        Document correct usage of cross-asset features with train/val/test splits.

        This test serves as documentation for proper usage patterns.
        """
        # SCENARIO: Using cross-asset features AFTER splitting (e.g., in Phase 2)

        # Arrange - Simulate full dataset and train split
        full_size = 1000
        train_size = 700

        # Full dataset arrays (what Stage 3 would have)
        mes_full = np.random.randn(full_size) * 10 + 4500
        mgc_full = np.random.randn(full_size) * 10 + 2000

        # Train split dataframe
        df_train = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=train_size, freq='5min'),
            'close': np.random.randn(train_size) * 10 + 100
        })

        # CORRECT: Use subset arrays matching the train split
        mes_train = mes_full[:train_size]
        mgc_train = mgc_full[:train_size]

        feature_metadata = {}

        # Act - This is the CORRECT way
        result_correct = add_cross_asset_features(
            df_train,
            feature_metadata,
            mes_close=mes_train,
            mgc_close=mgc_train,
            current_symbol='MES'
        )

        # Assert - Should work without issues
        assert len(result_correct) == train_size
        # Cross-asset features should be computed (not all NaN after warmup)
        # Check last values which should be valid after warmup period
        assert not result_correct['mes_mgc_correlation_20'].iloc[-1:].isna().all()

        # INCORRECT: Using full arrays on train subset would cause leakage
        # This is what we're PREVENTING with the length validation
        # The function will detect the mismatch and skip features (set to NaN)
        result_incorrect = add_cross_asset_features(
            df_train,
            {},
            mes_close=mes_full,  # WRONG - using full array
            mgc_close=mgc_full,  # WRONG - using full array
            current_symbol='MES'
        )

        # The function detects this and skips features (sets to NaN)
        assert result_incorrect['mes_mgc_correlation_20'].isna().all()


# =============================================================================
# CRITICAL ISSUE #8: Last Bars Edge Case Tests
# =============================================================================

class TestLastBarsLeakagePrevention:
    """Tests for last bars edge case leakage prevention."""

    def test_last_max_bars_marked_invalid(self):
        """Test that last max_bars samples are marked with label=-99."""
        # Arrange
        n = 100
        max_bars = 20

        close = np.ones(n) * 100.0
        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0

        k_up = 2.0
        k_down = 2.0

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up, k_down, max_bars
        )

        # Assert - Last max_bars samples should be -99 (invalid)
        expected_invalid_start = n - max_bars
        assert np.all(labels[expected_invalid_start:] == -99)

        # Assert - Earlier samples should be valid (not -99)
        # Note: They might be 0, 1, or -1 depending on barrier hits
        assert np.all(labels[:expected_invalid_start] != -99)

    def test_invalid_samples_excluded_from_statistics(self):
        """Test that apply_triple_barrier excludes invalid samples from stats."""
        # Arrange
        n = 100
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': np.ones(n) * 100.0,
            'high': np.ones(n) * 101.0,
            'low': np.ones(n) * 99.0,
            'open': np.ones(n) * 100.0,
            'atr_14': np.ones(n) * 2.0
        })

        horizon = 5
        max_bars = 20

        # Act
        result_df = apply_triple_barrier(
            df,
            horizon=horizon,
            k_up=2.0,
            k_down=2.0,
            max_bars=max_bars
        )

        # Assert - Last max_bars should be invalid
        label_col = f'label_h{horizon}'
        invalid_mask = result_df[label_col] == -99
        num_invalid = invalid_mask.sum()

        assert num_invalid == max_bars

        # Assert - Invalid samples are at the end
        expected_invalid_start = n - max_bars
        assert np.all(result_df[label_col].iloc[expected_invalid_start:] == -99)

    def test_no_spurious_timeout_at_end(self):
        """Test that last bars are NOT labeled as timeout (0)."""
        # Arrange
        n = 100
        max_bars = 20

        close = np.ones(n) * 100.0
        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0

        k_up = 2.0
        k_down = 2.0

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up, k_down, max_bars
        )

        # Assert - Last bars should NOT be labeled as 0 (timeout)
        # They should be -99 (invalid)
        expected_invalid_start = n - max_bars
        assert not np.any(labels[expected_invalid_start:] == 0)
        assert np.all(labels[expected_invalid_start:] == -99)

    def test_valid_samples_count_correct(self):
        """Test that the count of valid samples is correct."""
        # Arrange
        n = 100
        max_bars = 20

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': np.ones(n) * 100.0,
            'high': np.ones(n) * 101.0,
            'low': np.ones(n) * 99.0,
            'open': np.ones(n) * 100.0,
            'atr_14': np.ones(n) * 2.0
        })

        # Act
        result_df = apply_triple_barrier(
            df,
            horizon=5,
            k_up=2.0,
            k_down=2.0,
            max_bars=max_bars
        )

        # Assert - Number of valid samples
        label_col = 'label_h5'
        valid_samples = result_df[result_df[label_col] != -99]

        expected_valid_count = n - max_bars
        assert len(valid_samples) == expected_valid_count

    def test_invalid_labels_filtering_example(self):
        """
        Example test showing how to filter invalid labels for model training.

        This test serves as documentation for downstream usage.
        """
        # Arrange
        n = 100
        max_bars = 20

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': np.random.randn(n) * 10 + 100,
            'high': np.random.randn(n) * 10 + 101,
            'low': np.random.randn(n) * 10 + 99,
            'open': np.random.randn(n) * 10 + 100,
            'atr_14': np.ones(n) * 2.0,
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n)
        })

        # Apply labeling
        df = apply_triple_barrier(
            df,
            horizon=5,
            k_up=2.0,
            k_down=2.0,
            max_bars=max_bars
        )

        # Act - CORRECT way to filter for model training
        label_col = 'label_h5'
        df_train_ready = df[df[label_col] != -99].copy()

        # Assert - Filtered dataset is ready for training
        assert len(df_train_ready) == n - max_bars
        assert not (df_train_ready[label_col] == -99).any()
        assert df_train_ready[label_col].isin([-1, 0, 1]).all()

    def test_different_horizons_different_invalid_counts(self):
        """Test that different horizons have different numbers of invalid samples."""
        # Arrange
        n = 100
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': np.ones(n) * 100.0,
            'high': np.ones(n) * 101.0,
            'low': np.ones(n) * 99.0,
            'open': np.ones(n) * 100.0,
            'atr_14': np.ones(n) * 2.0
        })

        # Apply different horizons with different max_bars
        df = apply_triple_barrier(df, horizon=5, k_up=2.0, k_down=2.0, max_bars=10)
        df = apply_triple_barrier(df, horizon=20, k_up=2.0, k_down=2.0, max_bars=60)

        # Assert - H5 should have 10 invalid, H20 should have 60 invalid
        h5_invalid = (df['label_h5'] == -99).sum()
        h20_invalid = (df['label_h20'] == -99).sum()

        assert h5_invalid == 10
        assert h20_invalid == 60


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestLeakagePreventionIntegration:
    """Integration tests for both leakage prevention fixes."""

    def test_combined_cross_asset_and_labeling_workflow(self):
        """
        Test complete workflow: cross-asset features + labeling with proper handling.
        """
        # Arrange - Create aligned multi-symbol data
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='5min')

        # MES data
        mes_df = pd.DataFrame({
            'datetime': dates,
            'close': np.random.randn(n) * 10 + 4500,
            'high': np.random.randn(n) * 10 + 4510,
            'low': np.random.randn(n) * 10 + 4490,
            'open': np.random.randn(n) * 10 + 4500,
            'atr_14': np.ones(n) * 20.0
        })

        # MGC data (aligned)
        mgc_df = pd.DataFrame({
            'datetime': dates,
            'close': np.random.randn(n) * 10 + 2000,
        })

        # Act - Add cross-asset features (Stage 3 behavior)
        feature_metadata = {}
        mes_df = add_cross_asset_features(
            mes_df,
            feature_metadata,
            mes_close=mes_df['close'].values,
            mgc_close=mgc_df['close'].values,
            current_symbol='MES'
        )

        # Apply labeling (Stage 4 behavior)
        max_bars = 60
        mes_df = apply_triple_barrier(
            mes_df,
            horizon=20,
            k_up=2.0,
            k_down=2.0,
            max_bars=max_bars
        )

        # Filter for training (what Phase 2 would do)
        mes_train = mes_df[mes_df['label_h20'] != -99].copy()

        # Assert - Training data is clean
        assert len(mes_train) == n - max_bars
        assert not (mes_train['label_h20'] == -99).any()
        assert mes_train['label_h20'].isin([-1, 0, 1]).all()

        # Assert - Cross-asset features are present
        assert 'mes_mgc_correlation_20' in mes_train.columns
        assert 'mes_mgc_spread_zscore' in mes_train.columns
        assert 'mes_mgc_beta' in mes_train.columns
        assert 'relative_strength' in mes_train.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
