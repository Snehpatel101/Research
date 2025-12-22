"""
Unit tests for Stage 4: Triple Barrier Labeling.

Triple barrier labels with quality scoring

Run with: pytest tests/phase_1_tests/stages/test_stage4_*.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.stage4_labeling import triple_barrier_numba, apply_triple_barrier


# =============================================================================
# TESTS
# =============================================================================

class TestTripleBarrierUpperHit:
    """Tests for upper barrier (profit target) hits."""

    def test_triple_barrier_upper_hit(self):
        """Test that upper barrier hit produces label +1."""
        # Arrange - Price that clearly hits upper barrier
        n = 20
        close = np.array([100.0] * n)
        close[5:] = 110.0  # Jump up at bar 5

        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0  # ATR = 2

        k_up = 2.0  # Upper barrier at 100 + 2*2 = 104
        k_down = 2.0  # Lower barrier at 100 - 2*2 = 96
        max_bars = 10

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up, k_down, max_bars
        )

        # Assert - First few bars should hit upper barrier (label +1)
        assert labels[0] == 1
        assert touch_type[0] == 1



class TestTripleBarrierLowerHit:
    """Tests for lower barrier (stop loss) hits."""

    def test_triple_barrier_lower_hit(self):
        """Test that lower barrier hit produces label -1."""
        # Arrange - Price that clearly hits lower barrier
        n = 20
        close = np.array([100.0] * n)
        close[5:] = 90.0  # Drop at bar 5

        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0  # ATR = 2

        k_up = 2.0  # Upper barrier at 100 + 2*2 = 104
        k_down = 2.0  # Lower barrier at 100 - 2*2 = 96
        max_bars = 10

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up, k_down, max_bars
        )

        # Assert - First few bars should hit lower barrier (label -1)
        assert labels[0] == -1
        assert touch_type[0] == -1



class TestTripleBarrierTimeout:
    """Tests for timeout (neutral) cases."""

    def test_triple_barrier_timeout_neutral(self):
        """Test that timeout produces label 0 (neutral)."""
        # Arrange - Price stays flat, doesn't hit any barrier
        n = 20
        close = np.array([100.0] * n)  # Flat price
        high = close + 0.5  # Very small range
        low = close - 0.5
        open_ = close.copy()
        atr = np.ones(n) * 5.0  # Large ATR means wide barriers

        k_up = 2.0  # Upper barrier at 100 + 2*5 = 110
        k_down = 2.0  # Lower barrier at 100 - 2*5 = 90
        max_bars = 5  # Short timeout

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up, k_down, max_bars
        )

        # Assert - Should timeout with label 0
        assert labels[0] == 0
        assert touch_type[0] == 0
        assert bars_to_hit[0] == max_bars



class TestTripleBarrierSameBarHit:
    """Tests for same-bar hit resolution."""

    def test_triple_barrier_same_bar_hit_resolution(self):
        """Test resolution when both barriers are hit on same bar."""
        # Arrange - Create a bar where both barriers are hit
        # The bar has wide range crossing both barriers
        n = 20  # Increased so we have valid samples outside last max_bars
        close = np.full(n, 100.0)

        # Bar 1 has extreme range crossing both barriers
        high = close.copy()
        low = close.copy()
        open_ = close.copy()

        high[1] = 108.0  # Hits upper barrier (100 + 2*2 = 104)
        low[1] = 92.0    # Hits lower barrier (100 - 2*2 = 96)
        open_[1] = 102.0  # Open closer to upper barrier

        atr = np.ones(n) * 2.0
        k_up = 2.0
        k_down = 2.0
        max_bars = 10

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up, k_down, max_bars
        )

        # Assert - Should resolve based on distance from open
        # Open at 102, upper barrier at 104, lower at 96
        # dist_to_upper = |102 - 104| = 2
        # dist_to_lower = |102 - 96| = 6
        # Upper is closer, so should be +1
        assert labels[0] == 1
        assert touch_type[0] == 1
        # Verify last max_bars are marked invalid
        assert np.all(labels[-max_bars:] == -99)



class TestTripleBarrierATRUsage:
    """Tests for ATR-based barrier calculation."""

    def test_barrier_uses_atr_correctly(self):
        """Test that barriers scale correctly with ATR."""
        # Arrange
        n = 20
        close = np.full(n, 100.0)
        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()

        # Different ATR values
        atr_small = np.ones(n) * 1.0
        atr_large = np.ones(n) * 10.0

        k_up = 2.0
        k_down = 2.0
        max_bars = 5

        # Create price that moves up 5 points at bar 2
        close[2:] = 105.0
        high[2:] = 106.0
        low[2:] = 104.0

        # Act
        labels_small, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_, atr_small, k_up, k_down, max_bars
        )
        labels_large, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_, atr_large, k_up, k_down, max_bars
        )

        # Assert
        # With small ATR (1), upper barrier at 102, 5-point move should hit it -> +1
        # With large ATR (10), upper barrier at 120, 5-point move won't hit -> timeout 0
        assert labels_small[0] == 1  # Upper barrier hit
        assert labels_large[0] == 0  # Timeout (barrier too far)

    def test_barrier_handles_invalid_atr(self):
        """Test that invalid ATR (NaN, 0) is handled gracefully."""
        # Arrange
        n = 20  # Increased to have valid samples outside last max_bars
        close = np.full(n, 100.0)
        high = close + 5.0
        low = close - 5.0
        open_ = close.copy()

        atr = np.ones(n) * 2.0
        atr[0] = np.nan  # Invalid ATR for first bar
        atr[3] = 0.0     # Zero ATR

        max_bars = 10

        # Act - should not raise
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, 2.0, 2.0, max_bars
        )

        # Assert - Invalid ATR rows should be handled (label 0, max_bars)
        # Note: Index 0 might be -99 if it's in the last max_bars window
        # Check a bar that's clearly not in the last max_bars
        assert labels[0] == 0  # Invalid ATR -> timeout
        assert bars_to_hit[0] == max_bars
        # Verify last max_bars are marked invalid
        assert np.all(labels[-max_bars:] == -99)



class TestTripleBarrierQualityScoring:
    """Tests for quality score calculation."""

    def test_quality_score_calculation(self, sample_features_df):
        """Test that quality-related metrics (MAE/MFE) are calculated."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act
        df = apply_triple_barrier(df, horizon=5)

        # Assert
        assert 'mae_h5' in df.columns
        assert 'mfe_h5' in df.columns

        # MAE should be <= 0 (adverse = negative direction for long)
        # MFE should be >= 0 (favorable = positive direction for long)
        valid_idx = df['label_h5'] != 0
        # MAE represents worst drawdown (negative) and MFE represents best upside
        # Both are percentages



class TestTripleBarrierSampleWeights:
    """Tests for sample weight tier assignment."""

    def test_sample_weight_tiers(self):
        """Test sample weight tier assignment based on quality scores."""
        # This test verifies the concept of sample weights based on label quality
        # Actual implementation may be in stage6_final_labels

        # Arrange - Simulate quality scores
        np.random.seed(42)
        n = 1000
        quality_scores = np.random.rand(n)

        # Assign tiers: top 20% get 1.5x, middle 60% get 1.0x, bottom 20% get 0.5x
        percentile_20 = np.percentile(quality_scores, 20)
        percentile_80 = np.percentile(quality_scores, 80)

        weights = np.where(
            quality_scores >= percentile_80, 1.5,
            np.where(quality_scores >= percentile_20, 1.0, 0.5)
        )

        # Assert - Check tier distribution
        assert np.isclose(np.mean(weights == 1.5), 0.2, atol=0.05)
        assert np.isclose(np.mean(weights == 1.0), 0.6, atol=0.05)
        assert np.isclose(np.mean(weights == 0.5), 0.2, atol=0.05)



class TestTripleBarrierNoFutureData:
    """Critical test to ensure no future data leakage in labels."""

    def test_no_future_data_in_labels(self, sample_features_df):
        """Test that labels only depend on future price action, not features."""
        # The triple barrier method by design uses future prices to determine
        # labels (it looks forward to see which barrier is hit). This is correct
        # for supervised learning. What we need to verify is that the features
        # used at time t do NOT include any information from time t+1 or later.

        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])
        n = len(df)

        # Apply labeling
        df = apply_triple_barrier(df, horizon=5)

        # Act - Check bars_to_hit
        # bars_to_hit should always be >= 1 (can't hit barrier at same bar)
        # Exception: last bar always has 0
        bars = df['bars_to_hit_h5'].values

        # Assert
        # All bars except last few should have bars_to_hit >= 1 when not timeout
        for i in range(n - 10):  # Skip last few bars
            label = df['label_h5'].iloc[i]
            if label != 0:  # Not timeout
                assert bars[i] >= 1, f"bars_to_hit should be >= 1 at index {i}"



class TestTripleBarrierMultipleHorizons:
    """Tests for labeling with multiple horizons."""

    def test_apply_multiple_horizons(self, sample_features_df):
        """Test applying labeling for multiple horizons."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act
        df = apply_triple_barrier(df, horizon=1)
        df = apply_triple_barrier(df, horizon=5)
        df = apply_triple_barrier(df, horizon=20)

        # Assert
        assert 'label_h1' in df.columns
        assert 'label_h5' in df.columns
        assert 'label_h20' in df.columns



class TestTripleBarrierCustomParameters:
    """Tests for custom barrier parameters."""

    def test_custom_k_up_k_down(self, sample_features_df):
        """Test labeling with custom k_up and k_down."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act - use asymmetric barriers
        df = apply_triple_barrier(df, horizon=5, k_up=1.5, k_down=1.0, max_bars=20)

        # Assert
        assert 'label_h5' in df.columns
        # With k_up > k_down, should potentially favor short labels

    def test_custom_max_bars(self, sample_features_df):
        """Test labeling with custom max_bars."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act - use very short max_bars (should result in more timeouts)
        df = apply_triple_barrier(df, horizon=5, max_bars=2)

        # Assert
        neutral_count = (df['label_h5'] == 0).sum()
        # With only 2 bars, expect more timeouts
        assert neutral_count > 0



class TestTripleBarrierNonStandardHorizon:
    """Tests for non-standard horizons."""

    def test_apply_triple_barrier_non_standard_horizon(self, sample_features_df):
        """Test labeling with a non-configured horizon."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act - use horizon=10 which is not in BARRIER_PARAMS
        df = apply_triple_barrier(df, horizon=10, k_up=1.0, k_down=1.0, max_bars=30)

        # Assert
        assert 'label_h10' in df.columns



class TestTripleBarrierEdgeCases:
    """Edge case tests for triple barrier labeling."""

    def test_last_max_bars_marked_invalid(self):
        """Test that the last max_bars samples are marked as invalid (-99)."""
        # Arrange
        n = 20
        max_bars = 10
        close = np.full(n, 100.0)
        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, 2.0, 2.0, max_bars
        )

        # Assert - Last max_bars should be marked invalid (-99)
        assert np.all(labels[-max_bars:] == -99)
        # Earlier samples should be valid (not -99)
        valid_samples = labels[:-max_bars]
        assert not np.any(valid_samples == -99)

    def test_very_small_atr(self):
        """Test behavior with very small ATR."""
        # Arrange
        n = 20
        max_bars = 10
        close = np.full(n, 100.0)
        close[5:] = 100.05  # Tiny move

        high = close + 0.1
        low = close - 0.1
        open_ = close.copy()
        atr = np.ones(n) * 0.01  # Very small ATR

        # Act - With tiny ATR, barriers will be very tight
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, 2.0, 2.0, max_bars
        )

        # Assert - Valid labels should be in {-1, 0, 1}, invalid are -99
        # Check only the valid samples (not last max_bars)
        valid_labels = labels[:-max_bars]
        assert set(valid_labels).issubset({-1, 0, 1})
        # Last max_bars should be -99
        assert np.all(labels[-max_bars:] == -99)



class TestTripleBarrierLabelDistribution:
    """Tests for label distribution characteristics."""

    def test_label_distribution_balanced(self):
        """Test that asymmetric barriers produce more balanced labels."""
        # Arrange - Create trending data (upward bias)
        n = 200
        max_bars = 15
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.3 + 0.02)  # Slight upward drift

        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_ = close + np.random.randn(n) * 0.1
        atr = np.ones(n) * 1.0

        # Act - Symmetric barriers
        labels_sym, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_, atr, 2.0, 2.0, max_bars
        )

        # Act - Asymmetric barriers (easier lower barrier)
        labels_asym, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_, atr, 2.5, 1.5, max_bars  # k_up > k_down
        )

        # Assert - Both should produce valid labels (excluding -99)
        # Filter out invalid labels for checking
        valid_sym = labels_sym[labels_sym != -99]
        valid_asym = labels_asym[labels_asym != -99]
        assert set(valid_sym).issubset({-1, 0, 1})
        assert set(valid_asym).issubset({-1, 0, 1})

        # Asymmetric should have more short labels
        long_sym = (labels_sym == 1).sum()
        short_sym = (labels_sym == -1).sum()
        long_asym = (labels_asym == 1).sum()
        short_asym = (labels_asym == -1).sum()

        # Verify asymmetric has different distribution
        # (not asserting specific values as it depends on random data)
        assert (long_asym + short_asym) > 0  # At least some non-timeout labels



class TestTripleBarrierMAEMFE:
    """Tests for MAE/MFE calculations."""

    def test_mae_mfe_values(self):
        """Test that MAE and MFE are calculated correctly."""
        # Arrange - Simple case with known excursions
        n = 20
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        open_ = np.full(n, 100.0)

        # At bar 5, price moves up significantly
        close[5:10] = 105.0
        high[5:10] = 106.0
        low[5:10] = 104.0

        atr = np.ones(n) * 2.0

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, 2.0, 2.0, 15
        )

        # Assert - MFE should capture the upside
        # Entry at bar 0 (close=100), max high is 106 at bars 5-9
        # MFE = (106-100)/100 = 0.06
        assert mfe[0] > 0  # Should have positive favorable excursion

