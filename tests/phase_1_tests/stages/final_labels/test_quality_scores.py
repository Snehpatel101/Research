"""
Comprehensive unit tests for quality score calculation logic.

Tests the direction-aware MAE/MFE logic for LONG, SHORT, and NEUTRAL trades
in the compute_quality_scores function.
"""
import numpy as np
import pytest

from src.phase1.stages.final_labels.core import compute_quality_scores


class TestQualityScoresLongTrades:
    """Test quality score calculation for LONG trades (label=1)."""

    def test_long_trade_favorable_positive_mfe(self):
        """Test LONG trade with positive MFE (favorable movement)."""
        # Setup: Single LONG trade with positive MFE (price went up)
        bars_to_hit = np.array([10], dtype=np.int32)
        mae = np.array([-0.5], dtype=np.float32)  # Max downside (negative)
        mfe = np.array([1.0], dtype=np.float32)   # Max upside (positive)
        labels = np.array([1], dtype=np.int8)     # LONG
        horizon = 20

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, horizon, symbol='MES'
        )

        # Assertions:
        # For LONG: favorable = max(mfe, 0) = max(1.0, 0) = 1.0
        #           adverse = |min(mae, 0)| = |min(-0.5, 0)| = 0.5
        # Pain-to-gain = adverse / favorable = 0.5 / 1.0 = 0.5
        assert quality_scores[0] > 0, "Quality score should be positive"
        assert pain_to_gain[0] == pytest.approx(0.5, rel=1e-3)

    def test_long_trade_zero_mfe(self):
        """Test LONG trade with zero MFE (no favorable movement)."""
        bars_to_hit = np.array([15], dtype=np.int32)
        mae = np.array([-1.0], dtype=np.float32)  # Price went down
        mfe = np.array([0.0], dtype=np.float32)   # No upside
        labels = np.array([1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For LONG: favorable = max(0.0, 0) = 0.0
        #           adverse = |min(-1.0, 0)| = 1.0
        # Pain-to-gain = 1.0 / max(0.0, 1e-6) = 1.0 / 1e-6 (capped to 1.0 after normalization)
        assert quality_scores[0] >= 0, "Quality score should be non-negative"
        assert pain_to_gain[0] > 1.0, "Pain-to-gain should be high when MFE is zero"

    def test_long_trade_zero_mae(self):
        """Test LONG trade with zero MAE (no adverse movement)."""
        bars_to_hit = np.array([12], dtype=np.int32)
        mae = np.array([0.0], dtype=np.float32)   # No downside
        mfe = np.array([2.0], dtype=np.float32)   # Price went up
        labels = np.array([1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For LONG: favorable = max(2.0, 0) = 2.0
        #           adverse = |min(0.0, 0)| = 0.0
        # Pain-to-gain = 0.0 / 2.0 = 0.0 (ideal trade)
        assert quality_scores[0] > 0.5, "Quality score should be high with no adverse movement"
        assert pain_to_gain[0] == pytest.approx(0.0, abs=1e-6)

    def test_long_trade_positive_mae(self):
        """Test LONG trade with positive MAE (edge case - should be treated as zero adverse)."""
        bars_to_hit = np.array([10], dtype=np.int32)
        mae = np.array([0.3], dtype=np.float32)   # Positive (unusual but possible)
        mfe = np.array([1.5], dtype=np.float32)   # Price went up
        labels = np.array([1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For LONG: favorable = max(1.5, 0) = 1.5
        #           adverse = |min(0.3, 0)| = 0.0 (positive MAE means no downside)
        # Pain-to-gain = 0.0 / 1.5 = 0.0
        assert quality_scores[0] > 0, "Quality score should be positive"
        assert pain_to_gain[0] == pytest.approx(0.0, abs=1e-6)

    def test_long_trade_multiple_samples(self):
        """Test LONG trades with multiple samples showing varying quality."""
        bars_to_hit = np.array([10, 15, 20], dtype=np.int32)
        mae = np.array([-0.5, -1.0, -0.2], dtype=np.float32)
        mfe = np.array([1.0, 0.5, 2.0], dtype=np.float32)
        labels = np.array([1, 1, 1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # Sample 0: pain=0.5, gain=1.0, ptg=0.5
        # Sample 1: pain=1.0, gain=0.5, ptg=2.0
        # Sample 2: pain=0.2, gain=2.0, ptg=0.1
        assert len(quality_scores) == 3
        assert pain_to_gain[0] == pytest.approx(0.5, rel=1e-3)
        assert pain_to_gain[1] == pytest.approx(2.0, rel=1e-3)
        assert pain_to_gain[2] == pytest.approx(0.1, rel=1e-3)


class TestQualityScoresShortTrades:
    """Test quality score calculation for SHORT trades (label=-1)."""

    def test_short_trade_favorable_negative_mae(self):
        """Test SHORT trade with negative MAE (favorable movement - price went down)."""
        bars_to_hit = np.array([10], dtype=np.int32)
        mae = np.array([-1.0], dtype=np.float32)  # Max downside = favorable for SHORT
        mfe = np.array([0.5], dtype=np.float32)   # Max upside = adverse for SHORT
        labels = np.array([-1], dtype=np.int8)    # SHORT
        horizon = 20

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, horizon, symbol='MES'
        )

        # For SHORT: favorable = |min(mae, 0)| = |min(-1.0, 0)| = 1.0
        #            adverse = max(mfe, 0) = max(0.5, 0) = 0.5
        # Pain-to-gain = adverse / favorable = 0.5 / 1.0 = 0.5
        assert quality_scores[0] > 0, "Quality score should be positive"
        assert pain_to_gain[0] == pytest.approx(0.5, rel=1e-3)

    def test_short_trade_zero_mae(self):
        """Test SHORT trade with zero MAE (no favorable movement)."""
        bars_to_hit = np.array([15], dtype=np.int32)
        mae = np.array([0.0], dtype=np.float32)   # No downside
        mfe = np.array([1.0], dtype=np.float32)   # Price went up (adverse)
        labels = np.array([-1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For SHORT: favorable = |min(0.0, 0)| = 0.0
        #            adverse = max(1.0, 0) = 1.0
        # Pain-to-gain = 1.0 / max(0.0, 1e-6) = very high
        assert quality_scores[0] >= 0, "Quality score should be non-negative"
        assert pain_to_gain[0] > 1.0, "Pain-to-gain should be high when MAE is zero"

    def test_short_trade_zero_mfe(self):
        """Test SHORT trade with zero MFE (no adverse movement)."""
        bars_to_hit = np.array([12], dtype=np.int32)
        mae = np.array([-2.0], dtype=np.float32)  # Price went down (favorable)
        mfe = np.array([0.0], dtype=np.float32)   # No upside
        labels = np.array([-1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For SHORT: favorable = |min(-2.0, 0)| = 2.0
        #            adverse = max(0.0, 0) = 0.0
        # Pain-to-gain = 0.0 / 2.0 = 0.0 (ideal trade)
        assert quality_scores[0] > 0.5, "Quality score should be high with no adverse movement"
        assert pain_to_gain[0] == pytest.approx(0.0, abs=1e-6)

    def test_short_trade_negative_mfe(self):
        """Test SHORT trade with negative MFE (edge case - should be treated as zero adverse)."""
        bars_to_hit = np.array([10], dtype=np.int32)
        mae = np.array([-1.5], dtype=np.float32)  # Price went down (favorable)
        mfe = np.array([-0.3], dtype=np.float32)  # Negative (unusual but possible)
        labels = np.array([-1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For SHORT: favorable = |min(-1.5, 0)| = 1.5
        #            adverse = max(-0.3, 0) = 0.0 (negative MFE means no upside)
        # Pain-to-gain = 0.0 / 1.5 = 0.0
        assert quality_scores[0] > 0, "Quality score should be positive"
        assert pain_to_gain[0] == pytest.approx(0.0, abs=1e-6)

    def test_short_trade_multiple_samples(self):
        """Test SHORT trades with multiple samples showing varying quality."""
        bars_to_hit = np.array([10, 15, 20], dtype=np.int32)
        mae = np.array([-1.0, -0.5, -2.0], dtype=np.float32)
        mfe = np.array([0.5, 1.0, 0.2], dtype=np.float32)
        labels = np.array([-1, -1, -1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # Sample 0: favorable=1.0, adverse=0.5, ptg=0.5
        # Sample 1: favorable=0.5, adverse=1.0, ptg=2.0
        # Sample 2: favorable=2.0, adverse=0.2, ptg=0.1
        assert len(quality_scores) == 3
        assert pain_to_gain[0] == pytest.approx(0.5, rel=1e-3)
        assert pain_to_gain[1] == pytest.approx(2.0, rel=1e-3)
        assert pain_to_gain[2] == pytest.approx(0.1, rel=1e-3)


class TestQualityScoresNeutralTrades:
    """Test quality score calculation for NEUTRAL trades (label=0)."""

    def test_neutral_trade_symmetric_mae_mfe(self):
        """Test NEUTRAL trade with symmetric MAE and MFE."""
        bars_to_hit = np.array([10], dtype=np.int32)
        mae = np.array([-1.0], dtype=np.float32)
        mfe = np.array([1.0], dtype=np.float32)
        labels = np.array([0], dtype=np.int8)     # NEUTRAL
        horizon = 20

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, horizon, symbol='MES'
        )

        # For NEUTRAL: favorable = max(|mfe|, |mae|) = max(1.0, 1.0) = 1.0
        #              adverse = min(|mfe|, |mae|) = min(1.0, 1.0) = 1.0
        # Pain-to-gain = 1.0 (default for neutral)
        assert quality_scores[0] >= 0, "Quality score should be non-negative"
        assert pain_to_gain[0] == pytest.approx(1.0, rel=1e-3)

    def test_neutral_trade_mfe_greater_than_mae(self):
        """Test NEUTRAL trade where MFE > |MAE|."""
        bars_to_hit = np.array([15], dtype=np.int32)
        mae = np.array([-0.5], dtype=np.float32)
        mfe = np.array([2.0], dtype=np.float32)
        labels = np.array([0], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For NEUTRAL: favorable = max(2.0, 0.5) = 2.0
        #              adverse = min(2.0, 0.5) = 0.5
        assert quality_scores[0] >= 0, "Quality score should be non-negative"
        assert pain_to_gain[0] == pytest.approx(1.0, rel=1e-3)  # Default for neutral

    def test_neutral_trade_mae_greater_than_mfe(self):
        """Test NEUTRAL trade where |MAE| > MFE."""
        bars_to_hit = np.array([12], dtype=np.int32)
        mae = np.array([-2.0], dtype=np.float32)
        mfe = np.array([0.5], dtype=np.float32)
        labels = np.array([0], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For NEUTRAL: favorable = max(0.5, 2.0) = 2.0
        #              adverse = min(0.5, 2.0) = 0.5
        assert quality_scores[0] >= 0, "Quality score should be non-negative"
        assert pain_to_gain[0] == pytest.approx(1.0, rel=1e-3)  # Default for neutral

    def test_neutral_trade_zero_mae_mfe(self):
        """Test NEUTRAL trade with zero MAE and MFE."""
        bars_to_hit = np.array([10], dtype=np.int32)
        mae = np.array([0.0], dtype=np.float32)
        mfe = np.array([0.0], dtype=np.float32)
        labels = np.array([0], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For NEUTRAL: favorable = max(0.0, 0.0) = 0.0
        #              adverse = min(0.0, 0.0) = 0.0
        assert quality_scores[0] >= 0, "Quality score should be non-negative"
        assert pain_to_gain[0] == pytest.approx(1.0, rel=1e-3)  # Default for neutral


class TestQualityScoresEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_mixed_labels_batch(self):
        """Test batch processing with mixed LONG, SHORT, and NEUTRAL labels."""
        bars_to_hit = np.array([10, 12, 15, 18, 20], dtype=np.int32)
        mae = np.array([-0.5, -1.0, -0.3, -1.5, -0.8], dtype=np.float32)
        mfe = np.array([1.0, 0.5, 1.5, 0.3, 1.2], dtype=np.float32)
        labels = np.array([1, -1, 0, -1, 1], dtype=np.int8)
        horizon = 20

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, horizon, symbol='MES'
        )

        # All outputs should have same length as inputs
        assert len(quality_scores) == 5
        assert len(pain_to_gain) == 5
        assert len(time_weighted_dd) == 5

        # All quality scores should be non-negative
        assert np.all(quality_scores >= 0)

        # Specific checks for each sample:
        # Sample 0 (LONG): pain=0.5, gain=1.0, ptg=0.5
        assert pain_to_gain[0] == pytest.approx(0.5, rel=1e-2)

        # Sample 1 (SHORT): favorable=1.0, adverse=0.5, ptg=0.5
        assert pain_to_gain[1] == pytest.approx(0.5, rel=1e-2)

        # Sample 2 (NEUTRAL): ptg=1.0 (default)
        assert pain_to_gain[2] == pytest.approx(1.0, rel=1e-3)

        # Sample 3 (SHORT): favorable=1.5, adverse=0.3, ptg=0.2
        assert pain_to_gain[3] == pytest.approx(0.2, rel=1e-2)

        # Sample 4 (LONG): pain=0.8, gain=1.2, ptg=0.667
        assert pain_to_gain[4] == pytest.approx(0.667, rel=1e-2)

    def test_all_zeros(self):
        """Test edge case with all zero values."""
        bars_to_hit = np.array([0, 0, 0], dtype=np.int32)
        mae = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        mfe = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        labels = np.array([1, -1, 0], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # Should not crash and return valid arrays
        assert len(quality_scores) == 3
        assert np.all(np.isfinite(quality_scores))
        assert np.all(np.isfinite(pain_to_gain))

    def test_extreme_values(self):
        """Test with extreme MAE/MFE values."""
        bars_to_hit = np.array([10, 10], dtype=np.int32)
        mae = np.array([-100.0, -0.001], dtype=np.float32)
        mfe = np.array([100.0, 0.001], dtype=np.float32)
        labels = np.array([1, 1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # Should handle extreme values without overflow
        assert np.all(np.isfinite(quality_scores))
        assert np.all(np.isfinite(pain_to_gain))

        # Sample 0: pain=100, gain=100, ptg=1.0
        assert pain_to_gain[0] == pytest.approx(1.0, rel=1e-2)

        # Sample 1: pain=0.001, gain=0.001, ptg=1.0
        assert pain_to_gain[1] == pytest.approx(1.0, rel=1e-2)

    def test_single_sample(self):
        """Test with a single sample."""
        bars_to_hit = np.array([10], dtype=np.int32)
        mae = np.array([-0.5], dtype=np.float32)
        mfe = np.array([1.0], dtype=np.float32)
        labels = np.array([1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        assert len(quality_scores) == 1
        assert quality_scores[0] > 0
        assert 0 <= quality_scores[0] <= 1

    def test_large_batch(self):
        """Test with a large batch of samples."""
        n = 10000
        np.random.seed(42)

        bars_to_hit = np.random.randint(5, 50, size=n, dtype=np.int32)
        mae = -np.abs(np.random.randn(n).astype(np.float32))  # Always negative
        mfe = np.abs(np.random.randn(n).astype(np.float32))   # Always positive
        labels = np.random.choice([1, -1, 0], size=n).astype(np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # Should process large batch efficiently
        assert len(quality_scores) == n
        assert np.all(np.isfinite(quality_scores))
        assert np.all(quality_scores >= 0)
        assert np.all(quality_scores <= 1)

    def test_negative_bars_to_hit(self):
        """Test edge case with negative bars_to_hit."""
        bars_to_hit = np.array([-5, 0, 10], dtype=np.int32)
        mae = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        mfe = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        labels = np.array([1, 1, 1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # Should handle negative/zero bars_to_hit gracefully
        assert len(quality_scores) == 3
        assert np.all(np.isfinite(quality_scores))

    def test_symbol_parameter(self):
        """Test that symbol parameter is accepted."""
        bars_to_hit = np.array([10], dtype=np.int32)
        mae = np.array([-0.5], dtype=np.float32)
        mfe = np.array([1.0], dtype=np.float32)
        labels = np.array([1], dtype=np.int8)

        # Test with different symbols
        for symbol in ['MES', 'MGC', 'ES', 'GC']:
            quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
                bars_to_hit, mae, mfe, labels, 20, symbol=symbol
            )
            assert len(quality_scores) == 1
            assert quality_scores[0] > 0


class TestQualityScoresDirectionAwareCorrectness:
    """Explicit tests to verify direction-aware MAE/MFE interpretation."""

    def test_long_favorable_is_mfe(self):
        """Verify that for LONG trades, favorable excursion comes from positive MFE."""
        bars_to_hit = np.array([10], dtype=np.int32)
        mae = np.array([-0.5], dtype=np.float32)  # Downside
        mfe = np.array([2.0], dtype=np.float32)   # Upside (favorable for LONG)
        labels = np.array([1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For LONG: favorable should be MFE (2.0), adverse should be |MAE| (0.5)
        # Pain-to-gain = 0.5 / 2.0 = 0.25
        assert pain_to_gain[0] == pytest.approx(0.25, rel=1e-3)

    def test_short_favorable_is_negative_mae(self):
        """Verify that for SHORT trades, favorable excursion comes from negative MAE."""
        bars_to_hit = np.array([10], dtype=np.int32)
        mae = np.array([-2.0], dtype=np.float32)  # Downside (favorable for SHORT)
        mfe = np.array([0.5], dtype=np.float32)   # Upside (adverse for SHORT)
        labels = np.array([-1], dtype=np.int8)

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # For SHORT: favorable should be |MAE| (2.0), adverse should be MFE (0.5)
        # Pain-to-gain = 0.5 / 2.0 = 0.25
        assert pain_to_gain[0] == pytest.approx(0.25, rel=1e-3)

    def test_long_vs_short_same_price_movement(self):
        """Verify that LONG and SHORT trades with identical price movements have symmetric metrics."""
        bars_to_hit = np.array([10, 10], dtype=np.int32)
        mae = np.array([-1.0, -1.0], dtype=np.float32)
        mfe = np.array([1.0, 1.0], dtype=np.float32)
        labels = np.array([1, -1], dtype=np.int8)  # LONG, SHORT

        quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
            bars_to_hit, mae, mfe, labels, 20, symbol='MES'
        )

        # LONG: favorable=1.0, adverse=1.0, ptg=1.0
        # SHORT: favorable=1.0, adverse=1.0, ptg=1.0
        # Both should have same pain-to-gain ratio
        assert pain_to_gain[0] == pytest.approx(pain_to_gain[1], rel=1e-3)
        assert pain_to_gain[0] == pytest.approx(1.0, rel=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
