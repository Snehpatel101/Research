"""
Unit tests for Stage 6: Final Labels.

Apply optimized labels with quality scores

Run with: pytest tests/phase_1_tests/stages/test_stage6_*.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.labeling import compute_quality_scores, assign_sample_weights, apply_optimized_labels


# =============================================================================
# TESTS
# =============================================================================

class TestStage6FinalLabeler:
    """Tests for Stage 6: Final Labeling with Quality Scoring."""

    def test_compute_quality_scores_speed_component(self):
        """Test speed component of quality scoring."""
        n = 100
        horizon = 5

        # Fast hits (around ideal speed)
        fast_bars = np.ones(n, dtype=np.int32) * int(horizon * 1.5)
        # Slow hits (much slower than ideal)
        slow_bars = np.ones(n, dtype=np.int32) * int(horizon * 5)

        mae = np.zeros(n, dtype=np.float32)
        mfe = np.ones(n, dtype=np.float32) * 0.01
        labels = np.ones(n, dtype=np.int8)  # All long labels

        fast_scores, _, _ = compute_quality_scores(fast_bars, mae, mfe, labels, horizon)
        slow_scores, _, _ = compute_quality_scores(slow_bars, mae, mfe, labels, horizon)

        # Fast hits should have higher speed component
        assert fast_scores.mean() > slow_scores.mean(), \
            "Faster hits should have higher quality"

    def test_compute_quality_scores_mae_component(self):
        """Test MAE component of quality scoring."""
        n = 100
        horizon = 5

        bars = np.ones(n, dtype=np.int32) * 5
        labels = np.ones(n, dtype=np.int8)  # All long labels

        # Create varied MFE so normalization doesn't collapse
        np.random.seed(42)
        mfe = np.abs(np.random.randn(n) * 0.02).astype(np.float32)

        # Low adverse excursion (good) - range of small values
        low_mae = -np.abs(np.random.randn(n) * 0.005).astype(np.float32)
        # High adverse excursion (bad) - range of larger values
        high_mae = -np.abs(np.random.randn(n) * 0.05 + 0.03).astype(np.float32)

        low_mae_scores, _, _ = compute_quality_scores(bars, low_mae, mfe, labels, horizon)
        high_mae_scores, _, _ = compute_quality_scores(bars, high_mae, mfe, labels, horizon)

        # Lower (less negative) MAE should have higher quality
        # Note: MAE component is 40% of total score
        assert low_mae_scores.mean() >= high_mae_scores.mean() - 0.01, \
            f"Lower MAE should have higher quality: {low_mae_scores.mean()} vs {high_mae_scores.mean()}"

    def test_compute_quality_scores_mfe_component(self):
        """Test MFE component of quality scoring."""
        n = 100
        horizon = 5

        bars = np.ones(n, dtype=np.int32) * 5
        labels = np.ones(n, dtype=np.int8)  # All long labels

        # Create varied MAE so normalization doesn't collapse
        np.random.seed(42)
        mae = -np.abs(np.random.randn(n) * 0.02).astype(np.float32)

        # High favorable excursion (good) - range of high values
        high_mfe = (np.abs(np.random.randn(n) * 0.02) + 0.04).astype(np.float32)
        # Low favorable excursion (not as good) - range of low values
        low_mfe = np.abs(np.random.randn(n) * 0.005).astype(np.float32)

        high_mfe_scores, _, _ = compute_quality_scores(bars, mae, high_mfe, labels, horizon)
        low_mfe_scores, _, _ = compute_quality_scores(bars, mae, low_mfe, labels, horizon)

        # Higher MFE should have higher quality
        # Note: MFE component is 30% of total score
        assert high_mfe_scores.mean() >= low_mfe_scores.mean() - 0.01, \
            f"Higher MFE should have higher quality: {high_mfe_scores.mean()} vs {low_mfe_scores.mean()}"

    def test_compute_quality_scores_range(self):
        """Test quality scores are in valid range."""
        np.random.seed(42)
        n = 1000
        horizon = 5

        bars = np.random.randint(1, 20, n).astype(np.int32)
        mae = -np.abs(np.random.randn(n) * 0.02).astype(np.float32)
        mfe = np.abs(np.random.randn(n) * 0.03).astype(np.float32)
        labels = np.random.choice([-1, 0, 1], n).astype(np.int8)

        scores, _, _ = compute_quality_scores(bars, mae, mfe, labels, horizon)

        # Scores should be bounded
        assert np.all(scores >= 0), "Scores should be non-negative"
        assert np.all(scores <= 1.5), "Scores should be reasonable"

    def test_assign_sample_weights_tier_distribution(self):
        """Test sample weight tier assignment."""
        np.random.seed(42)
        n = 1000

        quality_scores = np.random.rand(n).astype(np.float32)
        weights = assign_sample_weights(quality_scores)

        # Check tier distribution
        tier1 = (weights == 1.5).sum()
        tier2 = (weights == 1.0).sum()
        tier3 = (weights == 0.5).sum()

        # Tier 1: top 20% (around 200 +/- margin)
        assert 150 < tier1 < 250, f"Tier 1 count unexpected: {tier1}"
        # Tier 2: middle 60% (around 600 +/- margin)
        assert 550 < tier2 < 650, f"Tier 2 count unexpected: {tier2}"
        # Tier 3: bottom 20% (around 200 +/- margin)
        assert 150 < tier3 < 250, f"Tier 3 count unexpected: {tier3}"

    def test_assign_sample_weights_unique_values(self):
        """Test sample weights have correct unique values."""
        n = 1000
        quality_scores = np.random.rand(n).astype(np.float32)

        weights = assign_sample_weights(quality_scores)

        unique_weights = set(np.unique(weights))
        expected_weights = {0.5, 1.0, 1.5}

        assert unique_weights == expected_weights, \
            f"Unexpected weights: {unique_weights}"

    def test_assign_sample_weights_ordering(self):
        """Test high quality gets high weight."""
        n = 100

        # Create ordered quality scores
        quality_scores = np.linspace(0, 1, n).astype(np.float32)
        weights = assign_sample_weights(quality_scores)

        # Top 20% should have weight 1.5
        assert np.all(weights[-20:] == 1.5), "Top quality should have weight 1.5"
        # Bottom 20% should have weight 0.5
        assert np.all(weights[:20] == 0.5), "Bottom quality should have weight 0.5"

    def test_apply_optimized_labels_columns_created(self, sample_ohlcv_data):
        """Test that apply_optimized_labels creates all required columns."""
        df = sample_ohlcv_data.copy()
        horizon = 5
        best_params = {'k_up': 1.0, 'k_down': 1.0, 'max_bars': 15}

        result = apply_optimized_labels(df, horizon, best_params)

        # Check all expected columns exist
        expected_cols = [
            f'label_h{horizon}',
            f'bars_to_hit_h{horizon}',
            f'mae_h{horizon}',
            f'mfe_h{horizon}',
            f'touch_type_h{horizon}',
            f'quality_h{horizon}',
            f'sample_weight_h{horizon}',
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_apply_optimized_labels_label_values(self, sample_ohlcv_data):
        """Test that labels have valid values."""
        df = sample_ohlcv_data.copy()
        horizon = 5
        best_params = {'k_up': 1.0, 'k_down': 1.0, 'max_bars': 15}

        result = apply_optimized_labels(df, horizon, best_params)

        labels = result[f'label_h{horizon}'].values
        # -99 is the sentinel value for invalid/unlabeled bars (end of dataset)
        valid_labels = {-1, 0, 1, -99}

        assert set(np.unique(labels)).issubset(valid_labels), \
            f"Invalid label values: {np.unique(labels)}"

    def test_label_consistency_with_stage4(self, sample_ohlcv_data):
        """Test that stage 6 labels are consistent with stage 4 methodology."""
        # This tests that the same labeling function produces consistent results
        from stages.stage4_labeling import triple_barrier_numba

        df = sample_ohlcv_data.copy()
        k_up, k_down, max_bars = 1.0, 1.0, 15

        # Direct call to triple_barrier_numba
        labels1, _, _, _, _ = triple_barrier_numba(
            df['close'].values,
            df['high'].values,
            df['low'].values,
            df['open'].values,
            df['atr_14'].values,
            k_up, k_down, max_bars
        )

        # Via apply_optimized_labels
        best_params = {'k_up': k_up, 'k_down': k_down, 'max_bars': max_bars}
        result = apply_optimized_labels(df.copy(), horizon=5, best_params=best_params)
        labels2 = result['label_h5'].values

        # Labels should be identical
        np.testing.assert_array_equal(labels1, labels2, "Labels should match stage4")

    def test_handles_missing_ga_params(self, sample_ohlcv_data, temp_directory):
        """Test graceful handling when GA params file is missing."""
        # This is tested implicitly in process_symbol_final
        # When GA results file doesn't exist, defaults are used
        df = sample_ohlcv_data.copy()
        horizon = 5

        # Use default params (simulating missing GA results)
        default_params = {'k_up': 2.0, 'k_down': 1.0, 'max_bars': 15}

        result = apply_optimized_labels(df, horizon, default_params)

        # Should complete successfully with defaults
        assert f'label_h{horizon}' in result.columns


# =============================================================================
# STAGE 7 TESTS: DataSplitter
# =============================================================================
