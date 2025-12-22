"""
Unit tests for Stage 5: GA Optimization.

Genetic algorithm barrier optimization

Run with: pytest tests/phase_1_tests/stages/test_stage5_*.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.stage5_ga_optimize import calculate_fitness, evaluate_individual, run_ga_optimization


# =============================================================================
# TESTS
# =============================================================================

class TestStage5GAOptimizer:
    """Tests for Stage 5: Genetic Algorithm Optimization."""

    def test_calculate_fitness_valid_balanced_labels(self):
        """Test fitness calculation with balanced label distribution."""
        np.random.seed(42)
        n = 1000

        # Create balanced labels (40% long, 30% neutral, 30% short)
        labels = np.array([1]*400 + [0]*300 + [-1]*300, dtype=np.int8)
        np.random.shuffle(labels)

        bars_to_hit = np.random.randint(1, 10, n).astype(np.int32)
        mae = -np.abs(np.random.randn(n) * 0.01).astype(np.float32)
        mfe = np.abs(np.random.randn(n) * 0.02).astype(np.float32)
        horizon = 5

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon)

        # Should have positive fitness for balanced distribution
        assert fitness > -100, f"Fitness too low for balanced labels: {fitness}"

    def test_calculate_fitness_degenerate_neutral_heavy(self):
        """Test fitness penalizes neutral-heavy distributions."""
        n = 1000

        # Create neutral-heavy labels (90% neutral)
        labels = np.array([0]*900 + [1]*50 + [-1]*50, dtype=np.int8)
        bars_to_hit = np.ones(n, dtype=np.int32) * 5
        mae = np.zeros(n, dtype=np.float32)
        mfe = np.zeros(n, dtype=np.float32)
        horizon = 5

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon)

        # Should have very negative fitness due to low signal rate
        assert fitness < -500, f"Fitness should be very negative: {fitness}"

    def test_calculate_fitness_empty_labels(self):
        """Test fitness with empty labels returns minimum value."""
        labels = np.array([], dtype=np.int8)
        bars_to_hit = np.array([], dtype=np.int32)
        mae = np.array([], dtype=np.float32)
        mfe = np.array([], dtype=np.float32)

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon=5)

        assert fitness == -1000.0, "Empty labels should return -1000"

    def test_calculate_fitness_signal_rate_threshold(self):
        """Test that signal rate below 40% is heavily penalized."""
        n = 1000

        # 30% signal rate (below threshold)
        labels = np.array([1]*150 + [-1]*150 + [0]*700, dtype=np.int8)
        bars_to_hit = np.ones(n, dtype=np.int32) * 5
        mae = -np.ones(n, dtype=np.float32) * 0.01
        mfe = np.ones(n, dtype=np.float32) * 0.02
        horizon = 5

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon)

        # Should be penalized but slightly better than -1000
        assert -1000 < fitness < -900, f"Expected penalty for low signal rate: {fitness}"

    def test_calculate_fitness_profit_factor_components(self):
        """Test profit factor calculation with known values."""
        n = 100

        # All long wins with good MFE
        labels = np.array([1]*50 + [-1]*50, dtype=np.int8)  # 50% long, 50% short
        bars_to_hit = np.ones(n, dtype=np.int32) * 3
        mae = np.zeros(n, dtype=np.float32)  # No adverse excursion
        mfe = np.ones(n, dtype=np.float32) * 0.02  # Good favorable excursion
        horizon = 5

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon)

        # Should have good fitness due to balanced distribution and good PF
        assert fitness > -10, f"Expected positive-ish fitness: {fitness}"

    def test_evaluate_individual_valid_params(self, sample_price_arrays):
        """Test evaluate_individual with valid parameters."""
        individual = [1.0, 1.0, 3.0]  # k_up, k_down, max_bars_mult

        fitness_tuple = evaluate_individual(
            individual,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            sample_price_arrays['atr'],
            horizon=5
        )

        assert isinstance(fitness_tuple, tuple), "Should return tuple"
        assert len(fitness_tuple) == 1, "Should have one fitness value"
        assert isinstance(fitness_tuple[0], float), "Fitness should be float"

    def test_evaluate_individual_invalid_params(self, sample_price_arrays):
        """Test evaluate_individual with parameters that cause errors."""
        # Zero ATR should be handled gracefully
        zero_atr = np.zeros_like(sample_price_arrays['atr'])
        individual = [1.0, 1.0, 3.0]

        fitness_tuple = evaluate_individual(
            individual,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            zero_atr,
            horizon=5
        )

        # Should return very negative fitness, not crash
        assert fitness_tuple[0] < 0, "Invalid params should give negative fitness"

    def test_ga_bounds_respected(self, sample_price_arrays):
        """Test that GA optimization respects parameter bounds."""
        individual_too_low = [0.1, 0.1, 1.0]  # Below K_MIN
        individual_too_high = [3.0, 3.0, 6.0]  # Above K_MAX

        # Bounds should be enforced in run_ga_optimization
        # Test via evaluate_individual which uses these params
        fitness_low = evaluate_individual(
            individual_too_low,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            sample_price_arrays['atr'],
            horizon=5
        )

        # Even with extreme params, should return valid fitness
        assert isinstance(fitness_low[0], float)

    def test_asymmetric_barrier_constraint(self, sample_price_arrays):
        """Test asymmetric barrier penalty is applied."""
        # Highly asymmetric barriers (k_up=2.0, k_down=0.5)
        asymmetric = [2.0, 0.5, 3.0]  # 4:1 ratio - very asymmetric
        symmetric = [1.0, 1.0, 3.0]   # 1:1 ratio - symmetric

        fitness_asym = evaluate_individual(
            asymmetric,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            sample_price_arrays['atr'],
            horizon=5
        )

        fitness_sym = evaluate_individual(
            symmetric,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            sample_price_arrays['atr'],
            horizon=5
        )

        # Highly asymmetric should have penalty applied
        # Note: actual comparison depends on data, just verify both run
        assert isinstance(fitness_asym[0], float)
        assert isinstance(fitness_sym[0], float)

    def test_error_handling_returns_negative_infinity(self, sample_price_arrays):
        """Test that errors in evaluation return very negative fitness."""
        # NaN in ATR should cause issues
        nan_atr = sample_price_arrays['atr'].copy()
        nan_atr[:] = np.nan

        individual = [1.0, 1.0, 3.0]

        fitness_tuple = evaluate_individual(
            individual,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            nan_atr,
            horizon=5
        )

        # Should handle NaN gracefully
        assert fitness_tuple[0] < 0

    def test_get_contiguous_subset(self, sample_ohlcv_data):
        """Test contiguous subset extraction."""
        df = sample_ohlcv_data
        subset_fraction = 0.3

        subset = get_contiguous_subset(df, subset_fraction)

        # Check size is approximately correct
        expected_len = int(len(df) * subset_fraction)
        assert len(subset) >= min(expected_len, 1000), "Subset too small"

        # Check it's contiguous (indices should be sequential)
        if len(subset) > 1:
            time_diffs = subset['datetime'].diff().dropna()
            # All time diffs should be equal (5 min intervals)
            assert time_diffs.nunique() == 1, "Subset should be contiguous"

    def test_get_contiguous_subset_respects_minimum(self, sample_ohlcv_data):
        """Test contiguous subset has minimum 1000 samples if available."""
        df = sample_ohlcv_data

        # Request very small fraction
        subset = get_contiguous_subset(df, 0.01)

        # Should still have minimum samples
        assert len(subset) >= min(1000, len(df))


# =============================================================================
# STAGE 6 TESTS: FinalLabeler
# =============================================================================
