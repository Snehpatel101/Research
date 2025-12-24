"""
Unit tests for Optuna TPE barrier optimization.

Tests the run_optuna_optimization function and ConvergenceRecord.

Run with: pytest tests/phase_1_tests/stages/ga_optimize/test_optuna_optimizer.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.ga_optimize.optuna_optimizer import (
    run_optuna_optimization,
    ConvergenceRecord,
    get_seeded_trials,
    create_objective,
)


# =============================================================================
# FIXTURE DATA
# =============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame with ATR."""
    np.random.seed(42)
    n = 2000  # Need enough data for subset

    close = 4500 + np.cumsum(np.random.randn(n) * 5)
    high = close + np.abs(np.random.randn(n) * 2)
    low = close - np.abs(np.random.randn(n) * 2)
    open_prices = close + np.random.randn(n) * 1

    # Compute ATR-like value
    true_range = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    true_range[0] = high[0] - low[0]
    atr_14 = pd.Series(true_range).rolling(14).mean().fillna(true_range[0]).values

    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'atr_14': atr_14,
    })

    return df


# =============================================================================
# CONVERGENCE RECORD TESTS
# =============================================================================

class TestConvergenceRecord:
    """Tests for ConvergenceRecord class."""

    def test_add_trial(self):
        """Test adding trials to convergence record."""
        record = ConvergenceRecord()

        record.add_trial(
            trial_number=0,
            value=1.5,
            best_value=1.5,
            params={'k_up': 1.0, 'k_down': 1.0, 'max_bars_mult': 2.5}
        )

        assert len(record.trials) == 1
        assert record.trials[0]['trial'] == 0
        assert record.trials[0]['value'] == 1.5

    def test_to_convergence_list(self):
        """Test conversion to plotting-compatible format."""
        record = ConvergenceRecord()

        # Add multiple trials
        for i in range(10):
            record.add_trial(
                trial_number=i,
                value=float(i),
                best_value=float(i),
                params={'k_up': 1.0, 'k_down': 1.0, 'max_bars_mult': 2.5}
            )

        convergence_list = record.to_convergence_list()

        assert len(convergence_list) > 0
        assert 'gen' in convergence_list[0]
        assert 'avg' in convergence_list[0]
        assert 'max' in convergence_list[0]
        assert 'min' in convergence_list[0]
        assert 'std' in convergence_list[0]


# =============================================================================
# SEEDED TRIALS TESTS
# =============================================================================

class TestSeededTrials:
    """Tests for get_seeded_trials function."""

    def test_mes_seeded_trials_asymmetric(self):
        """Test MES seeded trials are asymmetric (k_up > k_down)."""
        seeds = get_seeded_trials('MES')

        assert len(seeds) > 0

        for seed in seeds:
            assert 'k_up' in seed
            assert 'k_down' in seed
            assert 'max_bars_mult' in seed
            # MES should have asymmetric barriers
            assert seed['k_up'] >= seed['k_down'], \
                f"MES k_up ({seed['k_up']}) should be >= k_down ({seed['k_down']})"

    def test_mgc_seeded_trials_symmetric(self):
        """Test MGC seeded trials are symmetric (k_up == k_down)."""
        seeds = get_seeded_trials('MGC')

        assert len(seeds) > 0

        for seed in seeds:
            assert 'k_up' in seed
            assert 'k_down' in seed
            # MGC should have symmetric barriers
            assert seed['k_up'] == seed['k_down'], \
                f"MGC k_up ({seed['k_up']}) should == k_down ({seed['k_down']})"


# =============================================================================
# OPTUNA OPTIMIZATION TESTS
# =============================================================================

class TestOptunaOptimization:
    """Tests for run_optuna_optimization function."""

    def test_run_optuna_optimization_returns_dict(self, sample_ohlcv_df):
        """Test run_optuna_optimization returns expected result structure."""
        results, convergence = run_optuna_optimization(
            df=sample_ohlcv_df,
            horizon=5,
            symbol='MES',
            n_trials=10,  # Small for speed
            subset_fraction=0.5,
            seed=42,
            show_progress=False,
        )

        # Check result structure
        assert isinstance(results, dict)
        assert 'horizon' in results
        assert 'best_k_up' in results
        assert 'best_k_down' in results
        assert 'best_max_bars' in results
        assert 'best_fitness' in results
        assert 'validation' in results
        assert 'convergence' in results
        assert results['optimizer'] == 'optuna_tpe'

    def test_run_optuna_optimization_validation_stats(self, sample_ohlcv_df):
        """Test validation statistics in results."""
        results, _ = run_optuna_optimization(
            df=sample_ohlcv_df,
            horizon=5,
            symbol='MES',
            n_trials=10,
            subset_fraction=0.5,
            seed=42,
            show_progress=False,
        )

        val = results['validation']
        assert 'n_total' in val
        assert 'n_long' in val
        assert 'n_short' in val
        assert 'n_neutral' in val
        assert 'pct_long' in val
        assert 'pct_short' in val
        assert 'pct_neutral' in val
        assert 'signal_rate' in val

        # Percentages should sum to approximately 100 (may have rounding)
        total_pct = val['pct_long'] + val['pct_short'] + val['pct_neutral']
        assert abs(total_pct - 100.0) < 1.0  # Allow 1% tolerance for rounding

    def test_run_optuna_optimization_bounds_respected(self, sample_ohlcv_df):
        """Test optimized parameters respect search bounds."""
        results, _ = run_optuna_optimization(
            df=sample_ohlcv_df,
            horizon=5,
            symbol='MES',
            n_trials=10,
            subset_fraction=0.5,
            seed=42,
            show_progress=False,
        )

        # K_MIN=0.8, K_MAX=2.5
        assert 0.8 <= results['best_k_up'] <= 2.5
        assert 0.8 <= results['best_k_down'] <= 2.5

        # max_bars should be between 2*horizon and 3*horizon
        assert 5 * 2 <= results['best_max_bars'] <= 5 * 3

    def test_run_optuna_optimization_reproducibility(self, sample_ohlcv_df):
        """Test optimization is reproducible with same seed."""
        results1, _ = run_optuna_optimization(
            df=sample_ohlcv_df,
            horizon=5,
            symbol='MES',
            n_trials=10,
            subset_fraction=0.5,
            seed=42,
            show_progress=False,
        )

        results2, _ = run_optuna_optimization(
            df=sample_ohlcv_df,
            horizon=5,
            symbol='MES',
            n_trials=10,
            subset_fraction=0.5,
            seed=42,
            show_progress=False,
        )

        assert results1['best_k_up'] == results2['best_k_up']
        assert results1['best_k_down'] == results2['best_k_down']
        assert results1['best_max_bars'] == results2['best_max_bars']
        assert results1['best_fitness'] == results2['best_fitness']


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with DEAP interface."""

    def test_run_ga_optimization_interface(self, sample_ohlcv_df):
        """Test run_ga_optimization (wrapper) returns same structure."""
        from src.phase1.stages.ga_optimize import run_ga_optimization

        results, logbook = run_ga_optimization(
            df=sample_ohlcv_df,
            horizon=5,
            symbol='MES',
            population_size=5,  # Small for speed
            generations=2,
            subset_fraction=0.5,
            seed=42,
        )

        # Should have DEAP-style fields for compatibility
        assert 'population_size' in results
        assert 'generations' in results

        # Should also have Optuna fields
        assert 'best_k_up' in results
        assert 'best_k_down' in results
        assert 'best_max_bars' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
