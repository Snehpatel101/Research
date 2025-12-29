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


# =============================================================================
# SAFE MODE TESTS (Test Data Leakage Prevention)
# =============================================================================

class TestSafeModeOptimization:
    """Tests for safe mode optimization that prevents test data leakage."""

    def test_run_optuna_optimization_safe_returns_dict(self, sample_ohlcv_df):
        """Test run_optuna_optimization_safe returns expected result structure."""
        from src.phase1.stages.ga_optimize.optuna_optimizer import run_optuna_optimization_safe

        results, convergence = run_optuna_optimization_safe(
            df=sample_ohlcv_df,
            horizon=5,
            symbol='MES',
            train_ratio=0.70,
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
        assert results['optimizer'] == 'optuna_tpe'

        # CRITICAL: Check safe mode markers
        assert results.get('safe_mode') is True
        assert results.get('train_ratio_used') == 0.70
        assert 'train_samples' in results
        assert 'total_samples' in results

    def test_safe_mode_uses_only_training_data(self, sample_ohlcv_df):
        """Test that safe mode only uses training portion of data."""
        from src.phase1.stages.ga_optimize.optuna_optimizer import run_optuna_optimization_safe

        n_total = len(sample_ohlcv_df)
        train_ratio = 0.70
        expected_train_samples = int(n_total * train_ratio)

        results, _ = run_optuna_optimization_safe(
            df=sample_ohlcv_df,
            horizon=5,
            symbol='MES',
            train_ratio=train_ratio,
            n_trials=10,
            subset_fraction=0.5,
            seed=42,
            show_progress=False,
        )

        # Verify the correct number of training samples were used
        assert results['train_samples'] == expected_train_samples
        assert results['total_samples'] == n_total

        # The validation stats should reflect ONLY training data
        # (n_total in validation should be less than full dataset)
        val = results['validation']
        assert val['n_total'] <= expected_train_samples  # May use subset of training

    def test_safe_mode_no_test_data_leakage(self, sample_ohlcv_df):
        """
        Test that safe mode optimization is NOT influenced by test data characteristics.

        This test creates a dataset where the test portion has very different
        characteristics than the training portion, and verifies that the optimized
        parameters are determined ONLY by the training data.
        """
        from src.phase1.stages.ga_optimize.optuna_optimizer import (
            run_optuna_optimization_safe,
            run_optuna_optimization,
        )

        # Create a dataset with artificially different train/test characteristics
        # Train portion: normal data
        # Test portion: data with very different volatility
        n = 2000
        train_end = int(n * 0.70)

        # Normal training data
        np.random.seed(42)
        close_train = 4500 + np.cumsum(np.random.randn(train_end) * 5)

        # Test data with VERY different characteristics (10x volatility)
        close_test = close_train[-1] + np.cumsum(np.random.randn(n - train_end) * 50)

        # Combine
        close = np.concatenate([close_train, close_test])
        high = close + np.abs(np.random.randn(n) * 2)
        low = close - np.abs(np.random.randn(n) * 2)
        open_prices = close + np.random.randn(n) * 1

        # ATR will be much higher in test portion
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

        # Run SAFE optimization (should only use train data)
        results_safe, _ = run_optuna_optimization_safe(
            df=df,
            horizon=5,
            symbol='MES',
            train_ratio=0.70,
            n_trials=15,
            subset_fraction=0.5,
            seed=42,
            show_progress=False,
        )

        # Run UNSAFE optimization on just the training portion
        df_train_only = df.iloc[:train_end].copy()
        results_train_only, _ = run_optuna_optimization(
            df=df_train_only,
            horizon=5,
            symbol='MES',
            n_trials=15,
            subset_fraction=0.5,
            seed=42,
            show_progress=False,
        )

        # The safe mode results should be VERY SIMILAR to train-only results
        # because both use only the training data
        # Allow some tolerance due to randomness in subset selection
        assert abs(results_safe['best_k_up'] - results_train_only['best_k_up']) < 0.5
        assert abs(results_safe['best_k_down'] - results_train_only['best_k_down']) < 0.5

    def test_run_ga_optimization_safe_wrapper(self, sample_ohlcv_df):
        """Test run_ga_optimization_safe wrapper works correctly."""
        from src.phase1.stages.ga_optimize import run_ga_optimization_safe

        results, logbook = run_ga_optimization_safe(
            df=sample_ohlcv_df,
            horizon=5,
            symbol='MES',
            train_ratio=0.70,
            population_size=5,  # Small for speed
            generations=2,
            subset_fraction=0.5,
            seed=42,
        )

        # Should have DEAP-style fields for backward compatibility
        assert 'population_size' in results
        assert 'generations' in results

        # Should have safe mode markers
        assert results.get('safe_mode') is True
        assert results.get('train_ratio_used') == 0.70

        # Should have Optuna fields
        assert 'best_k_up' in results
        assert 'best_k_down' in results
        assert 'best_max_bars' in results

    def test_safe_mode_minimum_samples_validation(self):
        """Test that safe mode rejects datasets with too few training samples."""
        from src.phase1.stages.ga_optimize.optuna_optimizer import run_optuna_optimization_safe

        # Create a very small dataset
        np.random.seed(42)
        n = 100  # Too small - 70% = 70 samples < 500 minimum
        close = 4500 + np.cumsum(np.random.randn(n) * 5)

        df = pd.DataFrame({
            'open': close + np.random.randn(n) * 1,
            'high': close + np.abs(np.random.randn(n) * 2),
            'low': close - np.abs(np.random.randn(n) * 2),
            'close': close,
            'atr_14': np.abs(np.random.randn(n) * 5) + 1,
        })

        # Should raise ValueError due to insufficient training data
        with pytest.raises(ValueError, match="Training portion too small"):
            run_optuna_optimization_safe(
                df=df,
                horizon=5,
                symbol='MES',
                train_ratio=0.70,
                n_trials=10,
                show_progress=False,
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
