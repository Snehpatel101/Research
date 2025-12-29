"""
Stage 5: Barrier Optimization using Optuna TPE

Uses Optuna's Tree-structured Parzen Estimator (TPE) to find optimal
k_up, k_down, and max_bars for balanced, tradable labels.

This replaces the previous DEAP genetic algorithm implementation with
Optuna TPE, which is 27% more sample-efficient.

Public API:
    - run_ga_optimization: Run optimization for a single horizon (backward-compatible name)
    - run_ga_optimization_safe: SAFE version that prevents test data leakage
    - run_optuna_optimization: Run Optuna TPE optimization directly
    - run_optuna_optimization_safe: SAFE Optuna TPE (train data only)
    - process_symbol_ga: Run optimization for all horizons for a symbol
    - calculate_fitness: Fitness function for label quality
    - evaluate_individual: Evaluate parameter set
    - plot_convergence: Plot optimization convergence curve
    - main: Entry point for Stage 5

CRITICAL - Test Data Leakage Prevention:
    Use run_ga_optimization_safe() or run_optuna_optimization_safe() to ensure
    barrier parameters are optimized ONLY on training data. This prevents the
    critical issue where test set performance is optimistically biased because
    labeling parameters were tuned on data that will later become the test set.
"""

from .fitness import calculate_fitness, evaluate_individual
from .operators import (
    check_bounds,
    get_contiguous_subset,
    get_seeded_individuals,
)
from .optimization import process_symbol_ga, run_ga_optimization, run_ga_optimization_safe
from .optuna_optimizer import (
    ConvergenceRecord,
    run_optuna_optimization,
    run_optuna_optimization_safe,
)
from .plotting import plot_convergence

__all__ = [
    # Core optimization functions
    "run_ga_optimization",  # Backward-compatible name (uses Optuna internally)
    "run_ga_optimization_safe",  # SAFE version - prevents test data leakage
    "run_optuna_optimization",  # Direct Optuna interface
    "run_optuna_optimization_safe",  # SAFE Optuna version - prevents test data leakage
    "process_symbol_ga",
    "main",
    # Fitness
    "calculate_fitness",
    "evaluate_individual",
    # Helpers
    "get_contiguous_subset",
    "check_bounds",
    "get_seeded_individuals",
    # Optuna types
    "ConvergenceRecord",
    # Plotting
    "plot_convergence",
]


def main():
    """Run Stage 5: Barrier optimization for all symbols using dynamic horizons."""
    from .optimization import main as _main

    _main()
