"""
Stage 5: Barrier Optimization using Optuna TPE

Uses Optuna's Tree-structured Parzen Estimator (TPE) to find optimal
k_up, k_down, and max_bars for balanced, tradable labels.

This replaces the previous DEAP genetic algorithm implementation with
Optuna TPE, which is 27% more sample-efficient.

Public API:
    - run_ga_optimization: Run optimization for a single horizon (backward-compatible name)
    - run_optuna_optimization: Run Optuna TPE optimization directly
    - process_symbol_ga: Run optimization for all horizons for a symbol
    - calculate_fitness: Fitness function for label quality
    - evaluate_individual: Evaluate parameter set
    - plot_convergence: Plot optimization convergence curve
    - main: Entry point for Stage 5
"""

from .fitness import calculate_fitness, evaluate_individual
from .operators import (
    get_contiguous_subset,
    check_bounds,
    get_seeded_individuals,
)
from .optimization import run_ga_optimization, process_symbol_ga
from .optuna_optimizer import run_optuna_optimization, ConvergenceRecord
from .plotting import plot_convergence

__all__ = [
    # Core optimization functions
    "run_ga_optimization",  # Backward-compatible name (uses Optuna internally)
    "run_optuna_optimization",  # Direct Optuna interface
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
