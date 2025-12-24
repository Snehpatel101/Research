"""
Stage 5: Genetic Algorithm Optimization of Labeling Parameters

Uses DEAP to find optimal k_up, k_down, and max_bars for balanced, tradable labels.

Public API:
    - run_ga_optimization: Run GA for a single horizon
    - process_symbol_ga: Run GA for all horizons for a symbol
    - calculate_fitness: Fitness function for label quality
    - evaluate_individual: Evaluate a GA individual
    - plot_convergence: Plot GA convergence curve
    - main: Entry point for Stage 5
"""

from .fitness import calculate_fitness, evaluate_individual
from .operators import (
    get_contiguous_subset,
    check_bounds,
    create_toolbox,
    get_seeded_individuals,
)
from .optimization import run_ga_optimization, process_symbol_ga
from .plotting import plot_convergence

__all__ = [
    # Core functions
    "run_ga_optimization",
    "process_symbol_ga",
    "main",
    # Fitness
    "calculate_fitness",
    "evaluate_individual",
    # Operators
    "get_contiguous_subset",
    "check_bounds",
    "create_toolbox",
    "get_seeded_individuals",
    # Plotting
    "plot_convergence",
]


def main():
    """Run Stage 5: GA optimization for all symbols using dynamic horizons."""
    from .optimization import main as _main

    _main()
