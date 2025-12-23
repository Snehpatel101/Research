"""
Stage 5: Genetic Algorithm Optimization of Labeling Parameters

Uses DEAP to find optimal k_up, k_down, and max_bars for balanced, tradable labels.

FIXES APPLIED:
1. Improved fitness function with 40% minimum signal rate
2. Horizon-specific neutral targets (20-30% neutral rate)
3. Fixed profit factor calculation using actual trade outcomes
4. Contiguous time block sampling instead of random (preserves temporal order)
5. Narrower search space bounds for more realistic parameters
6. SYMBOL-SPECIFIC asymmetry constraints:
   - MES: asymmetric (k_up > k_down) for equity drift
   - MGC: symmetric (k_up = k_down) for mean-reverting gold
7. TRANSACTION COST penalty in fitness (MES: 0.5 ticks, MGC: 0.3 ticks)
8. Wider barriers to target 20-30% neutral rate (was <2%)

This module is a thin wrapper around the ga_optimize submodule for backward
compatibility. All functionality has been refactored into:
    - ga_optimize/fitness.py: calculate_fitness(), evaluate_individual()
    - ga_optimize/operators.py: GA operators, seeded individuals, bounds checking
    - ga_optimize/optimization.py: run_ga_optimization(), process_symbol_ga()
    - ga_optimize/plotting.py: plot_convergence()
"""

# Re-export all public API from submodule for backward compatibility
from .ga_optimize import (
    calculate_fitness,
    check_bounds,
    create_toolbox,
    evaluate_individual,
    get_contiguous_subset,
    get_seeded_individuals,
    main,
    plot_convergence,
    process_symbol_ga,
    run_ga_optimization,
)

__all__ = [
    "calculate_fitness",
    "evaluate_individual",
    "get_contiguous_subset",
    "check_bounds",
    "create_toolbox",
    "get_seeded_individuals",
    "run_ga_optimization",
    "process_symbol_ga",
    "plot_convergence",
    "main",
]

if __name__ == "__main__":
    main()
