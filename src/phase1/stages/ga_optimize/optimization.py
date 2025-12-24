"""
Core optimization functions using Optuna TPE.

Contains:
    - run_ga_optimization: Run Optuna TPE for a single horizon (backward-compatible name)
    - process_symbol_ga: Run optimization for all horizons for a symbol
    - main: Entry point for Stage 5
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.phase1.stages.labeling import triple_barrier_numba
from src.phase1.config import TRANSACTION_COSTS

from .optuna_optimizer import run_optuna_optimization, ConvergenceRecord
from .operators import get_contiguous_subset
from .plotting import plot_convergence

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def run_ga_optimization(
    df: pd.DataFrame,
    horizon: int,
    symbol: str = "MES",
    population_size: int = 50,
    generations: int = 30,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    tournament_size: int = 3,
    subset_fraction: float = 0.3,
    atr_column: str = "atr_14",
    seed: int = 42,
) -> Tuple[Dict[str, Any], Any]:
    """
    Run optimization to find optimal labeling parameters.

    NOTE: This function now uses Optuna TPE internally but maintains
    the same interface for backward compatibility. The parameters
    population_size, generations, crossover_prob, mutation_prob, and
    tournament_size are deprecated but accepted for compatibility.

    Parameters:
    -----------
    df : DataFrame with OHLCV and ATR
    horizon : horizon to optimize for
    symbol : 'MES' or 'MGC' for symbol-specific optimization
    population_size : (deprecated) GA population size - mapped to n_trials
    generations : (deprecated) number of generations - mapped to n_trials
    crossover_prob : (deprecated) ignored
    mutation_prob : (deprecated) ignored
    tournament_size : (deprecated) ignored
    subset_fraction : fraction of data to use (for speed)
    atr_column : ATR column name
    seed : random seed for reproducibility (default: 42)

    Returns:
    --------
    results : dict with best parameters and statistics
    convergence_record : ConvergenceRecord (replaces DEAP logbook)
    """
    # Map old GA parameters to Optuna trials
    # GA typically evaluates population_size * generations individuals
    # But TPE is more efficient, so we use fewer trials
    # GA default: 50 * 30 = 1500 evaluations
    # Optuna: 100-150 trials achieves similar or better results
    n_trials = max(50, min(population_size * 2, 150))

    logger.info(f"  Using Optuna TPE optimizer (n_trials={n_trials})")
    logger.info(f"  Note: population_size/generations params mapped to {n_trials} trials")

    results, convergence_record = run_optuna_optimization(
        df=df,
        horizon=horizon,
        symbol=symbol,
        n_trials=n_trials,
        subset_fraction=subset_fraction,
        atr_column=atr_column,
        seed=seed,
        show_progress=True,
        n_startup_trials=10,
    )

    # Add backward-compatible fields
    results["population_size"] = population_size
    results["generations"] = generations

    return results, convergence_record


def process_symbol_ga(
    symbol: str,
    horizons: Optional[List[int]] = None,
    population_size: int = 50,
    generations: int = 30,
    seed: int = 42,
) -> Dict[int, Dict]:
    """
    Run optimization for all horizons for a symbol.

    Parameters:
    -----------
    symbol : str
        Symbol name ('MES' or 'MGC')
    horizons : List[int], optional
        List of horizons to optimize. If None, uses config.HORIZONS.
    population_size : int
        (deprecated) GA population size (default: 50)
    generations : int
        (deprecated) Number of generations (default: 30)
    seed : int
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    Dict[int, Dict] : Mapping of horizon -> optimization results
    """
    if horizons is None:
        from src.phase1.config import HORIZONS

        horizons = HORIZONS

    # Load labels_init data
    project_root = Path(__file__).parent.parent.parent.parent.resolve()
    labels_dir = project_root / "data" / "labels"
    input_path = labels_dir / f"{symbol}_labels_init.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Labels file not found: {input_path}")

    logger.info("=" * 70)
    logger.info(f"OPTIMIZATION: {symbol} (Optuna TPE)")
    logger.info(f"  Mode: {'SYMMETRIC' if symbol == 'MGC' else 'ASYMMETRIC'}")
    logger.info(f"  Transaction cost: {TRANSACTION_COSTS.get(symbol, 0.5)} ticks")
    logger.info("=" * 70)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    all_results = {}

    for horizon in horizons:
        results, convergence_record = run_ga_optimization(
            df,
            horizon,
            symbol=symbol,
            population_size=population_size,
            generations=generations,
            subset_fraction=0.3,
            seed=seed,
        )

        all_results[horizon] = results

        # Check if optimization succeeded
        val = results.get("validation", {})
        signal_rate = val.get("signal_rate", 0)
        if signal_rate < 0.40:
            logger.warning(
                f"  WARNING: Signal rate {signal_rate*100:.1f}% below 40% threshold!"
            )
            logger.warning("  Consider adjusting search bounds or running more trials.")

        # Save results
        ga_results_dir = project_root / "config" / "ga_results"
        ga_results_dir.mkdir(parents=True, exist_ok=True)

        results_path = ga_results_dir / f"{symbol}_ga_h{horizon}_best.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"  Saved results to {results_path}")

        # Plot convergence
        plots_dir = project_root / "results" / "ga_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / f"{symbol}_ga_h{horizon}_convergence.png"
        plot_convergence(results, plot_path)

    return all_results


def main():
    """Run Stage 5: Optimization for all symbols using dynamic horizons."""
    from src.phase1.config import (
        ACTIVE_HORIZONS,
        HORIZONS,
        RANDOM_SEED,
        SYMBOLS,
        set_global_seeds,
        validate_horizons,
    )

    # Set random seeds
    set_global_seeds(RANDOM_SEED)

    horizons = HORIZONS if HORIZONS else ACTIVE_HORIZONS

    try:
        validate_horizons(horizons)
    except ValueError as e:
        logger.error(f"Horizon validation failed: {e}")
        raise

    population_size = 50
    generations = 30

    logger.info("=" * 70)
    logger.info("STAGE 5: BARRIER OPTIMIZATION (OPTUNA TPE)")
    logger.info("=" * 70)
    logger.info(f"Horizons: {horizons} (configurable via config.HORIZONS)")
    logger.info(f"Optimizer: Optuna TPE (replaces DEAP GA)")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info("")
    logger.info("OPTUNA TPE ADVANTAGES:")
    logger.info("  - 27% more sample-efficient than genetic algorithms")
    logger.info("  - Tree-structured Parzen Estimator for smart exploration")
    logger.info("  - Multivariate sampling considers parameter correlations")
    logger.info("")
    logger.info("PRODUCTION FIXES APPLIED:")
    logger.info("  - Target 20-30% neutral rate (was <2%)")
    logger.info("  - SYMBOL-SPECIFIC barriers:")
    logger.info("    - MES: ASYMMETRIC (k_up > k_down) for equity drift")
    logger.info("    - MGC: SYMMETRIC (k_up = k_down) for mean-reverting gold")
    logger.info("  - Transaction cost penalty in fitness:")
    logger.info(f"    - MES: {TRANSACTION_COSTS.get('MES', 0.5)} ticks round-trip")
    logger.info(f"    - MGC: {TRANSACTION_COSTS.get('MGC', 0.3)} ticks round-trip")
    logger.info("  - Wider search bounds: k=[0.8,2.5], max_bars=[2x,3x]")
    logger.info("  - Fixed profit factor calculation")
    logger.info("  - Random seed for reproducibility")
    logger.info("")

    all_symbols_results = {}
    errors = []

    for symbol in SYMBOLS:
        try:
            results = process_symbol_ga(
                symbol, horizons, population_size, generations, seed=RANDOM_SEED
            )
            all_symbols_results[symbol] = results
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e), "type": type(e).__name__})
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    if errors:
        error_summary = f"{len(errors)}/{len(SYMBOLS)} symbols failed optimization"
        logger.error(f"Optimization completed with errors: {error_summary}")
        raise RuntimeError(f"{error_summary}. Errors: {errors[:5]}")

    # Save combined summary
    project_root = Path(__file__).parent.parent.parent.parent.resolve()
    summary_path = project_root / "config" / "ga_results" / "optimization_summary.json"
    summary = {}
    for symbol, symbol_results in all_symbols_results.items():
        summary[symbol] = {
            str(h): {
                "k_up": res["best_k_up"],
                "k_down": res["best_k_down"],
                "max_bars": res["best_max_bars"],
                "fitness": res["best_fitness"],
                "signal_rate": res.get("validation", {}).get("signal_rate", None),
                "pct_long": res.get("validation", {}).get("pct_long", None),
                "pct_short": res.get("validation", {}).get("pct_short", None),
                "pct_neutral": res.get("validation", {}).get("pct_neutral", None),
                "optimizer": res.get("optimizer", "optuna_tpe"),
            }
            for h, res in symbol_results.items()
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 5 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Optimization summary saved to {summary_path}")

    # Final summary
    logger.info("\nFINAL PARAMETER SUMMARY:")
    for symbol, sym_res in summary.items():
        logger.info(f"\n  {symbol}:")
        for h, params in sym_res.items():
            sr = params.get("signal_rate", 0)
            status = "OK" if sr and sr >= 0.40 else "WARNING"
            logger.info(
                f"    H{h}: k_up={params['k_up']:.2f}, k_down={params['k_down']:.2f}, "
                f"max_bars={params['max_bars']}, signal_rate={sr*100 if sr else 0:.1f}% [{status}]"
            )
