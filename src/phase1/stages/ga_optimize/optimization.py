"""
Core GA optimization functions.

Contains:
    - run_ga_optimization: Run GA for a single horizon
    - process_symbol_ga: Run GA for all horizons for a symbol
    - main: Entry point for Stage 5
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from deap import tools
from tqdm import tqdm

from src.phase1.stages.labeling import triple_barrier_numba
from src.phase1.config import TRANSACTION_COSTS

from .operators import (
    check_bounds,
    create_toolbox,
    get_contiguous_subset,
    get_seeded_individuals,
)
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
) -> Tuple[Dict, tools.Logbook]:
    """
    Run genetic algorithm to optimize labeling parameters.

    Parameters:
    -----------
    df : DataFrame with OHLCV and ATR
    horizon : horizon to optimize for
    symbol : 'MES' or 'MGC' for symbol-specific optimization
    population_size : GA population size
    generations : number of generations
    crossover_prob : crossover probability
    mutation_prob : mutation probability
    tournament_size : tournament selection size
    subset_fraction : fraction of data to use (for speed)
    atr_column : ATR column name
    seed : random seed for reproducibility (default: 42)

    Returns:
    --------
    results : dict with best parameters and statistics
    logbook : DEAP logbook with evolution history
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"\nOptimizing parameters for {symbol} horizon {horizon}")
    logger.info(f"  Population: {population_size}, Generations: {generations}")
    logger.info(
        f"  Symbol-specific mode: {'SYMMETRIC (gold)' if symbol == 'MGC' else 'ASYMMETRIC (equity)'}"
    )
    logger.info(f"  Random seed: {seed}")

    # Use contiguous time blocks instead of random sampling
    df_subset = get_contiguous_subset(df, subset_fraction, seed=seed)
    logger.info(
        f"  Using {len(df_subset):,} contiguous samples ({subset_fraction*100:.0f}% of data)"
    )

    # Extract arrays
    close = df_subset["close"].values
    high = df_subset["high"].values
    low = df_subset["low"].values
    open_prices = df_subset["open"].values
    atr = df_subset[atr_column].values

    # Create toolbox
    toolbox = create_toolbox(
        close, high, low, open_prices, atr, horizon, symbol, tournament_size
    )

    # Create initial population
    population = toolbox.population(n=population_size)

    # Seed with symbol-specific starting points
    seeded_individuals = get_seeded_individuals(symbol)
    for i, seed_ind in enumerate(seeded_individuals):
        if i < len(population):
            population[i][:] = seed_ind

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of fame
    hof = tools.HallOfFame(5)

    # Track statistics
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # Track consecutive errors
    MAX_CONSECUTIVE_ERRORS = 10
    consecutive_errors = 0
    total_errors = 0

    def count_errors_in_fitnesses(fitnesses_list):
        return sum(1 for fit in fitnesses_list if fit[0] == float("-inf"))

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    init_errors = count_errors_in_fitnesses(fitnesses)
    if init_errors > 0:
        logger.warning(f"  Initial population had {init_errors} evaluation errors")
        total_errors += init_errors

    hof.update(population)
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(population), **record)

    logger.info(f"  Gen 0: avg={record['avg']:.2f}, max={record['max']:.2f}")

    # Evolution loop
    for gen in tqdm(range(1, generations + 1), desc=f"  GA H={horizon}"):
        # Select next generation (leave room for elites)
        offspring = toolbox.select(population, len(population) - 2)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Apply bounds
        offspring = [check_bounds(ind) for ind in offspring]

        # Evaluate offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Track errors
        gen_errors = count_errors_in_fitnesses(fitnesses)
        if gen_errors > 0:
            consecutive_errors += 1
            total_errors += gen_errors
            logger.warning(
                f"  Gen {gen}: {gen_errors} evaluation errors (consecutive: {consecutive_errors})"
            )

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error(
                    f"  ABORTING: {consecutive_errors} consecutive generations with errors. "
                    f"Total errors: {total_errors}"
                )
                raise RuntimeError(
                    f"GA optimization aborted due to {consecutive_errors} consecutive "
                    "generations with evaluation errors"
                )
        else:
            consecutive_errors = 0

        # ELITISM: Add best 2 from hall of fame
        elite = [toolbox.clone(ind) for ind in hof[:2]]
        offspring.extend(elite)

        # Replace population
        population[:] = offspring
        hof.update(population)

        # Record statistics
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if gen % 10 == 0:
            logger.info(f"  Gen {gen}: avg={record['avg']:.2f}, max={record['max']:.2f}")

    # Get best individual
    best_ind = hof[0]
    best_fitness = best_ind.fitness.values[0]

    logger.info("\n  OPTIMIZATION COMPLETE")
    logger.info(f"  Best fitness: {best_fitness:.4f}")
    logger.info(
        f"  Best params: k_up={best_ind[0]:.3f}, k_down={best_ind[1]:.3f}, "
        f"max_bars={int(horizon * best_ind[2])}"
    )

    # Validate on full data
    logger.info("  Validating on full dataset...")
    full_close = df["close"].values
    full_high = df["high"].values
    full_low = df["low"].values
    full_open = df["open"].values
    full_atr = df[atr_column].values

    k_up, k_down, max_bars_mult = best_ind
    max_bars = int(horizon * max_bars_mult)

    val_labels, val_bars, val_mae, val_mfe, _ = triple_barrier_numba(
        full_close, full_high, full_low, full_open, full_atr, k_up, k_down, max_bars
    )

    n_total = len(val_labels)
    n_long = (val_labels == 1).sum()
    n_short = (val_labels == -1).sum()
    n_neutral = (val_labels == 0).sum()

    logger.info("  Full data label distribution:")
    logger.info(f"    Long:    {n_long:,} ({100*n_long/n_total:.1f}%)")
    logger.info(f"    Short:   {n_short:,} ({100*n_short/n_total:.1f}%)")
    logger.info(f"    Neutral: {n_neutral:,} ({100*n_neutral/n_total:.1f}%)")

    # Prepare results
    results = {
        "horizon": horizon,
        "best_k_up": float(best_ind[0]),
        "best_k_down": float(best_ind[1]),
        "best_max_bars": int(horizon * best_ind[2]),
        "best_fitness": float(best_fitness),
        "population_size": population_size,
        "generations": generations,
        "validation": {
            "n_total": int(n_total),
            "n_long": int(n_long),
            "n_short": int(n_short),
            "n_neutral": int(n_neutral),
            "pct_long": float(100 * n_long / n_total),
            "pct_short": float(100 * n_short / n_total),
            "pct_neutral": float(100 * n_neutral / n_total),
            "signal_rate": float((n_long + n_short) / n_total),
        },
        "convergence": [
            {
                "gen": record["gen"],
                "avg": float(record["avg"]),
                "max": float(record["max"]),
                "min": float(record["min"]),
                "std": float(record["std"]),
            }
            for record in logbook
        ],
    }

    return results, logbook


def process_symbol_ga(
    symbol: str,
    horizons: Optional[List[int]] = None,
    population_size: int = 50,
    generations: int = 30,
    seed: int = 42,
) -> Dict[int, Dict]:
    """
    Run GA optimization for all horizons for a symbol.

    Parameters:
    -----------
    symbol : str
        Symbol name ('MES' or 'MGC')
    horizons : List[int], optional
        List of horizons to optimize. If None, uses config.HORIZONS.
    population_size : int
        GA population size (default: 50)
    generations : int
        Number of generations (default: 30)
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
    logger.info(f"GA OPTIMIZATION: {symbol}")
    logger.info(f"  Mode: {'SYMMETRIC' if symbol == 'MGC' else 'ASYMMETRIC'}")
    logger.info(f"  Transaction cost: {TRANSACTION_COSTS.get(symbol, 0.5)} ticks")
    logger.info("=" * 70)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    all_results = {}

    for horizon in horizons:
        results, logbook = run_ga_optimization(
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
            logger.warning("  Consider adjusting search bounds or running more generations.")

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
    """Run Stage 5: GA optimization for all symbols using dynamic horizons."""
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
    logger.info("STAGE 5: GENETIC ALGORITHM OPTIMIZATION (DYNAMIC HORIZONS)")
    logger.info("=" * 70)
    logger.info(f"Horizons: {horizons} (configurable via config.HORIZONS)")
    logger.info(f"GA Config: pop_size={population_size}, gens={generations}")
    logger.info(f"Random seed: {RANDOM_SEED}")
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
        error_summary = f"{len(errors)}/{len(SYMBOLS)} symbols failed GA optimization"
        logger.error(f"GA optimization completed with errors: {error_summary}")
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
