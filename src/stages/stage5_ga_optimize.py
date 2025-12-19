"""
Stage 5: Genetic Algorithm Optimization of Labeling Parameters
Uses DEAP to find optimal k_up, k_down, and max_bars for balanced, tradable labels
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# DEAP imports
from deap import base, creator, tools, algorithms

# Import our labeling function
import sys
sys.path.insert(0, str(Path(__file__).parent))
from stage4_labeling import triple_barrier_numba

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_fitness(
    labels: np.ndarray,
    bars_to_hit: np.ndarray,
    mae: np.ndarray,
    mfe: np.ndarray,
    horizon: int
) -> float:
    """
    Calculate fitness score for a set of labels.

    Objective: Balance between label quality metrics:
    1. Label balance (not too imbalanced)
    2. Win rate realism (40-60% preferred)
    3. Average bars to hit (not too fast/slow)
    4. Profit factor from simple simulation

    Returns:
    --------
    fitness : float (higher is better)
    """
    # Count labels
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    total = len(labels)

    n_long = label_counts.get(1, 0)
    n_short = label_counts.get(-1, 0)
    n_neutral = label_counts.get(0, 0)

    # Avoid degenerate cases
    if total == 0 or (n_long + n_short) < total * 0.1:
        return -1000.0

    # 1. Label Balance Score (penalize extreme imbalance)
    # Ideal: roughly 30-40% each for long/short, 20-40% neutral
    long_pct = n_long / total
    short_pct = n_short / total
    neutral_pct = n_neutral / total

    # Balance penalty (prefer 0.2-0.5 for each non-neutral class)
    balance_score = 0.0
    for pct in [long_pct, short_pct]:
        if 0.20 <= pct <= 0.50:
            balance_score += 1.0
        else:
            balance_score -= abs(pct - 0.35) * 2  # penalty

    # Neutral should be reasonable (not too high)
    if neutral_pct > 0.6:
        balance_score -= (neutral_pct - 0.6) * 3

    # 2. Win Rate Realism (40-60% preferred)
    if (n_long + n_short) > 0:
        win_rate = n_long / (n_long + n_short)
        if 0.40 <= win_rate <= 0.60:
            win_rate_score = 2.0
        else:
            # Penalize extreme win rates
            win_rate_score = -abs(win_rate - 0.50) * 5
    else:
        win_rate_score = -5.0

    # 3. Speed Score (bars to hit should be reasonable)
    # Not too fast (< horizon/2) or too slow (> horizon*2)
    valid_bars = bars_to_hit[bars_to_hit > 0]
    if len(valid_bars) > 0:
        avg_bars = valid_bars.mean()
        ideal_bars = horizon * 1.5

        if horizon * 0.5 <= avg_bars <= horizon * 2.5:
            speed_score = 1.0
        else:
            speed_score = -abs(avg_bars - ideal_bars) / ideal_bars
    else:
        speed_score = -2.0

    # 4. Profit Factor (simple backtest)
    # Assume: long wins = +mfe, short losses = +|mae|
    # Wins: where label = 1
    # Losses: where label = -1
    winning_trades = mfe[labels == 1]
    losing_trades = np.abs(mae[labels == -1])

    total_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
    total_loss = losing_trades.sum() if len(losing_trades) > 0 else 0

    if total_loss > 0:
        profit_factor = total_profit / total_loss
        # Prefer profit factor between 1.2 and 2.0
        if 1.2 <= profit_factor <= 2.5:
            pf_score = 2.0
        elif profit_factor > 1.0:
            pf_score = 1.0
        else:
            pf_score = -2.0
    else:
        pf_score = 0.0

    # Combined fitness
    fitness = balance_score + win_rate_score + speed_score + pf_score

    return fitness


def evaluate_individual(
    individual: List[float],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    horizon: int
) -> Tuple[float]:
    """
    Evaluate a GA individual (parameter set).

    Parameters:
    -----------
    individual : [k_up, k_down, max_bars_multiplier]
    close, high, low, atr : price/indicator arrays
    horizon : horizon value

    Returns:
    --------
    (fitness,) : tuple with single fitness value (DEAP convention)
    """
    k_up, k_down, max_bars_mult = individual

    # Decode max_bars
    max_bars = int(horizon * max_bars_mult)
    max_bars = max(horizon, min(max_bars, horizon * 5))  # clamp

    # Run labeling
    try:
        labels, bars_to_hit, mae, mfe, _ = triple_barrier_numba(
            close, high, low, atr, k_up, k_down, max_bars
        )

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon)
        return (fitness,)

    except Exception as e:
        logger.error(f"Error evaluating individual: {e}")
        return (-1000.0,)


def run_ga_optimization(
    df: pd.DataFrame,
    horizon: int,
    population_size: int = 50,
    generations: int = 30,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    tournament_size: int = 3,
    subset_fraction: float = 0.2,
    atr_column: str = 'atr_14'
) -> Dict:
    """
    Run genetic algorithm to optimize labeling parameters.

    Parameters:
    -----------
    df : DataFrame with OHLCV and ATR
    horizon : horizon to optimize for
    population_size : GA population size
    generations : number of generations
    crossover_prob : crossover probability
    mutation_prob : mutation probability
    tournament_size : tournament selection size
    subset_fraction : fraction of data to use (for speed)
    atr_column : ATR column name

    Returns:
    --------
    results : dict with best parameters and statistics
    """
    logger.info(f"\nOptimizing parameters for horizon {horizon}")
    logger.info(f"  Population: {population_size}, Generations: {generations}")

    # Subsample data for speed
    n_subset = int(len(df) * subset_fraction)
    if n_subset < 1000:
        n_subset = min(1000, len(df))

    # Random sample (ensure temporal order preserved)
    indices = sorted(np.random.choice(len(df), n_subset, replace=False))
    df_subset = df.iloc[indices].copy()

    logger.info(f"  Using {len(df_subset):,} samples ({subset_fraction*100:.0f}% of data)")

    # Extract arrays
    close = df_subset['close'].values
    high = df_subset['high'].values
    low = df_subset['low'].values
    atr = df_subset[atr_column].values

    # Setup DEAP
    # Define fitness and individual
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximize
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Gene ranges:
    # k_up: [0.5, 3.0]
    # k_down: [0.5, 3.0]
    # max_bars_multiplier: [1.0, 5.0]
    toolbox.register("k_up", random.uniform, 0.5, 3.0)
    toolbox.register("k_down", random.uniform, 0.5, 3.0)
    toolbox.register("max_bars_mult", random.uniform, 1.0, 5.0)

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.k_up, toolbox.k_down, toolbox.max_bars_mult),
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function
    toolbox.register(
        "evaluate",
        evaluate_individual,
        close=close,
        high=high,
        low=low,
        atr=atr,
        horizon=horizon
    )

    # Genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Constraint function (ensure valid ranges)
    def check_bounds(individual):
        for i in range(len(individual)):
            if i < 2:  # k_up, k_down
                if individual[i] < 0.5:
                    individual[i] = 0.5
                elif individual[i] > 3.0:
                    individual[i] = 3.0
            else:  # max_bars_mult
                if individual[i] < 1.0:
                    individual[i] = 1.0
                elif individual[i] > 5.0:
                    individual[i] = 5.0
        return individual

    # Create initial population
    population = toolbox.population(n=population_size)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of fame (best individuals)
    hof = tools.HallOfFame(1)

    # Run evolution
    logger.info("  Starting evolution...")

    # Track statistics
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    hof.update(population)
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(population), **record)

    # Evolution loop
    for gen in tqdm(range(1, generations + 1), desc=f"  GA H={horizon}"):
        # Select next generation
        offspring = toolbox.select(population, len(population))
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
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        population[:] = offspring
        hof.update(population)

        # Record statistics
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

    # Get best individual
    best_ind = hof[0]
    best_fitness = best_ind.fitness.values[0]

    logger.info(f"  Best fitness: {best_fitness:.4f}")
    logger.info(f"  Best params: k_up={best_ind[0]:.3f}, k_down={best_ind[1]:.3f}, "
                f"max_bars={int(horizon * best_ind[2])}")

    # Prepare results
    results = {
        'horizon': horizon,
        'best_k_up': float(best_ind[0]),
        'best_k_down': float(best_ind[1]),
        'best_max_bars': int(horizon * best_ind[2]),
        'best_fitness': float(best_fitness),
        'population_size': population_size,
        'generations': generations,
        'convergence': [
            {
                'gen': record['gen'],
                'avg': float(record['avg']),
                'max': float(record['max']),
                'min': float(record['min']),
                'std': float(record['std'])
            }
            for record in logbook
        ]
    }

    return results, logbook


def plot_convergence(results: Dict, output_path: Path):
    """Plot GA convergence curve."""
    convergence = results['convergence']

    gens = [c['gen'] for c in convergence]
    avg_fits = [c['avg'] for c in convergence]
    max_fits = [c['max'] for c in convergence]
    min_fits = [c['min'] for c in convergence]

    plt.figure(figsize=(10, 6))
    plt.plot(gens, max_fits, 'g-', label='Best', linewidth=2)
    plt.plot(gens, avg_fits, 'b-', label='Average', linewidth=2)
    plt.plot(gens, min_fits, 'r-', label='Worst', linewidth=1, alpha=0.5)

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title(f'GA Convergence - Horizon {results["horizon"]}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"  Saved convergence plot to {output_path}")


def process_symbol_ga(
    symbol: str,
    horizons: List[int] = [1, 5, 20],
    population_size: int = 50,
    generations: int = 30
) -> Dict[int, Dict]:
    """
    Run GA optimization for all horizons for a symbol.

    Returns:
    --------
    all_results : dict mapping horizon -> results
    """
    # Load labels_init data (has features + initial labels)
    project_root = Path(__file__).parent.parent.parent.resolve()
    labels_dir = project_root / 'data' / 'labels'
    input_path = labels_dir / f"{symbol}_labels_init.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Labels file not found: {input_path}")

    logger.info("=" * 70)
    logger.info(f"GA OPTIMIZATION: {symbol}")
    logger.info("=" * 70)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    all_results = {}

    for horizon in horizons:
        results, logbook = run_ga_optimization(
            df, horizon,
            population_size=population_size,
            generations=generations,
            subset_fraction=0.2  # Use 20% of data for speed
        )

        all_results[horizon] = results

        # Save results
        ga_results_dir = project_root / 'config' / 'ga_results'
        ga_results_dir.mkdir(parents=True, exist_ok=True)

        results_path = ga_results_dir / f"{symbol}_ga_h{horizon}_best.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"  Saved results to {results_path}")

        # Plot convergence
        plots_dir = project_root / 'results' / 'ga_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / f"{symbol}_ga_h{horizon}_convergence.png"
        plot_convergence(results, plot_path)

    return all_results


def main():
    """Run Stage 5: GA optimization for all symbols."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import SYMBOLS

    horizons = [1, 5, 20]
    population_size = 50
    generations = 30

    logger.info("=" * 70)
    logger.info("STAGE 5: GENETIC ALGORITHM OPTIMIZATION")
    logger.info("=" * 70)
    logger.info(f"Horizons: {horizons}")
    logger.info(f"GA Config: pop_size={population_size}, gens={generations}")
    logger.info("")

    all_symbols_results = {}

    for symbol in SYMBOLS:
        try:
            results = process_symbol_ga(
                symbol, horizons, population_size, generations
            )
            all_symbols_results[symbol] = results
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    # Save combined summary
    project_root = Path(__file__).parent.parent.parent.resolve()
    summary_path = project_root / 'config' / 'ga_results' / 'optimization_summary.json'
    summary = {}
    for symbol, symbol_results in all_symbols_results.items():
        summary[symbol] = {
            str(h): {
                'k_up': res['best_k_up'],
                'k_down': res['best_k_down'],
                'max_bars': res['best_max_bars'],
                'fitness': res['best_fitness']
            }
            for h, res in symbol_results.items()
        }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 5 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Optimization summary saved to {summary_path}")


if __name__ == "__main__":
    main()
