"""
Stage 5: Genetic Algorithm Optimization of Labeling Parameters
Uses DEAP to find optimal k_up, k_down, and max_bars for balanced, tradable labels

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
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional
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

# Import config for transaction costs
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRANSACTION_COSTS, TICK_VALUES

# Configure logging - use NullHandler to avoid duplicate logs when imported as module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# IMPROVED FITNESS FUNCTION WITH TRANSACTION COSTS
# =============================================================================

def calculate_fitness(
    labels: np.ndarray,
    bars_to_hit: np.ndarray,
    mae: np.ndarray,
    mfe: np.ndarray,
    horizon: int,
    atr_mean: float,
    symbol: str = 'MES'
) -> float:
    """
    Calculate fitness score for a set of labels.

    IMPROVEMENTS:
    1. Require at least 60% directional signals (target 70-80% signal rate)
    2. Target 20-30% neutral rate (was <2%)
    3. Reward balanced long/short ratio
    4. Speed score prefers faster resolution
    5. FIXED profit factor using ACTUAL trade outcomes (not MAE/MFE confusion)
    6. TRANSACTION COST penalty (MES: 0.5 ticks, MGC: 0.3 ticks)

    Parameters:
    -----------
    labels : array of -1, 0, 1
    bars_to_hit : array of bars until barrier hit
    mae : Maximum Adverse Excursion (always negative or zero - worst drawdown)
    mfe : Maximum Favorable Excursion (always positive or zero - best gain)
    horizon : horizon identifier (5 or 20)
    atr_mean : mean ATR for the sample (for transaction cost normalization)
    symbol : 'MES' or 'MGC' (for symbol-specific transaction costs)

    Returns:
    --------
    fitness : float (higher is better)
    """
    total = len(labels)
    if total == 0:
        return -1000.0

    n_long = (labels == 1).sum()
    n_short = (labels == -1).sum()
    n_neutral = (labels == 0).sum()

    # ==========================================================================
    # 1. SIGNAL RATE REQUIREMENT (target 70-80% signals, 20-30% neutral)
    # ==========================================================================
    signal_rate = (n_long + n_short) / total
    neutral_pct = n_neutral / total

    # Target: 20-30% neutral rate
    if neutral_pct < 0.15:
        # Too few neutrals - trading on noise
        return -1000.0 + (neutral_pct * 100)
    elif neutral_pct > 0.40:
        # Too many neutrals - not enough trading signals
        return -1000.0 + ((1 - neutral_pct) * 100)

    # ==========================================================================
    # 2. NEUTRAL SCORE (target 20-30% neutral)
    # ==========================================================================
    # Perfect neutral range: 20-30%
    TARGET_NEUTRAL_LOW = 0.20
    TARGET_NEUTRAL_HIGH = 0.30
    TARGET_NEUTRAL_MID = 0.25

    if TARGET_NEUTRAL_LOW <= neutral_pct <= TARGET_NEUTRAL_HIGH:
        neutral_score = 3.0  # Perfect range
    else:
        # Penalize deviation from target range
        if neutral_pct < TARGET_NEUTRAL_LOW:
            deviation = TARGET_NEUTRAL_LOW - neutral_pct
        else:
            deviation = neutral_pct - TARGET_NEUTRAL_HIGH
        neutral_score = 3.0 - (deviation * 15.0)

    # ==========================================================================
    # 3. LONG/SHORT BALANCE SCORE
    # ==========================================================================
    if (n_long + n_short) > 0:
        long_ratio = n_long / (n_long + n_short)
        # Perfect balance = 0.5, max score = 2.0
        # Deviation from 0.5 penalized heavily
        balance_score = 2.0 - abs(long_ratio - 0.5) * 8.0

        # Additional penalty for extreme imbalance
        if long_ratio < 0.30 or long_ratio > 0.70:
            balance_score -= 3.0
    else:
        balance_score = -10.0

    # ==========================================================================
    # 4. SPEED SCORE (prefer faster resolution within reason)
    # ==========================================================================
    hit_mask = labels != 0
    if hit_mask.sum() > 0:
        avg_bars = bars_to_hit[hit_mask].mean()
        # Normalize by horizon - prefer hitting barriers in 1-2x horizon
        max_expected = horizon * 2.0
        speed_score = 1.0 - (avg_bars / max_expected)
        speed_score = np.clip(speed_score, -2.0, 1.5)
    else:
        speed_score = -2.0

    # ==========================================================================
    # 5. PROFIT FACTOR SCORE (FIXED CALCULATION)
    # ==========================================================================
    # CRITICAL FIX: The trade outcome is determined by which barrier was hit,
    # NOT by comparing MAE/MFE. The label tells us the outcome:
    # - label = 1 (Long): Upper barrier hit first -> PROFIT = upper_barrier level
    # - label = -1 (Short): Lower barrier hit first -> PROFIT = lower_barrier level
    # - label = 0 (Neutral): Timeout -> No trade taken
    #
    # For profit factor, we need to estimate:
    # - Total profits from winning trades
    # - Total losses from losing trades (drawdown before winning)
    #
    # Use MFE as profit potential and MAE as risk taken.
    # MFE is the max favorable move (profit), MAE is max adverse move (risk).

    long_mask = labels == 1
    short_mask = labels == -1

    # Long trades: MFE is profit (upside), |MAE| is the drawdown risk
    # Short trades: |MAE| represents the downside (profit for short), MFE is upside risk

    # CORRECT interpretation:
    # - For a LONG (label=1): The trade hit the UPPER barrier
    #   - Actual profit = the barrier level (approximated by MFE at hit)
    #   - Risk taken = |MAE| (max drawdown during trade)
    # - For a SHORT (label=-1): The trade hit the LOWER barrier
    #   - Actual profit = the barrier level (approximated by |MAE| magnitude)
    #   - Risk taken = MFE (max upside move during trade = max loss for short)

    if long_mask.sum() > 0:
        # Long profits = MFE (how far up it went)
        long_profits = np.maximum(mfe[long_mask], 0).sum()
        # Long risk = |MAE| (how far down it went before winning)
        long_risk = np.abs(mae[long_mask]).sum()
    else:
        long_profits = 0.0
        long_risk = 0.0

    if short_mask.sum() > 0:
        # Short profits = |MAE| (how far down it went = profit for short)
        short_profits = np.abs(mae[short_mask]).sum()
        # Short risk = MFE (how far up it went = loss for short)
        # FIX: Remove np.maximum wrapper - MFE represents upward movement (risk for shorts)
        short_risk = mfe[short_mask].sum()
    else:
        short_profits = 0.0
        short_risk = 0.0

    total_profit = long_profits + short_profits
    total_risk = long_risk + short_risk

    if total_risk > 1e-10:
        profit_factor = total_profit / total_risk

        # Prefer profit factor between 1.0 and 2.0 (realistic range)
        if 1.0 <= profit_factor <= 2.0:
            pf_score = 2.0
        elif 0.8 <= profit_factor < 1.0:
            pf_score = 0.5  # Near-profitable
        elif profit_factor > 2.0:
            pf_score = 1.5  # Good but might be overfit
        else:
            pf_score = -2.0  # Unprofitable
    else:
        pf_score = 0.0

    # ==========================================================================
    # 6. TRANSACTION COST PENALTY (NEW)
    # ==========================================================================
    # Penalize strategies where expected profit is small relative to costs
    # FIX: Convert transaction costs to price units to match avg_profit units
    cost_ticks = TRANSACTION_COSTS.get(symbol, 0.5)
    tick_value = TICK_VALUES.get(symbol, 1.0)
    cost_in_price_units = cost_ticks * tick_value  # Convert ticks to price units

    # Estimate average profit per trade (in price units)
    n_trades = n_long + n_short
    if n_trades > 0 and atr_mean > 0:
        avg_profit_per_trade = total_profit / n_trades

        # Transaction cost as fraction of average profit (both in price units now)
        # cost_ratio = transaction_cost / avg_profit (dimensionally correct)
        cost_ratio = cost_in_price_units / (avg_profit_per_trade + 1e-6)

        # Penalize if cost ratio is too high (>20% of expected profit)
        if cost_ratio > 0.20:
            transaction_penalty = -(cost_ratio - 0.20) * 10.0
        else:
            transaction_penalty = 0.5  # Bonus for low-cost strategy
    else:
        transaction_penalty = 0.0

    # ==========================================================================
    # COMBINED FITNESS
    # ==========================================================================
    fitness = neutral_score + balance_score + speed_score + pf_score + transaction_penalty

    return fitness


def evaluate_individual(
    individual: List[float],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_prices: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    symbol: str = 'MES'
) -> Tuple[float]:
    """
    Evaluate a GA individual (parameter set).

    Parameters:
    -----------
    individual : [k_up, k_down, max_bars_multiplier]
    close, high, low, open_prices, atr : price/indicator arrays
    horizon : horizon value
    symbol : 'MES' or 'MGC' for symbol-specific constraints

    Returns:
    --------
    (fitness,) : tuple with single fitness value (DEAP convention)
    """
    k_up, k_down, max_bars_mult = individual

    # ==========================================================================
    # SYMBOL-SPECIFIC ASYMMETRY CONSTRAINT
    # ==========================================================================
    # MES (equity): Allow asymmetric barriers (k_up > k_down) to counteract drift
    # MGC (gold): Enforce symmetric barriers (k_up ≈ k_down) for mean-reverting asset
    avg_k = (k_up + k_down) / 2.0

    if symbol == 'MGC':
        # MGC: Strict symmetry - penalize asymmetry > 10%
        if abs(k_up - k_down) > avg_k * 0.10:
            asymmetry_penalty = -abs(k_up - k_down) * 5.0  # Heavy penalty for asymmetry
        else:
            asymmetry_penalty = 0.5  # Bonus for symmetry
    else:
        # MES: Allow asymmetry but prefer k_up > k_down for equity drift
        # Penalize if k_down > k_up (wrong direction for equity)
        if k_down > k_up:
            asymmetry_penalty = -(k_down - k_up) * 3.0  # Wrong direction
        elif abs(k_up - k_down) > avg_k * 0.60:
            asymmetry_penalty = -abs(k_up - k_down) * 2.0  # Too asymmetric
        else:
            asymmetry_penalty = 0.0  # Good range of asymmetry

    # Decode max_bars - reduced from [2x, 4x] to [2x, 3x] for more neutrals
    max_bars = int(horizon * max_bars_mult)
    max_bars = max(horizon * 2, min(max_bars, horizon * 3))  # Clamp to [2x, 3x] horizon

    # Run labeling
    try:
        labels, bars_to_hit, mae, mfe, _ = triple_barrier_numba(
            close, high, low, open_prices, atr, k_up, k_down, max_bars
        )

        # Calculate mean ATR for transaction cost normalization
        atr_mean = np.mean(atr[atr > 0]) if np.any(atr > 0) else 1.0

        fitness = calculate_fitness(
            labels, bars_to_hit, mae, mfe, horizon,
            atr_mean=atr_mean,
            symbol=symbol
        )
        fitness += asymmetry_penalty

        return (fitness,)

    except Exception as e:
        # Use float('-inf') for invalid fitness to ensure it's never selected
        # Log as warning since this is acceptable GA behavior (invalid individuals get worst fitness)
        logger.warning(f"Fitness evaluation failed for individual [k_up={k_up:.3f}, k_down={k_down:.3f}, max_bars={max_bars}]: {e}")
        return (float('-inf'),)


def get_contiguous_subset(
    df: pd.DataFrame,
    subset_fraction: float,
    seed: int = 42
) -> pd.DataFrame:
    """
    Get a contiguous time block from the data instead of random sampling.

    This preserves temporal order which is critical for barrier calculations.
    Random sampling would create artificial gaps and invalidate the barriers.

    Parameters:
    -----------
    df : Full DataFrame
    subset_fraction : Fraction of data to use
    seed : Random seed for reproducibility (default: 42)

    Returns:
    --------
    df_subset : Contiguous slice of the data
    """
    total_len = len(df)
    subset_len = int(total_len * subset_fraction)

    if subset_len < 1000:
        subset_len = min(1000, total_len)

    # Choose a random starting point that allows for the full subset
    max_start = total_len - subset_len
    if max_start <= 0:
        return df.copy()

    # Set seed for reproducibility before random selection
    random.seed(seed)
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + subset_len

    return df.iloc[start_idx:end_idx].copy()


def run_ga_optimization(
    df: pd.DataFrame,
    horizon: int,
    symbol: str = 'MES',
    population_size: int = 50,
    generations: int = 30,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    tournament_size: int = 3,
    subset_fraction: float = 0.3,
    atr_column: str = 'atr_14',
    seed: int = 42
) -> Dict:
    """
    Run genetic algorithm to optimize labeling parameters.

    FIXES:
    - Uses contiguous time blocks instead of random sampling
    - WIDER search bounds: k_up/k_down [0.8, 2.0] for 20-30% neutral target
    - Symbol-specific asymmetry constraints (MES: asymmetric, MGC: symmetric)
    - Transaction cost penalty in fitness
    - Improved mutation with smaller sigma for finer tuning
    - Random seed for reproducibility

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
    """
    # Set random seeds for reproducibility at GA initialization
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"\nOptimizing parameters for {symbol} horizon {horizon}")
    logger.info(f"  Population: {population_size}, Generations: {generations}")
    logger.info(f"  Symbol-specific mode: {'SYMMETRIC (gold)' if symbol == 'MGC' else 'ASYMMETRIC (equity)'}")
    logger.info(f"  Random seed: {seed}")

    # ==========================================================================
    # FIX: Use contiguous time blocks instead of random sampling
    # ==========================================================================
    df_subset = get_contiguous_subset(df, subset_fraction, seed=seed)
    logger.info(f"  Using {len(df_subset):,} contiguous samples ({subset_fraction*100:.0f}% of data)")

    # Extract arrays
    close = df_subset['close'].values
    high = df_subset['high'].values
    low = df_subset['low'].values
    open_prices = df_subset['open'].values
    atr = df_subset[atr_column].values

    # Setup DEAP
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximize
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # ==========================================================================
    # FIX: WIDER search bounds for 20-30% neutral rate
    # ==========================================================================
    # Previous bounds [0.3, 1.5] were too tight, resulting in <2% neutral
    # New bounds [0.8, 2.5] allow for wider barriers = more timeouts = more neutrals
    # max_bars_mult: [2.0, 3.0] (shorter windows = more timeouts)

    K_MIN, K_MAX = 0.8, 2.5
    MAX_BARS_MIN, MAX_BARS_MAX = 2.0, 3.0

    toolbox.register("k_up", random.uniform, K_MIN, K_MAX)
    toolbox.register("k_down", random.uniform, K_MIN, K_MAX)
    toolbox.register("max_bars_mult", random.uniform, MAX_BARS_MIN, MAX_BARS_MAX)

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.k_up, toolbox.k_down, toolbox.max_bars_mult),
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function with symbol parameter
    toolbox.register(
        "evaluate",
        evaluate_individual,
        close=close,
        high=high,
        low=low,
        open_prices=open_prices,
        atr=atr,
        horizon=horizon,
        symbol=symbol
    )

    # Genetic operators - smaller sigma for finer tuning
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.20, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Constraint function with UPDATED bounds
    def check_bounds(individual):
        # k_up bounds
        individual[0] = max(K_MIN, min(K_MAX, individual[0]))
        # k_down bounds
        individual[1] = max(K_MIN, min(K_MAX, individual[1]))
        # max_bars_mult bounds
        individual[2] = max(MAX_BARS_MIN, min(MAX_BARS_MAX, individual[2]))
        return individual

    # Create initial population
    population = toolbox.population(n=population_size)

    # ==========================================================================
    # SEED with symbol-specific starting points
    # ==========================================================================
    if symbol == 'MGC':
        # MGC: Symmetric barriers for mean-reverting gold
        seeded_individuals = [
            [1.2, 1.2, 2.5],   # Symmetric, medium barriers
            [1.5, 1.5, 2.5],   # Symmetric, wider barriers
            [1.0, 1.0, 2.0],   # Symmetric, tighter barriers
            [1.3, 1.3, 2.8],   # Symmetric, medium
            [1.8, 1.8, 2.2],   # Symmetric, wide
        ]
    else:
        # MES: Asymmetric barriers for equity drift (k_up > k_down)
        seeded_individuals = [
            [1.5, 1.0, 2.5],   # Asymmetric, upper barrier 50% higher
            [1.8, 1.2, 2.5],   # Asymmetric, upper barrier 50% higher
            [1.3, 0.9, 2.0],   # Asymmetric, tighter barriers
            [2.0, 1.4, 2.8],   # Asymmetric, wider barriers
            [1.6, 1.1, 2.2],   # Asymmetric, medium
        ]

    for i, seed_ind in enumerate(seeded_individuals):
        if i < len(population):
            population[i][:] = seed_ind

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of fame (best individuals)
    hof = tools.HallOfFame(5)  # Keep top 5 for diversity

    # Run evolution
    logger.info("  Starting evolution...")

    # Track statistics
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # Track consecutive evaluation errors for early abort
    MAX_CONSECUTIVE_ERRORS = 10
    consecutive_errors = 0
    total_errors = 0

    def count_errors_in_fitnesses(fitnesses_list):
        """Count how many fitness values indicate errors (float('-inf'))."""
        return sum(1 for fit in fitnesses_list if fit[0] == float('-inf'))

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Check for errors in initial population
    init_errors = count_errors_in_fitnesses(fitnesses)
    if init_errors > 0:
        logger.warning(f"  Initial population had {init_errors} evaluation errors")
        total_errors += init_errors

    hof.update(population)
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(population), **record)

    # Log initial stats
    logger.info(f"  Gen 0: avg={record['avg']:.2f}, max={record['max']:.2f}")

    # Evolution loop with elitism
    for gen in tqdm(range(1, generations + 1), desc=f"  GA H={horizon}"):
        # Select next generation
        offspring = toolbox.select(population, len(population) - 2)  # Leave room for elites
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

        # Track consecutive errors and abort if too many
        gen_errors = count_errors_in_fitnesses(fitnesses)
        if gen_errors > 0:
            consecutive_errors += 1
            total_errors += gen_errors
            logger.warning(f"  Gen {gen}: {gen_errors} evaluation errors (consecutive: {consecutive_errors})")

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error(f"  ABORTING: {consecutive_errors} consecutive generations with errors. Total errors: {total_errors}")
                raise RuntimeError(f"GA optimization aborted due to {consecutive_errors} consecutive generations with evaluation errors")
        else:
            consecutive_errors = 0  # Reset on successful generation

        # ELITISM: Add best 2 from hall of fame
        elite = [toolbox.clone(ind) for ind in hof[:2]]
        offspring.extend(elite)

        # Replace population
        population[:] = offspring
        hof.update(population)

        # Record statistics
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        # Log progress every 10 generations
        if gen % 10 == 0:
            logger.info(f"  Gen {gen}: avg={record['avg']:.2f}, max={record['max']:.2f}")

    # Get best individual
    best_ind = hof[0]
    best_fitness = best_ind.fitness.values[0]

    logger.info(f"\n  OPTIMIZATION COMPLETE")
    logger.info(f"  Best fitness: {best_fitness:.4f}")
    logger.info(f"  Best params: k_up={best_ind[0]:.3f}, k_down={best_ind[1]:.3f}, "
                f"max_bars={int(horizon * best_ind[2])}")

    # ==========================================================================
    # VALIDATE best parameters on full data
    # ==========================================================================
    logger.info(f"  Validating on full dataset...")
    full_close = df['close'].values
    full_high = df['high'].values
    full_low = df['low'].values
    full_open = df['open'].values
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

    logger.info(f"  Full data label distribution:")
    logger.info(f"    Long:    {n_long:,} ({100*n_long/n_total:.1f}%)")
    logger.info(f"    Short:   {n_short:,} ({100*n_short/n_total:.1f}%)")
    logger.info(f"    Neutral: {n_neutral:,} ({100*n_neutral/n_total:.1f}%)")

    # Prepare results
    results = {
        'horizon': horizon,
        'best_k_up': float(best_ind[0]),
        'best_k_down': float(best_ind[1]),
        'best_max_bars': int(horizon * best_ind[2]),
        'best_fitness': float(best_fitness),
        'population_size': population_size,
        'generations': generations,
        'validation': {
            'n_total': int(n_total),
            'n_long': int(n_long),
            'n_short': int(n_short),
            'n_neutral': int(n_neutral),
            'pct_long': float(100 * n_long / n_total),
            'pct_short': float(100 * n_short / n_total),
            'pct_neutral': float(100 * n_neutral / n_total),
            'signal_rate': float((n_long + n_short) / n_total)
        },
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Convergence plot
    ax1.plot(gens, max_fits, 'g-', label='Best', linewidth=2)
    ax1.plot(gens, avg_fits, 'b-', label='Average', linewidth=2)
    ax1.plot(gens, min_fits, 'r-', label='Worst', linewidth=1, alpha=0.5)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.set_title(f'GA Convergence - Horizon {results["horizon"]}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Label distribution (if validation data available)
    if 'validation' in results:
        val = results['validation']
        labels = ['Long', 'Short', 'Neutral']
        sizes = [val['pct_long'], val['pct_short'], val['pct_neutral']]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']

        ax2.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1.2)
        ax2.axhline(y=40, color='orange', linestyle='--', label='40% min signal', alpha=0.7)
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_title(f'Label Distribution (Full Data)\nSignal Rate: {val["signal_rate"]*100:.1f}%',
                      fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 60)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for i, (label, pct) in enumerate(zip(labels, sizes)):
            ax2.text(i, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved convergence plot to {output_path}")


def process_symbol_ga(
    symbol: str,
    horizons: List[int] = [5, 20],  # Exclude H1 - not viable after transaction costs
    population_size: int = 50,
    generations: int = 30,
    seed: int = 42
) -> Dict[int, Dict]:
    """
    Run GA optimization for all horizons for a symbol.

    Parameters:
    -----------
    symbol : 'MES' or 'MGC'
    horizons : list of horizons to optimize (default excludes H1)
    population_size : GA population size
    generations : number of generations
    seed : random seed for reproducibility (default: 42)

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
    logger.info(f"  Mode: {'SYMMETRIC' if symbol == 'MGC' else 'ASYMMETRIC'}")
    logger.info(f"  Transaction cost: {TRANSACTION_COSTS.get(symbol, 0.5)} ticks")
    logger.info("=" * 70)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    all_results = {}

    for horizon in horizons:
        results, logbook = run_ga_optimization(
            df, horizon,
            symbol=symbol,  # Pass symbol for symbol-specific optimization
            population_size=population_size,
            generations=generations,
            subset_fraction=0.3,
            seed=seed
        )

        all_results[horizon] = results

        # Check if optimization succeeded
        val = results.get('validation', {})
        signal_rate = val.get('signal_rate', 0)
        if signal_rate < 0.40:
            logger.warning(f"  WARNING: Signal rate {signal_rate*100:.1f}% below 40% threshold!")
            logger.warning(f"  Consider adjusting search bounds or running more generations.")

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

    from config import SYMBOLS, ACTIVE_HORIZONS, RANDOM_SEED, set_global_seeds

    # Set random seeds for reproducibility
    set_global_seeds(RANDOM_SEED)

    # Use only ACTIVE_HORIZONS (excludes H1 which is not viable after costs)
    horizons = ACTIVE_HORIZONS  # [5, 20]
    population_size = 50
    generations = 30

    logger.info("=" * 70)
    logger.info("STAGE 5: GENETIC ALGORITHM OPTIMIZATION (PRODUCTION FIX)")
    logger.info("=" * 70)
    logger.info(f"Horizons: {horizons} (H1 excluded - not viable after transaction costs)")
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
            errors.append({
                'symbol': symbol,
                'error': str(e),
                'type': type(e).__name__
            })
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    if errors:
        error_summary = f"{len(errors)}/{len(SYMBOLS)} symbols failed GA optimization"
        logger.error(f"GA optimization completed with errors: {error_summary}")
        raise RuntimeError(f"{error_summary}. Errors: {errors[:5]}")

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
                'fitness': res['best_fitness'],
                'signal_rate': res.get('validation', {}).get('signal_rate', None),
                'pct_long': res.get('validation', {}).get('pct_long', None),
                'pct_short': res.get('validation', {}).get('pct_short', None),
                'pct_neutral': res.get('validation', {}).get('pct_neutral', None)
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

    # Final summary
    logger.info("\nFINAL PARAMETER SUMMARY:")
    for symbol, sym_res in summary.items():
        logger.info(f"\n  {symbol}:")
        for h, params in sym_res.items():
            sr = params.get('signal_rate', 0)
            status = "OK" if sr and sr >= 0.40 else "WARNING"
            logger.info(f"    H{h}: k_up={params['k_up']:.2f}, k_down={params['k_down']:.2f}, "
                       f"max_bars={params['max_bars']}, signal_rate={sr*100 if sr else 0:.1f}% [{status}]")


if __name__ == "__main__":
    main()
