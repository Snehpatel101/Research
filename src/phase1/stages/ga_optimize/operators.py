"""
GA operators and helper functions.

Contains:
    - get_contiguous_subset: Extract contiguous time block for temporal integrity
    - check_bounds: Enforce parameter bounds on individuals
    - create_toolbox: Create and configure DEAP toolbox
    - get_seeded_individuals: Get symbol-specific seeded starting points
"""

import logging
import random
from typing import List

import numpy as np
import pandas as pd
from deap import base, creator, tools

from .fitness import evaluate_individual

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Search space bounds
K_MIN, K_MAX = 0.8, 2.5
MAX_BARS_MIN, MAX_BARS_MAX = 2.0, 3.0


def get_contiguous_subset(
    df: pd.DataFrame,
    subset_fraction: float,
    seed: int = 42,
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

    max_start = total_len - subset_len
    if max_start <= 0:
        return df.copy()

    random.seed(seed)
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + subset_len

    return df.iloc[start_idx:end_idx].copy()


def check_bounds(individual: List[float]) -> List[float]:
    """
    Enforce parameter bounds on an individual.

    Parameters:
    -----------
    individual : [k_up, k_down, max_bars_multiplier]

    Returns:
    --------
    individual : Clamped to valid bounds
    """
    # k_up bounds
    individual[0] = max(K_MIN, min(K_MAX, individual[0]))
    # k_down bounds
    individual[1] = max(K_MIN, min(K_MAX, individual[1]))
    # max_bars_mult bounds
    individual[2] = max(MAX_BARS_MIN, min(MAX_BARS_MAX, individual[2]))
    return individual


def create_toolbox(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_prices: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    symbol: str,
    tournament_size: int = 3,
) -> base.Toolbox:
    """
    Create and configure DEAP toolbox for GA optimization.

    Parameters:
    -----------
    close, high, low, open_prices, atr : Price/indicator arrays
    horizon : Horizon value
    symbol : Symbol name for symbol-specific constraints
    tournament_size : Tournament selection size

    Returns:
    --------
    toolbox : Configured DEAP toolbox
    """
    # Clean up any existing DEAP types
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Register attribute generators
    toolbox.register("k_up", random.uniform, K_MIN, K_MAX)
    toolbox.register("k_down", random.uniform, K_MIN, K_MAX)
    toolbox.register("max_bars_mult", random.uniform, MAX_BARS_MIN, MAX_BARS_MAX)

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.k_up, toolbox.k_down, toolbox.max_bars_mult),
        n=1,
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
        symbol=symbol,
    )

    # Genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.20, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    return toolbox


def get_seeded_individuals(symbol: str) -> List[List[float]]:
    """
    Get symbol-specific seeded starting points for the GA.

    Parameters:
    -----------
    symbol : 'MES' or 'MGC'

    Returns:
    --------
    List of seeded individuals [k_up, k_down, max_bars_mult]
    """
    if symbol == "MGC":
        # MGC: Symmetric barriers for mean-reverting gold
        return [
            [1.2, 1.2, 2.5],  # Symmetric, medium barriers
            [1.5, 1.5, 2.5],  # Symmetric, wider barriers
            [1.0, 1.0, 2.0],  # Symmetric, tighter barriers
            [1.3, 1.3, 2.8],  # Symmetric, medium
            [1.8, 1.8, 2.2],  # Symmetric, wide
        ]
    else:
        # MES: Asymmetric barriers for equity drift (k_up > k_down)
        return [
            [1.5, 1.0, 2.5],  # Asymmetric, upper barrier 50% higher
            [1.8, 1.2, 2.5],  # Asymmetric, upper barrier 50% higher
            [1.3, 0.9, 2.0],  # Asymmetric, tighter barriers
            [2.0, 1.4, 2.8],  # Asymmetric, wider barriers
            [1.6, 1.1, 2.2],  # Asymmetric, medium
        ]
