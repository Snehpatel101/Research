"""
Helper functions for barrier optimization.

Contains:
    - get_contiguous_subset: Extract contiguous time block for temporal integrity

NOTE: DEAP-specific functions (create_toolbox) have been removed.
      The optimization now uses Optuna TPE via optuna_optimizer.py.
"""

import logging
import random

import pandas as pd

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
