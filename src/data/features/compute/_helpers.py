"""
Shared helper functions for feature computation.

Phase 29: Consolidated duplicate functions to compute once and cache.
Previously had 4 duplicate _log_returns() definitions across:
- entropy.py
- volatility.py
- regime.py
- microstructure.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """
    Calculate log returns.

    This is the canonical implementation - all feature modules should import
    from here to avoid duplicate computation.

    Args:
        close: Series of close prices

    Returns:
        Series of log returns: log(close_t / close_{t-1})
    """
    return np.log(close / close.shift(1))
