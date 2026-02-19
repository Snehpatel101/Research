"""
Shared helper functions for pipeline feature engineering stages.

Phase H5: Consolidated duplicate ``_np_shift1`` definitions from:
- momentum.py
- trend.py
- volatility.py
- wavelets.py
"""

from __future__ import annotations

import numpy as np


def np_shift1(arr: np.ndarray) -> np.ndarray:
    """
    Shift array by 1 using numpy (avoids pd.Series overhead).

    Inserts ``np.nan`` at position 0 and shifts all other values forward by one
    index.  This is the canonical implementation -- all pipeline feature modules
    should import from here instead of defining their own ``_np_shift1``.

    Args:
        arr: Input 1-D numpy array.

    Returns:
        Shifted array with ``np.nan`` at the first position.
    """
    result = np.empty_like(arr, dtype=np.float64)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result
