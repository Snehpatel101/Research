"""
Symmetric CUSUM filter for event-driven sampling.

Only labels bars where cumulative price movements exceed a threshold,
reducing noise from non-informative bars. This improves label quality
by focusing on structural breaks rather than random fluctuations.

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning", Chapter 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit
def _cusum_inner(returns: np.ndarray, threshold: float) -> np.ndarray:
    """Numba-accelerated CUSUM inner loop.

    Tracks positive and negative cumulative sums of returns.
    Triggers an event when either crosses the threshold, then resets both.

    Args:
        returns: 1D array of returns (NaN-free).
        threshold: Cumulative deviation threshold to trigger an event.

    Returns:
        Array of integer indices where events occurred.
    """
    n = len(returns)
    events = np.empty(n, dtype=np.int64)
    count = 0
    s_pos = 0.0
    s_neg = 0.0
    for i in range(n):
        s_pos = max(0.0, s_pos + returns[i])
        s_neg = min(0.0, s_neg + returns[i])
        if s_pos > threshold or s_neg < -threshold:
            events[count] = i
            count += 1
            s_pos = 0.0
            s_neg = 0.0
    return events[:count]


def cusum_filter(returns: pd.Series, threshold: float = 0.01) -> np.ndarray:
    """Apply symmetric CUSUM filter to return series.

    Args:
        returns: Log returns or simple returns series.
        threshold: Cumulative deviation threshold to trigger an event.
                   Typical values: 0.005 to 0.02 for intraday.

    Returns:
        Array of integer indices (positional) where events occurred.

    Raises:
        ValueError: If threshold is not positive.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    clean = returns.dropna().values.astype(np.float64)
    if len(clean) == 0:
        return np.empty(0, dtype=np.int64)

    # Map positions in clean array back to positions in original series
    valid_mask = returns.notna().values
    valid_positions = np.where(valid_mask)[0]

    inner_indices = _cusum_inner(clean, threshold)
    if len(inner_indices) == 0:
        return np.empty(0, dtype=np.int64)

    return valid_positions[inner_indices]


def cusum_events_mask(returns: pd.Series, threshold: float = 0.01) -> pd.Series:
    """Return boolean mask aligned to returns index (True = event bar).

    Args:
        returns: Log returns or simple returns series.
        threshold: Cumulative deviation threshold to trigger an event.

    Returns:
        Boolean Series with same index as returns. True at event bars.
    """
    event_indices = cusum_filter(returns, threshold)
    mask = np.zeros(len(returns), dtype=bool)
    if len(event_indices) > 0:
        mask[event_indices] = True
    return pd.Series(mask, index=returns.index, name="cusum_event")


def get_cusum_threshold(
    returns: pd.Series,
    target_events_per_day: float = 5.0,
    bars_per_day: int = 78,
) -> float:
    """Auto-calibrate CUSUM threshold to achieve target event frequency.

    Uses binary search on threshold to match desired events/day.

    Args:
        returns: Log returns or simple returns series.
        target_events_per_day: Desired average number of events per trading day.
        bars_per_day: Number of bars in one trading day (default 78 for 5-min bars).

    Returns:
        Threshold value that produces closest to target_events_per_day.

    Raises:
        ValueError: If target_events_per_day is not positive or bars_per_day < 1.
    """
    if target_events_per_day <= 0:
        raise ValueError(f"target_events_per_day must be positive, got {target_events_per_day}")
    if bars_per_day < 1:
        raise ValueError(f"bars_per_day must be >= 1, got {bars_per_day}")

    n_bars = len(returns.dropna())
    if n_bars == 0:
        return 0.01  # default fallback

    n_days = max(n_bars / bars_per_day, 1.0)
    target_total_events = target_events_per_day * n_days

    lo = 0.0001
    hi = 0.1

    # Binary search: 50 iterations gives precision ~2^-50 * (hi-lo) ~ 1e-16
    for _ in range(50):
        mid = (lo + hi) / 2.0
        n_events = len(cusum_filter(returns, mid))
        if n_events > target_total_events:
            lo = mid  # threshold too low, raise it
        else:
            hi = mid  # threshold too high, lower it

    return (lo + hi) / 2.0
