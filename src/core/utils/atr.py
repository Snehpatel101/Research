"""
Canonical True Range and Wilder Average True Range.

One definition shared by the triple-barrier labeler, the backtester's
barriers and costs, the volatility regime detector and the ATR features, so
labels and backtest play the same game (same values, same warm-up).

Wilder ATR (Wilder 1978; TA-Lib ``ATR``):

- ``TR[i] = max(high[i] - low[i], |high[i] - close[i-1]|, |low[i] - close[i-1]|)``
  for ``i >= 1``. Bar 0 has no previous close, so it does not enter the ATR.
- ``ATR[period] = mean(TR[1 .. period])`` (the first ``period`` true ranges).
- ``ATR[i] = (ATR[i-1] * (period - 1) + TR[i]) / period`` for ``i > period``.
- ``ATR[i]`` is NaN for ``i < period``: bars without ``period`` true ranges
  have no ATR (the labeler marks them invalid, the backtester opens no
  barrier trade on them).

``ATR[i]`` uses bars ``0 .. i`` only (causal, no lookahead); consumers that
need the value known before bar ``i`` opens shift it themselves (features).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

__all__ = ["true_range", "wilder_atr"]


@njit(cache=True)
def _wilder_atr_kernel(
    high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], period: int
) -> NDArray[np.float64]:
    n = len(high)
    tr = np.zeros(n)
    atr = np.full(n, np.nan)
    if n <= period:
        return atr

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr[period] = np.mean(tr[1 : period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _as_float64(values: ArrayLike) -> NDArray[np.float64]:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def wilder_atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> np.ndarray:
    """Wilder Average True Range (float64, NaN before bar ``period``).

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Number of true ranges averaged (Wilder's ``N``), >= 1.

    Returns:
        Array of ``len(high)`` ATR values; the first ``period`` are NaN (all
        of them when there are not more than ``period`` bars).
    """
    if int(period) < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    h, lo, c = _as_float64(high), _as_float64(low), _as_float64(close)
    if not len(h) == len(lo) == len(c):
        raise ValueError(f"high/low/close lengths differ: {len(h)}, {len(lo)}, {len(c)}")
    return _wilder_atr_kernel(h, lo, c, int(period))


def true_range(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> np.ndarray:
    """True range per bar; bar 0 (no previous close) is ``high - low``."""
    h, lo, c = _as_float64(high), _as_float64(low), _as_float64(close)
    prev_close = np.empty_like(c)
    prev_close[:1] = np.nan
    prev_close[1:] = c[:-1]
    # fmax skips the NaN previous close on bar 0
    return np.fmax(h - lo, np.fmax(np.abs(h - prev_close), np.abs(lo - prev_close)))
