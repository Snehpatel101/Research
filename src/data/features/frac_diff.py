"""
Fixed-width window Fractional Differentiation (FFD).

Preserves memory in price series while achieving stationarity.
d=0 returns original series, d=1 returns first difference.
Optimal d is typically 0.2-0.4 for financial time series.

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning", Chapter 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit
def _get_weights_ffd(d: float, threshold: float = 1e-5, max_len: int = 1000) -> np.ndarray:
    """Compute FFD weights using the binomial series.

    w_k = -w_{k-1} * (d - k + 1) / k
    Truncate when |w_k| < threshold.
    """
    weights = np.empty(max_len, dtype=np.float64)
    weights[0] = 1.0
    k = 1
    while k < max_len:
        w = -weights[k - 1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights[k] = w
        k += 1
    return weights[:k]


@njit
def _frac_diff_inner(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply FFD weights to series (convolution). Returns NaN for warmup period."""
    n = len(x)
    w_len = len(weights)
    result = np.full(n, np.nan)
    for i in range(w_len - 1, n):
        val = 0.0
        for j in range(w_len):
            val += weights[j] * x[i - j]
        result[i] = val
    return result


def frac_diff_ffd(
    series: pd.Series,
    d: float = 0.3,
    threshold: float = 1e-5,
) -> pd.Series:
    """Compute fixed-width window fractional differentiation.

    Args:
        series: Input price series (typically log prices or close prices).
        d: Differentiation order. 0 = original, 1 = first diff.
           Typical range: 0.2-0.4 for stationarity with memory.
        threshold: Weight truncation threshold. Smaller = more precise but slower.

    Returns:
        pd.Series aligned to original index with NaN for warmup period.
    """
    if d == 0.0:
        return series.copy()

    clean = series.dropna()
    if len(clean) == 0:
        return series.copy()

    x = clean.values.astype(np.float64)
    # Cap window at half the data length to ensure sufficient output.
    # For d=0.3, threshold=1e-5 produces ~2000 weights, which is too long
    # for most financial series. A window of 200-500 is standard practice.
    max_window = min(500, len(x) // 2)
    weights = _get_weights_ffd(d, threshold, max_len=max(max_window, 2))
    result = _frac_diff_inner(x, weights)

    out = pd.Series(np.nan, index=series.index, dtype=np.float64)
    out.loc[clean.index] = result
    return out


def find_min_d(
    series: pd.Series,
    p_value_threshold: float = 0.05,
    d_range: tuple[float, float] = (0.0, 1.0),
    d_step: float = 0.05,
) -> float:
    """Find minimum d that makes series stationary (ADF test).

    Scans d values from d_range[0] to d_range[1] in steps of d_step.
    Returns the smallest d where the Augmented Dickey-Fuller test
    p-value < p_value_threshold (series is stationary).

    Args:
        series: Input price series.
        p_value_threshold: ADF p-value threshold for stationarity.
        d_range: Range of d values to search.
        d_step: Step size for d search.

    Returns:
        Minimum d value for stationarity. Returns 1.0 if no d found.
    """
    from statsmodels.tsa.stattools import adfuller

    d_values = np.arange(d_range[0], d_range[1] + d_step / 2, d_step)

    for d in d_values:
        diffed = series.dropna() if d == 0.0 else frac_diff_ffd(series, d=d).dropna()

        if len(diffed) < 20:
            continue

        try:
            adf_stat, p_value, *_ = adfuller(diffed, maxlag=1, autolag=None)
        except Exception:
            continue

        if p_value < p_value_threshold:
            return float(round(d, 10))

    return 1.0
