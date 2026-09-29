"""
Fixed-width window Fractional Differentiation (FFD).

Preserves memory in price series while achieving stationarity.
d=0 returns original series, d=1 returns first difference.
Optimal d is typically 0.2-0.4 for financial time series.

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning", Chapter 5.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)

# Smallest differentiation order the feature pipeline accepts (d=0 is the raw level)
MIN_FRAC_DIFF_D = 0.05


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


def ffd_weights(d: float, threshold: float = 1e-5, max_window: int = 100) -> np.ndarray:
    """FFD weights ``[w_0 = 1, w_1, ...]`` for order ``d``.

    The window is fixed by ``d``, ``threshold`` and ``max_window`` alone — never
    by the length of the series it is applied to — so a value computed on a
    long training history equals the value computed on a short serving window
    wherever both have the ``len(weights)`` bars of history it needs.
    """
    if max_window < 2:
        raise ValueError(f"max_window must be >= 2, got {max_window}")
    return _get_weights_ffd(float(d), float(threshold), int(max_window))


def frac_diff_ffd(
    series: pd.Series,
    d: float = 0.3,
    threshold: float = 1e-5,
    max_window: int | None = None,
) -> pd.Series:
    """Compute fixed-width window fractional differentiation.

    Causal: the value at bar ``t`` uses bars ``t - window + 1 .. t`` only.

    Args:
        series: Input price series (typically log prices or close prices).
        d: Differentiation order. 0 = original, 1 = first diff.
           Typical range: 0.2-0.4 for stationarity with memory.
        threshold: Weight truncation threshold. Smaller = more precise but slower.
        max_window: Hard cap on the window length. Pass an explicit value
            whenever the result must not depend on how much data is supplied
            (features replayed at inference). ``None`` keeps the legacy
            data-dependent cap ``min(500, len(series) // 2)``.

    Returns:
        pd.Series aligned to original index with NaN for warmup period.
    """
    if d == 0.0:
        return series.copy()

    clean = series.dropna()
    if len(clean) == 0:
        return series.copy()

    x = clean.values.astype(np.float64)
    if max_window is None:
        # Legacy: cap at half the data length to ensure sufficient output.
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
    threshold: float = 1e-5,
    max_window: int | None = None,
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
        threshold: FFD weight truncation threshold used for every candidate d.
        max_window: FFD window cap used for every candidate d. Pass the same
            value the features are computed with, so the d found is the d
            that makes THOSE features stationary.

    Returns:
        Minimum d value for stationarity. Returns 1.0 if no d found.

    Raises:
        ImportError: statsmodels (the ADF test) is not installed.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:
        raise ImportError(
            "find_min_d needs statsmodels for the ADF stationarity test. "
            "Install the optional extra: uv pip install -e '.[stats]' "
            "(or set an explicit frac_diff d instead of 'auto')."
        ) from exc

    d_values = np.arange(d_range[0], d_range[1] + d_step / 2, d_step)

    for d in d_values:
        diffed = (
            series.dropna()
            if d == 0.0
            else frac_diff_ffd(series, d=d, threshold=threshold, max_window=max_window).dropna()
        )

        if len(diffed) < 20:
            continue

        try:
            with warnings.catch_warnings():
                # statsmodels announces a future return type; the tuple form is fine here
                warnings.simplefilter("ignore", FutureWarning)
                adf_stat, p_value, *_ = adfuller(diffed, maxlag=1, autolag=None)
        except Exception:
            continue

        if p_value < p_value_threshold:
            return float(round(d, 10))

    return 1.0


def resolve_frac_diff_d(
    d: float | str,
    train_log_price: pd.Series,
    threshold: float = 1e-5,
    max_window: int = 100,
) -> float:
    """Resolve the configured ``d`` (a number, or ``"auto"``) into a frozen number.

    ``"auto"`` runs :func:`find_min_d` (ADF) on ``train_log_price`` — the
    TRAINING rows only — with the same window and threshold the features use.
    The result is frozen into the feature spec, so inference replays the same
    ``d`` and never re-fits it on serving data.

    Raises:
        ImportError: ``"auto"`` without statsmodels installed.
        ValueError: ``d`` is not ``"auto"`` or a number in (0, 1].
    """
    if isinstance(d, str):
        if d != "auto":
            raise ValueError(f"frac_diff d must be a number or 'auto', got {d!r}")
        found = find_min_d(train_log_price, threshold=threshold, max_window=max_window)
        if found < MIN_FRAC_DIFF_D:
            # The log price already passes ADF undifferenced (a random walk does
            # ~5% of the time by chance). d=0 would return the raw price level, which
            # the feature spec rejects, so keep the smallest supported order.
            logger.info(
                f"frac_diff d='auto': the training log price is already stationary "
                f"(ADF at d=0); using the smallest supported d={MIN_FRAC_DIFF_D}"
            )
            return MIN_FRAC_DIFF_D
        return found
    value = float(d)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"frac_diff d must be in (0, 1], got {value}")
    return value
