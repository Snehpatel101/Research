"""
Deterministic feature rankings.

Scores carry floating-point noise in their last bits (summation order, BLAS
kernels), and many features tie exactly. Ranking raw values lets that noise, or
the incidental order of the input, decide which near-tied feature makes a cut,
so two identical runs could select different features. ``rank_by_importance``
rounds every score to ``RANKING_SIGNIFICANT_DIGITS`` of its OWN magnitude
(so scores spanning many orders of magnitude, e.g. variances, keep their order)
and breaks ties by feature name.

Permutation importances additionally scatter around an exact 0 for features the
model barely uses; ``noise_floor`` (importance series only, never variances)
folds values that small relative to the largest score into 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Digits kept per value: far above float noise (~1e-16 relative), far below any
# meaningful score difference
RANKING_SIGNIFICANT_DIGITS = 12
# Permutation importances below this fraction of the largest |importance| are
# float noise around 0 (pass as ``noise_floor`` for importance series)
IMPORTANCE_NOISE_FLOOR = 1e-12


def quantize_importance(
    importance: pd.Series,
    significant_digits: int = RANKING_SIGNIFICANT_DIGITS,
    noise_floor: float | None = None,
) -> pd.Series:
    """
    Round each value to ``significant_digits`` of its own magnitude.

    Args:
        importance: Scores indexed by feature.
        significant_digits: Digits kept per value.
        noise_floor: When set, values with ``|x| < noise_floor * max|x|`` become
            0.0. For importance series only: a variance ranking has
            legitimately tiny values that must keep their order.
    """
    values = importance.to_numpy(dtype=float)
    rounded = np.array(
        [float(f"{v:.{significant_digits}g}") if np.isfinite(v) else v for v in values]
    )
    if noise_floor is not None:
        finite = np.abs(rounded[np.isfinite(rounded)])
        scale = float(finite.max()) if finite.size else 0.0
        rounded[np.abs(rounded) < noise_floor * scale] = 0.0
    return pd.Series(rounded + 0.0, index=importance.index)  # + 0.0 folds -0.0 into 0.0


def rank_by_importance(
    importance: pd.Series,
    significant_digits: int = RANKING_SIGNIFICANT_DIGITS,
    noise_floor: float | None = None,
) -> pd.Series:
    """
    ``importance`` ordered from most to least important, deterministically.

    Order: quantized score descending (NaN last), then feature name ascending.
    The returned values are the original, unrounded scores.
    """
    quantized = quantize_importance(importance, significant_digits, noise_floor)
    order = pd.DataFrame(
        {
            "neg": -quantized.to_numpy(dtype=float),
            "name": [str(i) for i in importance.index],
        }
    ).sort_values(["neg", "name"], kind="mergesort", na_position="last")
    return importance.iloc[order.index.to_numpy()]


def top_features(
    importance: pd.Series, n: int, noise_floor: float | None = IMPORTANCE_NOISE_FLOOR
) -> list[str]:
    """The ``n`` most important features (permutation-importance noise floor by default)."""
    return list(rank_by_importance(importance, noise_floor=noise_floor).index[:n])


__all__ = [
    "IMPORTANCE_NOISE_FLOOR",
    "RANKING_SIGNIFICANT_DIGITS",
    "quantize_importance",
    "rank_by_importance",
    "top_features",
]
