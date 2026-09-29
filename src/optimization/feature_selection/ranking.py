"""
Deterministic feature rankings.

Feature importances carry floating-point noise at the ~1e-16 level (summation
order, BLAS kernels), and many features tie exactly (a feature the model never
uses scores 0). Ranking raw values lets that noise, or the incidental order of
the input, decide which near-tied feature makes the cut, so two identical runs
can select different features. ``rank_by_importance`` quantizes importances to
``RANKING_SIGNIFICANT_DIGITS`` relative to the largest magnitude and breaks
ties by feature name.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Digits kept relative to the largest |importance|: far above float noise
# (~1e-16 relative), far below any meaningful importance difference
RANKING_SIGNIFICANT_DIGITS = 12


def quantize_importance(
    importance: pd.Series, significant_digits: int = RANKING_SIGNIFICANT_DIGITS
) -> pd.Series:
    """Round to ``significant_digits`` relative to the series' largest magnitude.

    Values below the resolution (noise around an exact 0) become 0.0.
    """
    finite = importance[np.isfinite(importance.to_numpy(dtype=float))]
    scale = float(finite.abs().max()) if len(finite) else 0.0
    if scale == 0.0:
        return importance.astype(float)
    decimals = significant_digits - 1 - math.floor(math.log10(scale))
    return importance.astype(float).round(decimals) + 0.0  # + 0.0 folds -0.0 into 0.0


def rank_by_importance(
    importance: pd.Series, significant_digits: int = RANKING_SIGNIFICANT_DIGITS
) -> pd.Series:
    """
    ``importance`` ordered from most to least important, deterministically.

    Order: quantized importance descending (NaN last), then feature name
    ascending. The returned values are the original, unrounded importances.
    """
    quantized = quantize_importance(importance, significant_digits)
    order = pd.DataFrame(
        {
            "neg": -quantized.to_numpy(dtype=float),
            "name": [str(i) for i in importance.index],
        }
    ).sort_values(["neg", "name"], kind="mergesort", na_position="last")
    return importance.iloc[order.index.to_numpy()]


def top_features(importance: pd.Series, n: int) -> list[str]:
    """The ``n`` most important features (deterministic tie-breaking)."""
    return list(rank_by_importance(importance).index[:n])


__all__ = [
    "RANKING_SIGNIFICANT_DIGITS",
    "quantize_importance",
    "rank_by_importance",
    "top_features",
]
