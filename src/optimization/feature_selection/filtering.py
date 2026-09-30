"""
Feature filters used by the orchestrator's train-only selection.

- ``filter_low_variance``: drop near-constant features
- ``select_decorrelated_by_rank``: greedy rank-ordered decorrelation
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_low_variance(
    df: pd.DataFrame, feature_cols: list[str], variance_threshold: float = 0.01
) -> tuple[list[str], list[str]]:
    """
    Remove features with variance below threshold.

    Near-constant features provide no discriminative power and should be removed.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        variance_threshold: Minimum variance to keep feature (default 0.01)

    Returns:
        Tuple of (features_to_keep, low_variance_features)
    """
    low_variance = []
    to_keep = []

    for col in feature_cols:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()

        if len(series) < 100:
            # Not enough data points, keep feature
            to_keep.append(col)
            continue

        variance = series.var()

        # Normalize variance by scale for fair comparison
        # Use coefficient of variation for features with non-zero mean
        mean_val = abs(series.mean())
        normalized_variance = variance / mean_val**2 if mean_val > 1e-10 else variance

        # Check if variance is below threshold
        if variance < variance_threshold and normalized_variance < variance_threshold:
            low_variance.append(col)
        else:
            to_keep.append(col)

    return to_keep, low_variance


def select_decorrelated_by_rank(
    df: pd.DataFrame,
    ranked_features: list[str],
    n_target: int,
    n_min: int = 0,
    correlation_threshold: float = 0.85,
    max_rows: int = 50_000,
) -> list[str]:
    """
    Greedy rank-ordered decorrelation.

    Walks the ranking best-first and keeps a feature unless its absolute
    correlation with an already-kept feature exceeds ``correlation_threshold``,
    stopping at ``n_target``. Cutting the ranking first and decorrelating
    afterwards starves the selection when the top features are correlated
    members of the same signal cluster; walking the whole ranking keeps the
    best representative of each cluster and then moves on to the next cluster.

    If fewer than ``n_min`` features survive, the best-ranked rejected ones are
    added back (in rank order) so every model's minimum is still met.

    Args:
        df: Feature data (only the ranked columns are used)
        ranked_features: Features ordered best first
        n_target: Number of features to select at most
        n_min: Number of features to select at least
        correlation_threshold: Max absolute correlation with a kept feature
        max_rows: Rows used to estimate correlations (evenly strided, in order)

    Returns:
        Selected features, best first
    """
    ranked = [f for f in ranked_features if f in df.columns]
    data = df[ranked]
    if len(data) > max_rows:
        data = data.iloc[:: int(np.ceil(len(data) / max_rows))]
    corr = data.replace([np.inf, -np.inf], np.nan).corr().abs().fillna(0.0).to_numpy()
    position = {f: i for i, f in enumerate(ranked)}

    selected: list[str] = []
    for feature in ranked:
        if len(selected) >= n_target:
            break
        i = position[feature]
        if all(corr[i, position[kept]] <= correlation_threshold for kept in selected):
            selected.append(feature)

    if len(selected) < n_min:
        chosen = set(selected)
        top_up = [f for f in ranked if f not in chosen][: n_min - len(selected)]
        selected = sorted(selected + top_up, key=position.__getitem__)
    return selected


__all__ = [
    "filter_low_variance",
    "select_decorrelated_by_rank",
]
