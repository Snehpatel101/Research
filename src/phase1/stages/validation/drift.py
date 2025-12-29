"""
Feature drift checks for scaled datasets.
"""

import numpy as np
import pandas as pd


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between two samples.
    """
    expected = expected.replace([np.inf, -np.inf], np.nan).dropna()
    actual = actual.replace([np.inf, -np.inf], np.nan).dropna()

    if expected.empty or actual.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(expected, quantiles))
    if len(bin_edges) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.clip(expected_pct, 1e-6, 1.0)
    actual_pct = np.clip(actual_pct, 1e-6, 1.0)

    psi = np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))
    return float(psi)


def check_feature_drift(
    train_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    feature_cols: list[str],
    bins: int = 10,
    psi_threshold: float = 0.2,
    max_features: int = 200
) -> dict:
    """
    Compute drift metrics between train and a comparison split.
    """
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty for drift checks")

    metrics: dict[str, float] = {}
    drifted: list[tuple[str, float]] = []

    for col in feature_cols[:max_features]:
        psi = compute_psi(train_df[col], compare_df[col], bins=bins)
        metrics[col] = psi
        if psi >= psi_threshold:
            drifted.append((col, psi))

    drifted_sorted = sorted(drifted, key=lambda item: item[1], reverse=True)

    return {
        "psi_threshold": psi_threshold,
        "bins": bins,
        "feature_count": len(feature_cols[:max_features]),
        "drifted_feature_count": len(drifted_sorted),
        "drifted_features": [
            {"feature": name, "psi": value}
            for name, value in drifted_sorted[:20]
        ],
        "psi_by_feature": metrics,
    }
