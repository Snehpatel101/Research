"""Deterministic feature rankings: float noise and input order never decide ranks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimization.feature_selection.ranking import (
    quantize_importance,
    rank_by_importance,
    top_features,
)
from src.optimization.feature_selection.walk_forward import WalkForwardFeatureSelector


class TestRankByImportance:
    def test_float_noise_does_not_order_tied_features(self) -> None:
        # Same importance up to summation noise, and noise around an exact 0
        imp = pd.Series({"b": 0.02 + 3e-17, "a": 0.02, "d": 1e-17, "c": -2e-17, "e": 0.05})
        assert list(rank_by_importance(imp).index) == ["e", "a", "b", "c", "d"]

    def test_permutation_invariant(self) -> None:
        rng = np.random.default_rng(0)
        values = np.round(rng.normal(size=50), 3)  # many exact ties
        imp = pd.Series(values, index=[f"f{i:02d}" for i in range(50)])
        expected = list(rank_by_importance(imp).index)
        for seed in range(5):
            shuffled = imp.sample(frac=1.0, random_state=seed)
            assert list(rank_by_importance(shuffled).index) == expected

    def test_returns_original_values_nan_last(self) -> None:
        imp = pd.Series({"x": np.nan, "y": 0.123456789012345, "z": 0.5})
        ranked = rank_by_importance(imp)
        assert list(ranked.index) == ["z", "y", "x"]
        assert ranked["y"] == 0.123456789012345

    def test_quantize_zeroes_noise_and_keeps_signal(self) -> None:
        q = quantize_importance(pd.Series([1e-3, 5e-18, -5e-18, 2.5e-4]))
        assert q.tolist() == [1e-3, 0.0, 0.0, 2.5e-4]
        assert quantize_importance(pd.Series([0.0, 0.0])).tolist() == [0.0, 0.0]

    def test_top_features(self) -> None:
        imp = pd.Series({"b": 1.0, "a": 1.0, "c": 0.5})
        assert top_features(imp, 2) == ["a", "b"]


def test_stable_features_order_independent_of_set_iteration() -> None:
    """Equal selection counts are ordered by name, not by set (hash) order."""
    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X["f0"] + 0.5 * rng.normal(size=n) > 0).astype(int))
    splits = [(np.arange(0, 300), np.arange(300, 450)), (np.arange(0, 450), np.arange(450, 600))]
    selector = WalkForwardFeatureSelector(
        n_features_to_select=6, n_estimators=10, min_feature_frequency=0.5, random_state=3
    )
    result = selector.select_features_walkforward(X, y, splits)
    counts = result.feature_counts
    assert result.selected_features == sorted(
        result.selected_features, key=lambda f: (-counts[f], f)
    )
    assert list(counts) == sorted(counts)
