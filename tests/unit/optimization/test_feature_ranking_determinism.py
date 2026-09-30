"""Deterministic feature rankings: float noise and input order never decide ranks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimization.feature_selection.ranking import (
    IMPORTANCE_NOISE_FLOOR,
    quantize_importance,
    rank_by_importance,
    top_features,
)
from src.optimization.feature_selection.walk_forward import WalkForwardFeatureSelector


class TestRankByImportance:
    def test_float_noise_does_not_order_tied_features(self) -> None:
        # Same importance up to summation noise, and noise around an exact 0
        imp = pd.Series({"b": 0.02 + 3e-17, "a": 0.02, "d": 1e-17, "c": -2e-17, "e": 0.05})
        ranked = rank_by_importance(imp, noise_floor=IMPORTANCE_NOISE_FLOOR)
        assert list(ranked.index) == ["e", "a", "b", "c", "d"]
        # Without the floor, values keep their own-magnitude order (b ~ a tie still)
        assert list(rank_by_importance(imp).index) == ["e", "a", "b", "d", "c"]

    def test_wide_range_scores_keep_their_order(self) -> None:
        """Variances spanning many orders of magnitude must not collapse to ties
        (review F1: rounding relative to the largest value made the variance
        ranking alphabetical for most features)."""
        rng = np.random.default_rng(3)
        values = 10.0 ** rng.uniform(-14, 6, size=171)
        names = [f"f{i:03d}" for i in rng.permutation(171)]
        variances = pd.Series(values, index=names)
        expected = list(variances.sort_values(ascending=False).index)
        assert list(rank_by_importance(variances).index) == expected

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
        values = pd.Series([1e-3, 5e-18, -5e-18, 2.5e-4])
        q = quantize_importance(values, noise_floor=IMPORTANCE_NOISE_FLOOR)
        assert q.tolist() == [1e-3, 0.0, 0.0, 2.5e-4]
        # No floor: every value keeps 12 significant digits of its own magnitude
        assert quantize_importance(values).tolist() == [1e-3, 5e-18, -5e-18, 2.5e-4]
        assert quantize_importance(pd.Series([1 / 3])).iloc[0] == float(f"{1 / 3:.12g}")
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
