"""
Tests for the feature-governance statistics: block-subsample stability and
label-perturbation robustness (both take importances from the caller, so they
are tested with synthetic ranking functions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.feature_selection.bootstrap_stability import (
    BootstrapFeatureStability,
    BootstrapStabilityResult,
)
from src.optimization.feature_selection.label_perturbation import (
    LabelPerturbationTester,
    PerturbationSummary,
)

FEATURES = [f"f{i}" for i in range(10)]


def _ranker(good: set[str], seen: list[tuple[int, int]] | None = None):
    """Importance ranking where ``good`` features always win; noise varies per block."""

    def rank(start: int, stop: int) -> pd.Series:
        if seen is not None:
            seen.append((start, stop))
        rng = np.random.default_rng(start)
        imp = pd.Series(rng.uniform(0, 0.1, len(FEATURES)), index=FEATURES)
        imp[list(good)] += 1.0
        return imp

    return rank


class TestBootstrapStability:
    def test_stable_features_are_the_consistently_ranked_ones(self) -> None:
        tester = BootstrapFeatureStability(n_bootstrap=6, top_k=3, min_window_rows=10)
        results = tester.evaluate(FEATURES, 1000, _ranker({"f0", "f1", "f2"}))
        assert all(isinstance(r, BootstrapStabilityResult) for r in results)
        stable = {r.feature_name for r in results if r.is_stable}
        assert stable == {"f0", "f1", "f2"}
        assert all(r.selection_frequency == 1.0 for r in results if r.is_stable)
        # Sorted: most frequently selected first
        assert results[0].selection_frequency >= results[-1].selection_frequency

    def test_blocks_are_contiguous_and_inside_the_data(self) -> None:
        seen: list[tuple[int, int]] = []
        tester = BootstrapFeatureStability(n_bootstrap=5, window_fraction=0.4, min_window_rows=10)
        tester.evaluate(FEATURES, 1000, _ranker({"f0"}, seen))
        assert len(seen) == 5
        for start, stop in seen:
            assert 0 <= start < stop <= 1000
            assert stop - start == 400

    def test_blocks_are_deterministic_per_seed(self) -> None:
        tester = BootstrapFeatureStability(n_bootstrap=4, min_window_rows=10, random_state=3)
        other = BootstrapFeatureStability(n_bootstrap=4, min_window_rows=10, random_state=4)
        assert tester.draw_windows(500) == tester.draw_windows(500)
        assert tester.draw_windows(500) != other.draw_windows(500)

    def test_too_small_data_yields_no_blocks(self) -> None:
        tester = BootstrapFeatureStability(n_bootstrap=4, min_window_rows=300)
        assert tester.draw_windows(400) == []
        assert tester.evaluate(FEATURES, 400, _ranker({"f0"})) == []

    def test_unrankable_blocks_are_skipped(self) -> None:
        calls = {"n": 0}

        def flaky(start: int, stop: int) -> pd.Series | None:
            calls["n"] += 1
            return None if calls["n"] % 2 else _ranker({"f0"})(start, stop)

        tester = BootstrapFeatureStability(n_bootstrap=6, top_k=2, min_window_rows=10)
        results = tester.evaluate(FEATURES, 1000, flaky)
        assert results  # half the blocks ranked
        assert next(r for r in results if r.feature_name == "f0").selection_frequency == 1.0

    def test_all_zero_ranking_does_not_make_everything_stable(self) -> None:
        zeros = pd.Series(0.0, index=FEATURES)
        tester = BootstrapFeatureStability(n_bootstrap=3, top_k=2, min_window_rows=10)
        results = tester.evaluate(FEATURES, 1000, lambda a, b: zeros)
        assert not any(r.is_stable for r in results)

    def test_invalid_parameters_rejected(self) -> None:
        with pytest.raises(ValueError):
            BootstrapFeatureStability(n_bootstrap=0)
        with pytest.raises(ValueError):
            BootstrapFeatureStability(window_fraction=1.5)
        with pytest.raises(ValueError):
            BootstrapFeatureStability(stability_threshold=0.0)


class TestLabelPerturbation:
    def test_unchanged_ranking_is_fully_robust(self) -> None:
        base = pd.Series(np.linspace(1, 0.1, 10), index=FEATURES)
        summary = LabelPerturbationTester().evaluate(base, {"a": base.copy(), "b": base.copy()})
        assert isinstance(summary, PerturbationSummary)
        assert all(r.is_robust and r.max_rank_change == 0 for r in summary.results)
        assert summary.rank_correlation == {"a": pytest.approx(1.0), "b": pytest.approx(1.0)}

    def test_reversed_ranking_is_flagged(self) -> None:
        base = pd.Series(np.linspace(1, 0.1, 10), index=FEATURES)
        reversed_imp = pd.Series(base.to_numpy()[::-1], index=FEATURES)
        summary = LabelPerturbationTester(rank_change_fraction=0.2).evaluate(
            base, {"flip": reversed_imp}
        )
        by_name = {r.feature_name: r for r in summary.results}
        assert not by_name["f0"].is_robust and by_name["f0"].max_rank_change == 9
        assert summary.rank_correlation["flip"] == pytest.approx(-1.0)
        # Middle features move little
        assert by_name["f4"].max_rank_change <= 1 and by_name["f4"].is_robust

    def test_unrankable_variants_are_skipped_and_none_means_no_verdict(self) -> None:
        base = pd.Series(np.linspace(1, 0.1, 10), index=FEATURES)
        summary = LabelPerturbationTester().evaluate(base, {"a": None})
        assert summary.variants_used == []
        assert not any(r.is_robust for r in summary.results)

    def test_tolerance_is_relative_to_feature_count(self) -> None:
        names = [f"g{i}" for i in range(100)]
        base = pd.Series(np.linspace(1, 0.1, 100), index=names)
        shifted = base.copy()
        shifted.iloc[10], shifted.iloc[20] = shifted.iloc[20], shifted.iloc[10]  # swap = 10 ranks
        summary = LabelPerturbationTester(rank_change_fraction=0.15).evaluate(base, {"s": shifted})
        assert all(r.is_robust for r in summary.results)  # 10 <= 15% of 100
        strict = LabelPerturbationTester(rank_change_fraction=0.05).evaluate(base, {"s": shifted})
        assert not next(r for r in strict.results if r.feature_name == "g10").is_robust
