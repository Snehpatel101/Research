"""
Tests for the feature-governance statistics: block-subsample stability and
label-perturbation robustness. Both take their rankings/selections from the
caller, so they are tested with synthetic functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.feature_selection.bootstrap_stability import (
    BootstrapFeatureStability,
    BootstrapStabilityResult,
    StabilitySummary,
)
from src.optimization.feature_selection.label_perturbation import (
    LabelPerturbationTester,
    PerturbationSummary,
)

FEATURES = [f"f{i}" for i in range(10)]


def _selector(keep: dict[str, list[str]], seen: list[tuple[int, int]] | None = None):
    """block_fn keeping a fixed feature set per model on every block."""

    def block_fn(start: int, stop: int) -> dict[str, list[str]]:
        if seen is not None:
            seen.append((start, stop))
        return keep

    return block_fn


def _by_name(summary: StabilitySummary) -> dict[str, BootstrapStabilityResult]:
    return {r.feature_name: r for r in summary.results}


class TestBootstrapStability:
    def test_consistently_kept_features_are_stable(self) -> None:
        tester = BootstrapFeatureStability(n_bootstrap=6, min_window_rows=10)
        summary = tester.evaluate(FEATURES, 1000, _selector({"m": ["f0", "f1", "f2"]}))
        assert isinstance(summary, StabilitySummary)
        assert summary.n_blocks_used == 6
        stable = {r.feature_name for r in summary.results if r.is_stable}
        assert stable == {"f0", "f1", "f2"}
        assert all(r.selection_frequency == 1.0 for r in summary.results if r.is_stable)
        assert summary.results[0].selection_frequency >= summary.results[-1].selection_frequency

    def test_frequency_counts_how_often_the_real_selection_keeps_a_feature(self) -> None:
        # "f5" is a lower-ranked cluster representative: kept in 2 of every 3 blocks;
        # "f0" is kept every time. Stability is about the selection, not the rank.
        calls = {"n": 0}

        def block_fn(start: int, stop: int) -> dict[str, list[str]]:
            calls["n"] += 1
            return {"m": ["f0", "f5"] if calls["n"] % 3 else ["f0"]}

        tester = BootstrapFeatureStability(
            n_bootstrap=6, stability_threshold=0.6, min_window_rows=10
        )
        found = _by_name(tester.evaluate(FEATURES, 1000, block_fn))
        assert found["f0"].selection_frequency == 1.0
        assert found["f5"].selection_frequency == pytest.approx(4 / 6)
        assert found["f5"].is_stable
        assert found["f9"].selection_frequency == 0.0 and not found["f9"].is_stable

    def test_per_model_and_union_verdicts(self) -> None:
        keep = {"a": ["f0", "f1"], "b": ["f0"]}
        tester = BootstrapFeatureStability(n_bootstrap=4, min_window_rows=10)
        found = _by_name(tester.evaluate(FEATURES, 1000, _selector(keep)))
        assert found["f1"].group_frequency == {"a": 1.0, "b": 0.0}
        assert found["f1"].group_stable == {"a": True, "b": False}
        assert found["f1"].is_stable  # kept by SOME model every block
        assert found["f0"].group_stable == {"a": True, "b": True}

    def test_blocks_are_contiguous_and_inside_the_data(self) -> None:
        seen: list[tuple[int, int]] = []
        tester = BootstrapFeatureStability(n_bootstrap=5, window_fraction=0.4, min_window_rows=10)
        tester.evaluate(FEATURES, 1000, _selector({"m": ["f0"]}, seen))
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
        summary = tester.evaluate(FEATURES, 400, _selector({"m": ["f0"]}))
        assert summary.results == [] and summary.n_blocks_used == 0

    def test_unrankable_blocks_are_skipped(self) -> None:
        calls = {"n": 0}

        def flaky(start: int, stop: int) -> dict[str, list[str]] | None:
            calls["n"] += 1
            return None if calls["n"] % 2 else {"m": ["f0"]}

        tester = BootstrapFeatureStability(n_bootstrap=6, min_window_rows=10)
        summary = tester.evaluate(FEATURES, 1000, flaky)
        assert summary.n_blocks_drawn == 6 and summary.n_blocks_used == 3
        assert _by_name(summary)["f0"].selection_frequency == 1.0

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
        assert summary.control_rank_correlation is None

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

    def test_shifts_are_measured_against_the_control(self) -> None:
        names = [f"g{i}" for i in range(100)]
        live = pd.Series(np.linspace(1, 0.1, 100), index=names)
        # The control relabel already moved g10 far from the live ranking...
        control = live.copy()
        control.iloc[10], control.iloc[60] = control.iloc[60], control.iloc[10]
        # ...and the variant merely reproduces the control's ranking.
        summary = LabelPerturbationTester(rank_change_fraction=0.05).evaluate(
            live, {"v": control.copy()}, control=control
        )
        g10 = next(r for r in summary.results if r.feature_name == "g10")
        assert g10.max_rank_change == 0  # variant vs control: no perturbation effect
        assert g10.control_shift == 50  # the relabel procedure's own shift
        assert g10.is_robust
        assert summary.control_rank_correlation is not None
        assert summary.rank_correlation["v"] == pytest.approx(1.0)

    def test_only_shifts_above_the_controls_own_shift_are_flagged(self) -> None:
        names = [f"g{i}" for i in range(100)]
        live = pd.Series(np.linspace(1, 0.1, 100), index=names)
        control = live.copy()
        control.iloc[10], control.iloc[30] = control.iloc[30], control.iloc[10]  # own shift 20
        small = control.copy()
        small.iloc[10], small.iloc[25] = small.iloc[25], small.iloc[10]  # g10: 5 ranks from control
        large = control.copy()
        large.iloc[10], large.iloc[90] = (
            large.iloc[90],
            large.iloc[10],
        )  # g10: 60 ranks from control
        tester = LabelPerturbationTester(rank_change_fraction=0.05)  # threshold 5
        ok = tester.evaluate(live, {"v": small}, control=control)
        bad = tester.evaluate(live, {"v": large}, control=control)
        # g10: control rank 30, small variant rank 25 -> 5 <= threshold -> robust
        assert next(r for r in ok.results if r.feature_name == "g10").is_robust
        # large variant moves g10 by ~60+ ranks, above both threshold and control shift
        assert not next(r for r in bad.results if r.feature_name == "g10").is_robust
