"""
Tests for Phase 100: Feature Lifecycle, Feature Registry, Drawdown-Adjusted Sizing.

Covers:
- E6: Feature lifecycle states, transition table and per-run promotion policy
- E7: Feature registry with JSON persistence (CRUD, save/load, record_run)
- Drawdown-adjusted position sizing (scaling, integration)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.inference.backtesting.position_sizing import (
    DrawdownAdjustedSizer,
    FixedFractional,
    PositionSizingMethod,
)
from src.optimization.feature_selection.lifecycle import (
    VALID_TRANSITIONS,
    FeatureLifecycleState,
    next_state,
)
from src.optimization.feature_selection.registry import (
    VALID_STATES,
    FeatureRecord,
    FeatureRegistry,
)

S = FeatureLifecycleState

# ---------------------------------------------------------------------------
# E6: Feature Lifecycle policy
# ---------------------------------------------------------------------------


class TestLifecyclePolicy:
    """E6: transition table and next_state."""

    def test_retired_is_terminal(self) -> None:
        assert VALID_TRANSITIONS[S.RETIRED] == frozenset()

    def test_every_state_has_a_table_entry(self) -> None:
        assert set(VALID_TRANSITIONS) == set(S)

    @pytest.mark.parametrize(
        ("state", "selected", "stable", "expected"),
        [
            (S.CANDIDATE, True, None, S.SELECTED),
            (S.CANDIDATE, False, None, None),
            (S.SELECTED, True, None, S.ACTIVE),
            (S.SELECTED, True, True, S.ACTIVE),
            (S.SELECTED, True, False, None),  # unstable: not promoted
            (S.SELECTED, False, None, None),
            (S.ACTIVE, True, True, None),
            (S.ACTIVE, True, None, None),
            (S.ACTIVE, False, None, S.DEGRADED),
            (S.ACTIVE, True, False, S.DEGRADED),
            (S.DEGRADED, True, True, S.ACTIVE),  # recovery
            (S.DEGRADED, False, None, None),  # first failing run: stay
            (S.RETIRED, True, True, None),
        ],
    )
    def test_next_state(self, state, selected, stable, expected) -> None:
        new_state, _reason = next_state(
            state, selected=selected, stable=stable, degraded_runs=1, max_degraded_runs=3
        )
        assert new_state == expected

    def test_degraded_retires_after_max_failing_runs(self) -> None:
        assert next_state(S.DEGRADED, selected=False, stable=None, degraded_runs=2)[0] is None
        new_state, reason = next_state(S.DEGRADED, selected=False, stable=None, degraded_runs=3)
        assert new_state == S.RETIRED and "3" in reason

    def test_policy_only_proposes_allowed_transitions(self) -> None:
        for state in S:
            for selected in (True, False):
                for stable in (True, False, None):
                    for runs in (1, 5):
                        new_state, _ = next_state(
                            state, selected=selected, stable=stable, degraded_runs=runs
                        )
                        if new_state is not None:
                            assert new_state in VALID_TRANSITIONS[state]


# ---------------------------------------------------------------------------
# E7: Feature Registry
# ---------------------------------------------------------------------------


class TestFeatureRegistry:
    """E7: Feature registry CRUD and persistence."""

    def test_register_new_feature(self) -> None:
        reg = FeatureRegistry()
        record = reg.register("rsi_14", mda_score=0.75)
        assert record.feature_name == "rsi_14"
        assert record.mda_score == 0.75
        assert record.state == "candidate"

    def test_register_updates_existing(self) -> None:
        reg = FeatureRegistry()
        reg.register("rsi_14", mda_score=0.5)
        reg.register("rsi_14", mda_score=0.8, stability_score=0.9)
        record = reg.get("rsi_14")
        assert record is not None
        assert record.mda_score == 0.8
        assert record.stability_score == 0.9

    def test_register_invalid_score_field(self) -> None:
        reg = FeatureRegistry()
        with pytest.raises(ValueError, match="Unknown score"):
            reg.register("rsi_14", bogus_score=0.5)

    def test_update_state(self) -> None:
        reg = FeatureRegistry()
        reg.register("rsi_14")
        reg.update_state("rsi_14", "selected", "passed threshold")
        record = reg.get("rsi_14")
        assert record is not None
        assert record.state == "selected"
        assert len(record.transition_history) == 1
        assert record.transition_history[0]["to_state"] == "selected"
        assert record.transition_history[0]["reason"] == "passed threshold"

    def test_update_state_invalid_state(self) -> None:
        reg = FeatureRegistry()
        reg.register("rsi_14")
        with pytest.raises(ValueError, match="Invalid state"):
            reg.update_state("rsi_14", "bogus_state")

    def test_update_state_missing_feature(self) -> None:
        reg = FeatureRegistry()
        with pytest.raises(KeyError, match="not in registry"):
            reg.update_state("nonexistent", "active")

    def test_get_by_state(self) -> None:
        reg = FeatureRegistry()
        reg.register("rsi_14")
        reg.register("atr_14")
        reg.update_state("rsi_14", "selected")
        selected = reg.get_by_state("selected")
        assert len(selected) == 1
        assert selected[0].feature_name == "rsi_14"

    def test_get_active_features(self) -> None:
        reg = FeatureRegistry()
        reg.register("rsi_14")
        reg.register("atr_14")
        reg.update_state("rsi_14", "selected")
        reg.update_state("rsi_14", "active")
        active = reg.get_active_features()
        assert active == ["rsi_14"]

    def test_get_degraded_features(self) -> None:
        reg = FeatureRegistry()
        reg.register("rsi_14")
        reg.update_state("rsi_14", "selected")
        reg.update_state("rsi_14", "active")
        reg.update_state("rsi_14", "degraded")
        degraded = reg.get_degraded_features()
        assert degraded == ["rsi_14"]

    def test_all_features(self) -> None:
        reg = FeatureRegistry()
        reg.register("a")
        reg.register("b")
        reg.register("c")
        assert len(reg.all_features()) == 3

    def test_len(self) -> None:
        reg = FeatureRegistry()
        reg.register("a")
        reg.register("b")
        assert len(reg) == 2

    def test_json_persistence(self, tmp_path: Path) -> None:
        """Save/load round-trip with JSON file."""
        path = tmp_path / "registry.json"
        reg1 = FeatureRegistry(registry_path=path)
        reg1.register("rsi_14", mda_score=0.8, composite_score=0.7)
        reg1.update_state("rsi_14", "selected", "test")
        reg1.register("atr_14", stability_score=0.9)
        reg1.save()

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == "1.0"
        assert "rsi_14" in data["features"]

        # Load into fresh registry
        reg2 = FeatureRegistry(registry_path=path)
        assert len(reg2) == 2
        record = reg2.get("rsi_14")
        assert record is not None
        assert record.mda_score == 0.8
        assert record.state == "selected"
        assert len(record.transition_history) == 1

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Loading from non-existent file should work (empty registry)."""
        path = tmp_path / "nonexistent.json"
        reg = FeatureRegistry(registry_path=path)
        assert len(reg) == 0

    def test_save_no_path(self) -> None:
        """Save with no path is a no-op."""
        reg = FeatureRegistry()
        reg.register("a")
        reg.save()  # Should not raise

    def test_to_dict_from_dict_roundtrip(self) -> None:
        reg1 = FeatureRegistry()
        reg1.register("rsi_14", mda_score=0.6)
        reg1.update_state("rsi_14", "selected")
        data = reg1.to_dict()
        reg2 = FeatureRegistry.from_dict(data)
        assert len(reg2) == 1
        record = reg2.get("rsi_14")
        assert record is not None
        assert record.mda_score == 0.6
        assert record.state == "selected"

    def test_valid_states_complete(self) -> None:
        expected = {"candidate", "selected", "active", "degraded", "retired"}
        assert expected == VALID_STATES

    def test_feature_record_invalid_state(self) -> None:
        with pytest.raises(ValueError, match="Invalid state"):
            FeatureRecord(feature_name="bad", state="invalid")

    def test_selected_updates_last_selected_at(self) -> None:
        reg = FeatureRegistry()
        reg.register("rsi_14")
        reg.update_state("rsi_14", "selected")
        record = reg.get("rsi_14")
        assert record is not None
        assert record.last_selected_at is not None

    def test_update_state_rejects_disallowed_transition(self) -> None:
        reg = FeatureRegistry()
        reg.register("rsi_14")
        with pytest.raises(ValueError, match="Invalid transition"):
            reg.update_state("rsi_14", "active")  # candidate -> active skips selected
        assert reg.get("rsi_14").state == "candidate"  # type: ignore[union-attr]

    def test_corrupt_file_is_moved_aside_not_overwritten(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.json"
        path.write_text("{not json")
        reg = FeatureRegistry(registry_path=path)
        assert len(reg) == 0
        (kept,) = list(tmp_path.glob("registry.json.corrupt.*"))
        assert kept.read_text() == "{not json"
        reg.register("a")
        reg.save()
        assert len(FeatureRegistry(registry_path=path)) == 1
        assert not list(tmp_path.glob("*.tmp")), "no temp file left behind"

    @pytest.mark.parametrize(
        "content",
        [
            "[1, 2, 3]",  # wrong top-level shape
            '{"version": "1.0", "features": [1]}',  # features not a mapping
            '{"version": "1.0", "features": {"a": 5}}',  # record not a mapping
            '{"version": "1.0", "features": {"a": {"state": "bogus"}}}',
            '{"version": "9.9", "features": {}}',  # unknown version
            '{"version": "1.0", "recorded_runs": "r1", "features": {}}',
        ],
    )
    def test_wrong_shape_or_version_is_moved_aside(self, tmp_path: Path, content: str) -> None:
        path = tmp_path / "registry.json"
        path.write_text(content)
        assert len(FeatureRegistry(registry_path=path)) == 0
        assert not path.exists()
        (kept,) = list(tmp_path.glob("registry.json.corrupt.*"))
        assert kept.read_text() == content

    def test_repeated_corruption_keeps_every_copy(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.json"
        for i in range(2):
            path.write_text(f"{{bad {i}")
            FeatureRegistry(registry_path=path)
        assert len(list(tmp_path.glob("registry.json.corrupt.*"))) == 2

    def test_unknown_record_fields_are_dropped_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        reg = FeatureRegistry(tmp_path / "registry.json")
        reg.register("a", mda_score=0.5)
        data = reg.to_dict()
        data["features"]["a"]["field_from_the_future"] = 1
        (tmp_path / "registry.json").write_text(json.dumps(data))
        with caplog.at_level("WARNING"):
            loaded = FeatureRegistry(tmp_path / "registry.json")
        assert loaded.get("a").mda_score == 0.5  # type: ignore[union-attr]
        assert "field_from_the_future" in caplog.text
        assert not list(tmp_path.glob("*.corrupt.*"))


class TestRegistryRecordRun:
    """record_run folds one run's selection + stability verdicts into lifecycles."""

    @staticmethod
    def _scores(*names: str) -> dict[str, dict[str, float]]:
        return {n: {"mda_score": 0.1} for n in names}

    def _run(self, reg, run_id, pool, selected, stable=None, **kw):
        return reg.record_run(
            run_id, scores=self._scores(*pool), selected=selected, stable=stable, **kw
        )

    def test_first_run_registers_and_selects(self) -> None:
        reg = FeatureRegistry()
        update = self._run(reg, "r1", ["a", "b", "c"], ["a", "b"])
        assert update.n_new == 3
        assert [r.feature_name for r in reg.get_by_state("selected")] == ["a", "b"]
        assert [r.feature_name for r in reg.get_by_state("candidate")] == ["c"]
        assert reg.get("a").transition_history[0]["reason"] == "r1: selected"  # type: ignore[union-attr]

    def test_reselected_stable_feature_becomes_active(self) -> None:
        reg = FeatureRegistry()
        self._run(reg, "r1", ["a"], ["a"])
        self._run(reg, "r2", ["a"], ["a"], stable={"a": True})
        assert reg.get_active_features() == ["a"]

    def test_unstable_reselection_is_not_promoted(self) -> None:
        reg = FeatureRegistry()
        self._run(reg, "r1", ["a"], ["a"])
        self._run(reg, "r2", ["a"], ["a"], stable={"a": False})
        assert reg.get("a").state == "selected"  # type: ignore[union-attr]

    def test_active_degrades_recovers_and_retires(self) -> None:
        reg = FeatureRegistry()
        self._run(reg, "r1", ["a"], ["a"])
        self._run(reg, "r2", ["a"], ["a"])
        assert reg.get("a").state == "active"  # type: ignore[union-attr]
        self._run(reg, "r3", ["a"], [])  # dropped
        assert reg.get_degraded_features() == ["a"]
        self._run(reg, "r4", ["a"], ["a"], stable={"a": True})  # recovery
        assert reg.get("a").state == "active"  # type: ignore[union-attr]
        assert reg.get("a").degraded_runs == 0  # type: ignore[union-attr]

        for run in ["r5", "r6", "r7", "r8"]:
            update = self._run(reg, run, ["a"], [], max_degraded_runs=3)
        assert reg.get("a").state == "retired"  # type: ignore[union-attr]
        assert update.transitions == []  # r8: already retired, no further move
        history = [t["to_state"] for t in reg.get("a").transition_history]  # type: ignore[union-attr]
        assert history == ["selected", "active", "degraded", "active", "degraded", "retired"]

    def test_retired_feature_is_reported_not_revived(self) -> None:
        reg = FeatureRegistry()
        self._run(reg, "r1", ["a"], ["a"])
        reg.update_state("a", "retired", "manual")
        update = self._run(reg, "r2", ["a"], ["a"], stable={"a": True})
        assert update.retired_but_selected == ["a"]
        assert reg.get("a").state == "retired"  # type: ignore[union-attr]

    def test_feature_missing_from_the_pool_counts_as_not_selected(self) -> None:
        reg = FeatureRegistry()
        self._run(reg, "r1", ["a"], ["a"])
        self._run(reg, "r2", ["a"], ["a"])
        self._run(reg, "r3", ["b"], ["b"])  # "a" no longer computed
        assert reg.get("a").state == "degraded"  # type: ignore[union-attr]

    def test_persists_across_registry_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "reg.json"
        reg = FeatureRegistry(path)
        self._run(reg, "r1", ["a"], ["a"])
        reg.save()
        reopened = FeatureRegistry(path)
        self._run(reopened, "r2", ["a"], ["a"])
        assert reopened.get("a").state == "active"  # type: ignore[union-attr]
        assert reopened.get("a").last_run_id == "r2"  # type: ignore[union-attr]

    def test_record_run_is_idempotent_even_across_a_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "reg.json"
        with FeatureRegistry.transaction(path) as reg:
            self._run(reg, "r1", ["a"], ["a"])
        with FeatureRegistry.transaction(path) as reg:
            first = self._run(reg, "r2", ["a"], ["a"])
        assert first.skipped is None and reg.get("a").state == "active"  # type: ignore[union-attr]

        # A resumed / re-run "r2" must not advance anything again (active -> degraded here)
        with FeatureRegistry.transaction(path) as reg:
            again = self._run(reg, "r2", ["a"], [])
        assert again.skipped and "already recorded" in again.skipped
        assert again.transitions == [] and again.n_new == 0
        reopened = FeatureRegistry(path)
        assert reopened.get("a").state == "active"  # type: ignore[union-attr]
        assert len(reopened.get("a").transition_history) == 2  # type: ignore[union-attr]

    def test_context_change_is_skipped_not_judged_as_decay(self) -> None:
        reg = FeatureRegistry()
        mes = {"symbol": "MES", "bar_timeframe": "5min", "models": ["xgboost"]}
        self._run(reg, "r1", ["a"], ["a"], context=mes)
        self._run(reg, "r2", ["a"], ["a"], context=dict(reversed(list(mes.items()))))
        assert reg.get("a").state == "active"  # type: ignore[union-attr]

        other = {**mes, "bar_timeframe": "15min"}
        update = self._run(reg, "r3", ["a", "z"], [], context=other)
        assert update.skipped and "context changed" in update.skipped
        assert reg.get("a").state == "active"  # not degraded by the changed setup
        assert reg.get("z") is None, "nothing from a different context is recorded"
        # The same experiment can still continue afterwards
        assert self._run(reg, "r4", ["a"], [], context=mes).skipped is None
        assert reg.get("a").state == "degraded"  # type: ignore[union-attr]

    def test_context_persists_and_pins_the_registry(self, tmp_path: Path) -> None:
        path = tmp_path / "reg.json"
        with FeatureRegistry.transaction(path) as reg:
            self._run(reg, "r1", ["a"], ["a"], context={"symbol": "MES"})
        assert json.loads(path.read_text())["context"] == {"symbol": "MES"}
        with FeatureRegistry.transaction(path) as reg:
            skipped = self._run(reg, "r2", ["a"], ["a"], context={"symbol": "MGC"})
        assert skipped.skipped

    def test_failed_transaction_saves_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "reg.json"
        with pytest.raises(RuntimeError), FeatureRegistry.transaction(path) as reg:
            self._run(reg, "r1", ["a"], ["a"])
            raise RuntimeError("boom")
        assert not path.exists()


def _registry_worker(args: tuple[str, int, int]) -> None:
    """Process target: record ``n`` distinct runs, each inside one locked transaction."""
    path, worker, n = args
    for k in range(n):
        with FeatureRegistry.transaction(path) as reg:
            reg.record_run(
                f"w{worker}-r{k}",
                scores={f"f{worker}": {"mda_score": 0.1}, "shared": {"mda_score": 0.2}},
                selected=["shared"],
            )


class TestRegistryConcurrency:
    def test_concurrent_processes_lose_no_updates(self, tmp_path: Path) -> None:
        import multiprocessing

        path = str(tmp_path / "reg.json")
        n_workers, n_runs = 4, 6
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            pool.map(_registry_worker, [(path, w, n_runs) for w in range(n_workers)])

        data = json.loads(Path(path).read_text())  # never a torn / partial file
        assert len(data["recorded_runs"]) == n_workers * n_runs
        assert len(set(data["recorded_runs"])) == n_workers * n_runs
        assert set(data["features"]) == {"shared", *(f"f{w}" for w in range(n_workers))}
        assert not list(Path(path).parent.glob("*.tmp"))
        assert not list(Path(path).parent.glob("*.corrupt.*"))


# ---------------------------------------------------------------------------
# Drawdown-Adjusted Position Sizing
# ---------------------------------------------------------------------------


class TestDrawdownAdjustedSizer:
    """Drawdown-adjusted position sizing wrapper."""

    def _make_sizer(
        self, max_dd: float = 0.10, power: float = 2.0, min_scale: float = 0.0
    ) -> DrawdownAdjustedSizer:
        inner = FixedFractional(risk_per_trade=0.02, point_value=5.0)
        return DrawdownAdjustedSizer(
            inner_sizer=inner,
            max_drawdown_threshold=max_dd,
            scaling_power=power,
            min_scale=min_scale,
        )

    def test_no_drawdown_full_scale(self) -> None:
        sizer = self._make_sizer()
        assert sizer.drawdown_scale(0.0) == 1.0

    def test_at_threshold_zero_scale(self) -> None:
        sizer = self._make_sizer(max_dd=0.10)
        assert sizer.drawdown_scale(0.10) == 0.0

    def test_beyond_threshold_min_scale(self) -> None:
        sizer = self._make_sizer(max_dd=0.10, min_scale=0.1)
        assert sizer.drawdown_scale(0.15) == 0.1

    def test_half_drawdown_quadratic(self) -> None:
        """At 50% of threshold with power=2, scale = 1 - 0.5^2 = 0.75."""
        sizer = self._make_sizer(max_dd=0.10, power=2.0)
        scale = sizer.drawdown_scale(0.05)
        assert abs(scale - 0.75) < 1e-10

    def test_half_drawdown_linear(self) -> None:
        """At 50% of threshold with power=1, scale = 1 - 0.5 = 0.5."""
        sizer = self._make_sizer(max_dd=0.10, power=1.0)
        scale = sizer.drawdown_scale(0.05)
        assert abs(scale - 0.5) < 1e-10

    def test_min_scale_floor(self) -> None:
        sizer = self._make_sizer(max_dd=0.10, min_scale=0.2)
        # At threshold, should return min_scale
        assert sizer.drawdown_scale(0.10) == 0.2
        # Beyond threshold, should return min_scale
        assert sizer.drawdown_scale(0.20) == 0.2

    def test_position_size_reduces_with_drawdown(self) -> None:
        sizer = self._make_sizer(max_dd=0.10)
        # With no drawdown
        size_full = sizer.calculate_position_size(
            account_equity=100000.0, current_drawdown=0.0, stop_distance=20.0
        )
        # With 5% drawdown (quadratic: scale=0.75)
        size_reduced = sizer.calculate_position_size(
            account_equity=100000.0, current_drawdown=0.05, stop_distance=20.0
        )
        assert size_full > 0
        assert size_reduced <= size_full

    def test_position_size_zero_at_threshold(self) -> None:
        sizer = self._make_sizer(max_dd=0.10)
        size = sizer.calculate_position_size(
            account_equity=100000.0, current_drawdown=0.10, stop_distance=20.0
        )
        assert size == 0

    def test_enum_has_drawdown_adjusted(self) -> None:
        assert PositionSizingMethod.DRAWDOWN_ADJUSTED == "drawdown_adjusted"

    def test_negative_drawdown_treated_as_zero(self) -> None:
        """Negative drawdown (above high-water mark) should give full scale."""
        sizer = self._make_sizer()
        assert sizer.drawdown_scale(-0.05) == 1.0

    def test_drawdown_scale_monotonic(self) -> None:
        """Scale should decrease monotonically as drawdown increases."""
        sizer = self._make_sizer(max_dd=0.20, power=2.0)
        drawdowns = np.linspace(0.0, 0.20, 50)
        scales = [sizer.drawdown_scale(dd) for dd in drawdowns]
        for i in range(1, len(scales)):
            assert scales[i] <= scales[i - 1] + 1e-10
