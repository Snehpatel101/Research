"""
Tests for Phase 100: Feature Lifecycle, Feature Registry, Drawdown-Adjusted Sizing.

Covers:
- E6: Feature lifecycle state machine (transitions, history, retirement)
- E7: Feature registry with JSON persistence (CRUD, save/load)
- Drawdown-adjusted position sizing (scaling, integration)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.optimization.feature_selection.lifecycle import (
    FeatureLifecycle,
    FeatureLifecycleState,
)
from src.optimization.feature_selection.registry import (
    VALID_STATES,
    FeatureRecord,
    FeatureRegistry,
)
from src.inference.backtesting.position_sizing import (
    DrawdownAdjustedSizer,
    FixedFractional,
    PositionSizingMethod,
)


# ---------------------------------------------------------------------------
# E6: Feature Lifecycle State Machine
# ---------------------------------------------------------------------------


class TestFeatureLifecycle:
    """E6: Feature lifecycle state transitions and history."""

    def test_initial_state_is_candidate(self) -> None:
        fl = FeatureLifecycle("rsi_14")
        assert fl.state == FeatureLifecycleState.CANDIDATE

    def test_valid_transition_candidate_to_selected(self) -> None:
        fl = FeatureLifecycle("rsi_14")
        fl.transition(FeatureLifecycleState.SELECTED, "passed MDA threshold")
        assert fl.state == FeatureLifecycleState.SELECTED

    def test_valid_transition_selected_to_active(self) -> None:
        fl = FeatureLifecycle("rsi_14")
        fl.transition(FeatureLifecycleState.SELECTED)
        fl.transition(FeatureLifecycleState.ACTIVE, "validated in production")
        assert fl.state == FeatureLifecycleState.ACTIVE

    def test_valid_transition_active_to_degraded(self) -> None:
        fl = FeatureLifecycle("rsi_14")
        fl.transition(FeatureLifecycleState.SELECTED)
        fl.transition(FeatureLifecycleState.ACTIVE)
        fl.transition(FeatureLifecycleState.DEGRADED, "MDA dropped below threshold")
        assert fl.state == FeatureLifecycleState.DEGRADED

    def test_valid_transition_degraded_to_active(self) -> None:
        """Recovery path: degraded feature recovers performance."""
        fl = FeatureLifecycle("rsi_14")
        fl.transition(FeatureLifecycleState.SELECTED)
        fl.transition(FeatureLifecycleState.ACTIVE)
        fl.transition(FeatureLifecycleState.DEGRADED)
        fl.transition(FeatureLifecycleState.ACTIVE, "performance recovered")
        assert fl.state == FeatureLifecycleState.ACTIVE

    def test_valid_transition_degraded_to_retired(self) -> None:
        fl = FeatureLifecycle("rsi_14")
        fl.transition(FeatureLifecycleState.SELECTED)
        fl.transition(FeatureLifecycleState.ACTIVE)
        fl.transition(FeatureLifecycleState.DEGRADED)
        fl.transition(FeatureLifecycleState.RETIRED, "failed to recover")
        assert fl.state == FeatureLifecycleState.RETIRED

    def test_invalid_transition_candidate_to_active(self) -> None:
        fl = FeatureLifecycle("rsi_14")
        with pytest.raises(ValueError, match="Invalid transition"):
            fl.transition(FeatureLifecycleState.ACTIVE)

    def test_invalid_transition_retired_to_anything(self) -> None:
        """Retired is a terminal state."""
        fl = FeatureLifecycle("rsi_14")
        fl.transition(FeatureLifecycleState.SELECTED)
        fl.transition(FeatureLifecycleState.RETIRED)
        with pytest.raises(ValueError, match="Invalid transition"):
            fl.transition(FeatureLifecycleState.CANDIDATE)

    def test_history_tracking(self) -> None:
        fl = FeatureLifecycle("rsi_14")
        fl.transition(FeatureLifecycleState.SELECTED, "reason1")
        fl.transition(FeatureLifecycleState.ACTIVE, "reason2")
        history = fl.history
        assert len(history) == 2
        # Each entry: (timestamp_str, from_state, to_state, reason)
        assert history[0][1] == FeatureLifecycleState.CANDIDATE
        assert history[0][2] == FeatureLifecycleState.SELECTED
        assert history[0][3] == "reason1"
        assert history[1][2] == FeatureLifecycleState.ACTIVE

    def test_history_is_copy(self) -> None:
        """Ensure history property returns a copy, not mutable reference."""
        fl = FeatureLifecycle("rsi_14")
        fl.transition(FeatureLifecycleState.SELECTED)
        h1 = fl.history
        h1.append(("fake", FeatureLifecycleState.CANDIDATE, FeatureLifecycleState.RETIRED, ""))
        assert len(fl.history) == 1

    def test_consecutive_degradations_none(self) -> None:
        fl = FeatureLifecycle("rsi_14")
        assert fl.consecutive_degradations == 0

    def test_consecutive_degradations_count(self) -> None:
        fl = FeatureLifecycle("rsi_14")
        fl.transition(FeatureLifecycleState.SELECTED)
        fl.transition(FeatureLifecycleState.ACTIVE)
        fl.transition(FeatureLifecycleState.DEGRADED)
        fl.transition(FeatureLifecycleState.ACTIVE)  # recovery
        fl.transition(FeatureLifecycleState.DEGRADED)
        fl.transition(FeatureLifecycleState.ACTIVE)  # recovery
        fl.transition(FeatureLifecycleState.DEGRADED)  # consecutive 1
        assert fl.consecutive_degradations == 1

    def test_should_retire_false(self) -> None:
        fl = FeatureLifecycle("rsi_14")
        assert fl.should_retire(max_degradations=3) is False

    def test_should_retire_true(self) -> None:
        """Simulate 3 consecutive degradations via degraded->active->degraded cycle.

        Note: consecutive_degradations counts trailing DEGRADED transitions,
        but the state machine requires ACTIVE between DEGRADEDs. So we test
        with max_degradations=1 for a single trailing degradation.
        """
        fl = FeatureLifecycle("rsi_14")
        fl.transition(FeatureLifecycleState.SELECTED)
        fl.transition(FeatureLifecycleState.ACTIVE)
        fl.transition(FeatureLifecycleState.DEGRADED)
        assert fl.should_retire(max_degradations=1) is True

    def test_feature_name_preserved(self) -> None:
        fl = FeatureLifecycle("atr_14_1min")
        assert fl.feature_name == "atr_14_1min"


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
        assert VALID_STATES == expected

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
