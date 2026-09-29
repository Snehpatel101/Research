"""Feature lifecycle: states, allowed transitions and the per-run promotion policy.

    CANDIDATE -> SELECTED -> ACTIVE <-> DEGRADED -> RETIRED

A feature is a CANDIDATE while it has only ever been scored, SELECTED the first
time a run's selection keeps it, ACTIVE once a later run keeps it again (and its
stability did not contradict that), DEGRADED when a run drops it or finds it
unstable, and RETIRED after it stayed degraded for several consecutive runs.
RETIRED is terminal: the registry reports a retired feature that a later run
selects again so a human can decide, rather than silently reviving it.

The transition table and ``next_state`` are pure; ``FeatureRegistry`` owns the
per-feature state and history.
"""

from __future__ import annotations

from enum import StrEnum


class FeatureLifecycleState(StrEnum):
    """States a feature can occupy in its lifecycle."""

    CANDIDATE = "candidate"
    SELECTED = "selected"
    ACTIVE = "active"
    DEGRADED = "degraded"
    RETIRED = "retired"


VALID_TRANSITIONS: dict[FeatureLifecycleState, frozenset[FeatureLifecycleState]] = {
    FeatureLifecycleState.CANDIDATE: frozenset({FeatureLifecycleState.SELECTED}),
    FeatureLifecycleState.SELECTED: frozenset(
        {FeatureLifecycleState.ACTIVE, FeatureLifecycleState.RETIRED}
    ),
    FeatureLifecycleState.ACTIVE: frozenset(
        {FeatureLifecycleState.DEGRADED, FeatureLifecycleState.RETIRED}
    ),
    FeatureLifecycleState.DEGRADED: frozenset(
        {FeatureLifecycleState.ACTIVE, FeatureLifecycleState.RETIRED}
    ),
    FeatureLifecycleState.RETIRED: frozenset(),
}

# Consecutive failing runs in DEGRADED before a feature is retired
DEFAULT_MAX_DEGRADED_RUNS = 3


def next_state(
    state: FeatureLifecycleState,
    *,
    selected: bool,
    stable: bool | None,
    degraded_runs: int,
    max_degraded_runs: int = DEFAULT_MAX_DEGRADED_RUNS,
) -> tuple[FeatureLifecycleState | None, str]:
    """State a feature moves to after one more run, or None to stay put.

    Args:
        state: Current state.
        selected: The run's selection kept the feature.
        stable: Stability verdict of the run (None = not measured).
        degraded_runs: Consecutive failing runs already spent in DEGRADED,
            counting this one (the caller increments before asking).
        max_degraded_runs: Failing DEGRADED runs that trigger retirement.

    Returns:
        ``(new_state, reason)``; ``new_state`` is None when nothing changes.
    """
    healthy = selected and stable is not False
    if state is FeatureLifecycleState.CANDIDATE:
        return (FeatureLifecycleState.SELECTED, "selected") if selected else (None, "")
    if state is FeatureLifecycleState.SELECTED:
        return (FeatureLifecycleState.ACTIVE, "selected again") if healthy else (None, "")
    if state is FeatureLifecycleState.ACTIVE:
        if healthy:
            return None, ""
        return FeatureLifecycleState.DEGRADED, "not selected" if not selected else "unstable"
    if state is FeatureLifecycleState.DEGRADED:
        if healthy:
            return FeatureLifecycleState.ACTIVE, "recovered"
        if degraded_runs >= max_degraded_runs:
            return FeatureLifecycleState.RETIRED, f"degraded for {degraded_runs} consecutive runs"
        return None, ""
    return None, ""


__all__ = [
    "DEFAULT_MAX_DEGRADED_RUNS",
    "VALID_TRANSITIONS",
    "FeatureLifecycleState",
    "next_state",
]
