"""
Feature lifecycle state machine for tracking feature status across training cycles.

Manages state transitions for individual features through their lifecycle:
CANDIDATE -> SELECTED -> ACTIVE -> DEGRADED -> RETIRED

Each transition is validated against allowed transitions and recorded
with timestamp and reason for full auditability.

Reference: Phase E6 of Feature Governance roadmap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

logger = logging.getLogger(__name__)


class FeatureLifecycleState(StrEnum):
    """States a feature can occupy in its lifecycle."""

    CANDIDATE = "candidate"
    SELECTED = "selected"
    ACTIVE = "active"
    DEGRADED = "degraded"
    RETIRED = "retired"


# Valid state transitions: source -> set of allowed destinations
_VALID_TRANSITIONS: dict[FeatureLifecycleState, set[FeatureLifecycleState]] = {
    FeatureLifecycleState.CANDIDATE: {FeatureLifecycleState.SELECTED},
    FeatureLifecycleState.SELECTED: {
        FeatureLifecycleState.ACTIVE,
        FeatureLifecycleState.RETIRED,
    },
    FeatureLifecycleState.ACTIVE: {
        FeatureLifecycleState.DEGRADED,
        FeatureLifecycleState.RETIRED,
    },
    FeatureLifecycleState.DEGRADED: {
        FeatureLifecycleState.ACTIVE,
        FeatureLifecycleState.RETIRED,
    },
    FeatureLifecycleState.RETIRED: set(),
}


@dataclass
class FeatureLifecycle:
    """Manage lifecycle state transitions for a single feature.

    Tracks the current state, validates transitions, and maintains
    a full history of state changes with timestamps and reasons.
    """

    feature_name: str
    _state: FeatureLifecycleState = field(default=FeatureLifecycleState.CANDIDATE)
    _history: list[tuple[str, FeatureLifecycleState, FeatureLifecycleState, str]] = field(
        default_factory=list, repr=False
    )

    @property
    def state(self) -> FeatureLifecycleState:
        """Current lifecycle state."""
        return self._state

    @property
    def history(
        self,
    ) -> list[tuple[str, FeatureLifecycleState, FeatureLifecycleState, str]]:
        """Full transition history as (timestamp, from_state, to_state, reason) tuples."""
        return list(self._history)

    def transition(self, new_state: FeatureLifecycleState, reason: str = "") -> None:
        """Transition to a new state.

        Args:
            new_state: Target state to transition to.
            reason: Human-readable reason for the transition.

        Raises:
            ValueError: If the transition is not allowed from the current state.
        """
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition for '{self.feature_name}': "
                f"{self._state.value} -> {new_state.value}. "
                f"Allowed targets: {sorted(s.value for s in allowed) or 'none (terminal state)'}"
            )

        old_state = self._state
        self._state = new_state
        timestamp = datetime.now(tz=UTC).isoformat()
        self._history.append((timestamp, old_state, new_state, reason))

        logger.debug(
            "Feature '%s': %s -> %s%s",
            self.feature_name,
            old_state.value,
            new_state.value,
            f" ({reason})" if reason else "",
        )

    @property
    def consecutive_degradations(self) -> int:
        """Count consecutive DEGRADED transitions from the end of history."""
        count = 0
        for _, _, to_state, _ in reversed(self._history):
            if to_state == FeatureLifecycleState.DEGRADED:
                count += 1
            else:
                break
        return count

    def should_retire(self, max_degradations: int = 3) -> bool:
        """Check if the feature should be retired due to repeated degradation.

        Args:
            max_degradations: Number of consecutive degradations that triggers
                retirement recommendation. Defaults to 3.

        Returns:
            True if consecutive_degradations >= max_degradations.
        """
        return self.consecutive_degradations >= max_degradations


__all__ = ["FeatureLifecycleState", "FeatureLifecycle"]
