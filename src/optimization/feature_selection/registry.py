"""
Feature registry with JSON persistence.

Tracks feature lifecycle state, scores, and transition history across runs.
Each feature is stored as a FeatureRecord with composite/MDA/stability/regime
scores and a full state transition log. ``record_run`` applies one run's
selection + stability verdicts through the lifecycle policy
(``lifecycle.next_state``); ``update_state`` only allows transitions from
``lifecycle.VALID_TRANSITIONS``.

Persistence is optional: pass a registry_path for JSON file storage,
or None for in-memory only operation. The file is last-writer-wins: two runs
sharing one registry concurrently keep only the later save.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .lifecycle import (
    DEFAULT_MAX_DEGRADED_RUNS,
    VALID_TRANSITIONS,
    FeatureLifecycleState,
    next_state,
)

logger = logging.getLogger(__name__)

VALID_STATES = frozenset(state.value for state in FeatureLifecycleState)
_SCORE_FIELDS = frozenset({"composite_score", "mda_score", "stability_score", "regime_score"})


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


@dataclass
class FeatureRecord:
    """A single feature's lifecycle state, latest scores and transition log."""

    feature_name: str
    state: str = "candidate"
    composite_score: float = 0.0
    mda_score: float = 0.0
    stability_score: float = 0.0
    regime_score: float = 0.0
    created_at: str = ""
    last_selected_at: str | None = None
    transition_history: list[dict] = field(default_factory=list)
    degraded_runs: int = 0  # consecutive failing runs while DEGRADED
    last_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()
        if self.state not in VALID_STATES:
            raise ValueError(
                f"Invalid state '{self.state}' for feature '{self.feature_name}'. "
                f"Must be one of: {sorted(VALID_STATES)}"
            )


@dataclass
class RunUpdate:
    """What one ``record_run`` changed."""

    transitions: list[dict] = field(default_factory=list)
    n_new: int = 0
    retired_but_selected: list[str] = field(default_factory=list)


class FeatureRegistry:
    """Registry tracking feature lifecycle with optional JSON persistence.

    Args:
        registry_path: Path to JSON file for persistence. None for in-memory only.
    """

    def __init__(self, registry_path: Path | str | None = None) -> None:
        self._path: Path | None = Path(registry_path) if registry_path is not None else None
        self._features: dict[str, FeatureRecord] = {}
        self.load()

    def register(self, feature_name: str, **scores: float) -> FeatureRecord:
        """Add or update a feature in the registry.

        If the feature already exists, scores are updated in place.
        If new, a FeatureRecord is created with state 'candidate'.

        Args:
            feature_name: Name of the feature.
            **scores: Any of composite_score, mda_score, stability_score,
                regime_score.

        Returns:
            The created or updated FeatureRecord.
        """
        invalid = set(scores) - _SCORE_FIELDS
        if invalid:
            raise ValueError(f"Unknown score fields: {invalid}. Allowed: {sorted(_SCORE_FIELDS)}")

        if feature_name in self._features:
            record = self._features[feature_name]
            for key, value in scores.items():
                setattr(record, key, value)
            logger.debug(f"Updated feature '{feature_name}' scores: {scores}")
        else:
            record = FeatureRecord(feature_name=feature_name, **scores)
            self._features[feature_name] = record
            logger.debug(f"Registered new feature '{feature_name}'")

        return record

    def update_state(self, feature_name: str, new_state: str, reason: str = "") -> None:
        """Transition a feature to a new lifecycle state.

        Appends an entry to the feature's transition_history.

        Args:
            feature_name: Name of the feature to update.
            new_state: Target state (must be in VALID_STATES).
            reason: Optional reason for the transition.

        Raises:
            KeyError: If feature is not registered.
            ValueError: If new_state is not a valid state, or the move is not
                allowed from the feature's current state.
        """
        if new_state not in VALID_STATES:
            raise ValueError(f"Invalid state '{new_state}'. Must be one of: {sorted(VALID_STATES)}")
        if feature_name not in self._features:
            raise KeyError(f"Feature '{feature_name}' not in registry")

        record = self._features[feature_name]
        old_state = record.state
        if (
            FeatureLifecycleState(new_state)
            not in VALID_TRANSITIONS[FeatureLifecycleState(old_state)]
        ):
            raise ValueError(f"Invalid transition for '{feature_name}': {old_state} -> {new_state}")
        record.state = new_state
        record.transition_history.append(
            {
                "timestamp": _now_iso(),
                "from_state": old_state,
                "to_state": new_state,
                "reason": reason,
            }
        )

        if new_state == "selected":
            record.last_selected_at = _now_iso()

        logger.info(
            f"Feature '{feature_name}': {old_state} -> {new_state}"
            + (f" ({reason})" if reason else "")
        )

    def record_run(
        self,
        run_id: str,
        *,
        scores: Mapping[str, Mapping[str, float]],
        selected: Collection[str],
        stable: Mapping[str, bool] | None = None,
        max_degraded_runs: int = DEFAULT_MAX_DEGRADED_RUNS,
    ) -> RunUpdate:
        """Fold one run into the registry: refresh scores, then advance lifecycles.

        Every registered feature is judged, including ones this run no longer
        produced (they count as not selected). Retired features are never
        revived; a retired feature that is selected again is reported.

        Args:
            run_id: Identifier recorded in transition reasons and ``last_run_id``.
            scores: Candidate pool: feature -> score fields (see ``register``).
            selected: Features the run's selection kept.
            stable: Stability verdict per feature (missing = not measured).
            max_degraded_runs: Consecutive failing DEGRADED runs before retirement.
        """
        update = RunUpdate()
        selected_set = set(selected)
        stable = stable or {}
        for name, feature_scores in scores.items():
            if name not in self._features:
                update.n_new += 1
            self.register(name, **feature_scores)

        for name, record in self._features.items():
            is_selected = name in selected_set
            verdict = stable.get(name)
            record.last_run_id = run_id
            if is_selected:
                record.last_selected_at = _now_iso()
            state = FeatureLifecycleState(record.state)
            if state is FeatureLifecycleState.RETIRED:
                if is_selected:
                    update.retired_but_selected.append(name)
                continue
            if state is FeatureLifecycleState.DEGRADED and not (
                is_selected and verdict is not False
            ):
                record.degraded_runs += 1
            new_state, reason = next_state(
                state,
                selected=is_selected,
                stable=verdict,
                degraded_runs=record.degraded_runs,
                max_degraded_runs=max_degraded_runs,
            )
            if new_state is None:
                continue
            self.update_state(name, new_state.value, reason=f"{run_id}: {reason}")
            record.degraded_runs = 1 if new_state is FeatureLifecycleState.DEGRADED else 0
            update.transitions.append(
                {"feature": name, "from": state.value, "to": new_state.value, "reason": reason}
            )
        return update

    def get(self, feature_name: str) -> FeatureRecord | None:
        """Look up a feature by name. Returns None if not found."""
        return self._features.get(feature_name)

    def get_by_state(self, state: str) -> list[FeatureRecord]:
        """Return all features currently in the given state."""
        return [r for r in self._features.values() if r.state == state]

    def get_active_features(self) -> list[str]:
        """Return names of all features in ACTIVE state."""
        return [r.feature_name for r in self._features.values() if r.state == "active"]

    def get_degraded_features(self) -> list[str]:
        """Return names of all features in DEGRADED state."""
        return [r.feature_name for r in self._features.values() if r.state == "degraded"]

    def all_features(self) -> list[FeatureRecord]:
        """Return all registered features."""
        return list(self._features.values())

    # -- Persistence ----------------------------------------------------------

    def save(self) -> None:
        """Write registry to JSON file. No-op if no path configured."""
        if self._path is None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, self._path)
        logger.info(f"Registry saved: {len(self._features)} features -> {self._path}")

    def load(self) -> None:
        """Read registry from JSON file. No-op if no path or file missing."""
        if self._path is None or not self._path.exists():
            return

        try:
            data = json.loads(self._path.read_text())
            loaded = FeatureRegistry.from_dict(data, registry_path=self._path)
            self._features = loaded._features
            logger.info(f"Registry loaded: {len(self._features)} features from {self._path}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            # Keep the unreadable file for inspection instead of overwriting it on save
            broken = self._path.with_suffix(self._path.suffix + ".corrupt")
            os.replace(self._path, broken)
            logger.warning(f"Unreadable registry {self._path} ({exc}); moved to {broken}")

    def to_dict(self) -> dict:
        """Serialize registry to a dict suitable for JSON."""
        return {
            "version": "1.0",
            "features": {name: asdict(record) for name, record in self._features.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, registry_path: Path | str | None = None) -> FeatureRegistry:
        """Deserialize a registry from a dict.

        Args:
            data: Dict with 'version' and 'features' keys.
            registry_path: Optional path for future saves.

        Returns:
            A new FeatureRegistry populated with the data.
        """
        registry = cls.__new__(cls)
        registry._path = Path(registry_path) if registry_path is not None else None
        registry._features = {}

        features_data = data.get("features", {})
        for name, record_dict in features_data.items():
            record_dict.pop("feature_name", None)
            registry._features[name] = FeatureRecord(feature_name=name, **record_dict)

        return registry

    def __len__(self) -> int:
        return len(self._features)

    def __repr__(self) -> str:
        return f"FeatureRegistry({len(self._features)} features, path={self._path})"


__all__ = ["FeatureRecord", "FeatureRegistry", "RunUpdate", "VALID_STATES"]
