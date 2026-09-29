"""
Feature registry with JSON persistence.

Tracks feature lifecycle state, scores, and transition history across runs.
Each feature is stored as a FeatureRecord with composite/MDA/stability/regime
scores and a full state transition log. ``record_run`` applies one run's
selection + stability verdicts through the lifecycle policy
(``lifecycle.next_state``); ``update_state`` only allows transitions from
``lifecycle.VALID_TRANSITIONS``.

Persistence is optional: pass a registry_path for JSON file storage,
or None for in-memory only operation.

Cross-run safety:
- ``record_run`` is idempotent per run id (a resumed / re-run run is a no-op).
- It only advances lifecycles across runs with the same ``context`` fingerprint
  (symbol, bar timeframe, ranking label, MTF set, models): a changed setup is a
  different experiment, not feature decay.
- ``FeatureRegistry.transaction(path)`` serialises load-modify-save between
  concurrent runs with an advisory file lock, and saves atomically.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Collection, Iterator, Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # POSIX advisory locks; without fcntl (Windows) runs are unserialised
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from .lifecycle import (
    DEFAULT_MAX_DEGRADED_RUNS,
    VALID_TRANSITIONS,
    FeatureLifecycleState,
    next_state,
)

logger = logging.getLogger(__name__)

VALID_STATES = frozenset(state.value for state in FeatureLifecycleState)
_SCORE_FIELDS = frozenset({"composite_score", "mda_score", "stability_score", "regime_score"})
REGISTRY_VERSION = "1.0"


def context_fingerprint(context: Mapping[str, Any]) -> str:
    """Short stable hash of an experiment context (order-independent)."""
    payload = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Exclusive advisory lock on ``lock_path`` (no-op where fcntl is unavailable)."""
    if fcntl is None:  # pragma: no cover
        logger.debug("fcntl unavailable: registry updates are not serialised")
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


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
    skipped: str | None = None  # why nothing was recorded (duplicate run, context change)


class FeatureRegistry:
    """Registry tracking feature lifecycle with optional JSON persistence.

    Args:
        registry_path: Path to JSON file for persistence. None for in-memory only.
    """

    def __init__(self, registry_path: Path | str | None = None) -> None:
        self._path: Path | None = Path(registry_path) if registry_path is not None else None
        self._features: dict[str, FeatureRecord] = {}
        self._recorded_runs: list[str] = []
        self._context: dict[str, Any] | None = None
        self._context_fingerprint: str | None = None
        self.load()

    @classmethod
    @contextlib.contextmanager
    def transaction(cls, registry_path: Path | str) -> Iterator[FeatureRegistry]:
        """Load, let the caller modify, then save -- under an exclusive file lock.

        Two runs updating the same registry file cannot lose each other's
        changes. Nothing is saved if the body raises.
        """
        path = Path(registry_path)
        with _file_lock(path.with_name(path.name + ".lock")):
            registry = cls(path)
            yield registry
            registry.save()

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
        context: Mapping[str, Any] | None = None,
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
            context: Experiment setup (symbol, bar timeframe, ranking label, MTF
                set, models). The first recorded context pins the registry;
                a run with a different one is skipped, not judged as decay.

        Returns:
            What changed; ``skipped`` is set (and nothing changes) when ``run_id``
            was already recorded or the context differs from the registry's.
        """
        update = RunUpdate()
        if run_id in self._recorded_runs:
            update.skipped = f"run '{run_id}' already recorded"
            return update
        fingerprint = context_fingerprint(context) if context is not None else None
        if fingerprint is not None:
            if self._context_fingerprint not in (None, fingerprint):
                update.skipped = (
                    f"context changed (registry {self._context_fingerprint}, run {fingerprint}); "
                    "use a separate registry for a different setup"
                )
                logger.warning(f"Feature registry: {update.skipped}")
                return update
            self._context_fingerprint = fingerprint
            self._context = dict(context or {})
        self._recorded_runs.append(run_id)
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
        """Write registry to JSON file atomically. No-op if no path configured."""
        if self._path is None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2) + "\n"
        # Unique temp file in the same directory, then an atomic replace
        with tempfile.NamedTemporaryFile(
            "w", dir=self._path.parent, prefix=self._path.name + ".", suffix=".tmp", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            try:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
        try:
            os.replace(tmp_path, self._path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        logger.info(f"Registry saved: {len(self._features)} features -> {self._path}")

    def load(self) -> None:
        """Read registry from JSON file. No-op if no path or file missing.

        An unreadable file (bad JSON, wrong shape, unknown version) is kept as a
        timestamped ``.corrupt.<ts>`` copy and the registry starts empty.
        """
        if self._path is None or not self._path.exists():
            return

        try:
            data = json.loads(self._path.read_text())
            loaded = FeatureRegistry.from_dict(data, registry_path=self._path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            broken = self._path.with_name(f"{self._path.name}.corrupt.{stamp}")
            os.replace(self._path, broken)
            logger.warning(f"Unreadable registry {self._path} ({exc!r}); moved to {broken}")
            return
        self._features = loaded._features
        self._recorded_runs = loaded._recorded_runs
        self._context = loaded._context
        self._context_fingerprint = loaded._context_fingerprint
        logger.info(f"Registry loaded: {len(self._features)} features from {self._path}")

    def to_dict(self) -> dict:
        """Serialize registry to a dict suitable for JSON."""
        return {
            "version": REGISTRY_VERSION,
            "context_fingerprint": self._context_fingerprint,
            "context": self._context,
            "recorded_runs": list(self._recorded_runs),
            "features": {name: asdict(record) for name, record in self._features.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, registry_path: Path | str | None = None) -> FeatureRegistry:
        """Deserialize a registry from a dict.

        Args:
            data: Dict as produced by ``to_dict``.
            registry_path: Optional path for future saves.

        Raises:
            ValueError: Unsupported version.
            TypeError / AttributeError / KeyError: Wrong-shaped content.

        Returns:
            A new FeatureRegistry populated with the data. Record fields this
            version does not know are dropped with a warning.
        """
        version = data.get("version")
        if version != REGISTRY_VERSION:
            raise ValueError(f"unsupported registry version {version!r}, want {REGISTRY_VERSION}")
        registry = cls.__new__(cls)
        registry._path = Path(registry_path) if registry_path is not None else None
        registry._features = {}
        recorded = data.get("recorded_runs", [])
        if not isinstance(recorded, list):
            raise TypeError("recorded_runs must be a list")
        registry._recorded_runs = [str(r) for r in recorded]
        context = data.get("context")
        if context is not None and not isinstance(context, dict):
            raise TypeError("context must be a mapping")
        registry._context = context
        registry._context_fingerprint = data.get("context_fingerprint")

        features_data = data.get("features", {})
        if not isinstance(features_data, dict):
            raise TypeError("features must be a mapping")
        known = {f.name for f in fields(FeatureRecord)} - {"feature_name"}
        for name, record_dict in features_data.items():
            if not isinstance(record_dict, dict):
                raise TypeError(f"record for '{name}' must be a mapping")
            unknown = set(record_dict) - known - {"feature_name"}
            if unknown:
                logger.warning(
                    f"Registry record '{name}': dropping unknown fields {sorted(unknown)}"
                )
            kwargs = {k: v for k, v in record_dict.items() if k in known}
            registry._features[name] = FeatureRecord(feature_name=name, **kwargs)

        return registry

    def __len__(self) -> int:
        return len(self._features)

    def __repr__(self) -> str:
        return f"FeatureRegistry({len(self._features)} features, path={self._path})"


__all__ = [
    "REGISTRY_VERSION",
    "VALID_STATES",
    "FeatureRecord",
    "FeatureRegistry",
    "RunUpdate",
    "context_fingerprint",
]
