"""
Feature registry with JSON persistence.

Tracks feature lifecycle state, scores, and transition history.
Each feature is stored as a FeatureRecord with composite/MDA/stability/regime
scores and a full state transition log.

Persistence is optional: pass a registry_path for JSON file storage,
or None for in-memory only operation.

Reference: Phase E7 of Feature Governance roadmap.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_STATES = frozenset({"candidate", "selected", "active", "degraded", "retired"})


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


@dataclass
class FeatureRecord:
    """Immutable record of a single feature's lifecycle state and scores."""

    feature_name: str
    state: str = "candidate"
    composite_score: float = 0.0
    mda_score: float = 0.0
    stability_score: float = 0.0
    regime_score: float = 0.0
    created_at: str = ""
    last_selected_at: str | None = None
    transition_history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()
        if self.state not in VALID_STATES:
            raise ValueError(
                f"Invalid state '{self.state}' for feature '{self.feature_name}'. "
                f"Must be one of: {sorted(VALID_STATES)}"
            )


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
        allowed_scores = {"composite_score", "mda_score", "stability_score", "regime_score"}
        invalid = set(scores) - allowed_scores
        if invalid:
            raise ValueError(f"Unknown score fields: {invalid}. Allowed: {sorted(allowed_scores)}")

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
            ValueError: If new_state is not a valid state.
        """
        if new_state not in VALID_STATES:
            raise ValueError(f"Invalid state '{new_state}'. Must be one of: {sorted(VALID_STATES)}")
        if feature_name not in self._features:
            raise KeyError(f"Feature '{feature_name}' not in registry")

        record = self._features[feature_name]
        old_state = record.state
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
        self._path.write_text(json.dumps(data, indent=2) + "\n")
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
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(f"Failed to load registry from {self._path}: {exc}")

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


__all__ = ["FeatureRecord", "FeatureRegistry", "VALID_STATES"]
