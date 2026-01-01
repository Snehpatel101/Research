"""
Feature selection result and persistence.

Provides data structures for storing and persisting feature selection results
alongside model artifacts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PersistedFeatureSelection:
    """
    Persisted feature selection result for model artifacts.

    Stores the selected features and selection metadata to be saved
    alongside model checkpoints. Enables applying the same feature
    selection at inference time.

    Attributes:
        selected_features: Ordered list of selected feature names
        feature_indices: Mapping from feature name to original column index
        selection_method: Method used for selection (mda, mdi, hybrid)
        n_features_original: Original number of features before selection
        n_features_selected: Number of features after selection
        stability_scores: Feature stability scores (fraction of folds selected)
        importance_scores: Final feature importance scores
        metadata: Additional selection metadata
    """

    selected_features: list[str]
    feature_indices: dict[str, int]
    selection_method: str
    n_features_original: int
    n_features_selected: int
    stability_scores: dict[str, float] = field(default_factory=dict)
    importance_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result consistency."""
        if self.n_features_selected != len(self.selected_features):
            raise ValueError(
                f"n_features_selected ({self.n_features_selected}) does not match "
                f"len(selected_features) ({len(self.selected_features)})"
            )

    @property
    def is_empty(self) -> bool:
        """Check if no features were selected."""
        return len(self.selected_features) == 0

    @property
    def reduction_ratio(self) -> float:
        """Calculate feature reduction ratio (0 = no reduction, 1 = all removed)."""
        if self.n_features_original == 0:
            return 0.0
        return 1 - (self.n_features_selected / self.n_features_original)

    def get_feature_mask(self, all_features: list[str]) -> list[bool]:
        """
        Create boolean mask for selected features.

        Args:
            all_features: Complete list of feature names

        Returns:
            Boolean mask where True indicates selected features
        """
        selected_set = set(self.selected_features)
        return [f in selected_set for f in all_features]

    def get_column_indices(self, all_features: list[str]) -> list[int]:
        """
        Get column indices for selected features.

        Args:
            all_features: Complete list of feature names

        Returns:
            List of integer indices for selected features
        """
        feature_to_idx = {f: i for i, f in enumerate(all_features)}
        return [feature_to_idx[f] for f in self.selected_features if f in feature_to_idx]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "selected_features": self.selected_features,
            "feature_indices": self.feature_indices,
            "selection_method": self.selection_method,
            "n_features_original": self.n_features_original,
            "n_features_selected": self.n_features_selected,
            "stability_scores": self.stability_scores,
            "importance_scores": self.importance_scores,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistedFeatureSelection:
        """Create from dictionary."""
        return cls(
            selected_features=data["selected_features"],
            feature_indices=data["feature_indices"],
            selection_method=data["selection_method"],
            n_features_original=data["n_features_original"],
            n_features_selected=data["n_features_selected"],
            stability_scores=data.get("stability_scores", {}),
            importance_scores=data.get("importance_scores", {}),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        """
        Save feature selection result to JSON file.

        Args:
            path: Path to save the JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.debug(f"Saved feature selection to {path}")

    @classmethod
    def load(cls, path: Path) -> PersistedFeatureSelection:
        """
        Load feature selection result from JSON file.

        Args:
            path: Path to load from

        Returns:
            PersistedFeatureSelection instance

        Raises:
            FileNotFoundError: If path does not exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Feature selection file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        logger.debug(f"Loaded feature selection from {path}")
        return cls.from_dict(data)

    @classmethod
    def passthrough(cls, all_features: list[str]) -> PersistedFeatureSelection:
        """
        Create a passthrough result that keeps all features.

        Useful for models that don't use feature selection (e.g., sequence models).

        Args:
            all_features: List of all feature names

        Returns:
            PersistedFeatureSelection that passes through all features
        """
        return cls(
            selected_features=list(all_features),
            feature_indices={f: i for i, f in enumerate(all_features)},
            selection_method="passthrough",
            n_features_original=len(all_features),
            n_features_selected=len(all_features),
            stability_scores={f: 1.0 for f in all_features},
            importance_scores={},
            metadata={"passthrough": True},
        )

    def __repr__(self) -> str:
        return (
            f"PersistedFeatureSelection("
            f"n_selected={self.n_features_selected}, "
            f"n_original={self.n_features_original}, "
            f"method={self.selection_method}, "
            f"reduction={self.reduction_ratio:.1%})"
        )


__all__ = ["PersistedFeatureSelection"]
