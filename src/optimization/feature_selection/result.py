"""
Feature Selection Results.

Provides data structures for storing and persisting feature selection results.
This module consolidates result classes from:
- src/cross_validation/feature_selector.py (FeatureSelectionResult)
- src/models/feature_selection/result.py (PersistedFeatureSelection)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """
    Canonical result container for all feature selection operations.

    This class unifies three previous implementations to provide a single
    consistent interface for feature selection results across the codebase:
    - Walk-forward feature selection (cross_validation)
    - Correlation/variance filtering (phase1/utils)
    - OHLCV-specific selection with stability/regime analysis (feature_selection)

    Attributes:
        # === Core Fields (always populated) ===
        selected_features: Final list of selected feature names (alias: stable_features)

        # === Walk-Forward Selection Fields ===
        feature_counts: How many folds each feature was selected in (walk-forward)
        per_fold_selections: List of feature sets selected in each fold
        importance_history: Per-fold importance scores from walk-forward selection
        n_folds: Total number of CV folds used

        # === Correlation/Variance Filtering Fields ===
        removed_features: Dict mapping removed feature name to removal reason
        original_count: Number of features before selection (alias: n_original)
        final_count: Number of features after selection (alias: n_selected)
        correlation_groups: Groups of correlated features (lists, not sets for JSON)
        low_variance_features: Features removed due to low variance

        # === OHLCV/Stability Analysis Fields ===
        feature_importances: Dict mapping features to aggregated importance scores
        stability_scores: Dict mapping features to stability across folds (0-1)
        correlation_clusters: Alias for correlation_groups (OHLCV naming)
        regime_importances: Per-regime importance scores (regime-conditional selection)
        selection_metadata: Additional metadata about selection process
    """

    # === Core Field (required) ===
    selected_features: list[str]

    # === Walk-Forward Selection Fields (optional) ===
    feature_counts: dict[str, int] = field(default_factory=dict)
    per_fold_selections: list[set[str]] = field(default_factory=list)
    importance_history: list[dict[str, Any]] = field(default_factory=list)
    n_folds: int = 0

    # === Correlation/Variance Filtering Fields (optional) ===
    removed_features: dict[str, str] = field(default_factory=dict)
    original_count: int = 0
    final_count: int = 0
    correlation_groups: list[list[str]] = field(default_factory=list)
    low_variance_features: list[str] = field(default_factory=list)

    # === OHLCV/Stability Analysis Fields (optional) ===
    feature_importances: dict[str, float] = field(default_factory=dict)
    stability_scores: dict[str, float] = field(default_factory=dict)
    regime_importances: dict[int, dict[str, float]] | None = None
    selection_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-compute derived fields if not set."""
        # Auto-set n_folds from per_fold_selections if not provided
        if self.n_folds == 0 and self.per_fold_selections:
            self.n_folds = len(self.per_fold_selections)

        # Auto-set final_count from selected_features if not provided
        if self.final_count == 0 and self.selected_features:
            self.final_count = len(self.selected_features)

    # === Compatibility Properties ===

    @property
    def stable_features(self) -> list[str]:
        """Alias for selected_features (walk-forward naming convention)."""
        return self.selected_features

    @property
    def n_original(self) -> int:
        """Alias for original_count (OHLCV naming convention)."""
        return self.original_count

    @property
    def n_selected(self) -> int:
        """Alias for final_count (OHLCV naming convention)."""
        return self.final_count

    @property
    def correlation_clusters(self) -> list[list[str]]:
        """Alias for correlation_groups (OHLCV naming convention)."""
        return self.correlation_groups

    # === Walk-Forward Methods ===

    def get_stability_scores(self) -> dict[str, float]:
        """
        Return stability score (fraction of folds selected) for each feature.

        Used by walk-forward feature selection to measure consistency.
        """
        if self.n_folds == 0:
            return self.stability_scores if self.stability_scores else {}
        return {f: count / self.n_folds for f, count in self.feature_counts.items()}

    # === Correlation/Variance Filtering Methods ===

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Used by phase1 validation pipeline for reporting.
        """
        result: dict[str, Any] = {
            "selected_features": self.selected_features,
            "removed_features": self.removed_features,
            "original_count": self.original_count,
            "final_count": self.final_count,
            "reduction_pct": (
                round((1 - self.final_count / self.original_count) * 100, 1)
                if self.original_count > 0
                else 0
            ),
            "correlation_groups": self.correlation_groups,
            "low_variance_features": self.low_variance_features,
        }

        # Include additional fields if populated
        if self.feature_importances:
            result["feature_importances"] = self.feature_importances
        if self.stability_scores:
            result["stability_scores"] = self.stability_scores
        if self.regime_importances:
            result["regime_importances"] = self.regime_importances
        if self.selection_metadata:
            result["selection_metadata"] = self.selection_metadata
        if self.n_folds > 0:
            result["n_folds"] = self.n_folds

        return result

    # === OHLCV/Stability Analysis Methods ===

    def get_category_breakdown(self) -> dict[str, int]:
        """
        Get count of selected features per category.

        Requires OHLCV category utilities from feature_selection package.
        """
        try:
            from src.feature_selection.ohlcv_selector import get_feature_categories

            return {
                cat: len(feats)
                for cat, feats in get_feature_categories(self.selected_features).items()
            }
        except ImportError:
            return {}

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N features by importance."""
        if not self.feature_importances:
            return []
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:n]


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


__all__ = [
    "FeatureSelectionResult",
    "PersistedFeatureSelection",
]
