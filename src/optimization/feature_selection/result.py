"""Result of walk-forward feature selection (``WalkForwardFeatureSelector``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureSelectionResult:
    """
    Features selected across the folds of a walk-forward selection.

    Attributes:
        selected_features: Features selected in at least ``min_feature_frequency``
            of the folds, most frequent first (alias: ``stable_features``)
        feature_counts: How many folds selected each feature
        per_fold_selections: Feature set selected in each fold
        importance_history: Per-fold importance scores
        n_folds: Number of CV folds (defaults to ``len(per_fold_selections)``)
    """

    selected_features: list[str]
    feature_counts: dict[str, int] = field(default_factory=dict)
    per_fold_selections: list[set[str]] = field(default_factory=list)
    importance_history: list[dict[str, Any]] = field(default_factory=list)
    n_folds: int = 0

    def __post_init__(self) -> None:
        """Default ``n_folds`` to the number of per-fold selections."""
        if self.n_folds == 0 and self.per_fold_selections:
            self.n_folds = len(self.per_fold_selections)

    @property
    def stable_features(self) -> list[str]:
        """Alias for ``selected_features`` (walk-forward naming)."""
        return self.selected_features


__all__ = [
    "FeatureSelectionResult",
]
