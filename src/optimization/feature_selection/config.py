"""Configuration of the walk-forward feature selector (``ml cv`` per-fold selection)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureSelectorConfig:
    """
    Configuration for walk-forward feature selection.

    Used by WalkForwardFeatureSelector.

    Attributes:
        n_features_to_select: Number of top features to select per fold
        selection_method: Method for computing importance (mda, mdi, hybrid)
        n_estimators: Number of trees in importance estimator
        min_feature_frequency: Minimum fraction of folds feature must appear in
        use_clustered_importance: Whether to use clustered MDA for correlated features
        max_clusters: Maximum number of feature clusters (if using clustered)
    """

    n_features_to_select: int = 50
    selection_method: str = "mda"  # mda, mdi, or hybrid
    n_estimators: int = 50
    mda_n_repeats: int = 5
    min_feature_frequency: float = 0.6
    use_clustered_importance: bool = False
    max_clusters: int = 20

    def __post_init__(self) -> None:
        if self.n_features_to_select <= 0:
            raise ValueError(f"n_features_to_select must be > 0, got {self.n_features_to_select}")
        if self.selection_method not in ("mda", "mdi", "hybrid"):
            raise ValueError(
                f"selection_method must be mda/mdi/hybrid, got {self.selection_method}"
            )
        if self.mda_n_repeats <= 0:
            raise ValueError(f"mda_n_repeats must be > 0, got {self.mda_n_repeats}")
        if not 0 < self.min_feature_frequency <= 1:
            raise ValueError(
                f"min_feature_frequency must be in (0, 1], got {self.min_feature_frequency}"
            )


__all__ = [
    "FeatureSelectorConfig",
]
