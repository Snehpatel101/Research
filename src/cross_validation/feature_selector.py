"""
Walk-Forward Feature Selection for Time Series.

Prevents lookahead bias by selecting features using only historical data
at each point in time. Features that appear consistently across folds
are considered stable and used for final model training.

Methods:
- MDI (Mean Decrease in Impurity): Built-in RF importance, fast but biased
- MDA (Mean Decrease in Accuracy): Permutation importance, more reliable
- Hybrid: Combination of MDI and MDA rankings

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning", Chapter 8
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class FeatureSelectorConfig:
    """
    Configuration for walk-forward feature selection.

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
    n_estimators: int = 100
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
        if not 0 < self.min_feature_frequency <= 1:
            raise ValueError(
                f"min_feature_frequency must be in (0, 1], got {self.min_feature_frequency}"
            )


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

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Used by phase1 validation pipeline for reporting.
        """
        result = {
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


# =============================================================================
# WALK-FORWARD FEATURE SELECTOR
# =============================================================================


class WalkForwardFeatureSelector:
    """
    Feature selection with walk-forward methodology.

    Prevents lookahead bias by selecting features using only
    historical data at each point in time. Features that appear
    consistently across multiple folds are considered stable.

    Example:
        >>> selector = WalkForwardFeatureSelector(n_features_to_select=50)
        >>> cv_splits = list(cv.split(X, y))
        >>> result = selector.select_features_walkforward(X, y, cv_splits)
        >>> print(f"Stable features: {len(result.stable_features)}")
    """

    def __init__(
        self,
        n_features_to_select: int = 50,
        selection_method: str = "mda",
        n_estimators: int = 100,
        min_feature_frequency: float = 0.6,
        use_clustered_importance: bool = False,
        max_clusters: int = 20,
        random_state: int = 42,
    ) -> None:
        """
        Initialize WalkForwardFeatureSelector.

        Args:
            n_features_to_select: Number of top features per fold
            selection_method: Importance method (mda, mdi, hybrid)
            n_estimators: Number of trees for RF importance
            min_feature_frequency: Minimum fold frequency for stable features
            use_clustered_importance: Use clustered MDA for correlated features
            max_clusters: Max feature clusters (if clustered)
            random_state: Random seed for reproducibility
        """
        self.config = FeatureSelectorConfig(
            n_features_to_select=n_features_to_select,
            selection_method=selection_method,
            n_estimators=n_estimators,
            min_feature_frequency=min_feature_frequency,
            use_clustered_importance=use_clustered_importance,
            max_clusters=max_clusters,
        )
        self.random_state = random_state

    def select_features_walkforward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        sample_weights: pd.Series | None = None,
    ) -> FeatureSelectionResult:
        """
        Perform walk-forward feature selection across CV folds.

        For each fold:
        1. Compute feature importance on training data only
        2. Select top N features
        3. Track which features appear across folds

        Final stable features = features selected in >= min_frequency folds.

        Args:
            X: Feature DataFrame
            y: Labels
            cv_splits: List of (train_idx, test_idx) tuples from CV
            sample_weights: Optional sample weights

        Returns:
            FeatureSelectionResult with stable features and selection stats
        """
        feature_selections: list[set[str]] = []
        importance_history: list[dict[str, Any]] = []

        n_folds = len(cv_splits)
        logger.info(f"Running walk-forward feature selection across {n_folds} folds")

        for fold_idx, (train_idx, _) in enumerate(cv_splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            w_train = sample_weights.iloc[train_idx] if sample_weights is not None else None

            # Compute feature importance
            importance = self._compute_importance(X_train, y_train, w_train)

            # Select top features
            top_features = importance.nlargest(self.config.n_features_to_select).index.tolist()
            feature_selections.append(set(top_features))

            # Store importance history
            importance_history.append(
                {
                    "fold": fold_idx,
                    "n_features_evaluated": len(importance),
                    "top_feature": top_features[0] if top_features else None,
                    "top_importance": float(importance.max()) if len(importance) > 0 else 0.0,
                    "importance": importance.to_dict(),
                }
            )

            logger.debug(f"Fold {fold_idx}: selected {len(top_features)} features")

        # Find stable features (appear in >= min_frequency of folds)
        all_features = set().union(*feature_selections)
        feature_counts = {f: sum(f in s for s in feature_selections) for f in all_features}

        min_count = int(n_folds * self.config.min_feature_frequency)
        stable_features = [f for f, count in feature_counts.items() if count >= min_count]

        # Sort stable features by selection count (most stable first)
        stable_features.sort(key=lambda f: feature_counts[f], reverse=True)

        logger.info(
            f"Feature selection complete: {len(stable_features)} stable features "
            f"(selected in >= {min_count}/{n_folds} folds)"
        )

        return FeatureSelectionResult(
            selected_features=stable_features,
            feature_counts=feature_counts,
            per_fold_selections=feature_selections,
            importance_history=importance_history,
            n_folds=n_folds,
        )

    def _compute_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> pd.Series:
        """Compute feature importance using configured method."""
        if self.config.use_clustered_importance:
            return self._clustered_mda_importance(X, y, sample_weights)

        if self.config.selection_method == "mdi":
            return self._mdi_importance(X, y, sample_weights)
        elif self.config.selection_method == "mda":
            return self._mda_importance(X, y, sample_weights)
        else:  # hybrid
            mdi = self._mdi_importance(X, y, sample_weights)
            mda = self._mda_importance(X, y, sample_weights)
            # Combine by averaging ranks (robust to different scales)
            return (mdi.rank() + mda.rank()) / 2

    def _mdi_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Mean Decrease in Impurity (built-in RF importance).

        Fast but can be biased towards high-cardinality features.
        """
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=5,
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X, y, sample_weight=sample_weights)
        return pd.Series(rf.feature_importances_, index=X.columns)

    def _mda_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Mean Decrease in Accuracy (permutation importance).

        More reliable than MDI for correlated features.
        Reference: Lopez de Prado (2018)
        """
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=5,
            oob_score=True,
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X, y, sample_weight=sample_weights)

        # Use permutation importance
        result = permutation_importance(
            rf,
            X,
            y,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=-1,
            sample_weight=sample_weights,
        )

        return pd.Series(result.importances_mean, index=X.columns)

    def _clustered_mda_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        MDA importance with feature clustering.

        Groups correlated features and computes importance per cluster,
        then distributes importance within cluster. Handles multicollinearity.

        Reference: Lopez de Prado (2018), Chapter 8
        """
        # Compute correlation matrix
        corr = X.corr()

        # Convert NaN to 0 correlation (for constant features)
        corr = corr.fillna(0)

        # Hierarchical clustering on distance = 1 - |correlation|
        dist = 1 - corr.abs()
        np.fill_diagonal(dist.values, 0)  # Ensure diagonal is 0

        # Condense distance matrix and cluster
        dist_condensed = squareform(dist.values)
        linkage_matrix = linkage(dist_condensed, method="ward")
        clusters = fcluster(linkage_matrix, t=self.config.max_clusters, criterion="maxclust")

        # Map features to clusters
        feature_clusters = pd.Series(clusters, index=X.columns)

        # Compute importance per cluster
        cluster_importance: dict[int, float] = {}
        for cluster_id in np.unique(clusters):
            cluster_features = feature_clusters[feature_clusters == cluster_id].index.tolist()

            # Use mean of cluster features as representative
            X_cluster = X[cluster_features].mean(axis=1).to_frame("cluster_mean")

            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state,
            )
            rf.fit(X_cluster, y, sample_weight=sample_weights)
            cluster_importance[cluster_id] = float(rf.feature_importances_[0])

        # Distribute importance within cluster equally
        feature_importance: dict[str, float] = {}
        for feature in X.columns:
            cluster_id = feature_clusters[feature]
            n_features_in_cluster = int((feature_clusters == cluster_id).sum())
            feature_importance[feature] = cluster_importance[cluster_id] / n_features_in_cluster

        return pd.Series(feature_importance)


# =============================================================================
# CV-INTEGRATED FEATURE SELECTOR
# =============================================================================


class CVIntegratedFeatureSelector:
    """
    Integrate feature selection with CV to prevent lookahead.

    Performs feature selection and OOF prediction in a single pass,
    ensuring features are selected using only training data.

    Strategy:
    1. For each CV fold, select features using ONLY training data
    2. Train model on selected features
    3. Generate OOF predictions
    4. Track which features are stable across folds
    """

    def __init__(
        self,
        n_features: int = 50,
        min_frequency: float = 0.6,
        method: str = "mda",
        random_state: int = 42,
    ) -> None:
        """
        Initialize CVIntegratedFeatureSelector.

        Args:
            n_features: Number of features to select per fold
            min_frequency: Minimum fold frequency for stable features
            method: Feature importance method (mda, mdi)
            random_state: Random seed
        """
        self.selector = WalkForwardFeatureSelector(
            n_features_to_select=n_features,
            selection_method=method,
            min_feature_frequency=min_frequency,
            random_state=random_state,
        )
        self.n_features = n_features
        self.min_frequency = min_frequency

    def select_single_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> list[str]:
        """
        Select top N features for a single fold.

        Args:
            X_train: Training features
            y_train: Training labels
            sample_weights: Optional sample weights

        Returns:
            List of selected feature names
        """
        importance = self.selector._compute_importance(X_train, y_train, sample_weights)
        return importance.nlargest(self.n_features).index.tolist()


__all__ = [
    "FeatureSelectorConfig",
    "FeatureSelectionResult",
    "WalkForwardFeatureSelector",
    "CVIntegratedFeatureSelector",
]
