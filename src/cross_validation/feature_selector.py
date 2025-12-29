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
            raise ValueError(f"selection_method must be mda/mdi/hybrid, got {self.selection_method}")
        if not 0 < self.min_feature_frequency <= 1:
            raise ValueError(f"min_feature_frequency must be in (0, 1], got {self.min_feature_frequency}")


@dataclass
class FeatureSelectionResult:
    """
    Results from walk-forward feature selection.

    Attributes:
        stable_features: Features appearing in >= min_frequency folds
        feature_counts: How many folds each feature was selected in
        per_fold_selections: List of feature sets selected in each fold
        importance_history: Per-fold importance scores
        n_folds: Total number of folds
    """
    stable_features: list[str]
    feature_counts: dict[str, int]
    per_fold_selections: list[set[str]]
    importance_history: list[dict[str, Any]]
    n_folds: int = field(default=0)

    def __post_init__(self) -> None:
        if self.n_folds == 0:
            self.n_folds = len(self.per_fold_selections)

    def get_stability_scores(self) -> dict[str, float]:
        """Return stability score (fraction of folds selected) for each feature."""
        if self.n_folds == 0:
            return {}
        return {f: count / self.n_folds for f, count in self.feature_counts.items()}


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
            importance_history.append({
                "fold": fold_idx,
                "n_features_evaluated": len(importance),
                "top_feature": top_features[0] if top_features else None,
                "top_importance": float(importance.max()) if len(importance) > 0 else 0.0,
                "importance": importance.to_dict(),
            })

            logger.debug(f"Fold {fold_idx}: selected {len(top_features)} features")

        # Find stable features (appear in >= min_frequency of folds)
        all_features = set().union(*feature_selections)
        feature_counts = {f: sum(f in s for s in feature_selections) for f in all_features}

        min_count = int(n_folds * self.config.min_feature_frequency)
        stable_features = [
            f for f, count in feature_counts.items()
            if count >= min_count
        ]

        # Sort stable features by selection count (most stable first)
        stable_features.sort(key=lambda f: feature_counts[f], reverse=True)

        logger.info(
            f"Feature selection complete: {len(stable_features)} stable features "
            f"(selected in >= {min_count}/{n_folds} folds)"
        )

        return FeatureSelectionResult(
            stable_features=stable_features,
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
            rf, X, y,
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
        linkage_matrix = linkage(dist_condensed, method='ward')
        clusters = fcluster(
            linkage_matrix,
            t=self.config.max_clusters,
            criterion='maxclust'
        )

        # Map features to clusters
        feature_clusters = pd.Series(clusters, index=X.columns)

        # Compute importance per cluster
        cluster_importance: dict[int, float] = {}
        for cluster_id in np.unique(clusters):
            cluster_features = feature_clusters[feature_clusters == cluster_id].index.tolist()

            # Use mean of cluster features as representative
            X_cluster = X[cluster_features].mean(axis=1).to_frame('cluster_mean')

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
