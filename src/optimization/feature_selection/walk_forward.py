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

This module consolidates walk-forward feature selection from:
- src/cross_validation/feature_selector.py
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, make_scorer

from src.core.reproducibility import sequential_prediction

from .config import FeatureSelectorConfig
from .ranking import quantize_importance
from .ranking import top_features as rank_top_features
from .result import FeatureSelectionResult

logger = logging.getLogger(__name__)

# Log-loss floor: avoids -inf when a forest assigns exactly 0 to the true class.
_PROBA_EPS = 1e-15

# Fraction of the training fold held out when a caller supplies no holdout set.
_FALLBACK_HOLDOUT_FRAC = 0.25

# Scale of the within-cluster tie-break added to a cluster's importance.
_WITHIN_CLUSTER_TIEBREAK = 1e-9


def _neg_log_loss_scorer(classes: np.ndarray) -> Any:
    """Scorer returning negative (optionally weighted) log-loss.

    MDA must score probabilities, not argmax accuracy: on imbalanced classes
    accuracy is dominated by the majority class and is blind to features that
    sharpen probabilities without flipping the arg-max. ``labels`` is pinned to
    the fitted classes so a holdout missing a class does not raise.
    """
    return make_scorer(
        log_loss,
        greater_is_better=False,
        response_method="predict_proba",
        labels=np.asarray(classes),
    )


def _neg_log_loss(
    proba: np.ndarray,
    y_idx: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    """Negative weighted log-loss from class probabilities and class indices."""
    p_true = np.clip(proba[np.arange(len(y_idx)), y_idx], _PROBA_EPS, 1.0)
    ll = -np.log(p_true)
    if weights is None:
        return -float(ll.mean())
    return -float(np.average(ll, weights=weights))


def cluster_features(
    X: pd.DataFrame,
    max_clusters: int,
    distance_threshold: float,
) -> pd.Series:
    """Hierarchically cluster features on signed-correlation distance.

    Distance is ``sqrt(0.5 * (1 - rho))`` (Lopez de Prado 2020, "Machine Learning
    for Asset Managers", 4.2) using the *signed* correlation, so anti-correlated
    features are far apart (d=1 at rho=-1) and are not merged; the previous
    ``1 - |rho|`` distance averaged them into one cluster. Average linkage is used
    because Ward requires Euclidean geometry, which this distance matrix does not
    guarantee.

    Features are merged only while their linkage distance is below
    ``distance_threshold`` (``rho`` of roughly ``1 - 2 * threshold**2``). If that
    still leaves more than ``max_clusters`` clusters the tree is cut at exactly
    ``max_clusters`` to bound cost. Note that cap force-merges *uncorrelated*
    features (a noise column can then inherit a signal cluster's importance), so
    callers that care about ranking should pass ``max_clusters >= n_features``.

    Returns:
        Series of integer cluster ids indexed by feature name.
    """
    if X.shape[1] == 1:
        return pd.Series([1], index=X.columns)

    # Constant / NaN-correlated columns are treated as uncorrelated with everything.
    corr = X.corr().fillna(0.0).to_numpy(copy=True)
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    tree = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(tree, t=distance_threshold, criterion="distance")
    if len(np.unique(labels)) > max_clusters:
        labels = fcluster(tree, t=max_clusters, criterion="maxclust")
    return pd.Series(labels, index=X.columns)


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
        n_estimators: int = 50,
        mda_n_repeats: int = 5,
        min_feature_frequency: float = 0.6,
        use_clustered_importance: bool = False,
        max_clusters: int = 20,
        random_state: int = 42,
        cluster_distance_threshold: float = 0.5,
    ) -> None:
        """
        Initialize WalkForwardFeatureSelector.

        Args:
            n_features_to_select: Number of top features per fold
            selection_method: Importance method (mda, mdi, hybrid)
            n_estimators: Number of trees for RF importance
            mda_n_repeats: Number of permutation repeats for MDA
            min_feature_frequency: Minimum fold frequency for stable features
            use_clustered_importance: Use clustered MDA for correlated features
            max_clusters: Max feature clusters (if clustered)
            random_state: Random seed for reproducibility
            cluster_distance_threshold: Max ``sqrt(0.5*(1-rho))`` linkage distance at
                which features merge into one cluster (0.5 <=> rho of about 0.5)
        """
        self.config = FeatureSelectorConfig(
            n_features_to_select=n_features_to_select,
            selection_method=selection_method,
            n_estimators=n_estimators,
            mda_n_repeats=mda_n_repeats,
            min_feature_frequency=min_feature_frequency,
            use_clustered_importance=use_clustered_importance,
            max_clusters=max_clusters,
        )
        self.random_state = random_state
        self.cluster_distance_threshold = cluster_distance_threshold

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

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            w_train = sample_weights.iloc[train_idx] if sample_weights is not None else None

            # Use holdout set for MDA scoring to avoid overfitting bias
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            w_test = sample_weights.iloc[test_idx] if sample_weights is not None else None

            # Compute feature importance (MDA uses holdout for unbiased scoring)
            importance = self._compute_importance(
                X_train,
                y_train,
                w_train,
                X_test=X_test,
                y_test=y_test,
                w_test=w_test,
            )

            # Select top features
            top_features = rank_top_features(importance, self.config.n_features_to_select)
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

        # Find stable features (appear in >= min_frequency of folds). Sorted:
        # iterating a set of names would follow PYTHONHASHSEED
        all_features = sorted(set().union(*feature_selections))
        feature_counts = {f: sum(f in s for s in feature_selections) for f in all_features}

        min_count = int(n_folds * self.config.min_feature_frequency)
        stable_features = [f for f, count in feature_counts.items() if count >= min_count]

        # Most stable first; equal counts stay in name order (stable sort)
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
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        w_test: pd.Series | None = None,
    ) -> pd.Series:
        """Compute feature importance using configured method.

        Args:
            X: Training features (used for fitting).
            y: Training labels.
            sample_weights: Training sample weights.
            X_test: Holdout features for MDA scoring (avoids overfitting bias).
            y_test: Holdout labels for MDA scoring.
            w_test: Holdout sample weights for MDA scoring.
        """
        if self.config.use_clustered_importance:
            return self._clustered_mda_importance(
                X,
                y,
                sample_weights,
                X_test=X_test,
                y_test=y_test,
                w_test=w_test,
            )

        if self.config.selection_method == "mdi":
            return self._mdi_importance(X, y, sample_weights)
        elif self.config.selection_method == "mda":
            return self._mda_importance(
                X,
                y,
                sample_weights,
                X_test=X_test,
                y_test=y_test,
                w_test=w_test,
            )
        else:  # hybrid
            mdi = self._mdi_importance(X, y, sample_weights)
            mda = self._mda_importance(
                X,
                y,
                sample_weights,
                X_test=X_test,
                y_test=y_test,
                w_test=w_test,
            )
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
        rf.fit(X.to_numpy(dtype=float), y, sample_weight=sample_weights)
        # Thousands of small predict calls follow: a thread pool per call costs
        # far more than the prediction itself
        rf.set_params(n_jobs=1)
        return pd.Series(rf.feature_importances_, index=X.columns)

    def _mda_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        w_test: pd.Series | None = None,
    ) -> pd.Series:
        """
        Mean Decrease in Accuracy (permutation importance), scored by log-loss.

        Importance is the increase in (sample-weighted) log-loss when a feature is
        permuted, not the drop in accuracy: accuracy is insensitive on imbalanced
        classes. Fits the RF on training data but scores permutation importance on
        the holdout set (X_test, y_test) to avoid overfitting bias. Falls
        back to OOB scoring if no holdout is provided.

        Reference: Lopez de Prado (2018), Chapter 8
        """
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=5,
            oob_score=True,
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X, y, sample_weight=sample_weights)
        sequential_prediction(rf)  # bit-reproducible scoring

        # Score on holdout data to avoid overfitting bias (Critical Fix #6).
        # Training-set permutation importance inflates scores for overfit features.
        if X_test is not None and y_test is not None:
            score_X, score_y, score_w = X_test, y_test, w_test
        else:
            # Fallback: use training data (legacy behavior for direct callers)
            logger.warning(
                "MDA importance: no holdout set provided, falling back to "
                "training data. Results may have overfitting bias."
            )
            score_X, score_y, score_w = X, y, sample_weights

        result = permutation_importance(
            rf,
            score_X,
            score_y,
            scoring=_neg_log_loss_scorer(rf.classes_),
            n_repeats=self.config.mda_n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
            sample_weight=score_w,
        )

        return pd.Series(result.importances_mean, index=X.columns)

    def _clustered_mda_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        w_test: pd.Series | None = None,
    ) -> pd.Series:
        """
        Clustered MDA (Lopez de Prado 2020, "ML for Asset Managers", 6.5).

        1. Cluster features on train-fold signed-correlation distance.
        2. Fit ONE forest on all features of the (purged) train fold.
        3. On the held-out fold, permute all columns of a cluster JOINTLY (same
           row permutation, so intra-cluster structure is kept) and record the
           increase in log-loss; cluster importance is its mean over repeats.
           Joint permutation removes the substitution effect: a signal duplicated
           across correlated columns is no longer masked by its twins.
        4. Every feature inherits its cluster's importance. Ties inside a cluster
           are broken by the feature's own single-column MDA, which only orders
           members and is added at a 1e-9 scale so it can never reorder clusters.
           (Splitting the cluster score 1/size instead would rank pure-noise
           features by cluster size, which the previous implementation did.)

        Sample weights are used for both fitting and scoring. If no holdout is
        supplied, the last 25% of the (time-ordered) train rows is held out.
        """
        if X_test is None or y_test is None:
            logger.warning(
                "Clustered MDA: no holdout set provided, holding out the last "
                f"{_FALLBACK_HOLDOUT_FRAC:.0%} of the training rows."
            )
            cut = int(len(X) * (1 - _FALLBACK_HOLDOUT_FRAC))
            X_test, y_test = X.iloc[cut:], y.iloc[cut:]
            w_test = sample_weights.iloc[cut:] if sample_weights is not None else None
            X, y = X.iloc[:cut], y.iloc[:cut]
            sample_weights = sample_weights.iloc[:cut] if sample_weights is not None else None

        clusters = cluster_features(X, self.config.max_clusters, self.cluster_distance_threshold)

        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=5,
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X, y, sample_weight=sample_weights)
        sequential_prediction(rf)  # bit-reproducible scoring

        # Score only holdout rows whose class the forest has seen.
        class_to_idx = {c: i for i, c in enumerate(rf.classes_)}
        keep = y_test.isin(list(class_to_idx)).to_numpy()
        if not keep.any():
            return pd.Series(0.0, index=X.columns)
        values = X_test.loc[:, X.columns].to_numpy(dtype=float, copy=True)[keep]
        y_idx = y_test.map(class_to_idx).to_numpy()[keep].astype(int)
        w_eval = w_test.to_numpy(dtype=float)[keep] if w_test is not None else None

        base = _neg_log_loss(rf.predict_proba(values), y_idx, w_eval)
        col_pos = {c: i for i, c in enumerate(X.columns)}

        def group_importance(
            cols: list[str], seed: int, n_repeats: int = self.config.mda_n_repeats
        ) -> float:
            # Own copy: permutations run concurrently (forest prediction releases the GIL)
            local = values.copy()
            pos = [col_pos[c] for c in cols]
            saved = local[:, pos].copy()
            rng = np.random.default_rng(seed)
            drops = []
            for _ in range(n_repeats):
                local[:, pos] = saved[rng.permutation(len(saved))]
                drops.append(base - _neg_log_loss(rf.predict_proba(local), y_idx, w_eval))
            return float(np.mean(drops))

        def run_all(jobs: list[tuple[list[str], int, int]]) -> list[float]:
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
                return list(pool.map(lambda job: group_importance(*job), jobs))

        cluster_ids = [int(c) for c in np.unique(clusters.to_numpy())]
        cluster_members = {c: clusters.index[clusters == c].tolist() for c in cluster_ids}
        scores = run_all(
            [
                (cluster_members[c], self.random_state + c, self.config.mda_n_repeats)
                for c in cluster_ids
            ]
        )
        cluster_importance = dict(zip(cluster_ids, scores, strict=True))

        importance = pd.Series(
            {f: cluster_importance[int(clusters[f])] for f in X.columns}, dtype=float
        )

        # Within-cluster ordering (singleton clusters need none). One repeat
        # suffices: this only orders members inside a cluster.
        multi = [c for c in cluster_ids if len(cluster_members[c]) > 1]
        members_flat = [f for c in multi for f in cluster_members[c]]
        own_all = dict(
            zip(
                members_flat,
                run_all(
                    [([f], self.random_state + 7919 + i, 1) for i, f in enumerate(members_flat)]
                ),
                strict=True,
            )
        )
        for c in multi:
            # Quantized: float noise between near-tied members must not order them
            own = quantize_importance(pd.Series({f: own_all[f] for f in cluster_members[c]}))
            span = own.max() - own.min()
            rank01 = (own - own.min()) / span if span > 0 else own * 0.0
            importance[cluster_members[c]] += _WITHIN_CLUSTER_TIEBREAK * rank01

        return importance


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


__all__ = [
    "WalkForwardFeatureSelector",
    "CVIntegratedFeatureSelector",
]
