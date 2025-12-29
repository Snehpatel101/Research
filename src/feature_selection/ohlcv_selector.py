"""
Enhanced Feature Selection for OHLCV Time-Series ML.

This module provides advanced feature selection specifically designed for
OHLCV time-series data in ML trading applications. It extends the basic
walk-forward feature selection with:

- Time-series aware feature importance (walk-forward MDA)
- Regime-conditional feature selection
- Multi-timeframe correlation filtering
- Stability-weighted feature ranking
- OHLCV-specific feature category filtering

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CATEGORIES FOR OHLCV DATA
# =============================================================================

# Feature category patterns (prefix matching)
FEATURE_CATEGORIES: dict[str, list[str]] = {
    "momentum": [
        "rsi", "macd", "stochastic", "williams_r", "roc", "cci", "mfi",
        "momentum", "ppo", "tsi",
    ],
    "volatility": [
        "atr", "bollinger", "keltner", "volatility", "vol_", "parkinson",
        "garman_klass", "yang_zhang", "std_", "range_",
    ],
    "volume": [
        "obv", "vwap", "volume", "vol_ratio", "money_flow", "chaikin",
        "ad_line", "force_index", "ease_of_movement",
    ],
    "trend": [
        "sma", "ema", "adx", "supertrend", "trend_", "ma_", "dema", "tema",
        "kama", "linear_reg", "parabolic_sar",
    ],
    "microstructure": [
        "spread", "amihud", "roll", "kyle", "corwin", "imbalance",
        "trade_intensity", "efficiency", "realized_vol",
    ],
    "wavelet": [
        "wavelet", "dwt_", "cA_", "cD_", "energy_", "wave_",
    ],
    "mtf": [
        "_5min", "_15min", "_30min", "_1h", "_4h", "_daily", "_1d",
        "htf_", "ltf_", "mtf_",
    ],
    "regime": [
        "regime", "state_", "hidden_", "cluster_", "vol_regime", "trend_regime",
    ],
    "price": [
        "return", "log_return", "price_ratio", "close_", "open_", "high_", "low_",
        "hlc_", "ohlc_", "typical_price", "median_price",
    ],
    "temporal": [
        "hour", "day", "week", "month", "session", "time_", "dow_", "is_",
    ],
}


def categorize_feature(feature_name: str) -> str:
    """
    Categorize a feature by name pattern matching.

    Args:
        feature_name: Feature name to categorize

    Returns:
        Category name or 'other' if no match
    """
    feature_lower = feature_name.lower()
    for category, patterns in FEATURE_CATEGORIES.items():
        for pattern in patterns:
            if pattern in feature_lower:
                return category
    return "other"


def filter_ohlcv_features(
    feature_names: list[str],
    include_categories: list[str] | None = None,
    exclude_categories: list[str] | None = None,
) -> list[str]:
    """
    Filter features by OHLCV-specific categories.

    Args:
        feature_names: List of feature names to filter
        include_categories: Categories to include (None = include all)
        exclude_categories: Categories to exclude (None = exclude none)

    Returns:
        List of filtered feature names

    Categories:
        - 'momentum': RSI, MACD, ROC, etc.
        - 'volatility': ATR, Bollinger, etc.
        - 'volume': OBV, VWAP, etc.
        - 'trend': SMA, EMA, ADX, etc.
        - 'microstructure': Bid-ask, order flow, etc.
        - 'wavelet': Wavelet decomposition features
        - 'mtf': Multi-timeframe features
        - 'regime': Regime detection features
        - 'price': Price-derived features
        - 'temporal': Time-based features

    Example:
        >>> features = ['rsi_14', 'sma_20', 'atr_14', 'wavelet_energy']
        >>> filter_ohlcv_features(features, include_categories=['momentum', 'volatility'])
        ['rsi_14', 'atr_14']
    """
    filtered = []
    for name in feature_names:
        category = categorize_feature(name)
        if include_categories is not None and category not in include_categories:
            if category != "other" or "other" not in include_categories:
                continue
        if exclude_categories is not None and category in exclude_categories:
            continue
        filtered.append(name)
    return filtered


def get_feature_categories(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Get category breakdown of features.

    Args:
        feature_names: List of feature names

    Returns:
        Dict mapping category name to list of features in that category
    """
    result: dict[str, list[str]] = {}
    for name in feature_names:
        category = categorize_feature(name)
        if category not in result:
            result[category] = []
        result[category].append(name)
    return result


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class FeatureSelectionResult:
    """
    Result of OHLCV feature selection process.

    Attributes:
        selected_features: Final list of selected feature names
        feature_importances: Dict mapping features to aggregated importance scores
        stability_scores: Dict mapping features to stability across folds (0-1)
        correlation_clusters: List of correlated feature groups
        regime_importances: Optional dict of per-regime importance scores
        n_original: Number of original features
        n_selected: Number of selected features
        selection_metadata: Additional metadata about selection process
    """
    selected_features: list[str]
    feature_importances: dict[str, float]
    stability_scores: dict[str, float]
    correlation_clusters: list[list[str]]
    regime_importances: dict[int, dict[str, float]] | None
    n_original: int
    n_selected: int
    selection_metadata: dict[str, Any] = field(default_factory=dict)

    def get_category_breakdown(self) -> dict[str, int]:
        """Get count of selected features per category."""
        return {
            cat: len(feats)
            for cat, feats in get_feature_categories(self.selected_features).items()
        }

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N features by importance."""
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:n]


@dataclass
class StabilityMetrics:
    """Metrics for feature stability analysis."""
    rank_correlation: float  # Correlation of rankings across folds
    selection_frequency: float  # Fraction of folds feature was selected
    importance_std: float  # Standard deviation of importance across folds
    importance_cv: float  # Coefficient of variation (std/mean)


# =============================================================================
# OHLCV FEATURE SELECTOR
# =============================================================================

class OHLCVFeatureSelector:
    """
    Enhanced feature selection for OHLCV time-series ML.

    This selector extends basic walk-forward feature selection with:

    1. **Time-series aware MDA**: Uses PurgedKFold to respect temporal order
       and prevent lookahead bias in importance computation.

    2. **Stability scoring**: Tracks not just selection frequency, but the
       consistency of feature rankings across walk-forward folds. Features
       with high stability are more likely to generalize.

    3. **Correlation filtering**: Uses hierarchical clustering to identify
       redundant features and keeps only the most important in each cluster.

    4. **Regime conditioning**: Optionally computes feature importance
       separately for each market regime, identifying features that are
       predictive only in specific conditions.

    Example:
        >>> selector = OHLCVFeatureSelector(
        ...     n_splits=5,
        ...     min_stability_score=0.6,
        ...     correlation_threshold=0.85,
        ...     use_regime_conditioning=True,
        ... )
        >>> result = selector.select_features(X, y, feature_names, regimes=regime_labels)
        >>> print(f"Selected {result.n_selected} of {result.n_original} features")
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_stability_score: float = 0.5,
        correlation_threshold: float = 0.85,
        use_regime_conditioning: bool = False,
        n_features_per_fold: int = 50,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        """
        Initialize OHLCV feature selector.

        Args:
            n_splits: Number of time-series splits for stability testing
            min_stability_score: Minimum consistency score across folds (0-1)
            correlation_threshold: Remove features correlated above this (0-1)
            use_regime_conditioning: Separate feature importance by regime
            n_features_per_fold: Number of top features to consider per fold
            n_estimators: Number of trees for random forest importance
            random_state: Random seed for reproducibility
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not 0 <= min_stability_score <= 1:
            raise ValueError(f"min_stability_score must be in [0, 1], got {min_stability_score}")
        if not 0 < correlation_threshold <= 1:
            raise ValueError(f"correlation_threshold must be in (0, 1], got {correlation_threshold}")

        self.n_splits = n_splits
        self.min_stability_score = min_stability_score
        self.correlation_threshold = correlation_threshold
        self.use_regime_conditioning = use_regime_conditioning
        self.n_features_per_fold = n_features_per_fold
        self.n_estimators = n_estimators
        self.random_state = random_state

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        regimes: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> FeatureSelectionResult:
        """
        Select optimal features using multiple criteria.

        Selection process:
        1. Compute walk-forward MDA importance across folds
        2. Calculate stability scores (ranking consistency)
        3. Filter by minimum stability score
        4. Remove highly correlated features (keep most important per cluster)
        5. Optionally compute regime-conditional importance

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: List of feature names
            regimes: Optional regime labels for conditional selection
            sample_weights: Optional sample weights

        Returns:
            FeatureSelectionResult with selected features and metadata
        """
        n_samples, n_features = X.shape
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) != n_features ({n_features})"
            )

        logger.info(
            f"Starting OHLCV feature selection: {n_features} features, {n_samples} samples"
        )

        # Convert to DataFrame for easier handling
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        weights_series = pd.Series(sample_weights) if sample_weights is not None else None

        # Step 1: Compute walk-forward MDA importance
        fold_importances = self._compute_walk_forward_mda(
            X_df, y_series, feature_names, weights_series
        )

        # Step 2: Calculate stability scores
        stability_scores = self._compute_stability_scores(fold_importances, feature_names)

        # Aggregate importance (mean across folds)
        aggregated_importance = self._aggregate_importance(fold_importances, feature_names)

        # Step 3: Filter by stability
        stable_features = [
            f for f in feature_names
            if stability_scores.get(f, 0) >= self.min_stability_score
        ]
        logger.info(
            f"After stability filtering: {len(stable_features)} features "
            f"(min stability = {self.min_stability_score})"
        )

        # Step 4: Remove correlated features
        selected_features, correlation_clusters = self._remove_correlated_features(
            X_df[stable_features] if stable_features else X_df,
            stable_features if stable_features else feature_names,
            {f: aggregated_importance.get(f, 0) for f in (stable_features or feature_names)},
        )
        logger.info(
            f"After correlation filtering: {len(selected_features)} features "
            f"(threshold = {self.correlation_threshold})"
        )

        # Step 5: Regime-conditional importance (optional)
        regime_importances = None
        if self.use_regime_conditioning and regimes is not None:
            regime_importances = self._regime_conditional_selection(
                X_df[selected_features],
                y_series,
                selected_features,
                regimes,
                weights_series,
            )
            logger.info(f"Computed regime-conditional importance for {len(regime_importances)} regimes")

        # Build final result
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_importances={f: aggregated_importance.get(f, 0) for f in selected_features},
            stability_scores={f: stability_scores.get(f, 0) for f in selected_features},
            correlation_clusters=correlation_clusters,
            regime_importances=regime_importances,
            n_original=n_features,
            n_selected=len(selected_features),
            selection_metadata={
                "n_splits": self.n_splits,
                "min_stability_score": self.min_stability_score,
                "correlation_threshold": self.correlation_threshold,
                "use_regime_conditioning": self.use_regime_conditioning,
                "n_stable_before_correlation": len(stable_features),
                "category_breakdown": get_feature_categories(selected_features),
            },
        )

        logger.info(
            f"Feature selection complete: {result.n_selected}/{result.n_original} selected"
        )
        return result

    def _compute_walk_forward_mda(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        sample_weights: pd.Series | None = None,
    ) -> list[dict[str, float]]:
        """
        Compute Mean Decrease Accuracy using time-series splits.

        Unlike standard MDA, this uses time-series aware splits to respect
        temporal order and prevent using future data in importance calculation.

        Args:
            X: Feature DataFrame
            y: Labels
            feature_names: Feature names
            sample_weights: Optional sample weights

        Returns:
            List of importance dicts, one per fold
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        fold_importances: list[dict[str, float]] = []

        for fold_idx in range(self.n_splits):
            # Time-series split: train on past, test on future
            test_start = fold_idx * fold_size
            test_end = min((fold_idx + 1) * fold_size, n_samples)

            # Train on everything before test period
            if fold_idx == 0:
                # First fold: use minimal training (at least 20% of data)
                train_end = max(int(n_samples * 0.2), fold_size)
                train_idx = np.arange(train_end)
            else:
                train_idx = np.arange(test_start)

            test_idx = np.arange(test_start, test_end)

            if len(train_idx) < 100 or len(test_idx) < 50:
                logger.warning(
                    f"Fold {fold_idx}: insufficient samples (train={len(train_idx)}, test={len(test_idx)})"
                )
                continue

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            w_train = None
            if sample_weights is not None:
                w_train = sample_weights.iloc[train_idx].values

            # Train RF and compute permutation importance
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                n_jobs=-1,
                random_state=self.random_state + fold_idx,
            )
            rf.fit(X_train, y_train, sample_weight=w_train)

            # Permutation importance on test set (more reliable than MDI)
            result = permutation_importance(
                rf, X_test, y_test,
                n_repeats=10,
                random_state=self.random_state + fold_idx,
                n_jobs=-1,
            )

            importance = dict(zip(feature_names, result.importances_mean, strict=False))
            fold_importances.append(importance)

            logger.debug(
                f"Fold {fold_idx}: top feature = {max(importance, key=importance.get)}"
            )

        return fold_importances

    def _compute_stability_scores(
        self,
        fold_importances: list[dict[str, float]],
        feature_names: list[str],
    ) -> dict[str, float]:
        """
        Compute feature importance stability across walk-forward folds.

        Stability is measured by the Spearman rank correlation of feature
        rankings across consecutive folds. High stability indicates the
        feature's importance is consistent across time periods.

        Args:
            fold_importances: List of importance dicts from each fold
            feature_names: Feature names

        Returns:
            Dict mapping feature name to stability score (0-1)
        """
        if len(fold_importances) < 2:
            # Not enough folds for stability calculation
            return {f: 1.0 for f in feature_names}

        n_folds = len(fold_importances)

        # Build importance matrix (folds x features)
        importance_matrix = np.zeros((n_folds, len(feature_names)))
        for fold_idx, imp_dict in enumerate(fold_importances):
            for feat_idx, feat_name in enumerate(feature_names):
                importance_matrix[fold_idx, feat_idx] = imp_dict.get(feat_name, 0)

        # Compute rank matrix
        rank_matrix = np.zeros_like(importance_matrix)
        for fold_idx in range(n_folds):
            rank_matrix[fold_idx] = pd.Series(importance_matrix[fold_idx]).rank().values

        # Stability = average pairwise rank correlation
        stability_scores: dict[str, float] = {}
        for feat_idx, feat_name in enumerate(feature_names):
            ranks = rank_matrix[:, feat_idx]

            # Compute pairwise correlations between consecutive folds
            pairwise_corrs = []
            for i in range(len(ranks) - 1):
                # Compare this feature's rank to next fold
                corr, _ = spearmanr(
                    importance_matrix[i],
                    importance_matrix[i + 1],
                )
                if not np.isnan(corr):
                    pairwise_corrs.append(corr)

            if pairwise_corrs:
                # Average correlation, normalized to [0, 1]
                stability = (np.mean(pairwise_corrs) + 1) / 2
            else:
                stability = 0.5

            # Also consider variance in ranking position
            rank_std = np.std(ranks)
            max_rank = len(feature_names)
            rank_stability = 1 - (rank_std / (max_rank / 2))  # Normalize

            # Combine both metrics
            stability_scores[feat_name] = (stability + max(0, rank_stability)) / 2

        return stability_scores

    def _aggregate_importance(
        self,
        fold_importances: list[dict[str, float]],
        feature_names: list[str],
    ) -> dict[str, float]:
        """Aggregate importance across folds (mean)."""
        aggregated: dict[str, float] = {}
        for feat_name in feature_names:
            values = [imp.get(feat_name, 0) for imp in fold_importances]
            aggregated[feat_name] = float(np.mean(values))
        return aggregated

    def _remove_correlated_features(
        self,
        X: pd.DataFrame,
        feature_names: list[str],
        importances: dict[str, float],
    ) -> tuple[list[str], list[list[str]]]:
        """
        Remove highly correlated features, keeping most important in each cluster.

        Uses hierarchical clustering on the correlation matrix to identify
        groups of correlated features. For each group, keeps only the feature
        with highest importance.

        Args:
            X: Feature DataFrame
            feature_names: Feature names
            importances: Feature importance scores

        Returns:
            Tuple of (selected_features, correlation_clusters)
        """
        if len(feature_names) <= 1:
            return feature_names, [[f] for f in feature_names]

        # Compute correlation matrix
        corr = X[feature_names].corr()
        corr = corr.fillna(0)

        # Distance = 1 - |correlation|
        dist = 1 - corr.abs()
        np.fill_diagonal(dist.values, 0)

        # Handle edge cases (all same values, etc.)
        dist = dist.clip(lower=0)

        try:
            # Hierarchical clustering
            dist_condensed = squareform(dist.values, checks=False)
            linkage_matrix = linkage(dist_condensed, method="ward")

            # Cluster at correlation threshold
            distance_threshold = 1 - self.correlation_threshold
            clusters = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, returning all features")
            return feature_names, [[f] for f in feature_names]

        # Group features by cluster
        cluster_groups: dict[int, list[str]] = {}
        for feat_name, cluster_id in zip(feature_names, clusters, strict=False):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(feat_name)

        # Select best feature from each cluster
        selected: list[str] = []
        correlation_clusters: list[list[str]] = []

        for cluster_id, features in cluster_groups.items():
            if len(features) == 1:
                selected.append(features[0])
                correlation_clusters.append(features)
            else:
                # Keep feature with highest importance
                best_feature = max(features, key=lambda f: importances.get(f, 0))
                selected.append(best_feature)
                correlation_clusters.append(features)

        return selected, correlation_clusters

    def _regime_conditional_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        regimes: np.ndarray,
        sample_weights: pd.Series | None = None,
    ) -> dict[int, dict[str, float]]:
        """
        Compute feature importance separately for each regime.

        Some features may only be predictive in certain market conditions
        (e.g., volatility-based features in high-vol regimes).

        Args:
            X: Feature DataFrame
            y: Labels
            feature_names: Feature names
            regimes: Regime labels per sample
            sample_weights: Optional sample weights

        Returns:
            Dict mapping regime id to feature importance dict
        """
        unique_regimes = np.unique(regimes)
        regime_importances: dict[int, dict[str, float]] = {}

        for regime_id in unique_regimes:
            regime_mask = regimes == regime_id
            n_regime_samples = regime_mask.sum()

            if n_regime_samples < 100:
                logger.warning(
                    f"Regime {regime_id}: only {n_regime_samples} samples, skipping"
                )
                continue

            X_regime = X.loc[regime_mask]
            y_regime = y.loc[regime_mask]
            w_regime = None
            if sample_weights is not None:
                w_regime = sample_weights.loc[regime_mask].values

            # Train RF and get feature importance
            rf = RandomForestClassifier(
                n_estimators=min(50, self.n_estimators),
                max_depth=5,
                n_jobs=-1,
                random_state=self.random_state,
            )
            rf.fit(X_regime, y_regime, sample_weight=w_regime)

            importance = dict(zip(feature_names, rf.feature_importances_, strict=False))
            regime_importances[int(regime_id)] = importance

            logger.debug(
                f"Regime {regime_id}: {n_regime_samples} samples, "
                f"top feature = {max(importance, key=importance.get)}"
            )

        return regime_importances


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_ohlcv_selector(
    n_splits: int = 5,
    min_stability: float = 0.5,
    correlation_threshold: float = 0.85,
    use_regimes: bool = False,
    random_state: int = 42,
) -> OHLCVFeatureSelector:
    """
    Factory function for OHLCVFeatureSelector.

    Args:
        n_splits: Number of walk-forward splits
        min_stability: Minimum stability score for feature selection
        correlation_threshold: Correlation threshold for filtering
        use_regimes: Enable regime-conditional selection
        random_state: Random seed

    Returns:
        Configured OHLCVFeatureSelector instance
    """
    return OHLCVFeatureSelector(
        n_splits=n_splits,
        min_stability_score=min_stability,
        correlation_threshold=correlation_threshold,
        use_regime_conditioning=use_regimes,
        random_state=random_state,
    )


__all__ = [
    # Main class
    "OHLCVFeatureSelector",
    # Result classes
    "FeatureSelectionResult",
    "StabilityMetrics",
    # Category utilities
    "FEATURE_CATEGORIES",
    "categorize_feature",
    "filter_ohlcv_features",
    "get_feature_categories",
    # Factory
    "create_ohlcv_selector",
]
