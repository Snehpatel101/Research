"""
Purged Feature Selector - Bridge between OHLCVFeatureSelector and PurgedKFold.

This module provides integration between the enhanced OHLCV feature selector
and the existing PurgedKFold cross-validation infrastructure to ensure
proper temporal handling with purge and embargo.

Key difference from standard OHLCVFeatureSelector:
- Uses PurgedKFold splits instead of simple time-series splits
- Respects purge and embargo constraints
- Integrates with existing CV infrastructure
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.feature_selection.ohlcv_selector import (
    FeatureSelectionResult,
    OHLCVFeatureSelector,
    get_feature_categories,
)

logger = logging.getLogger(__name__)


class PurgedFeatureSelector:
    """
    Feature selector that uses PurgedKFold for proper temporal handling.

    This selector wraps OHLCVFeatureSelector but uses PurgedKFold splits
    instead of simple time-series splits, ensuring:
    - Proper purge bars before test set (prevents label leakage)
    - Proper embargo bars after test set (breaks serial correlation)

    Example:
        >>> from src.cross_validation import PurgedKFold, PurgedKFoldConfig
        >>> from src.feature_selection import PurgedFeatureSelector
        >>>
        >>> cv_config = PurgedKFoldConfig(n_splits=5, purge_bars=60, embargo_bars=1440)
        >>> cv = PurgedKFold(cv_config)
        >>>
        >>> selector = PurgedFeatureSelector(cv=cv)
        >>> result = selector.select_features(X, y, feature_names)
    """

    def __init__(
        self,
        cv: Any,  # PurgedKFold instance
        min_stability_score: float = 0.5,
        correlation_threshold: float = 0.85,
        use_regime_conditioning: bool = False,
        n_features_per_fold: int = 50,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        """
        Initialize PurgedFeatureSelector.

        Args:
            cv: PurgedKFold instance (or any CV with split() method)
            min_stability_score: Minimum consistency score across folds (0-1)
            correlation_threshold: Remove features correlated above this (0-1)
            use_regime_conditioning: Separate feature importance by regime
            n_features_per_fold: Number of top features to consider per fold
            n_estimators: Number of trees for random forest importance
            random_state: Random seed for reproducibility
        """
        self.cv = cv
        self.min_stability_score = min_stability_score
        self.correlation_threshold = correlation_threshold
        self.use_regime_conditioning = use_regime_conditioning
        self.n_features_per_fold = n_features_per_fold
        self.n_estimators = n_estimators
        self.random_state = random_state

        # Create base selector for non-CV-specific operations
        # Use n_splits=2 (minimum valid) since we use CV splits instead
        self._base_selector = OHLCVFeatureSelector(
            n_splits=2,  # Minimum valid; actual splits come from self.cv
            min_stability_score=min_stability_score,
            correlation_threshold=correlation_threshold,
            use_regime_conditioning=use_regime_conditioning,
            n_features_per_fold=n_features_per_fold,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        regimes: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> FeatureSelectionResult:
        """
        Select optimal features using PurgedKFold splits.

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
            f"Starting PurgedKFold feature selection: {n_features} features, {n_samples} samples"
        )

        # Convert to DataFrame for easier handling
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        weights_series = pd.Series(sample_weights) if sample_weights is not None else None

        # Step 1: Compute importance using PurgedKFold splits
        fold_importances = self._compute_purged_mda(
            X_df, y_series, feature_names, weights_series
        )

        # Step 2: Calculate stability scores
        stability_scores = self._base_selector._compute_stability_scores(
            fold_importances, feature_names
        )

        # Aggregate importance (mean across folds)
        aggregated_importance = self._base_selector._aggregate_importance(
            fold_importances, feature_names
        )

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
        selected_features, correlation_clusters = self._base_selector._remove_correlated_features(
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
            regime_importances = self._base_selector._regime_conditional_selection(
                X_df[selected_features],
                y_series,
                selected_features,
                regimes,
                weights_series,
            )

        # Build result
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_importances={f: aggregated_importance.get(f, 0) for f in selected_features},
            stability_scores={f: stability_scores.get(f, 0) for f in selected_features},
            correlation_clusters=correlation_clusters,
            regime_importances=regime_importances,
            n_original=n_features,
            n_selected=len(selected_features),
            selection_metadata={
                "n_splits": self._get_n_splits(),
                "purge_bars": self._get_purge_bars(),
                "embargo_bars": self._get_embargo_bars(),
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

    def _compute_purged_mda(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        sample_weights: pd.Series | None = None,
    ) -> list[dict[str, float]]:
        """
        Compute MDA importance using PurgedKFold splits.

        Uses the configured PurgedKFold to generate proper train/test splits
        with purge and embargo, then computes permutation importance.
        """
        fold_importances: list[dict[str, float]] = []

        # Get splits from PurgedKFold
        splits = list(self.cv.split(X, y))

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
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

            # Permutation importance on test set
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

    def _get_n_splits(self) -> int:
        """Get number of splits from CV."""
        if hasattr(self.cv, "config") and hasattr(self.cv.config, "n_splits"):
            return self.cv.config.n_splits
        elif hasattr(self.cv, "n_splits"):
            return self.cv.n_splits
        return 5

    def _get_purge_bars(self) -> int:
        """Get purge bars from CV."""
        if hasattr(self.cv, "config") and hasattr(self.cv.config, "purge_bars"):
            return self.cv.config.purge_bars
        return 60

    def _get_embargo_bars(self) -> int:
        """Get embargo bars from CV."""
        if hasattr(self.cv, "config") and hasattr(self.cv.config, "embargo_bars"):
            return self.cv.config.embargo_bars
        return 1440


def create_purged_selector(
    cv: Any,
    min_stability: float = 0.5,
    correlation_threshold: float = 0.85,
    use_regimes: bool = False,
    random_state: int = 42,
) -> PurgedFeatureSelector:
    """
    Factory function for PurgedFeatureSelector.

    Args:
        cv: PurgedKFold instance
        min_stability: Minimum stability score
        correlation_threshold: Correlation threshold for filtering
        use_regimes: Enable regime-conditional selection
        random_state: Random seed

    Returns:
        Configured PurgedFeatureSelector instance
    """
    return PurgedFeatureSelector(
        cv=cv,
        min_stability_score=min_stability,
        correlation_threshold=correlation_threshold,
        use_regime_conditioning=use_regimes,
        random_state=random_state,
    )


__all__ = [
    "PurgedFeatureSelector",
    "create_purged_selector",
]
