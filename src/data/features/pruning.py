"""
Feature Pruning Module - Optuna-based feature pruning for PHASE_1B.

This module provides intelligent feature pruning using Optuna's optimization
framework combined with multiple pruning strategies: importance-based,
correlation-based, and null importance testing.

PHASE_1B: Optimized feature pruning with MedianPruner for early stopping
on bad feature combinations and comprehensive null importance testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Dict, Optional, Any, Tuple

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

from src.core.constants import (
    DEFAULT_FEATURE_PRUNING_TRIALS,
    DEFAULT_MIN_FEATURES,
    DEFAULT_OPTUNA_RANDOM_STATE,
    DEFAULT_N_SPLITS,
)


logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class FeaturePruningResult:
    """Result of Optuna-based feature pruning.

    Attributes:
        original_features: List of original feature names before pruning.
        pruned_features: List of feature names after pruning (kept features).
        removed_features: List of feature names that were removed.
        n_original: Number of original features.
        n_pruned: Number of features after pruning.
        n_removed: Number of features removed.
        pruning_ratio: Ratio of removed to original features.
        score_before: Cross-validation score before pruning.
        score_after: Cross-validation score after pruning.
        improvement: Relative score improvement (positive = better after pruning).
        feature_importance: Dictionary mapping feature names to importance scores.
        correlation_matrix: Optional correlation matrix used for correlation pruning.
        null_importance_results: Optional null importance test results.
        study: The Optuna study object for further analysis.
    """

    original_features: List[str]
    pruned_features: List[str]
    removed_features: List[str]
    n_original: int
    n_pruned: int
    n_removed: int
    pruning_ratio: float
    score_before: float
    score_after: float
    improvement: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Optional[np.ndarray] = None
    null_importance_results: Optional[Dict[str, float]] = None
    study: Optional[optuna.Study] = None

    def __repr__(self) -> str:
        return (
            f"FeaturePruningResult("
            f"pruned={self.n_pruned}/{self.n_original}, "
            f"removed={self.n_removed}, "
            f"ratio={self.pruning_ratio:.2%}, "
            f"score_before={self.score_before:.4f}, "
            f"score_after={self.score_after:.4f}, "
            f"improvement={self.improvement:+.2%})"
        )

    def get_removed_by_importance(self) -> List[Tuple[str, float]]:
        """Get removed features sorted by importance.

        Returns:
            List of (feature_name, importance) tuples for removed features.
        """
        return sorted(
            [(f, self.feature_importance.get(f, 0.0)) for f in self.removed_features],
            key=lambda x: x[1],
            reverse=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "original_features": self.original_features,
            "pruned_features": self.pruned_features,
            "removed_features": self.removed_features,
            "n_original": self.n_original,
            "n_pruned": self.n_pruned,
            "n_removed": self.n_removed,
            "pruning_ratio": self.pruning_ratio,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "improvement": self.improvement,
            "feature_importance": self.feature_importance,
        }


# =============================================================================
# FEATURE PRUNER
# =============================================================================


class FeaturePruner:
    """Optuna-based feature pruning.

    Uses Optuna's optimization framework to intelligently prune features
    that don't contribute significantly to model performance.

    Methods:
        - importance: Remove lowest importance features based on model's
          feature_importances_ or permutation importance.
        - correlation: Remove highly correlated features, keeping the one
          with higher importance.
        - null: Remove features with null importance (permutation test shows
          they don't contribute more than random).

    Args:
        n_trials: Number of Optuna trials for importance-based pruning.
        pruning_strategy: Pruning strategy ("importance", "correlation", "null").
        min_features: Minimum number of features to keep.
        importance_threshold: Threshold below which features are candidates
            for removal (for importance strategy).
        correlation_threshold: Correlation threshold above which one of
            the correlated features is removed (for correlation strategy).
        cv_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 for all cores).

    Example:
        >>> pruner = FeaturePruner(
        ...     n_trials=50,
        ...     pruning_strategy="importance",
        ...     importance_threshold=0.01,
        ... )
        >>> result = pruner.prune_features(
        ...     X=X_train,
        ...     y=y_train,
        ...     feature_names=feature_names,
        ...     model_fn=lambda: XGBClassifier(n_estimators=100),
        ... )
        >>> print(f"Pruned {result.n_removed} features")
    """

    VALID_STRATEGIES = ("importance", "correlation", "null")

    def __init__(
        self,
        n_trials: int = DEFAULT_FEATURE_PRUNING_TRIALS,
        pruning_strategy: str = "importance",
        min_features: int = DEFAULT_MIN_FEATURES,
        importance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        cv_folds: int = DEFAULT_N_SPLITS,
        random_state: int = DEFAULT_OPTUNA_RANDOM_STATE,
        n_jobs: int = -1,
    ) -> None:
        if pruning_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid pruning_strategy: {pruning_strategy}. "
                f"Must be one of {self.VALID_STRATEGIES}"
            )

        self.n_trials = n_trials
        self.pruning_strategy = pruning_strategy
        self.min_features = min_features
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

    def prune_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_fn: Callable,
        scoring: str = "f1_weighted",
        storage_path: Optional[str] = None,
    ) -> FeaturePruningResult:
        """Run feature pruning optimization.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
            feature_names: List of feature names corresponding to columns of X.
            model_fn: Callable that returns a sklearn-compatible model instance.
            scoring: Scoring metric for cross-validation.
            storage_path: Optional path to SQLite file for Optuna study persistence.
                If provided, the study can be resumed if interrupted.
                If None, uses in-memory storage.

        Returns:
            FeaturePruningResult with pruned features and metadata.

        Raises:
            ValueError: If inputs are invalid.
        """
        # Validate inputs
        self._validate_inputs(X, y, feature_names)

        n_features = len(feature_names)
        effective_min = min(self.min_features, n_features)

        # Skip if too few features
        if n_features <= effective_min:
            logger.info(
                f"Skipping pruning: n_features ({n_features}) <= min_features ({effective_min})"
            )
            return self._create_trivial_result(X, y, feature_names, model_fn, scoring)

        # Compute score before pruning
        logger.info(f"Computing score before pruning with {n_features} features...")
        score_before = self._compute_score(X, y, model_fn, scoring)
        logger.info(f"Score before pruning: {score_before:.4f}")

        # Get feature importance
        logger.info("Computing feature importance...")
        feature_importance = self._get_feature_importance(
            X, y, feature_names, model_fn, scoring
        )

        # Apply pruning strategy
        if self.pruning_strategy == "correlation":
            pruned_features = self.prune_by_correlation(X, feature_names)
            study = None
            correlation_matrix = np.corrcoef(X.T)
            null_results = None

        elif self.pruning_strategy == "null":
            pruned_features = self.null_importance_test(
                X, y, feature_names, model_fn, n_permutations=20
            )
            study = None
            correlation_matrix = None
            null_results = {
                f: feature_importance.get(f, 0.0) for f in feature_names
            }

        else:  # importance
            pruned_features, study = self._importance_pruning(
                X, y, feature_names, model_fn, scoring, feature_importance, storage_path
            )
            correlation_matrix = None
            null_results = None

        # Ensure minimum features
        if len(pruned_features) < effective_min:
            # Add back features by importance until minimum is met
            sorted_by_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            for feat, _ in sorted_by_importance:
                if feat not in pruned_features:
                    pruned_features.append(feat)
                if len(pruned_features) >= effective_min:
                    break

        # Get indices of pruned features
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        pruned_indices = [name_to_idx[f] for f in pruned_features if f in name_to_idx]

        # Compute score after pruning
        X_pruned = X[:, pruned_indices]
        score_after = self._compute_score(X_pruned, y, model_fn, scoring)

        # Calculate removed features
        removed_features = [f for f in feature_names if f not in pruned_features]

        # Calculate improvement
        improvement = (
            (score_after - score_before) / score_before if score_before > 0 else 0.0
        )

        logger.info(
            f"Pruning complete: {len(pruned_features)}/{n_features} features kept, "
            f"score_after={score_after:.4f}, improvement={improvement:+.2%}"
        )

        return FeaturePruningResult(
            original_features=feature_names,
            pruned_features=pruned_features,
            removed_features=removed_features,
            n_original=n_features,
            n_pruned=len(pruned_features),
            n_removed=len(removed_features),
            pruning_ratio=len(removed_features) / n_features,
            score_before=score_before,
            score_after=score_after,
            improvement=improvement,
            feature_importance=feature_importance,
            correlation_matrix=correlation_matrix,
            null_importance_results=null_results,
            study=study,
        )

    def _validate_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> None:
        """Validate input arguments."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if len(y) != X.shape[0]:
            raise ValueError(
                f"X and y size mismatch: X has {X.shape[0]} samples, y has {len(y)}"
            )

        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) != X columns ({X.shape[1]})"
            )

    def _compute_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fn: Callable,
        scoring: str,
    ) -> float:
        """Compute cross-validation score."""
        try:
            model = model_fn()
            scores = cross_val_score(
                model, X, y, cv=self.cv_folds, scoring=scoring, n_jobs=self.n_jobs
            )
            return float(np.mean(scores))
        except Exception as e:
            logger.warning(f"Failed to compute score: {e}")
            return 0.0

    def _create_trivial_result(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_fn: Callable,
        scoring: str,
    ) -> FeaturePruningResult:
        """Create result when no pruning is needed."""
        score = self._compute_score(X, y, model_fn, scoring)

        return FeaturePruningResult(
            original_features=feature_names.copy(),
            pruned_features=feature_names.copy(),
            removed_features=[],
            n_original=len(feature_names),
            n_pruned=len(feature_names),
            n_removed=0,
            pruning_ratio=0.0,
            score_before=score,
            score_after=score,
            improvement=0.0,
            feature_importance={f: 1.0 for f in feature_names},
        )

    def _get_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_fn: Callable,
        scoring: str,
    ) -> Dict[str, float]:
        """Get feature importance from trained model.

        Uses model's feature_importances_ if available, otherwise falls back
        to permutation importance.

        Args:
            X: Feature matrix.
            y: Target labels.
            feature_names: Feature names.
            model_fn: Model factory function.
            scoring: Scoring metric.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        try:
            model = model_fn()
            model.fit(X, y)

            # Try to get native feature importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                # Fall back to permutation importance
                result = permutation_importance(
                    model,
                    X,
                    y,
                    n_repeats=10,
                    random_state=self.random_state,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                )
                importances = result.importances_mean

            # Normalize to 0-1
            max_imp = importances.max() if importances.max() > 0 else 1.0
            normalized = importances / max_imp

            return {feature_names[i]: float(normalized[i]) for i in range(len(feature_names))}

        except Exception as e:
            logger.warning(f"Failed to compute feature importance: {e}")
            return {f: 1.0 for f in feature_names}

    def _importance_pruning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_fn: Callable,
        scoring: str,
        feature_importance: Dict[str, float],
        storage_path: Optional[str] = None,
    ) -> Tuple[List[str], optuna.Study]:
        """Optuna-based importance pruning.

        Uses Optuna to find optimal threshold for removing low-importance features.
        """
        n_features = len(feature_names)
        effective_min = min(self.min_features, n_features)

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Create Optuna study with MedianPruner
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1,
        )

        # Configure storage for study persistence
        storage = None
        if storage_path is not None:
            storage_dir = Path(storage_path).parent
            storage_dir.mkdir(parents=True, exist_ok=True)
            storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        def objective(trial: optuna.Trial) -> float:
            # Suggest importance threshold
            threshold = trial.suggest_float("threshold", 0.0, 0.5)

            # Suggest number of features to keep
            n_keep = trial.suggest_int("n_keep", effective_min, n_features)

            # Select features above threshold, up to n_keep
            selected_features = []
            for feat, imp in sorted_features:
                if imp >= threshold:
                    selected_features.append(feat)
                if len(selected_features) >= n_keep:
                    break

            # Ensure minimum features
            if len(selected_features) < effective_min:
                for feat, _ in sorted_features:
                    if feat not in selected_features:
                        selected_features.append(feat)
                    if len(selected_features) >= effective_min:
                        break

            # Store selected features
            trial.set_user_attr("selected_features", selected_features)

            # Get indices and create subset
            indices = [name_to_idx[f] for f in selected_features]
            X_subset = X[:, indices]

            # Evaluate
            try:
                model = model_fn()
                scores = cross_val_score(
                    model,
                    X_subset,
                    y,
                    cv=self.cv_folds,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                )

                mean_score = float(np.mean(scores))

                # Report for pruning
                trial.report(mean_score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return mean_score

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("-inf")

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info(f"Starting importance-based pruning with {self.n_trials} trials...")

        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,),
        )

        # Get best trial's selected features
        best_trial = study.best_trial
        pruned_features = best_trial.user_attrs.get(
            "selected_features", feature_names.copy()
        )

        return pruned_features, study

    def prune_by_correlation(
        self, X: np.ndarray, feature_names: List[str]
    ) -> List[str]:
        """Remove highly correlated features.

        For each pair of highly correlated features (above threshold),
        removes the one that appears later in the list (assuming features
        are ordered by importance or preference).

        Args:
            X: Feature matrix (n_samples, n_features).
            feature_names: List of feature names.

        Returns:
            List of feature names after removing correlated features.
        """
        n_features = len(feature_names)

        if n_features <= 1:
            return feature_names.copy()

        # Compute correlation matrix
        correlation_matrix = np.corrcoef(X.T)

        # Handle NaN correlations (constant features)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

        # Track which features to keep
        features_to_keep = set(range(n_features))

        # Find highly correlated pairs
        for i in range(n_features):
            if i not in features_to_keep:
                continue

            for j in range(i + 1, n_features):
                if j not in features_to_keep:
                    continue

                if abs(correlation_matrix[i, j]) > self.correlation_threshold:
                    # Remove the feature with index j (later in the list)
                    features_to_keep.discard(j)
                    logger.debug(
                        f"Removing '{feature_names[j]}' due to high correlation "
                        f"({correlation_matrix[i, j]:.3f}) with '{feature_names[i]}'"
                    )

        # Ensure minimum features
        if len(features_to_keep) < self.min_features:
            remaining = sorted(set(range(n_features)) - features_to_keep)
            for idx in remaining:
                features_to_keep.add(idx)
                if len(features_to_keep) >= self.min_features:
                    break

        pruned_features = [feature_names[i] for i in sorted(features_to_keep)]

        logger.info(
            f"Correlation pruning: {len(pruned_features)}/{n_features} features kept "
            f"(threshold={self.correlation_threshold})"
        )

        return pruned_features

    def null_importance_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_fn: Callable,
        n_permutations: int = 20,
    ) -> List[str]:
        """Remove features with null importance using permutation test.

        A feature has null importance if shuffling it doesn't significantly
        reduce model performance (i.e., the model doesn't rely on it).

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels (n_samples,).
            feature_names: List of feature names.
            model_fn: Callable returning sklearn-compatible model.
            n_permutations: Number of permutations for the test.

        Returns:
            List of feature names that have significant importance.
        """
        n_features = len(feature_names)

        logger.info(
            f"Running null importance test with {n_permutations} permutations..."
        )

        try:
            # Train model on all features
            model = model_fn()
            model.fit(X, y)

            # Compute permutation importance
            result = permutation_importance(
                model,
                X,
                y,
                n_repeats=n_permutations,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

            importances_mean = result.importances_mean
            importances_std = result.importances_std

            # Features with importance > 2 standard deviations above 0 are significant
            significant_features = []
            for i, (mean, std) in enumerate(zip(importances_mean, importances_std)):
                # Feature is significant if mean importance > threshold
                # or if confidence interval doesn't include 0
                is_significant = mean > self.importance_threshold or (
                    mean - 2 * std > 0
                )
                if is_significant:
                    significant_features.append(feature_names[i])
                else:
                    logger.debug(
                        f"Feature '{feature_names[i]}' has null importance "
                        f"(mean={mean:.4f}, std={std:.4f})"
                    )

            # Ensure minimum features
            if len(significant_features) < self.min_features:
                # Add features by importance until minimum is met
                sorted_by_importance = sorted(
                    zip(feature_names, importances_mean),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for feat, _ in sorted_by_importance:
                    if feat not in significant_features:
                        significant_features.append(feat)
                    if len(significant_features) >= self.min_features:
                        break

            logger.info(
                f"Null importance test: {len(significant_features)}/{n_features} "
                f"features have significant importance"
            )

            return significant_features

        except Exception as e:
            logger.warning(f"Null importance test failed: {e}. Returning all features.")
            return feature_names.copy()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def prune_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_fn: Callable,
    n_trials: int = DEFAULT_FEATURE_PRUNING_TRIALS,
    strategy: str = "importance",
    min_features: int = DEFAULT_MIN_FEATURES,
    importance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    scoring: str = "f1_weighted",
    cv_folds: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_OPTUNA_RANDOM_STATE,
) -> FeaturePruningResult:
    """Convenience function for feature pruning.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target labels (n_samples,).
        feature_names: List of feature names.
        model_fn: Callable returning sklearn-compatible model.
        n_trials: Number of Optuna trials (for importance strategy).
        strategy: Pruning strategy ("importance", "correlation", "null").
        min_features: Minimum features to keep.
        importance_threshold: Threshold for importance-based pruning.
        correlation_threshold: Threshold for correlation-based pruning.
        scoring: Scoring metric for CV.
        cv_folds: Number of CV folds.
        random_state: Random seed.

    Returns:
        FeaturePruningResult with pruned features.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> result = prune_features(
        ...     X, y, feature_names,
        ...     model_fn=lambda: RandomForestClassifier(n_estimators=100),
        ...     strategy="importance",
        ...     n_trials=50,
        ... )
        >>> print(f"Removed {result.n_removed} features")
    """
    pruner = FeaturePruner(
        n_trials=n_trials,
        pruning_strategy=strategy,
        min_features=min_features,
        importance_threshold=importance_threshold,
        correlation_threshold=correlation_threshold,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    return pruner.prune_features(
        X=X,
        y=y,
        feature_names=feature_names,
        model_fn=model_fn,
        scoring=scoring,
    )


def prune_correlated_features(
    X: np.ndarray,
    feature_names: List[str],
    correlation_threshold: float = 0.95,
    min_features: int = DEFAULT_MIN_FEATURES,
) -> List[str]:
    """Convenience function for correlation-based pruning only.

    This is a fast method that doesn't require model training.

    Args:
        X: Feature matrix (n_samples, n_features).
        feature_names: List of feature names.
        correlation_threshold: Correlation threshold for removal.
        min_features: Minimum features to keep.

    Returns:
        List of feature names after removing highly correlated features.

    Example:
        >>> pruned = prune_correlated_features(X, feature_names, threshold=0.90)
        >>> print(f"Kept {len(pruned)} features")
    """
    pruner = FeaturePruner(
        pruning_strategy="correlation",
        correlation_threshold=correlation_threshold,
        min_features=min_features,
    )

    return pruner.prune_by_correlation(X, feature_names)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FeaturePruningResult",
    "FeaturePruner",
    "prune_features",
    "prune_correlated_features",
]
