"""
Feature Selection Module - Optuna-based feature selection for PHASE_1B.

This module provides intelligent feature selection using Optuna's hyperparameter
optimization framework. It supports multiple selection strategies and integrates
with sklearn-compatible models.

PHASE_1B: Optimized feature selection with binary categorical suggestions
for each feature, family-based selection, and importance-guided search.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

from src.core.constants import (
    DEFAULT_FEATURE_SELECTION_TRIALS,
    DEFAULT_MIN_FEATURES,
    DEFAULT_N_SPLITS,
    DEFAULT_OPTUNA_RANDOM_STATE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class FeatureSelectionResult:
    """Result of Optuna-based feature selection.

    Attributes:
        selected_features: List of selected feature names after optimization.
        all_features: List of all available feature names.
        n_selected: Number of features selected.
        n_total: Total number of available features.
        selection_ratio: Ratio of selected to total features.
        best_score: Best cross-validation score achieved.
        feature_importance: Dictionary mapping feature names to selection frequency
            in top trials (higher = more important).
        study: The Optuna study object for further analysis.
        baseline_score: Score with all features (for comparison).
        improvement: Relative improvement over baseline.
        trial_history: List of (n_features, score) tuples from all trials.
    """

    selected_features: list[str]
    all_features: list[str]
    n_selected: int
    n_total: int
    selection_ratio: float
    best_score: float
    feature_importance: dict[str, float]
    study: optuna.Study
    baseline_score: float = 0.0
    improvement: float = 0.0
    trial_history: list[tuple] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"FeatureSelectionResult("
            f"selected={self.n_selected}/{self.n_total}, "
            f"ratio={self.selection_ratio:.2%}, "
            f"best_score={self.best_score:.4f}, "
            f"improvement={self.improvement:+.2%})"
        )

    def get_top_features(self, n: int = 20) -> list[str]:
        """Get top N features by importance.

        Args:
            n: Number of top features to return.

        Returns:
            List of feature names sorted by importance.
        """
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_features[:n]]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "selected_features": self.selected_features,
            "all_features": self.all_features,
            "n_selected": self.n_selected,
            "n_total": self.n_total,
            "selection_ratio": self.selection_ratio,
            "best_score": self.best_score,
            "feature_importance": self.feature_importance,
            "baseline_score": self.baseline_score,
            "improvement": self.improvement,
            "n_trials": len(self.study.trials) if self.study else 0,
        }


# =============================================================================
# FEATURE SELECTOR
# =============================================================================


class FeatureSelector:
    """Optuna-based feature selection.

    Uses Optuna's TPE sampler to efficiently search the feature space and find
    an optimal subset of features for a given model.

    Strategies:
        - binary: Include/exclude each feature individually using categorical
          suggestions. Most flexible but slower for large feature sets.
        - family: Select entire feature families at once. Faster for datasets
          with clear family structure.
        - importance: Use pre-computed model importance to guide feature
          selection. Combines initial importance ranking with Optuna search.

    Args:
        n_trials: Number of Optuna trials to run.
        min_features: Minimum number of features to select.
        max_features: Maximum number of features to select.
        selection_strategy: Selection strategy ("binary", "family", "importance").
        cv_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.
        timeout: Optional timeout in seconds for optimization.
        n_jobs: Number of parallel jobs for cross-validation (-1 for all cores).

    Example:
        >>> selector = FeatureSelector(n_trials=100, selection_strategy="binary")
        >>> result = selector.select_features(
        ...     X=X_train,
        ...     y=y_train,
        ...     feature_names=feature_names,
        ...     model_fn=lambda: XGBClassifier(n_estimators=100),
        ...     scoring="f1_weighted",
        ... )
        >>> print(f"Selected {result.n_selected} features with score {result.best_score:.4f}")
    """

    VALID_STRATEGIES = ("binary", "family", "importance")

    def __init__(
        self,
        n_trials: int = DEFAULT_FEATURE_SELECTION_TRIALS,
        min_features: int = DEFAULT_MIN_FEATURES,
        max_features: int = 150,
        selection_strategy: str = "binary",
        cv_folds: int = DEFAULT_N_SPLITS,
        random_state: int = DEFAULT_OPTUNA_RANDOM_STATE,
        timeout: int | None = None,
        n_jobs: int = -1,
    ) -> None:
        if selection_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid selection_strategy: {selection_strategy}. "
                f"Must be one of {self.VALID_STRATEGIES}"
            )

        self.n_trials = n_trials
        self.min_features = min_features
        self.max_features = max_features
        self.selection_strategy = selection_strategy
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.timeout = timeout
        self.n_jobs = n_jobs

        # Internal state
        self._feature_names: list[str] = []
        self._feature_families: dict[str, list[int]] = {}
        self._initial_importance: dict[str, float] | None = None

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        model_fn: Callable,
        scoring: str = "f1_weighted",
        feature_families: dict[str, list[str]] | None = None,
        initial_importance: dict[str, float] | None = None,
        storage_path: str | None = None,
    ) -> FeatureSelectionResult:
        """Run feature selection optimization.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
            feature_names: List of feature names corresponding to columns of X.
            model_fn: Callable that returns a sklearn-compatible model instance.
                Must support fit(), predict(), and optionally predict_proba().
            scoring: Scoring metric for cross-validation (e.g., "f1_weighted",
                "accuracy", "roc_auc").
            feature_families: Optional mapping of family name to list of feature
                names. Required for "family" strategy.
            initial_importance: Optional pre-computed importance scores for
                "importance" strategy.
            storage_path: Optional path to SQLite database for study persistence.
                If provided, enables resuming interrupted studies. If None,
                uses in-memory storage.

        Returns:
            FeatureSelectionResult with selected features and metadata.

        Raises:
            ValueError: If inputs are invalid or incompatible.
        """
        # Validate inputs
        self._validate_inputs(X, y, feature_names, feature_families)

        self._feature_names = feature_names
        self._initial_importance = initial_importance
        n_features = len(feature_names)

        # Adjust max_features if needed
        effective_max = min(self.max_features, n_features)
        effective_min = min(self.min_features, n_features)

        # Skip optimization if we have fewer features than min_features
        if n_features <= effective_min:
            logger.info(
                f"Skipping selection: n_features ({n_features}) <= min_features ({effective_min})"
            )
            return self._create_trivial_result(X, y, feature_names, model_fn, scoring)

        # Build family index for family strategy
        if self.selection_strategy == "family" and feature_families:
            self._feature_families = self._build_family_index(feature_names, feature_families)

        # Compute baseline score with all features
        logger.info(f"Computing baseline score with {n_features} features...")
        baseline_score = self._compute_baseline_score(X, y, model_fn, scoring)
        logger.info(f"Baseline score: {baseline_score:.4f}")

        # Create Optuna study with MedianPruner for early stopping
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1,
        )

        # Set up storage for study persistence
        storage = None
        if storage_path:
            Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
            storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name="feature_selection",
            storage=storage,
            load_if_exists=True,
        )

        # Track trial history
        trial_history: list[tuple] = []

        def objective(trial: optuna.Trial) -> float:
            # Select features based on strategy
            if self.selection_strategy == "binary":
                selected_indices = self._binary_selection(
                    trial, n_features, effective_min, effective_max
                )
            elif self.selection_strategy == "family":
                selected_indices = self._family_selection(trial, effective_min, effective_max)
            else:  # importance
                selected_indices = self._importance_selection(
                    trial, n_features, effective_min, effective_max
                )

            # Ensure valid selection
            if len(selected_indices) < effective_min:
                return float("-inf")

            # Store selected indices
            trial.set_user_attr("selected_indices", sorted(selected_indices))

            # Create feature subset
            X_subset = X[:, selected_indices]

            # Evaluate with cross-validation
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

                # Report intermediate value for pruning
                mean_score = float(np.mean(scores))
                trial.report(mean_score, step=0)

                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Track history
                trial_history.append((len(selected_indices), mean_score))

                return mean_score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("-inf")

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info(
            f"Starting feature selection with {self.n_trials} trials, "
            f"strategy={self.selection_strategy}"
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
            catch=(Exception,),
        )

        # Get best trial results
        best_trial = study.best_trial
        best_indices = best_trial.user_attrs.get("selected_indices", list(range(n_features)))
        selected_features = [feature_names[i] for i in best_indices]

        # Compute feature importance based on selection frequency
        feature_importance = self._compute_selection_frequency(study, feature_names)

        # Calculate improvement
        improvement = (
            (study.best_value - baseline_score) / baseline_score if baseline_score > 0 else 0.0
        )

        logger.info(
            f"Selection complete: {len(selected_features)}/{n_features} features, "
            f"score={study.best_value:.4f}, improvement={improvement:+.2%}"
        )

        return FeatureSelectionResult(
            selected_features=selected_features,
            all_features=feature_names,
            n_selected=len(selected_features),
            n_total=n_features,
            selection_ratio=len(selected_features) / n_features,
            best_score=study.best_value,
            feature_importance=feature_importance,
            study=study,
            baseline_score=baseline_score,
            improvement=improvement,
            trial_history=trial_history,
        )

    def _validate_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        feature_families: dict[str, list[str]] | None,
    ) -> None:
        """Validate input arguments."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if len(y) != X.shape[0]:
            raise ValueError(f"X and y size mismatch: X has {X.shape[0]} samples, y has {len(y)}")

        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) != X columns ({X.shape[1]})"
            )

        if self.selection_strategy == "family" and not feature_families:
            raise ValueError("feature_families required for 'family' selection strategy")

    def _compute_baseline_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fn: Callable,
        scoring: str,
    ) -> float:
        """Compute baseline score with all features."""
        try:
            model = model_fn()
            scores = cross_val_score(
                model, X, y, cv=self.cv_folds, scoring=scoring, n_jobs=self.n_jobs
            )
            return float(np.mean(scores))
        except Exception as e:
            logger.warning(f"Failed to compute baseline: {e}")
            return 0.0

    def _create_trivial_result(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        model_fn: Callable,
        scoring: str,
    ) -> FeatureSelectionResult:
        """Create result when no selection is needed."""
        score = self._compute_baseline_score(X, y, model_fn, scoring)

        # Create empty study for consistency
        study = optuna.create_study(direction="maximize")

        return FeatureSelectionResult(
            selected_features=feature_names.copy(),
            all_features=feature_names.copy(),
            n_selected=len(feature_names),
            n_total=len(feature_names),
            selection_ratio=1.0,
            best_score=score,
            feature_importance={f: 1.0 for f in feature_names},
            study=study,
            baseline_score=score,
            improvement=0.0,
            trial_history=[],
        )

    def _build_family_index(
        self,
        feature_names: list[str],
        feature_families: dict[str, list[str]],
    ) -> dict[str, list[int]]:
        """Build mapping of family name to feature indices."""
        family_index: dict[str, list[int]] = {}
        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        for family, members in feature_families.items():
            indices = [name_to_idx[m] for m in members if m in name_to_idx]
            if indices:
                family_index[family] = indices

        return family_index

    def _binary_selection(
        self,
        trial: optuna.Trial,
        n_features: int,
        min_features: int,
        max_features: int,
    ) -> list[int]:
        """Binary feature selection - include/exclude each feature."""
        selected_indices: list[int] = []

        for i in range(n_features):
            if trial.suggest_categorical(f"include_{i}", [True, False]):
                selected_indices.append(i)

        # Ensure minimum features
        if len(selected_indices) < min_features:
            unselected = [i for i in range(n_features) if i not in selected_indices]
            np.random.seed(self.random_state + trial.number)
            needed = min_features - len(selected_indices)
            if unselected:
                extra = list(
                    np.random.choice(unselected, size=min(needed, len(unselected)), replace=False)
                )
                selected_indices.extend(extra)

        # Enforce maximum features
        if len(selected_indices) > max_features:
            np.random.seed(self.random_state + trial.number)
            selected_indices = list(
                np.random.choice(selected_indices, size=max_features, replace=False)
            )

        return selected_indices

    def _family_selection(
        self,
        trial: optuna.Trial,
        min_features: int,
        max_features: int,
    ) -> list[int]:
        """Family-based feature selection - select entire families."""
        selected_indices: list[int] = []

        for family, indices in self._feature_families.items():
            if trial.suggest_categorical(f"include_family_{family}", [True, False]):
                selected_indices.extend(indices)

        # Ensure minimum features by adding random families
        families_added = set()
        while len(selected_indices) < min_features and len(families_added) < len(
            self._feature_families
        ):
            for family, indices in self._feature_families.items():
                if family not in families_added and not any(i in selected_indices for i in indices):
                    selected_indices.extend(indices)
                    families_added.add(family)
                    if len(selected_indices) >= min_features:
                        break

        # Enforce maximum by removing random features
        if len(selected_indices) > max_features:
            np.random.seed(self.random_state + trial.number)
            selected_indices = list(
                np.random.choice(selected_indices, size=max_features, replace=False)
            )

        return selected_indices

    def _importance_selection(
        self,
        trial: optuna.Trial,
        n_features: int,
        min_features: int,
        max_features: int,
    ) -> list[int]:
        """Importance-guided feature selection."""
        if self._initial_importance is None:
            # Fall back to binary if no importance available
            return self._binary_selection(trial, n_features, min_features, max_features)

        # Sort features by importance
        sorted_features = sorted(self._initial_importance.items(), key=lambda x: x[1], reverse=True)

        name_to_idx = {name: i for i, name in enumerate(self._feature_names)}

        # Suggest number of features to include
        n_selected = trial.suggest_int("n_features", min_features, max_features)

        # Use importance to guide selection with some randomness
        selected_indices: list[int] = []

        for feat_name, importance in sorted_features:
            if feat_name not in name_to_idx:
                continue

            idx = name_to_idx[feat_name]

            # Higher importance features have higher probability of inclusion
            threshold = trial.suggest_float(f"thresh_{idx}", 0.0, 1.0)
            normalized_importance = importance / (sorted_features[0][1] + 1e-10)

            if normalized_importance >= threshold:
                selected_indices.append(idx)

            if len(selected_indices) >= n_selected:
                break

        # Ensure minimum
        if len(selected_indices) < min_features:
            remaining = [
                name_to_idx[f]
                for f, _ in sorted_features
                if f in name_to_idx and name_to_idx[f] not in selected_indices
            ]
            needed = min_features - len(selected_indices)
            selected_indices.extend(remaining[:needed])

        return selected_indices

    def _compute_selection_frequency(
        self, study: optuna.Study, feature_names: list[str]
    ) -> dict[str, float]:
        """Compute feature importance based on selection frequency in top trials.

        Features that appear more frequently in high-scoring trials are
        considered more important.

        Args:
            study: Completed Optuna study.
            feature_names: List of all feature names.

        Returns:
            Dictionary mapping feature names to importance scores (0-1).
        """
        n_features = len(feature_names)
        selection_counts = np.zeros(n_features)
        weighted_counts = np.zeros(n_features)

        # Get completed trials sorted by value (best first)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if not completed_trials:
            return {name: 0.0 for name in feature_names}

        completed_trials.sort(key=lambda t: t.value or 0, reverse=True)

        # Consider top 30% of trials for importance calculation
        n_top = max(1, len(completed_trials) // 3)
        top_trials = completed_trials[:n_top]

        best_value = top_trials[0].value or 1.0

        for trial in top_trials:
            indices = trial.user_attrs.get("selected_indices", [])
            weight = (trial.value or 0) / (best_value + 1e-10)

            for idx in indices:
                if idx < n_features:
                    selection_counts[idx] += 1
                    weighted_counts[idx] += weight

        # Normalize to 0-1 range
        max_weighted = weighted_counts.max() if weighted_counts.max() > 0 else 1.0
        importance_scores = weighted_counts / max_weighted

        return {feature_names[i]: float(importance_scores[i]) for i in range(n_features)}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_fn: Callable,
    n_trials: int = DEFAULT_FEATURE_SELECTION_TRIALS,
    min_features: int = DEFAULT_MIN_FEATURES,
    max_features: int = 150,
    strategy: str = "binary",
    scoring: str = "f1_weighted",
    cv_folds: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_OPTUNA_RANDOM_STATE,
) -> FeatureSelectionResult:
    """Convenience function for feature selection.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target labels (n_samples,).
        feature_names: List of feature names.
        model_fn: Callable returning sklearn-compatible model.
        n_trials: Number of Optuna trials.
        min_features: Minimum features to select.
        max_features: Maximum features to select.
        strategy: Selection strategy ("binary", "family", "importance").
        scoring: Scoring metric for CV.
        cv_folds: Number of CV folds.
        random_state: Random seed.

    Returns:
        FeatureSelectionResult with selected features.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> result = select_features(
        ...     X, y, feature_names,
        ...     model_fn=lambda: RandomForestClassifier(n_estimators=100),
        ...     n_trials=50,
        ...     scoring="f1_weighted",
        ... )
    """
    selector = FeatureSelector(
        n_trials=n_trials,
        min_features=min_features,
        max_features=max_features,
        selection_strategy=strategy,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    return selector.select_features(
        X=X,
        y=y,
        feature_names=feature_names,
        model_fn=model_fn,
        scoring=scoring,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FeatureSelectionResult",
    "FeatureSelector",
    "select_features",
]
