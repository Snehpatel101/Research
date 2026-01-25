"""
Feature Optimization - Selection and Pruning with Optuna.

This module provides two complementary feature optimization approaches:

1. Feature Selection: Binary include/exclude optimization
   - Each feature gets a boolean flag for inclusion
   - Optuna searches the combinatorial space efficiently

2. Feature Pruning: Importance-based progressive removal
   - Starts with all features, removes least important
   - Finds optimal feature count via Optuna

Key Components:
    FeatureSelectionResult: Result from binary feature selection
    FeaturePruningResult: Result from importance-based pruning
    FeatureOptimizer: Unified optimizer for both methods

Integration:
    Works with PipelineConfig settings:
    - optimize_features: Enable/disable feature optimization
    - feature_selection_trials: Trials for selection phase
    - feature_pruning_trials: Trials for pruning phase
    - min_features: Minimum features to retain

Usage:
    from src.optimization.features import FeatureOptimizer

    optimizer = FeatureOptimizer(
        selection_trials=100,
        pruning_trials=50,
        min_features=20,
    )

    # Binary selection
    selection_result = optimizer.optimize_selection(
        X_train, y_train, feature_names, model_factory
    )

    # Importance-based pruning
    pruning_result = optimizer.optimize_pruning(
        X_train, y_train, feature_names, model_factory
    )

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score

from src.core import (
    DEFAULT_FEATURE_PRUNING_TRIALS,
    DEFAULT_FEATURE_SELECTION_TRIALS,
    DEFAULT_MIN_FEATURES,
    DEFAULT_OPTUNA_RANDOM_STATE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class FeatureSelectionResult:
    """
    Result container for binary feature selection optimization.

    Attributes:
        selected_features: List of selected feature names
        selection_mask: Boolean mask for feature selection
        n_original: Original number of features
        n_selected: Number of selected features
        best_score: Best validation score achieved
        baseline_score: Score with all features
        improvement: Relative improvement over baseline
        n_trials: Number of optimization trials
        study: Optuna Study object
        optimization_time: Time taken in seconds
        feature_inclusion_rates: Dict of feature -> selection frequency across trials
    """

    selected_features: list[str]
    selection_mask: np.ndarray
    n_original: int
    n_selected: int
    best_score: float
    baseline_score: float
    improvement: float
    n_trials: int
    study: optuna.Study
    optimization_time: float = 0.0
    feature_inclusion_rates: dict[str, float] = field(default_factory=dict)

    @property
    def reduction_ratio(self) -> float:
        """Proportion of features removed."""
        if self.n_original == 0:
            return 0.0
        return 1 - (self.n_selected / self.n_original)

    def __repr__(self) -> str:
        return (
            f"FeatureSelectionResult("
            f"n_original={self.n_original}, "
            f"n_selected={self.n_selected}, "
            f"reduction={self.reduction_ratio:.1%}, "
            f"improvement={self.improvement:+.2%})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "selected_features": self.selected_features,
            "n_original": self.n_original,
            "n_selected": self.n_selected,
            "best_score": self.best_score,
            "baseline_score": self.baseline_score,
            "improvement": self.improvement,
            "n_trials": self.n_trials,
            "optimization_time": self.optimization_time,
            "feature_inclusion_rates": self.feature_inclusion_rates,
        }


@dataclass
class FeaturePruningResult:
    """
    Result container for importance-based feature pruning.

    Attributes:
        selected_features: List of selected feature names (after pruning)
        feature_importances: Dict of feature -> importance score
        pruning_order: Order in which features were pruned
        n_original: Original number of features
        n_selected: Number of remaining features
        optimal_n_features: Optimal feature count found
        best_score: Best validation score achieved
        baseline_score: Score with all features
        improvement: Relative improvement over baseline
        n_trials: Number of optimization trials
        study: Optuna Study object
        optimization_time: Time taken in seconds
    """

    selected_features: list[str]
    feature_importances: dict[str, float]
    pruning_order: list[str]
    n_original: int
    n_selected: int
    optimal_n_features: int
    best_score: float
    baseline_score: float
    improvement: float
    n_trials: int
    study: optuna.Study
    optimization_time: float = 0.0

    @property
    def reduction_ratio(self) -> float:
        """Proportion of features pruned."""
        if self.n_original == 0:
            return 0.0
        return 1 - (self.n_selected / self.n_original)

    def __repr__(self) -> str:
        return (
            f"FeaturePruningResult("
            f"n_original={self.n_original}, "
            f"n_selected={self.n_selected}, "
            f"reduction={self.reduction_ratio:.1%}, "
            f"improvement={self.improvement:+.2%})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "selected_features": self.selected_features,
            "feature_importances": self.feature_importances,
            "pruning_order": self.pruning_order,
            "n_original": self.n_original,
            "n_selected": self.n_selected,
            "optimal_n_features": self.optimal_n_features,
            "best_score": self.best_score,
            "baseline_score": self.baseline_score,
            "improvement": self.improvement,
            "n_trials": self.n_trials,
            "optimization_time": self.optimization_time,
        }


# =============================================================================
# FEATURE OPTIMIZER
# =============================================================================


class FeatureOptimizer:
    """
    Unified Optuna-based feature optimizer.

    Provides two optimization approaches:
    1. Selection: Binary include/exclude for each feature
    2. Pruning: Importance-based progressive feature removal

    Args:
        selection_trials: Trials for selection optimization (default: 100)
        pruning_trials: Trials for pruning optimization (default: 50)
        min_features: Minimum features to retain (default: 20)
        scoring: Scoring metric ("f1_weighted", "accuracy")
        random_state: Random seed for reproducibility
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)

    Example:
        >>> optimizer = FeatureOptimizer(selection_trials=100)
        >>> result = optimizer.optimize_selection(
        ...     X_train, y_train, feature_names, model_factory
        ... )
        >>> X_selected = X_train[:, result.selection_mask]
    """

    def __init__(
        self,
        selection_trials: int = DEFAULT_FEATURE_SELECTION_TRIALS,
        pruning_trials: int = DEFAULT_FEATURE_PRUNING_TRIALS,
        min_features: int = DEFAULT_MIN_FEATURES,
        scoring: str = "f1_weighted",
        random_state: int = DEFAULT_OPTUNA_RANDOM_STATE,
        verbose: int = 1,
    ):
        self.selection_trials = selection_trials
        self.pruning_trials = pruning_trials
        self.min_features = min_features
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose

        # Configure Optuna logging
        if verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)

    def _get_score_fn(self) -> Callable[[np.ndarray, np.ndarray], float]:
        """Get scoring function based on metric name."""
        if self.scoring == "accuracy":
            from sklearn.metrics import accuracy_score

            def acc_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return float(accuracy_score(y_true, y_pred))

            return acc_fn
        elif self.scoring == "f1_weighted":
            return lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="weighted"))
        elif self.scoring == "f1_macro":
            return lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="macro"))
        else:
            raise ValueError(f"Unknown scoring metric: {self.scoring}")

    def _compute_baseline_score(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_factory: Callable,
        score_fn: Callable,
    ) -> float:
        """Compute baseline score with all features."""
        try:
            model = model_factory({})
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return float(score_fn(y_val, preds))
        except Exception as e:
            logger.warning(f"Baseline score computation failed: {e}")
            return 0.0

    def optimize_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        model_factory: Callable[[dict[str, Any]], Any],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FeatureSelectionResult:
        """
        Optimize feature selection using binary include/exclude flags.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            feature_names: List of feature names
            model_factory: Factory function that creates model
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            FeatureSelectionResult with selected features

        Note:
            For large feature sets (>100), this method may be slow.
            Consider using optimize_pruning for better scalability.
        """
        start_time = time.time()
        n_features = len(feature_names)

        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print("Optimizing feature selection (binary)")
            print(f"  n_features={n_features}, n_trials={self.selection_trials}")
            print(f"  min_features={self.min_features}")
            print(f"{'='*60}")

        # Split data if no validation set provided
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_train,
            )

        score_fn = self._get_score_fn()

        # Compute baseline score
        baseline_score = self._compute_baseline_score(
            X_train, y_train, X_val, y_val, model_factory, score_fn
        )

        # Track feature inclusion across trials
        feature_inclusion_counts = dict.fromkeys(feature_names, 0)
        n_completed_trials = 0

        def objective(trial: optuna.Trial) -> float:
            nonlocal n_completed_trials

            # Suggest inclusion for each feature
            selected_indices = []
            for i, _fname in enumerate(feature_names):
                include = trial.suggest_categorical(f"include_{i}", [True, False])
                if include:
                    selected_indices.append(i)

            # Ensure minimum features
            if len(selected_indices) < self.min_features:
                # Add random features to meet minimum
                remaining = [i for i in range(n_features) if i not in selected_indices]
                np.random.seed(self.random_state + trial.number)
                np.random.shuffle(remaining)
                needed = self.min_features - len(selected_indices)
                selected_indices.extend(remaining[:needed])

            if len(selected_indices) == 0:
                return 0.0

            # Store selection
            trial.set_user_attr("selected_indices", sorted(selected_indices))

            # Create subset
            X_train_subset = X_train[:, selected_indices]
            X_val_subset = X_val[:, selected_indices]

            # Train and evaluate
            try:
                model = model_factory({})
                model.fit(X_train_subset, y_train)
                preds = model.predict(X_val_subset)
                score = score_fn(y_val, preds)

                # Track inclusion
                for idx in selected_indices:
                    feature_inclusion_counts[feature_names[idx]] += 1
                n_completed_trials += 1

                return score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return 0.0

        # Create study
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.selection_trials,
            show_progress_bar=(self.verbose >= 1),
        )

        optimization_time = time.time() - start_time

        # Get best trial
        best_trial = study.best_trial
        best_indices = best_trial.user_attrs.get("selected_indices", list(range(n_features)))
        selected_features = [feature_names[i] for i in best_indices]

        # Create selection mask
        selection_mask = np.zeros(n_features, dtype=bool)
        selection_mask[best_indices] = True

        # Compute feature inclusion rates
        feature_inclusion_rates = {}
        if n_completed_trials > 0:
            feature_inclusion_rates = {
                f: count / n_completed_trials for f, count in feature_inclusion_counts.items()
            }

        # Compute improvement
        best_score = study.best_value
        improvement = (best_score - baseline_score) / baseline_score if baseline_score > 0 else 0.0

        if self.verbose >= 1:
            print("\nSelection optimization complete:")
            print(f"  Selected {len(selected_features)}/{n_features} features")
            print(f"  Baseline score: {baseline_score:.4f}")
            print(f"  Best score: {best_score:.4f}")
            print(f"  Improvement: {improvement:+.2%}")
            print(f"  Time: {optimization_time:.1f}s")

        return FeatureSelectionResult(
            selected_features=selected_features,
            selection_mask=selection_mask,
            n_original=n_features,
            n_selected=len(selected_features),
            best_score=best_score,
            baseline_score=baseline_score,
            improvement=improvement,
            n_trials=len(study.trials),
            study=study,
            optimization_time=optimization_time,
            feature_inclusion_rates=feature_inclusion_rates,
        )

    def optimize_pruning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        model_factory: Callable[[dict[str, Any]], Any],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_importances: dict[str, float] | None = None,
    ) -> FeaturePruningResult:
        """
        Optimize feature count using importance-based pruning.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            feature_names: List of feature names
            model_factory: Factory function that creates model
            X_val: Optional validation features
            y_val: Optional validation labels
            feature_importances: Optional pre-computed importances

        Returns:
            FeaturePruningResult with pruned features

        Note:
            This method is more efficient than selection for large feature sets.
            It finds the optimal number of top-K features to keep.
        """
        start_time = time.time()
        n_features = len(feature_names)

        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print("Optimizing feature pruning (importance-based)")
            print(f"  n_features={n_features}, n_trials={self.pruning_trials}")
            print(f"  min_features={self.min_features}")
            print(f"{'='*60}")

        # Split data if no validation set provided
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_train,
            )

        score_fn = self._get_score_fn()

        # Compute baseline score
        baseline_score = self._compute_baseline_score(
            X_train, y_train, X_val, y_val, model_factory, score_fn
        )

        # Compute feature importances if not provided
        if feature_importances is None:
            if self.verbose >= 1:
                print("  Computing feature importances...")
            feature_importances = self._compute_feature_importances(
                X_train, y_train, feature_names, model_factory
            )

        # Sort features by importance (descending)
        sorted_features = sorted(
            feature_names,
            key=lambda f: feature_importances.get(f, 0),
            reverse=True,
        )

        # Pruning order (least important first)
        pruning_order = sorted_features[::-1]

        def objective(trial: optuna.Trial) -> float:
            # Suggest number of features to keep
            n_keep = trial.suggest_int(
                "n_features",
                self.min_features,
                n_features,
            )

            # Keep top-K features by importance
            selected_features = sorted_features[:n_keep]
            selected_indices = [feature_names.index(f) for f in selected_features]

            trial.set_user_attr("n_features", n_keep)
            trial.set_user_attr("selected_features", selected_features)

            # Create subset
            X_train_subset = X_train[:, selected_indices]
            X_val_subset = X_val[:, selected_indices]

            # Train and evaluate
            try:
                model = model_factory({})
                model.fit(X_train_subset, y_train)
                preds = model.predict(X_val_subset)
                return score_fn(y_val, preds)

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return 0.0

        # Create study
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.pruning_trials,
            show_progress_bar=(self.verbose >= 1),
        )

        optimization_time = time.time() - start_time

        # Get best trial
        best_trial = study.best_trial
        optimal_n_features = best_trial.params.get("n_features", n_features)
        selected_features = best_trial.user_attrs.get(
            "selected_features", sorted_features[:optimal_n_features]
        )

        # Compute improvement
        best_score = study.best_value
        improvement = (best_score - baseline_score) / baseline_score if baseline_score > 0 else 0.0

        if self.verbose >= 1:
            print("\nPruning optimization complete:")
            print(f"  Optimal features: {len(selected_features)}/{n_features}")
            print(f"  Baseline score: {baseline_score:.4f}")
            print(f"  Best score: {best_score:.4f}")
            print(f"  Improvement: {improvement:+.2%}")
            print(f"  Time: {optimization_time:.1f}s")

        return FeaturePruningResult(
            selected_features=selected_features,
            feature_importances=feature_importances,
            pruning_order=pruning_order,
            n_original=n_features,
            n_selected=len(selected_features),
            optimal_n_features=optimal_n_features,
            best_score=best_score,
            baseline_score=baseline_score,
            improvement=improvement,
            n_trials=len(study.trials),
            study=study,
            optimization_time=optimization_time,
        )

    def _compute_feature_importances(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        model_factory: Callable,
    ) -> dict[str, float]:
        """
        Compute feature importances using permutation importance.

        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Feature names
            model_factory: Model factory function

        Returns:
            Dict mapping feature name to importance score
        """
        from sklearn.model_selection import train_test_split

        # Split for importance calculation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_train,
        )

        # Train model
        try:
            model = model_factory({})
            model.fit(X_tr, y_tr)
        except Exception as e:
            logger.warning(f"Model training for importances failed: {e}")
            # Return uniform importances
            return {f: 1.0 / len(feature_names) for f in feature_names}

        # Try to get built-in feature importances first
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}

        # Fall back to permutation importance
        try:
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                model,
                X_val,
                y_val,
                n_repeats=5,
                random_state=self.random_state,
                scoring="f1_weighted",
            )

            return {
                feature_names[i]: max(0, float(result.importances_mean[i]))
                for i in range(len(feature_names))
            }

        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
            # Return uniform importances
            return {f: 1.0 / len(feature_names) for f in feature_names}

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        model_factory: Callable[[dict[str, Any]], Any],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        method: str = "both",
    ) -> tuple[FeatureSelectionResult, FeaturePruningResult]:
        """
        Run both selection and pruning optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Feature names
            model_factory: Model factory function
            X_val: Optional validation features
            y_val: Optional validation labels
            method: "selection", "pruning", or "both"

        Returns:
            Tuple of (FeatureSelectionResult, FeaturePruningResult)
        """
        if method not in ["selection", "pruning", "both"]:
            raise ValueError(f"Invalid method: {method}. Must be 'selection', 'pruning', or 'both'")

        selection_result: FeatureSelectionResult | None = None
        pruning_result: FeaturePruningResult | None = None

        if method in ["selection", "both"]:
            selection_result = self.optimize_selection(
                X_train, y_train, feature_names, model_factory, X_val, y_val
            )

        if method in ["pruning", "both"]:
            pruning_result = self.optimize_pruning(
                X_train, y_train, feature_names, model_factory, X_val, y_val
            )

        if selection_result is None or pruning_result is None:
            raise ValueError(f"Method '{method}' requires both selection and pruning to be run")

        return selection_result, pruning_result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FeatureSelectionResult",
    "FeaturePruningResult",
    "FeatureOptimizer",
]
