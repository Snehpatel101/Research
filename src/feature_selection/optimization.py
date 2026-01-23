"""
Feature subset optimization using Optuna.

Provides automated feature selection through hyperparameter optimization,
finding optimal feature subsets for given model and evaluation metric.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

from .result import FeatureSelectionResult

logger = logging.getLogger(__name__)


class FeatureOptimizer:
    """
    Optimize feature subsets using Optuna.

    Finds optimal feature subset by treating feature selection as
    hyperparameter optimization problem.
    """

    def __init__(
        self,
        n_trials: int = 50,
        metric: str = "f1",
        direction: str = "maximize",
        timeout: int | None = None,
        random_state: int = 42,
    ):
        """
        Initialize FeatureOptimizer.

        Args:
            n_trials: Number of optimization trials
            metric: Metric to optimize ("f1", "accuracy", "auc")
            direction: "maximize" or "minimize"
            timeout: Timeout in seconds (None = no timeout)
            random_state: Random seed for reproducibility
        """
        self.n_trials = n_trials
        self.metric = metric
        self.direction = direction
        self.timeout = timeout
        self.random_state = random_state

        self.study: optuna.Study | None = None
        self.best_features: list[str] | None = None
        self.best_score: float | None = None

    def optimize(
        self,
        feature_names: list[str],
        evaluation_fn: Callable[[list[str]], float],
        min_features: int = 5,
        max_features: int | None = None,
    ) -> FeatureSelectionResult:
        """
        Optimize feature subset.

        Args:
            feature_names: List of all available feature names
            evaluation_fn: Function that takes feature subset and returns metric score
            min_features: Minimum number of features to select
            max_features: Maximum number of features (None = len(feature_names))

        Returns:
            FeatureSelectionResult with optimized features
        """
        if max_features is None:
            max_features = len(feature_names)

        max_features = min(max_features, len(feature_names))
        min_features = max(1, min(min_features, max_features))

        logger.info(
            f"Starting feature optimization: {len(feature_names)} features, "
            f"range=[{min_features}, {max_features}], metric={self.metric}, "
            f"trials={self.n_trials}"
        )

        def objective(trial: optuna.Trial) -> float:
            n_features_to_select = trial.suggest_int("n_features", min_features, max_features)

            selected_indices = []
            for i in range(n_features_to_select):
                idx = trial.suggest_int(f"feature_{i}", 0, len(feature_names) - 1)
                if idx not in selected_indices:
                    selected_indices.append(idx)

            if len(selected_indices) == 0:
                return 0.0

            selected_features = [feature_names[i] for i in sorted(set(selected_indices))]

            try:
                score = evaluation_fn(selected_features)
                return score
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return 0.0

        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
        )

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        best_trial = self.study.best_trial
        n_features_selected = best_trial.params["n_features"]

        selected_indices = []
        for i in range(n_features_selected):
            idx = best_trial.params.get(f"feature_{i}")
            if idx is not None and idx not in selected_indices:
                selected_indices.append(idx)

        self.best_features = [feature_names[i] for i in sorted(set(selected_indices))]
        self.best_score = best_trial.value

        logger.info(
            f"Optimization complete: {len(self.best_features)} features selected, "
            f"{self.metric}={self.best_score:.4f}"
        )

        feature_importances = {
            feature: 1.0 if feature in self.best_features else 0.0 for feature in feature_names
        }

        return FeatureSelectionResult(
            selected_features=self.best_features,
            feature_importances=feature_importances,
            original_count=len(feature_names),
            final_count=len(self.best_features),
            selection_metadata={
                "selection_method": "optuna",
                "n_trials": self.n_trials,
                "best_trial": best_trial.number,
                "best_score": self.best_score,
                "metric": self.metric,
            },
        )

    def get_optimization_history(self) -> dict[str, Any]:
        """Get optimization history and statistics."""
        if self.study is None:
            return {}

        trials_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials_data.append(
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "n_features": trial.params.get("n_features", 0),
                    }
                )

        return {
            "n_trials": len(self.study.trials),
            "n_completed": len(
                [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "best_trial": self.study.best_trial.number if self.study.best_trial else None,
            "best_value": self.study.best_value if self.study.best_trial else None,
            "trials": trials_data,
        }


def optimize_feature_subset_simple(
    feature_names: list[str],
    feature_importances: dict[str, float],
    target_n_features: int,
    n_trials: int = 30,
) -> list[str]:
    """
    Simple feature optimization using importance scores.

    Args:
        feature_names: All feature names
        feature_importances: Dict of feature name -> importance score
        target_n_features: Target number of features
        n_trials: Number of optimization trials

    Returns:
        List of selected feature names
    """
    sorted_features = sorted(
        feature_names, key=lambda f: feature_importances.get(f, 0.0), reverse=True
    )

    if target_n_features >= len(sorted_features):
        return sorted_features

    top_features = sorted_features[:target_n_features]

    n_variations = min(n_trials, target_n_features // 2)

    if n_variations > 0:
        candidate_pool = sorted_features[target_n_features : target_n_features + n_variations]
        if candidate_pool:
            replace_idx = np.random.randint(0, min(n_variations, len(top_features)))
            top_features[replace_idx] = candidate_pool[0]

    return top_features
