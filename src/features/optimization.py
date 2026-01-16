from dataclasses import dataclass, field
from typing import Any

import optuna
import pandas as pd
import numpy as np

from .strategies import get_strategy_for_model


@dataclass
class OptimizationResult:
    """Result of feature optimization with Optuna."""

    model_name: str
    original_features: list[str]
    optimized_features: list[str]
    n_original: int
    n_optimized: int
    improvement: float
    best_trial_score: float
    n_trials: int
    study: optuna.Study | None = None
    # Keep backward compatibility aliases
    baseline_features: list[str] = field(default_factory=list, repr=False)
    best_score: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        # Set aliases for backward compatibility
        if not self.baseline_features:
            self.baseline_features = self.original_features.copy()
        if self.best_score == 0.0:
            self.best_score = self.best_trial_score

    def __repr__(self) -> str:
        return (
            f"OptimizationResult({self.model_name}, "
            f"original={self.n_original}, "
            f"optimized={self.n_optimized}, "
            f"improvement={self.improvement:+.2%}, "
            f"score={self.best_trial_score:.4f})"
        )


def optimize_features_for_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    cv_splits: int = 3,
) -> OptimizationResult:
    """Optimize features for a model using Optuna hyperparameter search.

    Args:
        model_name: Name of the model to optimize for.
        X_train: Training features DataFrame.
        y_train: Training labels Series.
        X_val: Validation features DataFrame.
        y_val: Validation labels Series.
        n_trials: Number of Optuna trials.
        cv_splits: Unused, kept for backward compatibility.

    Returns:
        OptimizationResult with the optimized feature set.
    """
    from src.models.registry import ModelRegistry
    from sklearn.metrics import f1_score

    strategy = get_strategy_for_model(model_name)
    baseline_features = strategy.baseline_features

    available_features = [f for f in baseline_features if f in X_train.columns]

    if len(available_features) < strategy.min_features:
        return OptimizationResult(
            model_name=model_name,
            original_features=baseline_features,
            optimized_features=available_features,
            n_original=len(baseline_features),
            n_optimized=len(available_features),
            improvement=0.0,
            best_trial_score=0.0,
            n_trials=0,
            study=None,
        )

    # Compute baseline score for improvement calculation
    baseline_score = 0.0
    try:
        baseline_model = ModelRegistry.create(model_name)
        X_train_baseline = X_train[available_features]
        X_val_baseline = X_val[available_features]
        baseline_model.fit(
            X_train=X_train_baseline.values,
            y_train=y_train.values,
            X_val=X_val_baseline.values,
            y_val=y_val.values,
        )
        baseline_preds = baseline_model.predict(X_val_baseline.values)
        baseline_score = f1_score(
            y_val.values, baseline_preds.class_predictions, average="macro"
        )
    except Exception:
        baseline_score = 0.0

    def objective(trial: optuna.Trial) -> float:
        # Select features using boolean flags for each feature
        selected_indices = []
        for i, feat in enumerate(available_features):
            if trial.suggest_categorical(f"include_{i}", [True, False]):
                selected_indices.append(i)

        # Ensure we have at least min_features
        if len(selected_indices) < strategy.min_features:
            # Add random features to meet minimum
            remaining = [i for i in range(len(available_features)) if i not in selected_indices]
            np.random.shuffle(remaining)
            selected_indices.extend(remaining[: strategy.min_features - len(selected_indices)])

        # Cap at max_features
        if len(selected_indices) > strategy.max_features:
            selected_indices = selected_indices[: strategy.max_features]

        if not selected_indices:
            return 0.0

        selected_features = [available_features[i] for i in selected_indices]
        trial.set_user_attr("selected_indices", selected_indices)

        X_train_subset = X_train[selected_features]
        X_val_subset = X_val[selected_features]

        model = ModelRegistry.create(model_name)
        model.fit(
            X_train=X_train_subset.values,
            y_train=y_train.values,
            X_val=X_val_subset.values,
            y_val=y_val.values,
        )

        predictions = model.predict(X_val_subset.values)
        score = f1_score(y_val.values, predictions.class_predictions, average="macro")

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_indices = best_params.get("feature_indices", [])
    if isinstance(best_indices, int):
        best_indices = [best_indices]

    optimized_features = [available_features[i] for i in best_indices]

    # Calculate improvement
    improvement = (
        (study.best_value - baseline_score) / baseline_score
        if baseline_score > 0
        else 0.0
    )

    return OptimizationResult(
        model_name=model_name,
        original_features=available_features,
        optimized_features=optimized_features,
        n_original=len(available_features),
        n_optimized=len(optimized_features),
        improvement=improvement,
        best_trial_score=study.best_value,
        n_trials=n_trials,
        study=study,
    )


def suggest_features(
    trial: optuna.Trial,
    model_name: str,
    available_features: list[str],
) -> list[str]:
    strategy = get_strategy_for_model(model_name)

    candidate_features = [f for f in strategy.baseline_features if f in available_features]

    if not candidate_features:
        return available_features

    n_features = trial.suggest_int(
        "n_features",
        min(strategy.min_features, len(candidate_features)),
        min(strategy.max_features, len(candidate_features)),
    )

    feature_indices = []
    for i in range(n_features):
        idx = trial.suggest_categorical(
            f"feature_{i}",
            list(range(len(candidate_features))),
        )
        if idx not in feature_indices:
            feature_indices.append(idx)

    return [candidate_features[i] for i in feature_indices]


class FeatureOptimizer:
    """Optuna-based feature optimizer that prunes from baseline to optimal subset.

    Uses TPE sampler to efficiently search the feature space and find an optimal
    subset of features for the given model.

    Args:
        model_name: Name of the model to optimize features for.
        n_trials: Number of Optuna trials to run.
        metric: Evaluation metric ('f1_weighted' or 'accuracy').
        min_features: Minimum number of features to select. If None, uses strategy default.
        random_seed: Random seed for reproducibility.

    Example:
        >>> optimizer = FeatureOptimizer("xgboost", n_trials=30)
        >>> result = optimizer.optimize(X_train, y_train, X_val, y_val, feature_names)
        >>> print(f"Optimized from {result.n_original} to {result.n_optimized} features")
    """

    def __init__(
        self,
        model_name: str,
        n_trials: int = 30,
        metric: str = "f1_weighted",
        min_features: int | None = None,
        random_seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.n_trials = n_trials
        self.metric = metric
        self.random_seed = random_seed

        # Get strategy to set min_features and max_features
        self._strategy = get_strategy_for_model(model_name)

        if min_features is not None:
            self.min_features = min_features
        else:
            self.min_features = self._strategy.min_features

        self.max_features = self._strategy.max_features

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str],
        sample_weights: np.ndarray | None = None,
    ) -> OptimizationResult:
        """Run Optuna optimization to find optimal feature subset.

        Args:
            X_train: Training feature matrix (n_samples, n_features).
            y_train: Training labels (n_samples,).
            X_val: Validation feature matrix.
            y_val: Validation labels.
            feature_names: List of feature names corresponding to columns.
            sample_weights: Optional sample weights for training.

        Returns:
            OptimizationResult with the optimized feature subset.
        """
        from src.models.registry import ModelRegistry
        from sklearn.metrics import f1_score, accuracy_score

        n_features = len(feature_names)

        # Skip optimization if n_features <= min_features
        if n_features <= self.min_features:
            return OptimizationResult(
                model_name=self.model_name,
                original_features=feature_names,
                optimized_features=feature_names,
                n_original=n_features,
                n_optimized=n_features,
                improvement=0.0,
                best_trial_score=0.0,
                n_trials=0,
                study=None,
            )

        # Select scoring function based on metric
        if self.metric == "accuracy":
            score_fn = accuracy_score
        else:
            # Default to f1_weighted
            def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return f1_score(y_true, y_pred, average="weighted")

        # Compute baseline score with all features
        baseline_score = 0.0
        try:
            baseline_model = ModelRegistry.create(self.model_name)
            baseline_model.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                sample_weights=sample_weights,
            )
            baseline_preds = baseline_model.predict(X_val)
            baseline_score = score_fn(y_val, baseline_preds.class_predictions)
        except Exception:
            baseline_score = 0.0

        def objective(trial: optuna.Trial) -> float:
            # Suggest number of features to select
            n_selected = trial.suggest_int(
                "n_features",
                self.min_features,
                min(self.max_features, n_features),
            )

            # Suggest inclusion for each feature
            selected_indices: list[int] = []
            for i, fname in enumerate(feature_names):
                include = trial.suggest_categorical(f"include_{fname}", [True, False])
                if include:
                    selected_indices.append(i)

            # Ensure we have at least min_features selected
            if len(selected_indices) < self.min_features:
                # Fill up with random unselected features
                unselected = [i for i in range(n_features) if i not in selected_indices]
                needed = self.min_features - len(selected_indices)
                np.random.seed(self.random_seed + trial.number)
                extra = list(np.random.choice(unselected, size=min(needed, len(unselected)), replace=False))
                selected_indices.extend(extra)

            # Limit to n_selected features (trim if over)
            if len(selected_indices) > n_selected:
                np.random.seed(self.random_seed + trial.number)
                selected_indices = list(
                    np.random.choice(selected_indices, size=n_selected, replace=False)
                )

            # Store selected indices for retrieval
            trial.set_user_attr("selected_indices", sorted(selected_indices))

            if len(selected_indices) == 0:
                return 0.0

            # Create subset arrays
            X_train_subset = X_train[:, selected_indices]
            X_val_subset = X_val[:, selected_indices]

            # Train model
            try:
                model = ModelRegistry.create(self.model_name)
                model.fit(
                    X_train=X_train_subset,
                    y_train=y_train,
                    X_val=X_val_subset,
                    y_val=y_val,
                    sample_weights=sample_weights,
                )

                # Evaluate
                predictions = model.predict(X_val_subset)
                score = score_fn(y_val, predictions.class_predictions)
                return score
            except Exception:
                return 0.0

        # Create study with TPE sampler for efficient search
        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Suppress Optuna logging for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        # Get best trial's selected features
        best_trial = study.best_trial
        best_indices = best_trial.user_attrs.get("selected_indices", list(range(n_features)))
        optimized_features = [feature_names[i] for i in best_indices]

        # Compute improvement vs baseline
        improvement = (
            (study.best_value - baseline_score) / baseline_score
            if baseline_score > 0
            else 0.0
        )

        return OptimizationResult(
            model_name=self.model_name,
            original_features=feature_names,
            optimized_features=optimized_features,
            n_original=n_features,
            n_optimized=len(optimized_features),
            improvement=improvement,
            best_trial_score=study.best_value,
            n_trials=self.n_trials,
            study=study,
        )
