from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd
import numpy as np

from .strategies import get_strategy_for_model


@dataclass
class OptimizationResult:
    model_name: str
    baseline_features: list[str]
    optimized_features: list[str]
    best_score: float
    n_trials: int
    study: optuna.Study

    def __repr__(self) -> str:
        return (
            f"OptimizationResult({self.model_name}, "
            f"baseline={len(self.baseline_features)}, "
            f"optimized={len(self.optimized_features)}, "
            f"score={self.best_score:.4f})"
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
    from src.models.registry import ModelRegistry
    from sklearn.metrics import f1_score

    strategy = get_strategy_for_model(model_name)
    baseline_features = strategy.baseline_features

    available_features = [f for f in baseline_features if f in X_train.columns]

    if len(available_features) < strategy.min_features:
        return OptimizationResult(
            model_name=model_name,
            baseline_features=baseline_features,
            optimized_features=available_features,
            best_score=0.0,
            n_trials=0,
            study=None,
        )

    def objective(trial: optuna.Trial) -> float:
        n_features = trial.suggest_int(
            "n_features",
            min(strategy.min_features, len(available_features)),
            min(strategy.max_features, len(available_features)),
        )

        feature_indices = trial.suggest_categorical(
            "feature_indices",
            list(range(len(available_features))),
            n_choices=n_features,
        )

        if isinstance(feature_indices, int):
            feature_indices = [feature_indices]

        selected_features = [available_features[i] for i in feature_indices]

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

    return OptimizationResult(
        model_name=model_name,
        baseline_features=baseline_features,
        optimized_features=optimized_features,
        best_score=study.best_value,
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
