"""
Optuna-based meta-learner selection for ensemble optimization.

Phase 16C: Auto Meta-Learner Selection - Uses Optuna to select the best
meta-learner type and hyperparameters based on validation performance.

Supported meta-learners:
- ridge: L2-regularized linear model (fast, interpretable)
- xgboost: Gradient boosting (handles non-linearity well)
- mlp: Neural network (captures complex interactions)
- calibrated_ridge: Ridge with isotonic calibration (calibrated probabilities)

Example:
    >>> from src.models.ensemble.meta_selection import (
    ...     MetaLearnerSelector,
    ...     select_meta_learner_with_optuna,
    ... )
    >>>
    >>> # Simple function interface
    >>> best_config = select_meta_learner_with_optuna(
    ...     oof_predictions=stacking_X,
    ...     labels=y_train,
    ...     n_trials=20,
    ... )
    >>> print(best_config)
    >>> # {'meta_learner_type': 'xgboost', 'params': {...}}
    >>>
    >>> # Class interface with more control
    >>> selector = MetaLearnerSelector(n_trials=30)
    >>> result = selector.select(oof_predictions, labels, X_val, y_val)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# Available meta-learner types for selection
AVAILABLE_META_LEARNER_TYPES = ["ridge", "xgboost", "mlp", "calibrated_ridge"]


@dataclass
class MetaLearnerSelectionResult:
    """
    Result from Optuna meta-learner selection.

    Attributes:
        best_meta_type: Best meta-learner type found.
        best_params: Best hyperparameters for that meta-learner.
        best_score: Validation score achieved.
        study: The Optuna study object (for analysis).
        all_trials: Summary of all trials.
    """

    best_meta_type: str
    best_params: dict[str, Any]
    best_score: float
    study: optuna.Study | None = None
    all_trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_meta_type": self.best_meta_type,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.all_trials) if self.all_trials else 0,
        }


def _suggest_ridge_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest hyperparameters for Ridge meta-learner."""
    return {
        "alpha": trial.suggest_float("ridge_alpha", 1e-4, 100.0, log=True),
        "fit_intercept": True,
        "class_weight": "balanced",
        "scale_features": True,
    }


def _suggest_xgboost_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest hyperparameters for XGBoost meta-learner."""
    return {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 20, 200),
        "max_depth": trial.suggest_int("xgb_max_depth", 2, 6),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 20),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("xgb_gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-4, 10.0, log=True),
    }


def _suggest_mlp_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest hyperparameters for MLP meta-learner."""
    n_layers = trial.suggest_int("mlp_n_layers", 1, 3)
    hidden_sizes = []
    for i in range(n_layers):
        size = trial.suggest_int(f"mlp_layer_{i}_size", 16, 128)
        hidden_sizes.append(size)

    return {
        "hidden_layer_sizes": tuple(hidden_sizes),
        "alpha": trial.suggest_float("mlp_alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True),
        "max_iter": trial.suggest_int("mlp_max_iter", 200, 1000),
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "activation": trial.suggest_categorical("mlp_activation", ["relu", "tanh"]),
        "solver": "adam",
        "scale_features": True,
    }


def _suggest_calibrated_ridge_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest hyperparameters for Calibrated Ridge meta-learner."""
    return {
        "base_estimator": "logistic",
        "method": trial.suggest_categorical("cal_method", ["isotonic", "sigmoid"]),
        "cv": trial.suggest_int("cal_cv", 3, 7),
        "ensemble": True,
        "scale_features": True,
    }


# Parameter suggestion functions by meta-learner type
PARAM_SUGGESTERS = {
    "ridge": _suggest_ridge_params,
    "xgboost": _suggest_xgboost_params,
    "mlp": _suggest_mlp_params,
    "calibrated_ridge": _suggest_calibrated_ridge_params,
}


def _train_and_evaluate_meta_learner(
    meta_type: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Train a meta-learner and return validation score.

    Args:
        meta_type: Type of meta-learner ('ridge', 'xgboost', etc.).
        params: Hyperparameters for the meta-learner.
        X_train: Training OOF predictions.
        y_train: Training labels.
        X_val: Validation OOF predictions.
        y_val: Validation labels.

    Returns:
        Validation F1 score (macro).
    """
    # Map to actual class names in the codebase
    meta_type_map = {
        "ridge": "ridge_meta",
        "xgboost": "xgboost_meta",
        "mlp": "mlp_meta",
        "calibrated_ridge": "calibrated_meta",
    }
    actual_meta_type = meta_type_map.get(meta_type, meta_type)

    # Import and create meta-learner
    from src.models.ensemble import MetaLearnerFactory

    factory = MetaLearnerFactory()
    meta_learner = factory.create(actual_meta_type, **params)

    # Train
    meta_learner.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    # Evaluate
    output = meta_learner.predict(X_val)
    y_pred = output.class_predictions

    # Use macro F1 as the scoring metric
    val_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

    return float(val_f1)


def select_meta_learner_with_optuna(
    oof_predictions: np.ndarray,
    labels: np.ndarray,
    n_trials: int = 20,
    val_ratio: float = 0.2,
    seed: int = 42,
    timeout: float | None = None,
    show_progress_bar: bool = False,
) -> MetaLearnerSelectionResult:
    """
    Use Optuna to select the best meta-learner type and hyperparameters.

    Phase 16C implementation: Automatically selects from ridge, xgboost, mlp,
    and calibrated_ridge based on validation performance.

    Args:
        oof_predictions: Shape (n_samples, n_features) - stacking features
            from base model OOF predictions.
        labels: Shape (n_samples,) - training labels.
        n_trials: Number of Optuna trials to run.
        val_ratio: Fraction of data to use for validation.
        seed: Random seed for reproducibility.
        timeout: Maximum time in seconds for optimization.
        show_progress_bar: Whether to show Optuna progress bar.

    Returns:
        MetaLearnerSelectionResult with best meta-learner config.

    Example:
        >>> result = select_meta_learner_with_optuna(
        ...     oof_predictions=stacking_X,
        ...     labels=y_train,
        ...     n_trials=30,
        ... )
        >>> print(f"Best: {result.best_meta_type} with score {result.best_score:.4f}")
    """
    # Time-series aware split (use last val_ratio for validation)
    n_samples = len(labels)
    n_train = int(n_samples * (1 - val_ratio))

    X_train = oof_predictions[:n_train]
    y_train = labels[:n_train]
    X_val = oof_predictions[n_train:]
    y_val = labels[n_train:]

    logger.info(
        f"Meta-learner selection: {n_samples} samples, "
        f"train={n_train}, val={n_samples - n_train}"
    )

    def objective(trial: optuna.Trial) -> float:
        # Select meta-learner type
        meta_type = trial.suggest_categorical(
            "meta_learner_type",
            AVAILABLE_META_LEARNER_TYPES,
        )

        # Get suggested parameters for this type
        param_suggester = PARAM_SUGGESTERS[meta_type]
        params = param_suggester(trial)

        try:
            score = _train_and_evaluate_meta_learner(
                meta_type=meta_type,
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )
            return score
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return worst score on failure

    # Create and run study
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="meta_learner_selection",
    )

    # Set verbosity
    optuna.logging.set_verbosity(
        optuna.logging.INFO if show_progress_bar else optuna.logging.WARNING
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
    )

    # Extract results
    best_trial = study.best_trial
    best_meta_type = best_trial.params["meta_learner_type"]
    best_params = {k: v for k, v in best_trial.params.items() if k != "meta_learner_type"}

    # Collect all trial summaries
    all_trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            all_trials.append(
                {
                    "number": trial.number,
                    "meta_type": trial.params.get("meta_learner_type", "unknown"),
                    "score": trial.value,
                    "params": {k: v for k, v in trial.params.items() if k != "meta_learner_type"},
                }
            )

    logger.info(f"Best meta-learner: {best_meta_type} with F1={study.best_value:.4f}")

    return MetaLearnerSelectionResult(
        best_meta_type=best_meta_type,
        best_params=best_params,
        best_score=study.best_value,
        study=study,
        all_trials=all_trials,
    )


class MetaLearnerSelector:
    """
    Class-based interface for Optuna meta-learner selection.

    Provides more control over the selection process and allows for
    cross-validation based selection.

    Example:
        >>> selector = MetaLearnerSelector(n_trials=30, n_cv_splits=3)
        >>> result = selector.select(oof_predictions, labels)
        >>> print(result.best_meta_type)
    """

    def __init__(
        self,
        n_trials: int = 20,
        n_cv_splits: int = 3,
        seed: int = 42,
        timeout: float | None = None,
        use_cv: bool = False,
    ) -> None:
        """
        Initialize MetaLearnerSelector.

        Args:
            n_trials: Number of Optuna trials.
            n_cv_splits: Number of time-series CV splits (if use_cv=True).
            seed: Random seed.
            timeout: Maximum optimization time in seconds.
            use_cv: If True, use time-series CV for evaluation.
        """
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.seed = seed
        self.timeout = timeout
        self.use_cv = use_cv

    def select(
        self,
        oof_predictions: np.ndarray,
        labels: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> MetaLearnerSelectionResult:
        """
        Select the best meta-learner.

        Args:
            oof_predictions: Stacking features from OOF predictions.
            labels: Training labels.
            X_val: Optional separate validation features.
            y_val: Optional separate validation labels.

        Returns:
            MetaLearnerSelectionResult with best configuration.
        """
        if self.use_cv:
            return self._select_with_cv(oof_predictions, labels)
        elif X_val is not None and y_val is not None:
            return self._select_with_holdout(oof_predictions, labels, X_val, y_val)
        else:
            # Use last 20% as validation
            return select_meta_learner_with_optuna(
                oof_predictions=oof_predictions,
                labels=labels,
                n_trials=self.n_trials,
                val_ratio=0.2,
                seed=self.seed,
                timeout=self.timeout,
            )

    def _select_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> MetaLearnerSelectionResult:
        """Select using time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        def objective(trial: optuna.Trial) -> float:
            meta_type = trial.suggest_categorical(
                "meta_learner_type",
                AVAILABLE_META_LEARNER_TYPES,
            )
            params = PARAM_SUGGESTERS[meta_type](trial)

            # Average score across CV folds
            scores = []
            for train_idx, val_idx in tscv.split(X):
                try:
                    score = _train_and_evaluate_meta_learner(
                        meta_type=meta_type,
                        params=params,
                        X_train=X[train_idx],
                        y_train=y[train_idx],
                        X_val=X[val_idx],
                        y_val=y[val_idx],
                    )
                    scores.append(score)
                except Exception:
                    scores.append(0.0)

            return float(np.mean(scores)) if scores else 0.0

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_trial = study.best_trial
        return MetaLearnerSelectionResult(
            best_meta_type=best_trial.params["meta_learner_type"],
            best_params={k: v for k, v in best_trial.params.items() if k != "meta_learner_type"},
            best_score=study.best_value,
            study=study,
            all_trials=[
                {
                    "number": t.number,
                    "meta_type": t.params.get("meta_learner_type"),
                    "score": t.value,
                }
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ],
        )

    def _select_with_holdout(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> MetaLearnerSelectionResult:
        """Select using provided holdout set."""

        def objective(trial: optuna.Trial) -> float:
            meta_type = trial.suggest_categorical(
                "meta_learner_type",
                AVAILABLE_META_LEARNER_TYPES,
            )
            params = PARAM_SUGGESTERS[meta_type](trial)

            try:
                return _train_and_evaluate_meta_learner(
                    meta_type=meta_type,
                    params=params,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                )
            except Exception:
                return 0.0

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_trial = study.best_trial
        return MetaLearnerSelectionResult(
            best_meta_type=best_trial.params["meta_learner_type"],
            best_params={k: v for k, v in best_trial.params.items() if k != "meta_learner_type"},
            best_score=study.best_value,
            study=study,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MetaLearnerSelectionResult",
    "MetaLearnerSelector",
    "select_meta_learner_with_optuna",
    "AVAILABLE_META_LEARNER_TYPES",
]
