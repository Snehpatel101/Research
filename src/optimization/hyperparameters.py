"""
Hyperparameter Optimization - PHASE_1B Complete Search Spaces.

This module provides Optuna-based hyperparameter optimization for ALL 23 models
in the ML Factory pipeline. Each model has carefully tuned search spaces based
on best practices and empirical results from financial ML applications.

Model Families (23 total):
    Boosting (3): xgboost, lightgbm, catboost
    Classical (3): random_forest, logistic, svm
    Neural RNN/CNN (10): lstm, gru, tcn, transformer, patchtst, itransformer,
                         tft, nbeats, inceptiontime, resnet1d
    Meta-learners (4): ridge_meta, mlp_meta, xgboost_meta, calibrated_meta
    Ensemble (3): voting, stacking, blending (minimal/no search spaces)

Search Space Types:
    ("int", min, max) - Integer parameter
    ("float", min, max) - Continuous parameter
    ("log_float", min, max) - Log-scale continuous parameter
    ("categorical", [options]) - Categorical parameter

Usage:
    from src.optimization.hyperparameters import (
        HyperparameterOptimizer,
        HYPERPARAMETER_SPACES,
    )

    optimizer = HyperparameterOptimizer(n_trials=100)
    result = optimizer.optimize("xgboost", X, y, model_factory)
    print(f"Best params: {result.best_params}")
    print(f"Improvement: {result.improvement:.2%}")

Reference: Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization"
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score

from src.core import (
    ALL_MODELS,
    DEFAULT_HYPERPARAM_TRIALS,
    DEFAULT_OPTUNA_RANDOM_STATE,
    MODEL_DATA_RANKS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class HyperparameterResult:
    """
    Result container for hyperparameter optimization.

    Attributes:
        model_name: Name of the optimized model
        best_params: Best hyperparameters found
        best_score: Best validation score achieved
        default_score: Score with default parameters
        improvement: Relative improvement over default
        n_trials: Number of trials completed
        study: Optuna Study object (for analysis)
        optimization_time: Time taken in seconds
        trial_history: List of (params, score) for each trial
    """

    model_name: str
    best_params: dict[str, Any]
    best_score: float
    default_score: float
    improvement: float
    n_trials: int
    study: optuna.Study
    optimization_time: float = 0.0
    trial_history: list[tuple[dict[str, Any], float]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"HyperparameterResult({self.model_name}, "
            f"best_score={self.best_score:.4f}, "
            f"improvement={self.improvement:+.2%}, "
            f"n_trials={self.n_trials})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "default_score": self.default_score,
            "improvement": self.improvement,
            "n_trials": self.n_trials,
            "optimization_time": self.optimization_time,
        }


# =============================================================================
# HYPERPARAMETER SEARCH SPACES - ALL 23 MODELS
# =============================================================================

# Type aliases for search space specification
# Each parameter can be one of:
#   - ("int", min, max) for integer range
#   - ("float", min, max) for float range
#   - ("log_float", min, max) for log-scale float range
#   - ("categorical", [choices]) for categorical options
#
# We use tuple[Any, ...] to allow variable length tuples with different
# element types. The runtime code validates the actual structure.
ParamSpec = tuple[Any, ...]
SearchSpaceType = dict[str, ParamSpec]

HYPERPARAMETER_SPACES: dict[str, SearchSpaceType] = {
    # =========================================================================
    # BOOSTING MODELS (3)
    # =========================================================================
    "xgboost": {
        "n_estimators": ("int", 100, 1000),
        "max_depth": ("int", 3, 10),
        "learning_rate": ("log_float", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "min_child_weight": ("int", 1, 10),
        "gamma": ("float", 0.0, 5.0),
        "reg_alpha": ("log_float", 1e-8, 10.0),
        "reg_lambda": ("log_float", 1e-8, 10.0),
    },
    "lightgbm": {
        "n_estimators": ("int", 100, 1000),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("log_float", 0.01, 0.3),
        "num_leaves": ("int", 20, 150),
        "feature_fraction": ("float", 0.6, 1.0),
        "bagging_fraction": ("float", 0.6, 1.0),
        "bagging_freq": ("int", 1, 7),
        "min_child_samples": ("int", 5, 100),
        "reg_alpha": ("log_float", 1e-8, 10.0),
        "reg_lambda": ("log_float", 1e-8, 10.0),
    },
    "catboost": {
        "iterations": ("int", 100, 1000),
        "depth": ("int", 4, 10),
        "learning_rate": ("log_float", 0.01, 0.3),
        "l2_leaf_reg": ("log_float", 1e-8, 10.0),
        "bagging_temperature": ("float", 0.0, 1.0),
        "random_strength": ("float", 0.0, 10.0),
        "border_count": ("int", 32, 255),
        "grow_policy": ("categorical", ["SymmetricTree", "Depthwise", "Lossguide"]),
    },
    # =========================================================================
    # CLASSICAL MODELS (3)
    # =========================================================================
    "random_forest": {
        "n_estimators": ("int", 100, 500),
        "max_depth": ("int", 5, 30),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("categorical", ["sqrt", "log2", None]),
        "bootstrap": ("categorical", [True, False]),
        "class_weight": ("categorical", ["balanced", "balanced_subsample", None]),
    },
    "logistic": {
        "C": ("log_float", 1e-4, 100.0),
        "penalty": ("categorical", ["l1", "l2", "elasticnet"]),
        "solver": ("categorical", ["saga", "lbfgs"]),
        "max_iter": ("int", 100, 1000),
        "class_weight": ("categorical", ["balanced", None]),
        "l1_ratio": ("float", 0.0, 1.0),  # Only used with elasticnet
    },
    "svm": {
        "C": ("log_float", 1e-3, 100.0),
        "kernel": ("categorical", ["rbf", "poly", "sigmoid"]),
        "gamma": ("categorical", ["scale", "auto"]),
        "degree": ("int", 2, 5),  # Only for poly kernel
        "class_weight": ("categorical", ["balanced", None]),
        "probability": ("categorical", [True]),  # Required for predict_proba
    },
    # =========================================================================
    # NEURAL RNN/CNN MODELS (10)
    # =========================================================================
    "lstm": {
        "hidden_size": ("int", 32, 256),
        "num_layers": ("int", 1, 4),
        "dropout": ("float", 0.0, 0.5),
        "bidirectional": ("categorical", [True, False]),
        "learning_rate": ("log_float", 1e-5, 1e-2),
        "batch_size": ("categorical", [32, 64, 128, 256]),
        "weight_decay": ("log_float", 1e-8, 1e-3),
        "gradient_clip": ("float", 0.5, 5.0),
    },
    "gru": {
        "hidden_size": ("int", 32, 256),
        "num_layers": ("int", 1, 4),
        "dropout": ("float", 0.0, 0.5),
        "bidirectional": ("categorical", [True, False]),
        "learning_rate": ("log_float", 1e-5, 1e-2),
        "batch_size": ("categorical", [32, 64, 128, 256]),
        "weight_decay": ("log_float", 1e-8, 1e-3),
        "gradient_clip": ("float", 0.5, 5.0),
    },
    "tcn": {
        "num_channels": ("categorical", [[32, 64], [64, 128], [32, 64, 128], [64, 128, 256]]),
        "kernel_size": ("int", 2, 7),
        "dropout": ("float", 0.0, 0.4),
        "learning_rate": ("log_float", 1e-5, 1e-2),
        "batch_size": ("categorical", [32, 64, 128, 256]),
        "weight_decay": ("log_float", 1e-8, 1e-3),
        "dilation_base": ("int", 2, 4),
    },
    "transformer": {
        "d_model": ("categorical", [32, 64, 128, 256]),
        "nhead": ("categorical", [2, 4, 8]),
        "num_encoder_layers": ("int", 1, 6),
        "dim_feedforward": ("int", 64, 512),
        "dropout": ("float", 0.0, 0.3),
        "learning_rate": ("log_float", 1e-5, 1e-3),
        "batch_size": ("categorical", [32, 64, 128]),
        "weight_decay": ("log_float", 1e-8, 1e-3),
        "warmup_steps": ("int", 100, 1000),
    },
    "patchtst": {
        "patch_len": ("int", 8, 32),
        "stride": ("int", 4, 16),
        "d_model": ("categorical", [64, 128, 256]),
        "nhead": ("categorical", [4, 8]),
        "num_encoder_layers": ("int", 2, 6),
        "d_ff": ("int", 128, 512),
        "dropout": ("float", 0.0, 0.3),
        "learning_rate": ("log_float", 1e-5, 1e-3),
        "batch_size": ("categorical", [32, 64, 128]),
        "weight_decay": ("log_float", 1e-8, 1e-3),
    },
    "itransformer": {
        "d_model": ("categorical", [64, 128, 256]),
        "nhead": ("categorical", [4, 8]),
        "num_encoder_layers": ("int", 2, 6),
        "d_ff": ("int", 128, 512),
        "dropout": ("float", 0.0, 0.3),
        "learning_rate": ("log_float", 1e-5, 1e-3),
        "batch_size": ("categorical", [32, 64, 128]),
        "weight_decay": ("log_float", 1e-8, 1e-3),
        "use_time_features": ("categorical", [True, False]),
    },
    "tft": {
        "hidden_size": ("int", 32, 256),
        "num_attention_heads": ("categorical", [1, 2, 4]),
        "dropout": ("float", 0.0, 0.3),
        "hidden_continuous_size": ("int", 8, 64),
        "learning_rate": ("log_float", 1e-5, 1e-3),
        "batch_size": ("categorical", [32, 64, 128]),
        "weight_decay": ("log_float", 1e-8, 1e-3),
        "gradient_clip": ("float", 0.1, 1.0),
    },
    "nbeats": {
        "num_stacks": ("int", 2, 8),
        "num_blocks": ("int", 1, 4),
        "num_layers": ("int", 2, 6),
        "layer_size": ("int", 64, 512),
        "sharing": ("categorical", [True, False]),
        "dropout": ("float", 0.0, 0.3),
        "learning_rate": ("log_float", 1e-5, 1e-3),
        "batch_size": ("categorical", [32, 64, 128, 256]),
        "weight_decay": ("log_float", 1e-8, 1e-3),
    },
    "inceptiontime": {
        "num_blocks": ("int", 1, 6),
        "num_filters": ("int", 16, 64),
        "use_residual": ("categorical", [True, False]),
        "use_bottleneck": ("categorical", [True, False]),
        "bottleneck_size": ("int", 16, 64),
        "kernel_sizes": ("categorical", [[10, 20, 40], [5, 10, 20], [20, 40, 80]]),
        "dropout": ("float", 0.0, 0.3),
        "learning_rate": ("log_float", 1e-5, 1e-3),
        "batch_size": ("categorical", [32, 64, 128]),
        "weight_decay": ("log_float", 1e-8, 1e-3),
    },
    "resnet1d": {
        "num_blocks": ("int", 2, 8),
        "base_filters": ("int", 32, 128),
        "kernel_size": ("int", 3, 15),
        "stride": ("int", 1, 2),
        "groups": ("int", 1, 4),
        "dropout": ("float", 0.0, 0.3),
        "learning_rate": ("log_float", 1e-5, 1e-3),
        "batch_size": ("categorical", [32, 64, 128]),
        "weight_decay": ("log_float", 1e-8, 1e-3),
    },
    # =========================================================================
    # META-LEARNERS (4)
    # =========================================================================
    "ridge_meta": {
        "alpha": ("log_float", 1e-4, 100.0),
        "fit_intercept": ("categorical", [True, False]),
        "normalize": ("categorical", [True, False]),
        "solver": ("categorical", ["auto", "svd", "cholesky", "lsqr", "sparse_cg"]),
    },
    "mlp_meta": {
        "hidden_layer_sizes": ("categorical", [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)]),
        "activation": ("categorical", ["relu", "tanh"]),
        "solver": ("categorical", ["adam", "sgd"]),
        "alpha": ("log_float", 1e-6, 1e-2),
        "learning_rate_init": ("log_float", 1e-5, 1e-2),
        "batch_size": ("categorical", [32, 64, 128, 256]),
        "early_stopping": ("categorical", [True]),
        "validation_fraction": ("float", 0.1, 0.2),
    },
    "xgboost_meta": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 2, 6),
        "learning_rate": ("log_float", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "reg_alpha": ("log_float", 1e-8, 1.0),
        "reg_lambda": ("log_float", 1e-8, 1.0),
    },
    "calibrated_meta": {
        "method": ("categorical", ["isotonic", "sigmoid"]),
        "cv": ("categorical", [3, 5]),
        "ensemble": ("categorical", [True, False]),
        # Base estimator params (ridge)
        "base_alpha": ("log_float", 1e-4, 100.0),
    },
    # =========================================================================
    # ENSEMBLE MODELS (3) - Minimal search spaces
    # =========================================================================
    "voting": {
        # Voting ensemble has no hyperparameters to tune
        # The weights could be tuned but are typically derived from OOF scores
    },
    "stacking": {
        # Stacking meta-learner params are in the meta-learner spaces above
        "passthrough": ("categorical", [True, False]),
    },
    "blending": {
        # Blending uses a simple weighted average, weights from OOF
        "blend_method": ("categorical", ["average", "weighted", "rank"]),
    },
}

# Verify all 23 models have search spaces
assert (
    len(HYPERPARAMETER_SPACES) == 23
), f"Expected 23 model search spaces, got {len(HYPERPARAMETER_SPACES)}"

# Verify all models in ALL_MODELS have search spaces
missing_models = set(ALL_MODELS) - set(HYPERPARAMETER_SPACES.keys())
assert not missing_models, f"Missing search spaces for models: {missing_models}"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def suggest_hyperparameters(
    trial: optuna.Trial,
    model_name: str,
    custom_space: SearchSpaceType | None = None,
) -> dict[str, Any]:
    """
    Suggest hyperparameters for a model using Optuna trial.

    Args:
        trial: Optuna Trial object
        model_name: Name of the model
        custom_space: Optional custom search space (overrides default)

    Returns:
        Dictionary of suggested hyperparameters

    Raises:
        ValueError: If model_name is unknown
    """
    model_name = model_name.lower()
    space = custom_space or HYPERPARAMETER_SPACES.get(model_name, {})

    if not space:
        if model_name not in HYPERPARAMETER_SPACES:
            raise ValueError(f"Unknown model: {model_name}")
        return {}

    params: dict[str, Any] = {}
    for param_name, spec in space.items():
        param_type = spec[0]

        if param_type == "int":
            # spec is ("int", min, max)
            int_low = int(spec[1])
            int_high = int(spec[2])
            params[param_name] = trial.suggest_int(param_name, int_low, int_high)

        elif param_type == "float":
            # spec is ("float", min, max)
            float_low = float(spec[1])
            float_high = float(spec[2])
            params[param_name] = trial.suggest_float(param_name, float_low, float_high)

        elif param_type == "log_float":
            # spec is ("log_float", min, max)
            log_low = float(spec[1])
            log_high = float(spec[2])
            params[param_name] = trial.suggest_float(param_name, log_low, log_high, log=True)

        elif param_type == "categorical":
            # spec is ("categorical", [choices])
            choices = list(spec[1])
            params[param_name] = trial.suggest_categorical(param_name, choices)

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return params


def get_default_hyperparameters(model_name: str) -> dict[str, Any]:
    """
    Get default hyperparameters for a model (middle of search space).

    Args:
        model_name: Name of the model

    Returns:
        Dictionary of default hyperparameters

    Note:
        Returns middle values for numeric params, first value for categorical.
    """
    model_name = model_name.lower()
    space = HYPERPARAMETER_SPACES.get(model_name, {})

    if not space:
        return {}

    defaults: dict[str, Any] = {}
    for param_name, spec in space.items():
        param_type = spec[0]

        if param_type == "int":
            # spec is ("int", min, max)
            int_low = int(spec[1])
            int_high = int(spec[2])
            defaults[param_name] = (int_low + int_high) // 2

        elif param_type == "float":
            # spec is ("float", min, max)
            float_low = float(spec[1])
            float_high = float(spec[2])
            defaults[param_name] = (float_low + float_high) / 2

        elif param_type == "log_float":
            # spec is ("log_float", min, max)
            log_low = float(spec[1])
            log_high = float(spec[2])
            # Geometric mean for log-scale
            defaults[param_name] = float(np.sqrt(log_low * log_high))

        elif param_type == "categorical":
            # spec is ("categorical", [choices])
            choices = spec[1]
            defaults[param_name] = choices[0]

    return defaults


# =============================================================================
# HYPERPARAMETER OPTIMIZER
# =============================================================================


class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimizer for ML Factory models.

    Uses TPESampler for efficient exploration and HyperbandPruner for
    early stopping of unpromising trials (neural networks).

    Args:
        n_trials: Number of optimization trials (default: 100)
        cv_folds: Cross-validation folds for evaluation (default: 3)
        scoring: Scoring metric ("f1_weighted", "accuracy") (default: "f1_weighted")
        use_pruning: Enable Hyperband pruning for neural models (default: True)
        random_state: Random seed for reproducibility (default: 42)
        timeout: Optional timeout in seconds per trial
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)

    Example:
        >>> optimizer = HyperparameterOptimizer(n_trials=100)
        >>> result = optimizer.optimize(
        ...     model_name="xgboost",
        ...     X=X_train,
        ...     y=y_train,
        ...     model_factory=lambda params: XGBClassifier(**params),
        ... )
        >>> print(f"Best: {result.best_params}")
    """

    def __init__(
        self,
        n_trials: int = DEFAULT_HYPERPARAM_TRIALS,
        cv_folds: int = 3,
        scoring: str = "f1_weighted",
        use_pruning: bool = True,
        random_state: int = DEFAULT_OPTUNA_RANDOM_STATE,
        timeout: int | None = None,
        verbose: int = 1,
    ):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.use_pruning = use_pruning
        self.random_state = random_state
        self.timeout = timeout
        self.verbose = verbose

        # Configure Optuna logging based on verbosity
        if verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)

    def _get_score_fn(self) -> Callable[[np.ndarray, np.ndarray], float]:
        """Get scoring function based on metric name."""
        if self.scoring == "accuracy":
            return lambda y_true, y_pred: float(accuracy_score(y_true, y_pred))
        elif self.scoring == "f1_weighted":
            return lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="weighted"))
        elif self.scoring == "f1_macro":
            return lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="macro"))
        else:
            raise ValueError(f"Unknown scoring metric: {self.scoring}")

    def _is_neural_model(self, model_name: str) -> bool:
        """Check if model is a neural network (for pruning)."""
        return MODEL_DATA_RANKS.get(model_name.lower(), 2) >= 3

    def optimize(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable[[dict[str, Any]], Any],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        search_space: SearchSpaceType | None = None,
        sample_weights: np.ndarray | None = None,
        storage_path: str | None = None,
    ) -> HyperparameterResult:
        """
        Optimize hyperparameters for a single model.

        Args:
            model_name: Name of the model to optimize
            X: Training features (n_samples, n_features) or (n_samples, seq_len, n_features)
            y: Training labels (n_samples,)
            model_factory: Factory function that creates model from params
            X_val: Optional validation features (if None, uses train/val split)
            y_val: Optional validation labels
            search_space: Optional custom search space
            sample_weights: Optional sample weights for training
            storage_path: Optional path to SQLite database for study persistence.
                If provided, enables resuming interrupted studies. If None, uses
                in-memory storage.

        Returns:
            HyperparameterResult with best params and scores

        Note:
            The model_factory should accept a dict of hyperparameters and return
            a model object with fit() and predict() methods.
        """
        model_name = model_name.lower()
        start_time = time.time()

        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"Optimizing hyperparameters for: {model_name}")
            print(f"  n_trials={self.n_trials}, scoring={self.scoring}")
            print(f"{'='*60}")

        # Get search space
        space = search_space or HYPERPARAMETER_SPACES.get(model_name, {})

        if not space:
            logger.warning(f"No search space for {model_name}, returning defaults")
            return HyperparameterResult(
                model_name=model_name,
                best_params={},
                best_score=0.0,
                default_score=0.0,
                improvement=0.0,
                n_trials=0,
                study=optuna.create_study(direction="maximize"),
                optimization_time=0.0,
            )

        # Get scoring function
        score_fn = self._get_score_fn()

        # Split data if no validation set provided
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y,
            )
            if sample_weights is not None:
                sw_train, sw_val = train_test_split(
                    sample_weights,
                    test_size=0.2,
                    random_state=self.random_state,
                )
            else:
                sw_train = None
        else:
            X_train, y_train = X, y
            sw_train = sample_weights

        # Calculate default score
        default_params = get_default_hyperparameters(model_name)
        default_score = 0.0
        try:
            default_model = model_factory(default_params)
            if hasattr(default_model, "fit"):
                if sw_train is not None and hasattr(default_model.fit, "__code__"):
                    # Check if fit accepts sample_weight
                    try:
                        default_model.fit(X_train, y_train, sample_weight=sw_train)
                    except TypeError:
                        default_model.fit(X_train, y_train)
                else:
                    default_model.fit(X_train, y_train)
                default_preds = default_model.predict(X_val)
                default_score = score_fn(y_val, default_preds)
        except Exception as e:
            logger.warning(f"Failed to compute default score: {e}")

        # Trial history
        trial_history: list[tuple[dict[str, Any], float]] = []

        # Define objective
        def objective(trial: optuna.Trial) -> float:
            params = suggest_hyperparameters(trial, model_name, space)

            try:
                model = model_factory(params)

                # Train
                if sw_train is not None:
                    try:
                        model.fit(X_train, y_train, sample_weight=sw_train)
                    except TypeError:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                # Evaluate
                preds = model.predict(X_val)
                score = score_fn(y_val, preds)

                trial_history.append((params.copy(), score))
                return score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return 0.0

        # Create study
        sampler = TPESampler(seed=self.random_state)

        # Use Hyperband pruner for neural models
        if self.use_pruning and self._is_neural_model(model_name):
            pruner = HyperbandPruner(
                min_resource=1,
                max_resource=self.n_trials,
                reduction_factor=3,
            )
        else:
            pruner = None

        # Configure storage for study persistence
        if storage_path is not None:
            storage_dir = Path(storage_path).parent
            storage_dir.mkdir(parents=True, exist_ok=True)
            storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")
        else:
            storage = None

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=(self.verbose >= 1),
        )

        optimization_time = time.time() - start_time

        # Calculate improvement
        best_score = study.best_value
        improvement = (best_score - default_score) / default_score if default_score > 0 else 0.0

        if self.verbose >= 1:
            print(f"\nOptimization complete for {model_name}:")
            print(f"  Default score: {default_score:.4f}")
            print(f"  Best score: {best_score:.4f}")
            print(f"  Improvement: {improvement:+.2%}")
            print(f"  Time: {optimization_time:.1f}s")
            print(f"  Best params: {study.best_params}")

        return HyperparameterResult(
            model_name=model_name,
            best_params=study.best_params,
            best_score=best_score,
            default_score=default_score,
            improvement=improvement,
            n_trials=len(study.trials),
            study=study,
            optimization_time=optimization_time,
            trial_history=trial_history,
        )

    def optimize_multiple(
        self,
        model_names: list[str],
        X: np.ndarray,
        y: np.ndarray,
        model_factories: dict[str, Callable[[dict[str, Any]], Any]],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, HyperparameterResult]:
        """
        Optimize hyperparameters for multiple models.

        Args:
            model_names: List of model names to optimize
            X: Training features
            y: Training labels
            model_factories: Dict mapping model_name -> factory function
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            Dict mapping model_name -> HyperparameterResult
        """
        results = {}

        for model_name in model_names:
            if model_name not in model_factories:
                logger.warning(f"No factory for {model_name}, skipping")
                continue

            result = self.optimize(
                model_name=model_name,
                X=X,
                y=y,
                model_factory=model_factories[model_name],
                X_val=X_val,
                y_val=y_val,
            )
            results[model_name] = result

        return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "HyperparameterResult",
    "HYPERPARAMETER_SPACES",
    "HyperparameterOptimizer",
    "suggest_hyperparameters",
    "get_default_hyperparameters",
]
