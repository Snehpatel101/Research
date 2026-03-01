"""
Hyperparameter Tuning for Time Series Cross-Validation.

Uses Optuna's TPE sampler with time-series aware objective.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.models.registry import ModelRegistry
from src.validation.deflated_sharpe import (
    compute_dsr_from_optuna_study,
    is_sharpe_like_metric,
)

from .param_spaces import (
    PARAM_SPACES,
    get_max_leaves_for_depth,
    validate_lightgbm_params,
)
from .purged_kfold import PurgedKFold

logger = logging.getLogger(__name__)

OPTUNA_MAX_SAMPLES = 50_000


class TimeSeriesOptunaTuner:
    """
    Hyperparameter tuning with purged cross-validation.

    Uses Optuna's TPE sampler with time-series aware objective.
    """

    def __init__(
        self,
        model_name: str,
        cv: PurgedKFold,
        n_trials: int = 50,
        direction: str = "maximize",
        metric: str = "f1",
        pruner: Any | None = None,
        max_epochs: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.cv = cv
        self.n_trials = n_trials
        self.direction = direction
        self.metric = metric
        self.pruner = pruner
        self.max_epochs = max_epochs

    def tune(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weights: pd.Series | None = None,
        param_space: dict | None = None,
        data_rank: int = 2,
    ) -> dict[str, Any]:
        """
        Run hyperparameter tuning.

        Args:
            X: Features (DataFrame for 2D, ndarray for 3D/4D)
            y: Labels (Series for 2D, ndarray for 3D/4D)
            sample_weights: Optional quality weights
            param_space: Search space (uses defaults if None)
            data_rank: Dimensionality of data (2, 3, or 4)

        Returns:
            Dict with best_params and study info
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            logger.warning("Optuna not installed, skipping tuning")
            return {"best_params": {}, "best_value": None, "skipped": True}

        # Get search space
        if param_space is None:
            param_space = PARAM_SPACES.get(self.model_name, {})

        if not param_space:
            logger.warning(f"No param space defined for {self.model_name}")
            return {"best_params": {}, "best_value": None, "skipped": True}

        # --- Memory guard: subsample large datasets before Optuna ---
        # For a 1.6M-row TCN dataset (13,620 flattened features), 100 trials x 5
        # folds each allocating numpy copies via fancy indexing consumes ~120 GB.
        # Capping at 50k rows keeps peak RSS under ~8 GB for float32 3D data.
        original_n = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        if original_n > OPTUNA_MAX_SAMPLES:
            logger.info(
                f"  Subsampling {original_n} -> {OPTUNA_MAX_SAMPLES} rows for Optuna (stratified)"
            )
            rng = np.random.default_rng(seed=42)
            y_arr = y if isinstance(y, np.ndarray) else y.to_numpy()
            classes, class_counts = np.unique(y_arr, return_counts=True)
            # Stratified: allocate quota per class proportional to its share
            class_fractions = class_counts / original_n
            class_quotas = np.floor(class_fractions * OPTUNA_MAX_SAMPLES).astype(int)
            # Any remainder goes to the largest class
            remainder = OPTUNA_MAX_SAMPLES - class_quotas.sum()
            class_quotas[np.argmax(class_counts)] += remainder
            sub_indices = []
            for cls, quota in zip(classes, class_quotas, strict=True):
                cls_idx = np.where(y_arr == cls)[0]
                chosen = rng.choice(cls_idx, size=min(quota, len(cls_idx)), replace=False)
                sub_indices.append(chosen)
            sub_indices = np.sort(np.concatenate(sub_indices))
            # Apply to X
            if isinstance(X, np.ndarray):
                X = X[sub_indices]
            else:
                X = X.iloc[sub_indices].reset_index(drop=True)
            # Apply to y
            if isinstance(y, np.ndarray):
                y = y[sub_indices]
            else:
                y = y.iloc[sub_indices].reset_index(drop=True)
            # Apply to sample_weights if present
            if sample_weights is not None:
                if isinstance(sample_weights, np.ndarray):
                    sample_weights = sample_weights[sub_indices]
                else:
                    sample_weights = sample_weights.iloc[sub_indices].reset_index(drop=True)

        # --- float32 conversion: halves memory for all 500 fold slices ---
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32, copy=False)
        else:
            X = X.astype(np.float32, copy=False)

        # Create study with optional pruner for early stopping of bad trials
        default_pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1,
        )
        study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=42),
            pruner=self.pruner or default_pruner,
        )

        # Callback to free memory between trials (prevents accumulation over 100 trials)
        def _trial_cleanup_callback(study: Any, trial: Any) -> None:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Get the scoring function based on configured metric
        from src.optimization.scoring import get_score_fn

        score_fn = get_score_fn(self.metric)

        # Precompute CV splits once — splits are deterministic so recomputing
        # them inside every trial is wasted work.
        # For 3D/4D data, create a lightweight index DataFrame for splitting.
        if data_rank >= 3 and isinstance(X, np.ndarray):
            if X.ndim < data_rank:
                raise ValueError(f"data_rank={data_rank} but X.ndim={X.ndim}")
            if X.shape[0] == 0:
                raise ValueError("X has 0 samples")
            if X.shape[0] != len(y):
                raise ValueError(f"X.shape[0]={X.shape[0]} != len(y)={len(y)}")
            n_samples = X.shape[0]
            X_for_cv = pd.DataFrame(index=range(n_samples))
            y_for_cv = pd.Series(y) if isinstance(y, np.ndarray) else y
            self._precomputed_splits = list(self.cv.split(X_for_cv, y_for_cv))
        else:
            self._precomputed_splits = list(self.cv.split(X, y))

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, param_space)

            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(self._precomputed_splits):
                # Rank-aware data indexing
                if data_rank >= 3 and isinstance(X, np.ndarray):
                    # 3D/4D: X is ndarray, index by sample axis (axis 0)
                    X_train = X[train_idx]
                    X_val = X[val_idx]
                    y_train = (
                        y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx].values
                    )
                    y_val = y[val_idx] if isinstance(y, np.ndarray) else y.iloc[val_idx].values
                else:
                    # 2D: X is DataFrame
                    X_train = X.iloc[train_idx].values
                    X_val = X.iloc[val_idx].values
                    y_train = y.iloc[train_idx].values
                    y_val = y.iloc[val_idx].values

                w_train = None
                if sample_weights is not None:
                    if isinstance(sample_weights, np.ndarray):
                        w_train = sample_weights[train_idx]
                    else:
                        w_train = sample_weights.iloc[train_idx].values

                # Train and evaluate - inject max_epochs if configured
                model_params = dict(params)
                if self.max_epochs is not None:
                    model_params["max_epochs"] = self.max_epochs
                    model_params["early_stopping_patience"] = max(1, self.max_epochs // 2)
                model = ModelRegistry.create(self.model_name, config=model_params)
                fit_config = {}
                if self.max_epochs is not None:
                    fit_config["max_epochs"] = self.max_epochs
                    fit_config["early_stopping_patience"] = max(1, self.max_epochs // 2)
                model.fit(X_train, y_train, X_val, y_val, sample_weights=w_train, config=fit_config)

                # Use configured metric instead of hardcoded val_f1
                # score_fn takes (y_true, y_pred) and returns a score
                pred_result = model.predict(X_val)
                y_pred = pred_result.class_predictions
                fold_score = score_fn(y_val, y_pred)
                scores.append(fold_score)

                # Free fold model to prevent memory accumulation over 100 trials
                del model, pred_result

                # Report intermediate value for pruning (prune bad trials early)
                trial.report(float(np.mean(scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            gc.collect()
            # Return mean score with variance penalty
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            penalty = 0.1 * std_score
            return float(mean_score - penalty)

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
            callbacks=[_trial_cleanup_callback],
        )

        # Compute Deflated Sharpe Ratio to correct for selection bias
        # DSR is only valid for Sharpe-like metrics (unbounded ratios).
        # For bounded metrics (F1, accuracy), DSR math is invalid — skip.
        dsr_result = None
        if is_sharpe_like_metric(self.metric):
            try:
                dsr_result = compute_dsr_from_optuna_study(
                    study, deployment_threshold=0.5, metric_name=self.metric
                )
                logger.info(
                    f"DSR computed: Raw={dsr_result.sharpe_ratio:.3f}, "
                    f"Deflated={dsr_result.deflated_sharpe:.3f}, "
                    f"Deploy={dsr_result.should_deploy}"
                )
            except Exception as e:
                logger.warning(f"Failed to compute DSR: {e}")
        else:
            logger.info(
                f"DSR skipped: metric '{self.metric}' is not a Sharpe-like ratio. "
                f"DSR deflation is only valid for unbounded Sharpe-like distributions."
            )

        result = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }

        # Add DSR metrics if computed successfully
        if dsr_result is not None:
            result["dsr"] = {
                "raw_sharpe": dsr_result.sharpe_ratio,
                "deflated_sharpe": dsr_result.deflated_sharpe,
                "deflation_pct": dsr_result.get_deflation_pct(),
                "is_significant": dsr_result.is_significant,
                "should_deploy": dsr_result.should_deploy,
                "risk_level": dsr_result.get_risk_level(),
            }

        return result

    def _sample_params(self, trial, param_space: dict) -> dict:
        """
        Sample parameters from search space with constraint enforcement.

        For LightGBM, enforces: num_leaves <= 2^max_depth
        """
        params = {}

        # For LightGBM, sample max_depth first to constrain num_leaves
        is_lightgbm = "num_leaves" in param_space and "max_depth" in param_space

        if is_lightgbm:
            # Sample max_depth first
            depth_spec = param_space["max_depth"]
            max_depth = trial.suggest_int("max_depth", depth_spec["low"], depth_spec["high"])
            params["max_depth"] = max_depth

            # Constrain num_leaves based on max_depth
            leaves_spec = param_space["num_leaves"]
            max_valid_leaves = get_max_leaves_for_depth(max_depth)
            # Use the smaller of: spec upper bound, 2^max_depth, or 128 (for regularization)
            constrained_high = min(leaves_spec["high"], max_valid_leaves, 128)
            constrained_low = min(leaves_spec["low"], constrained_high)

            params["num_leaves"] = trial.suggest_int(
                "num_leaves", constrained_low, constrained_high
            )

        # Sample remaining parameters
        for name, spec in param_space.items():
            if name in params:
                continue  # Already sampled (max_depth, num_leaves for LightGBM)

            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])

        # Apply validation as a safety net
        if is_lightgbm:
            params = validate_lightgbm_params(params)

        return params


__all__ = ["TimeSeriesOptunaTuner"]
