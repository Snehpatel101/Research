"""
Hyperparameter Tuning for Time Series Cross-Validation.

Uses Optuna's TPE sampler with time-series aware objective.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .param_spaces import (
    PARAM_SPACES,
    get_max_leaves_for_depth,
    validate_lightgbm_params,
)
from .purged_kfold import PurgedKFold
from src.models.registry import ModelRegistry
from src.validation.deflated_sharpe import compute_dsr_from_optuna_study

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.model_name = model_name
        self.cv = cv
        self.n_trials = n_trials
        self.direction = direction
        self.metric = metric

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        param_space: dict | None = None,
    ) -> dict[str, Any]:
        """
        Run hyperparameter tuning.

        Args:
            X: Features
            y: Labels
            sample_weights: Optional quality weights
            param_space: Search space (uses defaults if None)

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

        # Create study
        study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=42),
        )

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, param_space)

            scores = []
            for train_idx, val_idx in self.cv.split(X, y):
                X_train = X.iloc[train_idx].values
                X_val = X.iloc[val_idx].values
                y_train = y.iloc[train_idx].values
                y_val = y.iloc[val_idx].values

                w_train = None
                if sample_weights is not None:
                    w_train = sample_weights.iloc[train_idx].values

                # Train and evaluate
                model = ModelRegistry.create(self.model_name, config=params)
                metrics = model.fit(X_train, y_train, X_val, y_val, sample_weights=w_train)
                scores.append(metrics.val_f1)

            # Return mean score with variance penalty
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            penalty = 0.1 * std_score
            return mean_score - penalty

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        # Compute Deflated Sharpe Ratio to correct for selection bias
        dsr_result = None
        try:
            # Assumes the objective is maximizing a performance metric (e.g., F1 or Sharpe)
            # For deployment gating, DSR > 0.5 is recommended threshold
            dsr_result = compute_dsr_from_optuna_study(study, deployment_threshold=0.5)
            logger.info(
                f"DSR computed: Raw={dsr_result.sharpe_ratio:.3f}, "
                f"Deflated={dsr_result.deflated_sharpe:.3f}, "
                f"Deploy={dsr_result.should_deploy}"
            )
        except Exception as e:
            logger.warning(f"Failed to compute DSR: {e}")

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
