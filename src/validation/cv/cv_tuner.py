"""
Hyperparameter Tuning for Time Series Cross-Validation.

Uses Optuna's TPE sampler with time-series aware objective.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import logging
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.core.label_spans import LabelSpans
from src.models.registry import ModelRegistry
from src.validation.deflated_sharpe import (
    compute_dsr_from_optuna_study,
    is_sharpe_like_metric,
)

from .early_stopping_split import carve_early_stopping_split
from .fold_scaling import FoldAwareScaler, get_scaling_method_for_model
from .param_spaces import (
    PARAM_SPACES,
    get_max_leaves_for_depth,
    validate_lightgbm_params,
)
from .purged_kfold import PurgedKFold

logger = logging.getLogger(__name__)

# Default kept for backward compatibility when no OptunaConfig is provided.
_DEFAULT_MAX_SAMPLES = 50_000
_DEFAULT_VARIANCE_PENALTY = 0.1
_DEFAULT_N_STARTUP_TRIALS = 10


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
        metric: str = "f1_weighted",
        pruner: Any | None = None,
        max_epochs: int | None = None,
        n_startup_trials: int = _DEFAULT_N_STARTUP_TRIALS,
        variance_penalty: float = _DEFAULT_VARIANCE_PENALTY,
        max_samples: int = _DEFAULT_MAX_SAMPLES,
        timeout: int | None = None,
        purge_bars: int | None = None,
        scale_per_fold: bool = False,
    ) -> None:
        """
        Args:
            purge_bars: Bars dropped between a fold's fit rows and its
                early-stopping tail (default: ``cv.config.purge_bars``).
            scale_per_fold: Scale each fold's features with the model's scaler fit
                on that fold's fit rows only (2D data). Set it when ``X`` is
                unscaled (standalone ``ml cv``); leave it off when ``X`` was
                already scaled by the training pipeline.
        """
        self.model_name = model_name
        self.cv = cv
        if purge_bars is None:
            purge_bars = getattr(getattr(cv, "config", None), "purge_bars", 0)
        self.purge_bars = int(purge_bars or 0)
        self.n_trials = n_trials
        self.direction = direction
        self.metric = metric
        self.pruner = pruner
        self.max_epochs = max_epochs
        self.n_startup_trials = n_startup_trials
        self.variance_penalty = variance_penalty
        self.max_samples = max_samples
        self.timeout = timeout
        self.scale_per_fold = scale_per_fold

    def tune(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weights: pd.Series | np.ndarray | None = None,
        param_space: dict | None = None,
        data_rank: int = 2,
        label_spans: LabelSpans | None = None,
    ) -> dict[str, Any]:
        """
        Run hyperparameter tuning.

        Args:
            X: Features (DataFrame for 2D, ndarray for 3D/4D)
            y: Labels (Series for 2D, ndarray for 3D/4D)
            sample_weights: Optional quality weights
            param_space: Search space (uses defaults if None)
            data_rank: Dimensionality of data (2, 3, or 4)
            label_spans: Optional per-sample label spans (bar positions); CV
                folds purge training samples whose labels overlap validation

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
        # Capping at max_samples rows keeps peak RSS under ~8 GB for float32 3D data.
        #
        # IMPORTANT: Use strided (every-Nth) sampling instead of random to preserve
        # temporal structure. Random sampling collapses temporal gaps so that
        # purge/embargo becomes ~2 real bars on 1.6M rows.
        cv = self.cv
        max_samples = self.max_samples
        original_n = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        stride = 1
        if original_n > max_samples:
            stride = max(1, original_n // max_samples)
            sub_indices = np.arange(0, original_n, stride)[:max_samples]
            logger.info(
                f"  Subsampling {original_n} -> {len(sub_indices)} rows for Optuna "
                f"(strided, every {stride}th sample)"
            )
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
            # Spans stay in bar positions, so purging stays exact after striding
            if label_spans is not None:
                label_spans = label_spans.subset(sub_indices)
            # Apply to sample_weights if present
            if sample_weights is not None:
                if isinstance(sample_weights, np.ndarray):
                    sample_weights = sample_weights[sub_indices]
                else:
                    sample_weights = sample_weights.iloc[sub_indices].reset_index(drop=True)
            # Scale embargo proportionally: subsampled data has compressed indices
            # so the original embargo_bars must be scaled down by the same stride.
            # A copy: the caller's CV (shared across models and horizons) must keep
            # its embargo.
            if hasattr(cv, "config") and hasattr(cv.config, "embargo_bars"):
                orig_embargo = cv.config.embargo_bars
                scaled_embargo = max(1, orig_embargo // stride)
                cv = copy.copy(cv)
                cv.config = dataclasses.replace(cv.config, embargo_bars=scaled_embargo)
                logger.info(
                    f"  Scaled embargo: {orig_embargo} -> {scaled_embargo} " f"(stride={stride})"
                )

        # --- float32 conversion: halves memory for all 500 fold slices ---
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32, copy=False)
        else:
            X = X.astype(np.float32, copy=False)

        # Create study with optional pruner for early stopping of bad trials
        default_pruner = optuna.pruners.MedianPruner(
            n_startup_trials=self.n_startup_trials,
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
            from src.models.device import release_gpu_memory

            release_gpu_memory()

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
            self._precomputed_splits = list(cv.split(X_for_cv, y_for_cv, label_spans=label_spans))
        else:
            self._precomputed_splits = list(cv.split(X, y, label_spans=label_spans))

        # Early stopping (boosting rounds, best-epoch restore) selects on a
        # purged tail of each fold's TRAIN rows, never on the scored fold —
        # a trial stopped on the rows it is scored on is optimistic. After
        # strided subsampling one sample spans `stride` bars (ceil: never
        # under-purge).
        es_purge = -(-self.purge_bars // stride)
        fold_plans = [
            (carve_early_stopping_split(train_idx, es_purge), val_idx)
            for train_idx, val_idx in self._precomputed_splits
        ]

        # A fold whose fit or scored labels hold fewer than two classes
        # carries no hyperparameter signal: a constant predictor scores a
        # perfect F1 on it, so the trial would look like the best one found.
        # Labels are fixed across trials, so the check runs once; every trial
        # then scores the worst possible value and no model is fit.
        y_arr = np.asarray(y)
        degenerate_folds = [
            fold_idx
            for fold_idx, (es_split, val_idx) in enumerate(fold_plans)
            if np.unique(y_arr[es_split.fit_idx]).size < 2 or np.unique(y_arr[val_idx]).size < 2
        ]
        worst_value = float("-inf") if self.direction == "maximize" else float("inf")
        if degenerate_folds:
            logger.warning(
                f"  Degenerate labels: folds {degenerate_folds} have fewer than 2 classes "
                f"in their fit or scored rows — every trial scores {worst_value}"
            )

        # Rank-agnostic arrays, indexed by sample (axis 0)
        X_arr = X if isinstance(X, np.ndarray) else X.to_numpy()
        w_arr = np.asarray(sample_weights) if sample_weights is not None else None

        # Unscaled input (standalone `ml cv`): scale each fold with statistics from
        # its fit rows only. Fold scaling does not depend on the trial, so it is
        # computed once and reused by every trial.
        scaled_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
        if self.scale_per_fold and data_rank == 2 and not degenerate_folds:
            scaler = FoldAwareScaler(method=get_scaling_method_for_model(self.model_name))
            scaled_folds = []
            for es_split, val_idx in fold_plans:
                fit_idx, es_idx = es_split.fit_idx, es_split.es_idx
                scaled = scaler.fit_transform_fold(
                    X_arr[fit_idx], np.vstack([X_arr[es_idx], X_arr[val_idx]])
                )
                scaled_folds.append(
                    (
                        scaled.X_train_scaled,
                        scaled.X_val_scaled[: len(es_idx)],
                        scaled.X_val_scaled[len(es_idx) :],
                    )
                )

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, param_space)
            if degenerate_folds:
                return worst_value

            scores = []
            for fold_idx, (es_split, val_idx) in enumerate(fold_plans):
                fit_idx, es_idx = es_split.fit_idx, es_split.es_idx
                y_train, y_es, y_val = y_arr[fit_idx], y_arr[es_idx], y_arr[val_idx]
                if scaled_folds is not None:
                    X_train, X_es, X_val = scaled_folds[fold_idx]
                else:
                    X_train, X_es, X_val = X_arr[fit_idx], X_arr[es_idx], X_arr[val_idx]
                w_train = w_arr[fit_idx] if w_arr is not None else None

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
                model.fit(X_train, y_train, X_es, y_es, sample_weights=w_train, config=fit_config)

                # Score on the untouched fold with the configured metric
                # (score_fn takes (y_true, y_pred) and returns a score)
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
            penalty = self.variance_penalty * std_score
            return float(mean_score - penalty)

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        timeout = self.timeout if self.timeout and self.timeout > 0 else None
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=timeout,
            show_progress_bar=False,
            callbacks=[_trial_cleanup_callback],
        )

        if not np.isfinite(study.best_value):
            # Only degenerate trials completed — their params carry no signal,
            # so callers keep the model defaults.
            logger.warning("  No trial produced a finite score; keeping default hyperparameters")
            return {
                "best_params": {},
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "skipped": True,
            }

        # Compute Deflated Sharpe Ratio to correct for selection bias
        # DSR is only valid for Sharpe-like metrics (unbounded ratios).
        # For bounded metrics (F1, accuracy), DSR math is invalid — skip.
        # Trial values are annualized proxy Sharpes (src.optimization.scoring);
        # T is the number of distinct out-of-sample bars they were computed on.
        dsr_result = None
        if is_sharpe_like_metric(self.metric):
            try:
                from src.optimization.scoring import get_bars_per_year

                n_oos = len(np.unique(np.concatenate([v for _, v in fold_plans])))
                dsr_result = compute_dsr_from_optuna_study(
                    study,
                    n_observations=n_oos,
                    periods_per_year=get_bars_per_year(),
                    metric_name=self.metric,
                )
                logger.info(
                    f"DSR computed: per-bar Sharpe={dsr_result.sharpe_ratio:.4f}, "
                    f"DSR={dsr_result.dsr:.3f}, Deploy={dsr_result.should_deploy}"
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
                **dsr_result.to_dict(),
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
