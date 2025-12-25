"""
Cross-Validation Runner.

Orchestrates cross-validation for all models and horizons,
optionally including hyperparameter tuning with Optuna.
Generates OOF predictions and stacking datasets for Phase 4.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.cross_validation.purged_kfold import PurgedKFold, ModelAwareCV
from src.cross_validation.feature_selector import WalkForwardFeatureSelector
from src.cross_validation.oof_generator import OOFGenerator, OOFPrediction, StackingDataset
from src.cross_validation.param_spaces import PARAM_SPACES
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FoldMetrics:
    """Metrics from a single CV fold."""
    fold: int
    train_size: int
    val_size: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    training_time: float
    val_loss: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "training_time": self.training_time,
            "val_loss": self.val_loss,
        }


@dataclass
class CVResult:
    """
    Results from cross-validation run.

    Attributes:
        model_name: Name of the model
        horizon: Label horizon
        fold_metrics: List of per-fold metrics
        oos_predictions: DataFrame with OOF predictions
        feature_importance: Feature importance DataFrame
        tuned_params: Best hyperparameters from tuning
        selected_features: Features selected by walk-forward selection
        total_time: Total CV time in seconds
    """
    model_name: str
    horizon: int
    fold_metrics: List[FoldMetrics]
    oos_predictions: pd.DataFrame
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    tuned_params: Dict[str, Any] = field(default_factory=dict)
    selected_features: List[str] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def n_folds(self) -> int:
        return len(self.fold_metrics)

    @property
    def mean_accuracy(self) -> float:
        return np.mean([m.accuracy for m in self.fold_metrics])

    @property
    def mean_f1(self) -> float:
        return np.mean([m.f1 for m in self.fold_metrics])

    @property
    def std_f1(self) -> float:
        return np.std([m.f1 for m in self.fold_metrics])

    def get_stability_score(self) -> float:
        """Coefficient of variation for F1 score (lower = more stable)."""
        mean = self.mean_f1
        std = self.std_f1
        return std / mean if mean > 0 else float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "n_folds": self.n_folds,
            "mean_accuracy": self.mean_accuracy,
            "mean_f1": self.mean_f1,
            "std_f1": self.std_f1,
            "stability_score": self.get_stability_score(),
            "tuned_params": self.tuned_params,
            "n_selected_features": len(self.selected_features),
            "total_time": self.total_time,
            "fold_metrics": [m.to_dict() for m in self.fold_metrics],
        }


# =============================================================================
# HYPERPARAMETER TUNER
# =============================================================================

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
        sample_weights: Optional[pd.Series] = None,
        param_space: Optional[Dict] = None,
    ) -> Dict[str, Any]:
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

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }

    def _sample_params(self, trial, param_space: Dict) -> Dict:
        """Sample parameters from search space."""
        params = {}
        for name, spec in param_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"],
                    log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params


# =============================================================================
# CROSS-VALIDATION RUNNER
# =============================================================================

class CrossValidationRunner:
    """
    Orchestrates cross-validation for generating OOS predictions.

    Runs CV for all specified models and horizons, optionally including
    feature selection and hyperparameter tuning.

    Example:
        >>> cv_config = PurgedKFoldConfig(n_splits=5, purge_bars=60, embargo_bars=1440)
        >>> cv = PurgedKFold(cv_config)
        >>> runner = CrossValidationRunner(cv, models=["xgboost"], horizons=[5, 10, 20])
        >>> results = runner.run(container)
    """

    def __init__(
        self,
        cv: PurgedKFold,
        models: List[str],
        horizons: List[int],
        tune_hyperparams: bool = True,
        select_features: bool = True,
        n_features_to_select: int = 50,
        tuning_trials: int = 50,
    ) -> None:
        """
        Initialize CrossValidationRunner.

        Args:
            cv: PurgedKFold cross-validator
            models: List of model names to train
            horizons: List of horizons to process
            tune_hyperparams: Whether to run Optuna tuning
            select_features: Whether to run walk-forward feature selection
            n_features_to_select: Number of features to select
            tuning_trials: Number of Optuna trials per model
        """
        self.cv = cv
        self.models = models
        self.horizons = horizons
        self.tune_hyperparams = tune_hyperparams
        self.select_features = select_features
        self.n_features_to_select = n_features_to_select
        self.tuning_trials = tuning_trials

    def run(
        self,
        container: "TimeSeriesDataContainer",
    ) -> Dict[Tuple[str, int], CVResult]:
        """
        Run cross-validation for all models and horizons.

        Args:
            container: TimeSeriesDataContainer with data

        Returns:
            Dict mapping (model_name, horizon) to CVResult
        """
        from src.phase1.stages.datasets.container import TimeSeriesDataContainer

        results: Dict[Tuple[str, int], CVResult] = {}

        for model_name in self.models:
            for horizon in self.horizons:
                logger.info(f"Running CV for {model_name} on H{horizon}...")

                result = self._run_single_cv(container, model_name, horizon)
                results[(model_name, horizon)] = result

                logger.info(
                    f"  {model_name}/H{horizon}: "
                    f"F1={result.mean_f1:.3f} (+/- {result.std_f1:.3f})"
                )

        return results

    def _run_single_cv(
        self,
        container: "TimeSeriesDataContainer",
        model_name: str,
        horizon: int,
    ) -> CVResult:
        """Run CV for single model/horizon combination."""
        start_time = time.time()

        # Get data for this horizon
        X, y, weights = container.get_sklearn_arrays("train", return_df=True)

        # Get model family for CV adaptation
        try:
            model_info = ModelRegistry.get_model_info(model_name)
            model_family = model_info.get("family", "boosting")
        except ValueError:
            model_family = "boosting"

        # Adapt CV for model family
        model_cv = ModelAwareCV(model_family, self.cv)
        cv_splits = list(model_cv.get_cv_splits(X, y))

        # Feature selection (if enabled)
        selected_features = list(X.columns)
        if self.select_features:
            selector = WalkForwardFeatureSelector(
                n_features_to_select=self.n_features_to_select
            )
            selection_result = selector.select_features_walkforward(X, y, cv_splits)
            selected_features = selection_result.stable_features
            if selected_features:
                X = X[selected_features]
            logger.debug(f"  Selected {len(selected_features)} stable features")

        # Hyperparameter tuning (if enabled)
        tuned_params: Dict[str, Any] = {}
        if self.tune_hyperparams:
            tuner = TimeSeriesOptunaTuner(
                model_name=model_name,
                cv=self.cv,
                n_trials=self.tuning_trials,
            )
            tuning_result = tuner.tune(X, y, weights)
            tuned_params = tuning_result.get("best_params", {})
            logger.debug(f"  Tuned params: {tuned_params}")

        # Get default config and merge with tuned params
        try:
            default_config = ModelRegistry.get_model_info(model_name).get("default_config", {})
        except ValueError:
            default_config = {}
        config = {**default_config, **tuned_params}

        # Generate OOF predictions
        oof_generator = OOFGenerator(self.cv)
        oof_predictions = oof_generator.generate_oof_predictions(
            X=X,
            y=y,
            model_configs={model_name: config},
            sample_weights=weights,
        )

        oof_pred = oof_predictions[model_name]

        # Extract fold metrics from OOF generation
        fold_metrics = []
        for fi in oof_pred.fold_info:
            fold_metrics.append(FoldMetrics(
                fold=fi["fold"],
                train_size=fi["train_size"],
                val_size=fi["val_size"],
                accuracy=fi.get("val_accuracy", 0.0),
                f1=fi.get("val_f1", 0.0),
                precision=0.0,  # Not tracked in basic OOF
                recall=0.0,
                training_time=0.0,
            ))

        total_time = time.time() - start_time

        return CVResult(
            model_name=model_name,
            horizon=horizon,
            fold_metrics=fold_metrics,
            oos_predictions=oof_pred.predictions,
            tuned_params=tuned_params,
            selected_features=selected_features,
            total_time=total_time,
        )

    def build_stacking_datasets(
        self,
        cv_results: Dict[Tuple[str, int], CVResult],
        container: "TimeSeriesDataContainer",
    ) -> Dict[int, StackingDataset]:
        """
        Build stacking datasets from CV results.

        Args:
            cv_results: Results from run()
            container: Original data container

        Returns:
            Dict mapping horizon to StackingDataset
        """
        stacking_datasets: Dict[int, StackingDataset] = {}

        for horizon in self.horizons:
            # Collect all model predictions for this horizon
            oof_predictions: Dict[str, OOFPrediction] = {}

            for (model_name, h), result in cv_results.items():
                if h != horizon:
                    continue

                oof_predictions[model_name] = OOFPrediction(
                    model_name=model_name,
                    predictions=result.oos_predictions,
                    fold_info=[m.to_dict() for m in result.fold_metrics],
                    coverage=1.0,
                )

            if not oof_predictions:
                logger.warning(f"No predictions for horizon {horizon}")
                continue

            # Get true labels
            _, y, _ = container.get_sklearn_arrays("train", return_df=True)

            # Build stacking dataset
            oof_gen = OOFGenerator(self.cv)
            stacking_ds = oof_gen.build_stacking_dataset(
                oof_predictions=oof_predictions,
                y_true=y,
                horizon=horizon,
            )
            stacking_datasets[horizon] = stacking_ds

        return stacking_datasets

    def save_results(
        self,
        cv_results: Dict[Tuple[str, int], CVResult],
        stacking_datasets: Dict[int, StackingDataset],
        output_dir: Path,
    ) -> None:
        """
        Save all CV results and stacking datasets.

        Args:
            cv_results: Results from run()
            stacking_datasets: Stacking datasets from build_stacking_datasets()
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CV results summary
        summary = {
            "models": self.models,
            "horizons": self.horizons,
            "cv_config": {
                "n_splits": self.cv.config.n_splits,
                "purge_bars": self.cv.config.purge_bars,
                "embargo_bars": self.cv.config.embargo_bars,
            },
            "results": {},
        }

        for (model_name, horizon), result in cv_results.items():
            key = f"{model_name}_h{horizon}"
            summary["results"][key] = result.to_dict()

        with open(output_dir / "cv_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save stacking datasets
        stacking_dir = output_dir / "stacking"
        stacking_dir.mkdir(exist_ok=True)

        for horizon, stacking_ds in stacking_datasets.items():
            OOFGenerator(self.cv).save_stacking_dataset(stacking_ds, stacking_dir)

        # Save tuned parameters
        params_dir = output_dir / "tuned_params"
        params_dir.mkdir(exist_ok=True)

        for (model_name, horizon), result in cv_results.items():
            if result.tuned_params:
                params_path = params_dir / f"{model_name}_h{horizon}.json"
                with open(params_path, "w") as f:
                    json.dump(result.tuned_params, f, indent=2)

        logger.info(f"Saved CV results to {output_dir}")


# =============================================================================
# STABILITY ANALYSIS
# =============================================================================

def analyze_cv_stability(
    cv_results: Dict[Tuple[str, int], CVResult],
) -> pd.DataFrame:
    """
    Analyze stability of models across CV folds.

    Args:
        cv_results: Results from CrossValidationRunner

    Returns:
        DataFrame with stability metrics per model/horizon
    """
    stability_data = []

    for (model_name, horizon), result in cv_results.items():
        for metric_name in ["accuracy", "f1"]:
            values = [getattr(m, metric_name, 0.0) for m in result.fold_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val > 0 else float("inf")

            stability_data.append({
                "model": model_name,
                "horizon": horizon,
                "metric": metric_name,
                "mean": mean_val,
                "std": std_val,
                "cv": cv,
                "min": np.min(values),
                "max": np.max(values),
                "stability_grade": _grade_stability(cv),
            })

    return pd.DataFrame(stability_data)


def _grade_stability(cv: float) -> str:
    """Grade stability based on coefficient of variation."""
    if cv < 0.15:
        return "Excellent"
    elif cv < 0.25:
        return "Good"
    elif cv < 0.40:
        return "Acceptable"
    elif cv < 0.60:
        return "Poor"
    else:
        return "Unstable"


__all__ = [
    "FoldMetrics",
    "CVResult",
    "TimeSeriesOptunaTuner",
    "CrossValidationRunner",
    "analyze_cv_stability",
    "PARAM_SPACES",
]
