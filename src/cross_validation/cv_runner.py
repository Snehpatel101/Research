"""
Cross-Validation Runner.

Orchestrates cross-validation for all models and horizons,
optionally including hyperparameter tuning with Optuna.
Generates OOF predictions and stacking datasets for Phase 4.

CV + Feature Selection Interaction (LBL-006)
============================================

This module provides configurable feature selection behavior during cross-validation.
Understanding how feature selection interacts with CV is critical to avoid data leakage.

**Default Behavior (feature_selection_inside_fold=True):**
    Feature selection is performed INSIDE each CV fold using only training data.
    This is the recommended approach as it prevents validation data from influencing
    which features are selected, eliminating feature selection leakage.

    For each fold:
    1. Features are ranked using mutual information on TRAINING data only
    2. Top N features are selected (default: n_features_to_select=50)
    3. Model is trained on selected features
    4. Predictions are made on validation fold using same features

    The final "stable features" are those appearing in >= 60% of folds.

**Hyperparameter Tuning Trade-offs:**
    When both feature selection and hyperparameter tuning are enabled, there are
    two strategies controlled by the `tune_per_fold` parameter:

    1. tune_per_fold=False (default): FASTER but potentially suboptimal
       - Hyperparameters are tuned ONCE on the FULL feature set before CV
       - Each fold then uses its own selected features with shared HPs
       - Good when feature selection is stable across folds
       - Risk: HPs may not be optimal for reduced feature subsets

    2. tune_per_fold=True: More ACCURATE but significantly slower
       - For each fold, after feature selection, HPs are tuned on selected features
       - Ensures HPs match the actual feature subset used
       - n_folds * n_trials evaluations (e.g., 5 folds * 50 trials = 250 evaluations)
       - Recommended when feature selection varies significantly between folds

**Configuration Examples:**
    # Fast: Tune once on all features, select features per-fold
    runner = CrossValidationRunner(
        cv=cv, models=["xgboost"], horizons=[20],
        tune_hyperparams=True,
        select_features=True,
        tune_per_fold=False,  # default
    )

    # Accurate: Tune inside each fold after feature selection
    runner = CrossValidationRunner(
        cv=cv, models=["xgboost"], horizons=[20],
        tune_hyperparams=True,
        select_features=True,
        tune_per_fold=True,  # slower but more accurate
    )

    # No feature selection (use all features)
    runner = CrossValidationRunner(
        cv=cv, models=["xgboost"], horizons=[20],
        select_features=False,
    )

**Legacy Behavior (NOT RECOMMENDED):**
    Setting feature_selection_inside_fold=False triggers legacy behavior where
    feature selection is performed BEFORE CV using all data. This causes data
    leakage and should be avoided. A warning is logged if this option is used.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.cross_validation.cv_dataclasses import CVResult, FoldMetrics
from src.cross_validation.cv_feature_selection import run_cv_with_per_fold_feature_selection
from src.cross_validation.cv_stacking import (
    _grade_stability,
    analyze_cv_stability,
    build_stacking_datasets_from_cv_results,
    validate_stacking_consistency,
)
from src.cross_validation.cv_tuner import TimeSeriesOptunaTuner
from src.feature_selection import WalkForwardFeatureSelector
from src.cross_validation.oof_generator import OOFGenerator, OOFPrediction, StackingDataset
from src.cross_validation.param_spaces import (
    PARAM_SPACES,
    get_max_leaves_for_depth,
    validate_lightgbm_params,
)
from src.cross_validation.purged_kfold import ModelAwareCV, PurgedKFold

# Re-export ModelRegistry for backward compatibility
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class CrossValidationRunner:
    """
    Orchestrates cross-validation for generating OOS predictions.

    Runs CV for all specified models and horizons, optionally including
    feature selection and hyperparameter tuning.

    IMPORTANT: Feature selection is performed INSIDE each CV fold to prevent
    data leakage. Features are selected using only training data from each fold,
    ensuring validation data never influences feature selection.

    Hyperparameter Tuning Trade-offs:
        By default, hyperparameters are tuned on the FULL feature set before
        per-fold feature selection. This is an efficiency trade-off:

        - tune_per_fold=False (default): Faster, HPs tuned once on all features.
          Each fold then uses its own selected feature subset with shared HPs.
          This may be suboptimal if feature selection varies significantly.

        - tune_per_fold=True: More accurate but slower. HPs are tuned inside
          each fold AFTER feature selection, ensuring HPs match the actual
          feature subset used. Use when fold-level feature variation is high.

    Example:
        >>> cv_config = PurgedKFoldConfig(n_splits=5, purge_bars=60, embargo_bars=1440)
        >>> cv = PurgedKFold(cv_config)
        >>> runner = CrossValidationRunner(cv, models=["xgboost"], horizons=[5, 10, 20])
        >>> results = runner.run(container)
    """

    def __init__(
        self,
        cv: PurgedKFold,
        models: list[str],
        horizons: list[int],
        tune_hyperparams: bool = True,
        select_features: bool = True,
        n_features_to_select: int = 50,
        tuning_trials: int = 50,
        feature_selection_inside_fold: bool = True,
        tune_per_fold: bool = False,
    ) -> None:
        """
        Initialize CrossValidationRunner.

        Args:
            cv: PurgedKFold cross-validator
            models: List of model names to train
            horizons: List of horizons to process
            tune_hyperparams: Whether to run Optuna tuning. Default True.
            select_features: Whether to run walk-forward feature selection. Default True.
                When True, uses mutual information to rank and select top features.
            n_features_to_select: Number of features to select per fold. Default 50.
                Only used when select_features=True.
            tuning_trials: Number of Optuna trials per model. Default 50.
            feature_selection_inside_fold: If True (default), feature selection is
                performed inside each fold using only training data, preventing leakage.
                If False, uses legacy behavior (NOT RECOMMENDED - causes leakage).
            tune_per_fold: Controls when hyperparameters are tuned relative to feature
                selection. Only relevant when both tune_hyperparams=True and
                select_features=True. Default False.

                - False (default): Tune HPs once on FULL feature set BEFORE CV.
                  Faster (n_trials evaluations), but HPs may not be optimal for
                  the reduced feature subsets used in each fold.

                - True: Tune HPs inside each fold AFTER feature selection.
                  More accurate (HPs match actual features used), but slower
                  (n_folds * n_trials evaluations, e.g., 5*50=250).

        Notes:
            Default behavior summary:
            1. tune_hyperparams=True: Tune once on all features (tune_per_fold=False)
            2. select_features=True: Select features per-fold using only training data
            3. Feature selection uses mutual information for ranking
            4. "Stable features" are those appearing in >= 60% of folds
        """
        self.cv = cv
        self.models = models
        self.horizons = horizons
        self.tune_hyperparams = tune_hyperparams
        self.select_features = select_features
        self.n_features_to_select = n_features_to_select
        self.tuning_trials = tuning_trials
        self.feature_selection_inside_fold = feature_selection_inside_fold
        self.tune_per_fold = tune_per_fold
        self._label_end_times_warning_shown = False

    def _validate_label_end_times(
        self,
        label_end_times: pd.Series | None,
        model_family: str,
    ) -> None:
        """
        Warn if label_end_times not provided for trading models.

        For triple-barrier or other overlapping label schemes, label_end_times
        is critical for proper purging. Without it, the CV splitter cannot
        properly purge samples whose labels overlap with validation data,
        potentially causing label leakage.

        Args:
            label_end_times: Optional Series of datetime when each label is resolved
            model_family: Model family (boosting, neural, etc.)
        """
        if label_end_times is None and not self._label_end_times_warning_shown:
            logger.warning(
                "label_end_times not provided. This may cause label leakage "
                "for models with overlapping labels (e.g., triple-barrier labeling). "
                "Consider providing label_end_times to TimeSeriesDataContainer for proper purging. "
                "Set label_end_times=None explicitly to suppress this warning if labels are non-overlapping."
            )
            self._label_end_times_warning_shown = True

    def run(
        self,
        container,
    ) -> dict[tuple[str, int], CVResult]:
        """
        Run cross-validation for all models and horizons.

        Args:
            container: TimeSeriesDataContainer with data

        Returns:
            Dict mapping (model_name, horizon) to CVResult
        """

        results: dict[tuple[str, int], CVResult] = {}

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
        container,
        model_name: str,
        horizon: int,
    ) -> CVResult:
        """Run CV for single model/horizon combination."""
        start_time = time.time()

        # Get data for this horizon
        X, y, weights = container.get_sklearn_arrays("train", return_df=True)
        all_feature_names = list(X.columns)

        # Get model family for CV adaptation
        try:
            model_info = ModelRegistry.get_model_info(model_name)
            model_family = model_info.get("family", "boosting")
        except ValueError:
            model_family = "boosting"

        # Validate label_end_times is provided for proper purging
        label_end_times = None
        if hasattr(container, "get_label_end_times"):
            label_end_times = container.get_label_end_times("train")
        self._validate_label_end_times(label_end_times, model_family)

        # Adapt CV for model family
        model_cv = ModelAwareCV(model_family, self.cv)
        cv_splits = list(model_cv.get_cv_splits(X, y))

        # ==================================================================
        # HYPERPARAMETER TUNING (if enabled)
        # ==================================================================
        tuned_params: dict[str, Any] = {}
        if self.tune_hyperparams and not self.tune_per_fold:
            logger.debug(
                "Tuning hyperparameters on full feature set (tune_per_fold=False). "
                "Set tune_per_fold=True for per-fold HP tuning after feature selection."
            )
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

        # ==================================================================
        # LEAKAGE-FREE FEATURE SELECTION AND OOF GENERATION
        # ==================================================================

        if self.select_features and self.feature_selection_inside_fold:
            # Per-fold feature selection (LEAKAGE-FREE)
            oof_result = run_cv_with_per_fold_feature_selection(
                X=X,
                y=y,
                weights=weights,
                cv_splits=cv_splits,
                model_name=model_name,
                config=config,
                n_features_to_select=self.n_features_to_select,
                tune_per_fold=self.tune_per_fold and self.tune_hyperparams,
                cv=self.cv,
                tuning_trials=self.tuning_trials,
            )
            oof_pred = oof_result["oof_prediction"]
            selected_features = oof_result["selected_features"]
            fold_metrics = oof_result["fold_metrics"]
        else:
            # Legacy behavior or no feature selection
            selected_features = all_feature_names

            if self.select_features and not self.feature_selection_inside_fold:
                # Legacy behavior with warning (NOT RECOMMENDED)
                logger.warning(
                    "Feature selection with feature_selection_inside_fold=False "
                    "causes data leakage. Use feature_selection_inside_fold=True."
                )
                selector = WalkForwardFeatureSelector(
                    n_features_to_select=self.n_features_to_select
                )
                selection_result = selector.select_features_walkforward(X, y, cv_splits)
                selected_features = selection_result.stable_features
                if selected_features:
                    X = X[selected_features]

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
                fold_metrics.append(
                    FoldMetrics(
                        fold=fi["fold"],
                        train_size=fi["train_size"],
                        val_size=fi["val_size"],
                        accuracy=fi.get("val_accuracy", 0.0),
                        f1=fi.get("val_f1", 0.0),
                        precision=0.0,  # Not tracked in basic OOF
                        recall=0.0,
                        training_time=0.0,
                    )
                )

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
        cv_results: dict[tuple[str, int], CVResult],
        container,
    ) -> dict[int, StackingDataset]:
        """
        Build stacking datasets from CV results.

        Args:
            cv_results: Results from run()
            container: Original data container

        Returns:
            Dict mapping horizon to StackingDataset

        Raises:
            ValueError: If OOF predictions have inconsistent sample counts
        """
        return build_stacking_datasets_from_cv_results(
            cv_results=cv_results,
            horizons=self.horizons,
            cv=self.cv,
            container=container,
        )

    def save_results(
        self,
        cv_results: dict[tuple[str, int], CVResult],
        stacking_datasets: dict[int, StackingDataset],
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
# BACKWARD COMPATIBILITY EXPORTS
# =============================================================================

# Re-export all components from split modules for backward compatibility
__all__ = [
    # Core classes
    "CrossValidationRunner",
    # Dataclasses (from cv_dataclasses)
    "FoldMetrics",
    "CVResult",
    # Tuner (from cv_tuner)
    "TimeSeriesOptunaTuner",
    # Stacking utilities (from cv_stacking)
    "analyze_cv_stability",
    # Param spaces (from param_spaces)
    "PARAM_SPACES",
    "validate_lightgbm_params",
    "get_max_leaves_for_depth",
]
