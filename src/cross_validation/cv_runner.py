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
from typing import Any

import numpy as np
import pandas as pd

from src.cross_validation.feature_selector import WalkForwardFeatureSelector
from src.cross_validation.oof_generator import OOFGenerator, OOFPrediction, StackingDataset
from src.cross_validation.param_spaces import (
    PARAM_SPACES,
    get_max_leaves_for_depth,
    validate_lightgbm_params,
)
from src.cross_validation.purged_kfold import ModelAwareCV, PurgedKFold
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

    def to_dict(self) -> dict[str, Any]:
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
    fold_metrics: list[FoldMetrics]
    oos_predictions: pd.DataFrame
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    tuned_params: dict[str, Any] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)
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

    def to_dict(self) -> dict[str, Any]:
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

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }

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
                    name, spec["low"], spec["high"],
                    log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])

        # Apply validation as a safety net
        if is_lightgbm:
            params = validate_lightgbm_params(params)

        return params


# =============================================================================
# CROSS-VALIDATION RUNNER
# =============================================================================

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
            tune_hyperparams: Whether to run Optuna tuning
            select_features: Whether to run walk-forward feature selection
            n_features_to_select: Number of features to select
            tuning_trials: Number of Optuna trials per model
            feature_selection_inside_fold: If True (default), feature selection is
                performed inside each fold using only training data, preventing leakage.
                If False, uses legacy behavior (NOT RECOMMENDED - causes leakage).
            tune_per_fold: If True, hyperparameters are tuned inside each fold
                AFTER feature selection. More accurate but significantly slower
                (n_folds * n_trials evaluations). Default False tunes once on
                full feature set for efficiency.
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
        container: TimeSeriesDataContainer,
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
        container: TimeSeriesDataContainer,
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
        # By default (tune_per_fold=False), tuning is done on the FULL feature
        # set before per-fold feature selection. This is an efficiency trade-off:
        #
        # Trade-off: The tuned HPs may not be optimal for the per-fold feature
        # subsets if feature selection varies significantly between folds.
        # However, this approach is much faster (tuning once vs. n_folds times).
        #
        # For more accuracy at the cost of speed, set tune_per_fold=True to
        # tune HPs inside each fold after feature selection.
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
        # Feature selection is now performed INSIDE each fold to prevent
        # validation data from influencing feature selection.
        # ==================================================================

        if self.select_features and self.feature_selection_inside_fold:
            # Per-fold feature selection (LEAKAGE-FREE)
            oof_result = self._run_cv_with_per_fold_feature_selection(
                X=X,
                y=y,
                weights=weights,
                cv_splits=cv_splits,
                model_name=model_name,
                config=config,
                tune_per_fold=self.tune_per_fold and self.tune_hyperparams,
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

    def _run_cv_with_per_fold_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series | None,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        model_name: str,
        config: dict[str, Any],
        tune_per_fold: bool = False,
    ) -> dict[str, Any]:
        """
        Run CV with per-fold feature selection to prevent leakage.

        For each fold:
        1. Select features using ONLY training data from that fold
        2. Optionally tune hyperparameters on the selected features (if tune_per_fold=True)
        3. Train model on selected features
        4. Predict on validation set using selected features

        This ensures validation data never influences feature selection,
        eliminating the feature selection leakage issue.

        Args:
            X: Full feature DataFrame
            y: Labels
            weights: Sample weights
            cv_splits: List of (train_idx, val_idx) tuples
            model_name: Name of model to train
            config: Model configuration (may be overridden by per-fold tuning)
            tune_per_fold: If True, tune hyperparameters inside each fold after
                feature selection. More accurate but slower.

        Returns:
            Dict with oof_prediction, selected_features, fold_metrics
        """
        from collections import Counter

        from sklearn.feature_selection import mutual_info_classif

        n_samples = len(X)
        n_classes = 3
        all_features = list(X.columns)

        # Track which features are selected in each fold
        fold_selected_features: list[list[str]] = []

        # Initialize OOF prediction storage
        oof_predictions = np.full(n_samples, np.nan)
        oof_probabilities = np.full((n_samples, n_classes), np.nan)
        fold_metrics_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            fold_start = time.time()

            # Extract fold data
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            w_train = None
            if weights is not None:
                w_train = weights.iloc[train_idx].values

            # ============================================================
            # LEAKAGE-FREE FEATURE SELECTION: Use only training data
            # ============================================================
            # Use mutual information to rank features using ONLY training data
            mi_scores = mutual_info_classif(
                X_train_fold.values,
                y_train_fold.values,
                discrete_features=False,
                random_state=42,
            )

            # Select top N features based on MI scores from training data only
            feature_scores = list(zip(all_features, mi_scores, strict=False))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            fold_features = [f[0] for f in feature_scores[:self.n_features_to_select]]
            fold_selected_features.append(fold_features)

            # Subset data to selected features
            X_train_selected = X_train_fold[fold_features]
            X_val_selected = X_val_fold[fold_features]

            # ==============================================================
            # PER-FOLD HYPERPARAMETER TUNING (if enabled)
            # ==============================================================
            # When tune_per_fold=True, we tune hyperparameters on the selected
            # feature subset using only this fold's training data. This ensures
            # HPs are optimized for the actual features used in this fold.
            # ==============================================================
            fold_config = config.copy()
            if tune_per_fold:
                logger.debug(f"  Fold {fold_idx + 1}: Tuning HPs on {len(fold_features)} selected features...")
                # Create a mini CV for tuning within the training fold
                from src.cross_validation.purged_kfold import PurgedKFoldConfig
                inner_cv_config = PurgedKFoldConfig(
                    n_splits=min(3, len(train_idx) // 100),  # Fewer splits for inner CV
                    purge_bars=self.cv.config.purge_bars,
                    embargo_bars=self.cv.config.embargo_bars,
                )
                inner_cv = PurgedKFold(inner_cv_config)

                tuner = TimeSeriesOptunaTuner(
                    model_name=model_name,
                    cv=inner_cv,
                    n_trials=max(10, self.tuning_trials // 3),  # Fewer trials for inner tuning
                )
                tuning_result = tuner.tune(
                    X_train_selected,
                    y_train_fold,
                    weights.iloc[train_idx] if weights is not None else None,
                )
                fold_tuned_params = tuning_result.get("best_params", {})
                fold_config.update(fold_tuned_params)
                logger.debug(f"    Fold {fold_idx + 1} tuned params: {fold_tuned_params}")

            # Train model on selected features
            model = ModelRegistry.create(model_name, config=fold_config)
            model.fit(
                X_train=X_train_selected.values,
                y_train=y_train_fold.values,
                X_val=X_val_selected.values,
                y_val=y_val_fold.values,
                sample_weights=w_train,
            )

            # Generate OOF predictions for this fold's validation set
            output = model.predict(X_val_selected.values)
            oof_predictions[val_idx] = output.class_predictions
            oof_probabilities[val_idx] = output.class_probabilities

            # Compute fold metrics
            from sklearn.metrics import accuracy_score, f1_score
            fold_accuracy = accuracy_score(y_val_fold.values, output.class_predictions)
            fold_f1 = f1_score(y_val_fold.values, output.class_predictions, average="macro", zero_division=0)
            fold_time = time.time() - fold_start

            fold_metrics_list.append(FoldMetrics(
                fold=fold_idx,
                train_size=len(train_idx),
                val_size=len(val_idx),
                accuracy=fold_accuracy,
                f1=fold_f1,
                precision=0.0,
                recall=0.0,
                training_time=fold_time,
            ))

            logger.debug(
                f"  Fold {fold_idx + 1}: selected {len(fold_features)} features, "
                f"F1={fold_f1:.4f}"
            )

        # Aggregate selected features: keep features that appear in >= 60% of folds
        feature_counts = Counter()
        for fold_features in fold_selected_features:
            feature_counts.update(fold_features)

        min_frequency = 0.6
        min_count = int(min_frequency * len(cv_splits))
        stable_features = [
            f for f, count in feature_counts.items()
            if count >= min_count
        ]
        stable_features = stable_features[:self.n_features_to_select]  # Cap at max

        logger.debug(
            f"  Feature selection: {len(stable_features)} stable features "
            f"(appeared in >= {min_frequency*100:.0f}% of folds)"
        )

        # Build OOF prediction object
        oof_df = pd.DataFrame({
            "prediction": oof_predictions,
            "true_label": y.values,
        }, index=y.index)

        # Add probability columns
        for c in range(n_classes):
            oof_df[f"prob_class_{c}"] = oof_probabilities[:, c]

        oof_prediction = OOFPrediction(
            model_name=model_name,
            predictions=oof_df,
            fold_info=[m.to_dict() for m in fold_metrics_list],
            coverage=float(np.sum(~np.isnan(oof_predictions)) / n_samples),
        )

        return {
            "oof_prediction": oof_prediction,
            "selected_features": stable_features,
            "fold_metrics": fold_metrics_list,
        }

    def _validate_stacking_consistency(
        self,
        oof_predictions: dict[str, OOFPrediction],
        horizon: int,
    ) -> None:
        """
        Verify all OOF results used consistent CV settings.

        When building stacking datasets, all base models must have used the same
        CV configuration (purge/embargo settings, number of folds) to ensure the
        OOF predictions are comparable and properly aligned.

        Args:
            oof_predictions: Dict of OOF predictions by model name
            horizon: Label horizon being validated

        Raises:
            ValueError: If sample counts are inconsistent across models
        """
        if not oof_predictions:
            return

        first_key = next(iter(oof_predictions))
        reference = oof_predictions[first_key]
        ref_n_samples = len(reference.predictions)

        # Check for NaN patterns to detect sequence model gaps
        ref_pred_col = "prediction" if "prediction" in reference.predictions.columns else reference.predictions.columns[0]
        ref_valid_mask = ~reference.predictions[ref_pred_col].isna()

        for model_name, result in oof_predictions.items():
            # Check sample count matches
            n_samples = len(result.predictions)
            if n_samples != ref_n_samples:
                raise ValueError(
                    f"Inconsistent sample counts for horizon {horizon}: "
                    f"{model_name} has {n_samples} samples, expected {ref_n_samples} "
                    f"(based on {first_key}). All base models must use the same CV configuration."
                )

            # Check valid samples align (important for sequence models with different seq_len)
            pred_col = "prediction" if "prediction" in result.predictions.columns else result.predictions.columns[0]
            valid_mask = ~result.predictions[pred_col].isna()

            if not np.array_equal(ref_valid_mask.values, valid_mask.values):
                # Count mismatches for more informative warning
                mismatches = np.sum(ref_valid_mask.values != valid_mask.values)
                logger.warning(
                    f"OOF valid masks differ for {model_name} vs {first_key} "
                    f"(horizon {horizon}): {mismatches} samples differ. "
                    "This may occur with sequence models of different lengths. "
                    "Stacking dataset may have gaps for some samples."
                )

    def build_stacking_datasets(
        self,
        cv_results: dict[tuple[str, int], CVResult],
        container: TimeSeriesDataContainer,
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
        stacking_datasets: dict[int, StackingDataset] = {}

        for horizon in self.horizons:
            # Collect all model predictions for this horizon
            oof_predictions: dict[str, OOFPrediction] = {}

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

            # Validate that all OOF results are consistent
            self._validate_stacking_consistency(oof_predictions, horizon)

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
# STABILITY ANALYSIS
# =============================================================================

def analyze_cv_stability(
    cv_results: dict[tuple[str, int], CVResult],
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
    "validate_lightgbm_params",
    "get_max_leaves_for_depth",
]
