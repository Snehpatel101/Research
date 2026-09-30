"""
Per-Fold Feature Selection for Cross-Validation.

Provides leakage-free feature selection that runs inside each CV fold,
using only training data to rank and select features.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.feature_selection import mutual_info_classif  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, f1_score  # type: ignore[import-untyped]

from src.core.label_spans import LabelSpans
from src.models.registry import ModelRegistry

from .cv_dataclasses import FoldMetrics
from .early_stopping_split import carve_early_stopping_split
from .fold_scaling import FoldAwareScaler, get_scaling_method_for_model

# Import OOFPrediction directly from oof_core (where it's defined)
# to reduce import chain length and avoid going through oof_generator
from .oof_core import OOFPrediction, build_oof_frame
from .purged_kfold import PurgedKFold, PurgedKFoldConfig

logger = logging.getLogger(__name__)


def run_cv_with_per_fold_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series | None,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    config: dict[str, Any],
    n_features_to_select: int = 50,
    tune_per_fold: bool = False,
    cv: PurgedKFold | None = None,
    tuning_trials: int = 50,
    label_spans: LabelSpans | None = None,
    seed: int = 42,
    n_classes: int = 3,
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
        n_features_to_select: Number of features to select per fold
        tune_per_fold: If True, tune hyperparameters inside each fold after
            feature selection. More accurate but slower.
        cv: PurgedKFold instance (required if tune_per_fold=True)
        tuning_trials: Number of Optuna trials for per-fold tuning
        label_spans: Label spans of the rows of ``X`` (bar positions); inner tuning
            folds purge on them
        seed: Seeds mutual-information ranking and per-fold tuning
        n_classes: Label classes (2 for binary labels, 3 for short/neutral/long);
            fold models are built for it and the OOF frame has one probability
            column per class

    ``X`` is unscaled: every fold scales its features with the model's scaler fit on
    that fold's fit rows only (early-stopping tail and validation rows are transformed).

    Returns:
        Dict with oof_prediction, selected_features, fold_metrics
    """
    n_samples = len(X)
    all_features = list(X.columns)

    # Track which features are selected in each fold
    fold_selected_features: list[list[str]] = []

    # Initialize OOF prediction storage
    oof_predictions = np.full(n_samples, np.nan)
    oof_probabilities = np.full((n_samples, n_classes), np.nan)
    oof_confidence = np.full(n_samples, np.nan)
    oof_fold_ids = np.full(n_samples, -1, dtype=np.int32)
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
            random_state=seed,
        )

        # Select top N features based on MI scores from training data only
        feature_scores = list(zip(all_features, mi_scores, strict=False))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        fold_features = [f[0] for f in feature_scores[:n_features_to_select]]
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
        fold_config = {**config, "n_classes": n_classes}
        if tune_per_fold and cv is not None:
            logger.debug(
                f"  Fold {fold_idx + 1}: Tuning HPs on {len(fold_features)} selected features..."
            )
            # Create a mini CV for tuning within the training fold
            inner_cv_config = PurgedKFoldConfig(
                n_splits=min(3, len(train_idx) // 100),  # Fewer splits for inner CV
                purge_bars=cv.config.purge_bars,
                embargo_bars=cv.config.embargo_bars,
            )
            inner_cv = PurgedKFold(inner_cv_config)

            # Import here to avoid circular dependency
            from .cv_tuner import TimeSeriesOptunaTuner

            tuner = TimeSeriesOptunaTuner(
                model_name=model_name,
                cv=inner_cv,
                n_trials=max(10, tuning_trials // 3),  # Fewer trials for inner tuning
                scale_per_fold=True,
                seed=seed,
                n_classes=n_classes,
            )
            tuning_result = tuner.tune(
                X_train_selected,
                y_train_fold,
                weights.iloc[train_idx] if weights is not None else None,
                label_spans=label_spans.subset(train_idx) if label_spans is not None else None,
            )
            fold_tuned_params = tuning_result.get("best_params", {})
            fold_config.update(fold_tuned_params)
            logger.debug(f"    Fold {fold_idx + 1} tuned params: {fold_tuned_params}")

        # Train model on selected features. Early stopping selects on a
        # purged tail of the fold's train rows, never on the held-out fold.
        es_split = carve_early_stopping_split(
            train_idx, cv.config.purge_bars if cv is not None else 0
        )
        fit_pos = np.flatnonzero(np.isin(train_idx, es_split.fit_idx))
        es_pos = np.flatnonzero(np.isin(train_idx, es_split.es_idx))
        scaling = FoldAwareScaler(
            method=get_scaling_method_for_model(model_name)
        ).fit_transform_fold(
            X_train_selected.values[fit_pos],
            np.vstack([X_train_selected.values[es_pos], X_val_selected.values]),
        )
        X_es_scaled = scaling.X_val_scaled[: len(es_pos)]
        X_val_scaled = scaling.X_val_scaled[len(es_pos) :]
        model = ModelRegistry.create(model_name, config=fold_config)
        model.fit(
            X_train=scaling.X_train_scaled,
            y_train=y_train_fold.values[fit_pos],
            X_val=X_es_scaled,
            y_val=y_train_fold.values[es_pos],
            sample_weights=w_train[fit_pos] if w_train is not None else None,
        )

        # Generate OOF predictions for this fold's validation set
        output = model.predict(X_val_scaled)
        oof_predictions[val_idx] = output.class_predictions
        oof_probabilities[val_idx] = output.class_probabilities
        oof_confidence[val_idx] = output.confidence
        oof_fold_ids[val_idx] = fold_idx

        # Compute fold metrics
        fold_accuracy = accuracy_score(y_val_fold.values, output.class_predictions)
        fold_f1 = f1_score(
            y_val_fold.values, output.class_predictions, average="macro", zero_division=0
        )
        fold_time = time.time() - fold_start

        fold_metrics_list.append(
            FoldMetrics(
                fold=fold_idx,
                train_size=len(train_idx),
                val_size=len(val_idx),
                accuracy=fold_accuracy,
                f1=fold_f1,
                precision=0.0,
                recall=0.0,
                training_time=fold_time,
            )
        )

        logger.debug(
            f"  Fold {fold_idx + 1}: selected {len(fold_features)} features, " f"F1={fold_f1:.4f}"
        )

    # Aggregate selected features: keep features that appear in >= 60% of folds
    feature_counts: Counter[str] = Counter()
    for fold_features in fold_selected_features:
        feature_counts.update(fold_features)

    min_frequency = 0.6
    min_count = int(min_frequency * len(cv_splits))
    stable_features = [f for f, count in feature_counts.items() if count >= min_count]
    stable_features = stable_features[:n_features_to_select]  # Cap at max

    logger.debug(
        f"  Feature selection: {len(stable_features)} stable features "
        f"(appeared in >= {min_frequency*100:.0f}% of folds)"
    )

    # Same schema as every other OOF producer: stacking datasets combine them
    oof_df = build_oof_frame(
        model_name,
        X.index,
        y.values,
        oof_probabilities,
        oof_predictions,
        oof_confidence,
        oof_fold_ids,
    )

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


__all__ = [
    "run_cv_with_per_fold_feature_selection",
]
