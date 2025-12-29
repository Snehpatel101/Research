"""
Core OOF (Out-of-Fold) prediction generation.

Handles the main logic for generating out-of-sample predictions
for tabular (non-sequence) models.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.cross_validation.fold_scaling import FoldAwareScaler, get_scaling_method_for_model
from src.cross_validation.purged_kfold import PurgedKFold
from src.models.base import PredictionOutput
from src.models.calibration import CalibrationConfig, ProbabilityCalibrator
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class OOFPrediction:
    """
    Out-of-fold predictions for a single model.

    Attributes:
        model_name: Name of the model
        predictions: DataFrame with OOF predictions
        fold_info: Per-fold training information
        coverage: Fraction of samples with predictions
    """
    def __init__(
        self,
        model_name: str,
        predictions: pd.DataFrame,
        fold_info: list[dict[str, Any]],
        coverage: float = 1.0,
    ):
        self.model_name = model_name
        self.predictions = predictions
        self.fold_info = fold_info
        self.coverage = coverage

    def get_probabilities(self) -> np.ndarray:
        """Get probability matrix (n_samples, 3)."""
        return self.predictions[
            [f"{self.model_name}_prob_short",
             f"{self.model_name}_prob_neutral",
             f"{self.model_name}_prob_long"]
        ].values

    def get_class_predictions(self) -> np.ndarray:
        """Get predicted classes (-1, 0, 1)."""
        return self.predictions[f"{self.model_name}_pred"].values


# =============================================================================
# CORE OOF GENERATOR
# =============================================================================

class CoreOOFGenerator:
    """
    Core OOF prediction generator for tabular (non-sequence) models.

    Handles fold-aware scaling, training, and prediction generation.
    """

    def __init__(self, cv: PurgedKFold) -> None:
        """
        Initialize CoreOOFGenerator.

        Args:
            cv: PurgedKFold cross-validator
        """
        self.cv = cv

    def generate_tabular_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        config: dict[str, Any],
        sample_weights: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
    ) -> OOFPrediction:
        """
        Generate OOF predictions for a tabular model.

        Args:
            X: Feature DataFrame
            y: Labels
            model_name: Name of the model
            config: Model hyperparameters
            sample_weights: Optional quality weights
            label_end_times: Optional Series of datetime when each label is resolved.
                If provided, enables proper purging of overlapping labels in CV.

        Returns:
            OOFPrediction with predictions and fold info
        """
        n_samples = len(X)
        n_classes = 3  # short, neutral, long

        # Initialize OOF storage
        oof_probs = np.full((n_samples, n_classes), np.nan)
        oof_preds = np.full(n_samples, np.nan)
        oof_confidence = np.full(n_samples, np.nan)
        fold_info: list[dict[str, Any]] = []

        # Determine scaling method based on model requirements
        scaling_method = get_scaling_method_for_model(model_name)
        fold_scaler = FoldAwareScaler(method=scaling_method)

        # Generate predictions fold by fold (with label_end_times for overlapping label purge)
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv.split(X, y, label_end_times=label_end_times)
        ):
            logger.debug(f"  Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")

            # Extract fold data (raw, unscaled)
            X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # FOLD-AWARE SCALING: fit scaler on train-only, transform both
            scaling_result = fold_scaler.fit_transform_fold(
                X_train_raw.values, X_val_raw.values
            )
            X_train_scaled = scaling_result.X_train_scaled
            X_val_scaled = scaling_result.X_val_scaled

            # Handle sample weights
            if sample_weights is not None:
                w_train = sample_weights.iloc[train_idx].values
                w_val = sample_weights.iloc[val_idx].values
            else:
                w_train = None
                w_val = None

            # Create and train model
            model = ModelRegistry.create(model_name, config=config)

            # Use model's fit interface with scaled data
            training_metrics = model.fit(
                X_train=X_train_scaled,
                y_train=y_train.values,
                X_val=X_val_scaled,
                y_val=y_val.values,
                sample_weights=w_train,
            )

            # Generate predictions for validation fold (using scaled data)
            prediction_output: PredictionOutput = model.predict(X_val_scaled)

            # Store OOF predictions
            oof_probs[val_idx] = prediction_output.class_probabilities
            oof_preds[val_idx] = prediction_output.class_predictions
            oof_confidence[val_idx] = prediction_output.confidence

            # Track fold info
            fold_info.append({
                "fold": fold_idx,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "val_accuracy": training_metrics.val_accuracy,
                "val_f1": training_metrics.val_f1,
            })

        # Validate coverage
        coverage = float((~np.isnan(oof_preds)).mean())
        if coverage < 1.0:
            logger.warning(
                f"{model_name}: Only {coverage:.2%} coverage. "
                f"{int(np.isnan(oof_preds).sum())} samples missing predictions."
            )

        # Build result DataFrame
        oof_df = pd.DataFrame({
            "datetime": X.index if isinstance(X.index, pd.DatetimeIndex) else range(len(X)),
            f"{model_name}_prob_short": oof_probs[:, 0],
            f"{model_name}_prob_neutral": oof_probs[:, 1],
            f"{model_name}_prob_long": oof_probs[:, 2],
            f"{model_name}_pred": oof_preds,
            f"{model_name}_confidence": oof_confidence,
        })

        return OOFPrediction(
            model_name=model_name,
            predictions=oof_df,
            fold_info=fold_info,
            coverage=coverage,
        )

    def calibrate_oof_predictions(
        self,
        oof_results: dict[str, OOFPrediction],
        y_true: pd.Series,
        calibration_method: str = "auto",
    ) -> dict[str, OOFPrediction]:
        """
        Apply probability calibration to OOF predictions.

        This is leakage-safe because OOF predictions are truly out-of-sample:
        each prediction was made by a model that never saw that sample during
        training. The calibrator learns the probability mapping from these
        honest predictions.

        Args:
            oof_results: Dict of OOF predictions by model
            y_true: True labels
            calibration_method: Calibration method

        Returns:
            Dict of calibrated OOF predictions
        """
        logger.info("Applying probability calibration to OOF predictions...")

        y_array = y_true.values

        for model_name, oof_pred in oof_results.items():
            # Get probability columns
            prob_cols = [
                f"{model_name}_prob_short",
                f"{model_name}_prob_neutral",
                f"{model_name}_prob_long",
            ]
            probs = oof_pred.predictions[prob_cols].values

            # Handle NaN predictions (keep them as-is)
            valid_mask = ~np.isnan(probs[:, 0])
            if valid_mask.sum() == 0:
                logger.warning(f"  {model_name}: No valid predictions to calibrate")
                continue

            valid_probs = probs[valid_mask]
            valid_y = y_array[valid_mask]

            # Fit and apply calibrator
            cal_config = CalibrationConfig(method=calibration_method)
            calibrator = ProbabilityCalibrator(cal_config)
            metrics = calibrator.fit(valid_y, valid_probs)

            calibrated_probs = calibrator.calibrate(valid_probs)

            # Update predictions DataFrame
            oof_pred.predictions.loc[valid_mask, prob_cols] = calibrated_probs

            # Update confidence based on calibrated probabilities
            oof_pred.predictions.loc[valid_mask, f"{model_name}_confidence"] = (
                calibrated_probs.max(axis=1)
            )

            logger.info(
                f"  {model_name}: Brier {metrics.brier_before:.4f} -> {metrics.brier_after:.4f}, "
                f"ECE {metrics.ece_before:.4f} -> {metrics.ece_after:.4f}"
            )

        return oof_results


__all__ = [
    "OOFPrediction",
    "CoreOOFGenerator",
]
