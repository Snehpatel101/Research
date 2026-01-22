"""
Stacking Dataset Building Utilities for Cross-Validation.

Functions for building stacking datasets from OOF predictions,
validating consistency, and analyzing CV stability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .oof_generator import OOFGenerator, OOFPrediction, StackingDataset
from .purged_kfold import PurgedKFold

if TYPE_CHECKING:
    from .cv_dataclasses import CVResult

logger = logging.getLogger(__name__)


def validate_stacking_consistency(
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
    ref_pred_col = (
        "prediction"
        if "prediction" in reference.predictions.columns
        else reference.predictions.columns[0]
    )
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
        pred_col = (
            "prediction"
            if "prediction" in result.predictions.columns
            else result.predictions.columns[0]
        )
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


def build_stacking_datasets_from_cv_results(
    cv_results: dict[tuple[str, int], CVResult],
    horizons: list[int],
    cv: PurgedKFold,
    container,
) -> dict[int, StackingDataset]:
    """
    Build stacking datasets from CV results.

    Args:
        cv_results: Results from CrossValidationRunner.run()
        horizons: List of horizons to process
        cv: PurgedKFold cross-validator
        container: Original data container

    Returns:
        Dict mapping horizon to StackingDataset

    Raises:
        ValueError: If OOF predictions have inconsistent sample counts
    """
    stacking_datasets: dict[int, StackingDataset] = {}

    for horizon in horizons:
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
        validate_stacking_consistency(oof_predictions, horizon)

        # Get true labels
        _, y, _ = container.get_sklearn_arrays("train", return_df=True)

        # Build stacking dataset
        oof_gen = OOFGenerator(cv)
        stacking_ds = oof_gen.build_stacking_dataset(
            oof_predictions=oof_predictions,
            y_true=y,
            horizon=horizon,
        )
        stacking_datasets[horizon] = stacking_ds

    return stacking_datasets


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

            stability_data.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "metric": metric_name,
                    "mean": mean_val,
                    "std": std_val,
                    "cv": cv,
                    "min": np.min(values),
                    "max": np.max(values),
                    "stability_grade": _grade_stability(cv),
                }
            )

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
    "validate_stacking_consistency",
    "build_stacking_datasets_from_cv_results",
    "analyze_cv_stability",
]
