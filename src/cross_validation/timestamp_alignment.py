"""
Timestamp alignment validation for heterogeneous stacking.

Ensures that predictions from multiple base models align correctly
on their datetime indices before combining them for meta-learner training.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .oof_core import OOFPrediction

logger = logging.getLogger(__name__)


def validate_datetime_alignment(
    oof_predictions: dict[str, OOFPrediction],
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate that datetime indices align across all base model predictions.

    Args:
        oof_predictions: Dict of OOF predictions by model name
        strict: If True, require exact datetime alignment. If False, allow
            misalignment as long as there's sufficient overlap.

    Returns:
        Tuple of (is_valid, issues_list)
    """
    if not oof_predictions:
        return True, []

    if len(oof_predictions) == 1:
        return True, []

    issues = []
    model_names = list(oof_predictions.keys())

    base_model = model_names[0]
    base_datetime = oof_predictions[base_model].predictions["datetime"]

    for model_name in model_names[1:]:
        current_datetime = oof_predictions[model_name].predictions["datetime"]

        if len(base_datetime) != len(current_datetime):
            issue = (
                f"Length mismatch: {base_model}={len(base_datetime)} vs "
                f"{model_name}={len(current_datetime)}"
            )
            issues.append(issue)
            logger.warning(issue)

        if isinstance(base_datetime, pd.Series) and isinstance(current_datetime, pd.Series):
            if not base_datetime.equals(current_datetime):
                base_set = set(base_datetime)
                current_set = set(current_datetime)
                overlap = base_set & current_set
                n_misaligned = len(base_set) + len(current_set) - 2 * len(overlap)

                issue = (
                    f"Datetime mismatch between {base_model} and {model_name}: "
                    f"{n_misaligned} non-overlapping timestamps"
                )
                issues.append(issue)
                logger.warning(issue)

                overlap_pct = len(overlap) / max(len(base_datetime), len(current_datetime))
                if overlap_pct < 0.95 and strict:
                    issue = (
                        f"Insufficient overlap between {base_model} and {model_name}: "
                        f"{overlap_pct:.1%} (threshold: 95%)"
                    )
                    issues.append(issue)
                    logger.error(issue)

    is_valid = len(issues) == 0 if strict else all("Insufficient overlap" not in i for i in issues)

    if is_valid:
        logger.info(f"Timestamp alignment validated across {len(model_names)} models")
    else:
        logger.error(
            f"Timestamp alignment validation failed: {len(issues)} issues across "
            f"{len(model_names)} models"
        )

    return is_valid, issues


def align_predictions_on_datetime(
    oof_predictions: dict[str, OOFPrediction],
    method: str = "inner",
) -> dict[str, OOFPrediction]:
    """
    Align predictions from multiple models on common datetime indices.

    Args:
        oof_predictions: Dict of OOF predictions by model name
        method: Alignment method:
            - "inner": Keep only timestamps present in ALL models
            - "outer": Keep all timestamps, fill missing with NaN

    Returns:
        Dict of aligned OOF predictions
    """
    if not oof_predictions or len(oof_predictions) == 1:
        return oof_predictions

    model_names = list(oof_predictions.keys())

    if method == "inner":
        common_datetimes = set(oof_predictions[model_names[0]].predictions["datetime"])
        for model_name in model_names[1:]:
            model_datetimes = set(oof_predictions[model_name].predictions["datetime"])
            common_datetimes &= model_datetimes

        n_original = len(oof_predictions[model_names[0]].predictions)
        n_aligned = len(common_datetimes)

        if n_aligned < n_original:
            logger.info(
                f"Aligning predictions on common timestamps: {n_aligned}/{n_original} "
                f"({100 * n_aligned / n_original:.1f}%)"
            )

        aligned = {}
        for model_name, oof_pred in oof_predictions.items():
            mask = oof_pred.predictions["datetime"].isin(common_datetimes)
            aligned_df = oof_pred.predictions.loc[mask].reset_index(drop=True)
            aligned[model_name] = OOFPrediction(
                model_name=model_name,
                predictions=aligned_df,
                fold_info=oof_pred.fold_info,
                coverage=oof_pred.coverage * (len(aligned_df) / len(oof_pred.predictions)),
            )

        return aligned

    elif method == "outer":
        all_datetimes = set()
        for oof_pred in oof_predictions.values():
            all_datetimes.update(oof_pred.predictions["datetime"])

        all_datetimes_sorted = sorted(all_datetimes)
        logger.info(f"Outer alignment: expanding to {len(all_datetimes_sorted)} unique timestamps")

        aligned = {}
        for model_name, oof_pred in oof_predictions.items():
            base_df = pd.DataFrame({"datetime": all_datetimes_sorted})
            merged_df = base_df.merge(oof_pred.predictions, on="datetime", how="left")
            aligned[model_name] = OOFPrediction(
                model_name=model_name,
                predictions=merged_df,
                fold_info=oof_pred.fold_info,
                coverage=oof_pred.coverage,
            )

        return aligned

    else:
        raise ValueError(f"Unknown alignment method: {method}. Use 'inner' or 'outer'.")


def get_datetime_alignment_report(
    oof_predictions: dict[str, OOFPrediction],
) -> dict[str, Any]:
    """
    Generate detailed report on datetime alignment across models.

    Args:
        oof_predictions: Dict of OOF predictions by model name

    Returns:
        Dict with alignment statistics
    """
    if not oof_predictions:
        return {}

    model_names = list(oof_predictions.keys())
    datetime_sets = {
        name: set(pred.predictions["datetime"]) for name, pred in oof_predictions.items()
    }

    lengths = {name: len(dt_set) for name, dt_set in datetime_sets.items()}

    common_timestamps = set.intersection(*datetime_sets.values()) if datetime_sets else set()
    all_timestamps = set.union(*datetime_sets.values()) if datetime_sets else set()

    pairwise_overlaps = {}
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i + 1 :]:
            overlap = datetime_sets[model1] & datetime_sets[model2]
            overlap_pct = len(overlap) / max(len(datetime_sets[model1]), len(datetime_sets[model2]))
            pairwise_overlaps[f"{model1}_vs_{model2}"] = {
                "overlap_count": len(overlap),
                "overlap_pct": overlap_pct,
            }

    report = {
        "n_models": len(model_names),
        "model_names": model_names,
        "sample_counts": lengths,
        "common_timestamps": len(common_timestamps),
        "total_unique_timestamps": len(all_timestamps),
        "common_coverage": len(common_timestamps) / len(all_timestamps) if all_timestamps else 0.0,
        "pairwise_overlaps": pairwise_overlaps,
    }

    return report
