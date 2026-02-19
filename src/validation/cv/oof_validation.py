"""
OOF prediction validation and quality checks.

Validates coverage, correlation, ensemble diversity, and fold leakage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .oof_core import OOFPrediction

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION
# =============================================================================


class OOFValidator:
    """Validate OOF prediction quality, coverage, and fold leakage."""

    @staticmethod
    def validate_fold_leakage(
        fold_indices: list[tuple[np.ndarray, np.ndarray]],
        purge_bars: int,
        embargo_bars: int,
    ) -> dict[str, Any]:
        """
        Verify that no validation index falls within purge/embargo windows
        of training index boundaries.

        This is a post-training sanity check: after CV folds have been split,
        confirm that purge and embargo gaps were actually respected. Catches
        bugs in fold splitting logic that could allow data leakage.

        For each fold the check verifies:
        1. **Purge violation**: No val index appears within ``purge_bars``
           *before* the start of the validation block (i.e., in the zone that
           should have been purged from training data on the left side).
           Specifically, for every contiguous validation block, no training
           index should exist in [val_start - purge_bars, val_start).
        2. **Embargo violation**: No training index appears within
           ``embargo_bars`` immediately *after* the end of the validation
           block.  Specifically, for every contiguous validation block, no
           training index should exist in (val_end, val_end + embargo_bars].

        Args:
            fold_indices: List of (train_indices, val_indices) tuples, one per
                fold. Indices are integer position-based (0-indexed).
            purge_bars: Number of bars that should be purged before val set.
            embargo_bars: Number of bars to skip after val set.

        Returns:
            Dict with keys:
                - ``passed`` (bool): True if no violations found.
                - ``n_violations`` (int): Total violation count across folds.
                - ``fold_details`` (list[dict]): Per-fold breakdown with
                  ``purge_violations`` and ``embargo_violations`` lists.
        """
        fold_details: list[dict[str, Any]] = []
        total_violations = 0

        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            train_set = set(train_idx.tolist())
            val_sorted = np.sort(val_idx)

            purge_violations: list[dict[str, Any]] = []
            embargo_violations: list[dict[str, Any]] = []

            # Identify contiguous validation blocks (there is usually one per
            # fold, but the logic handles gaps gracefully).
            val_blocks = _contiguous_blocks(val_sorted)

            for block_start, block_end in val_blocks:
                # --- Purge check ---
                # The zone [block_start - purge_bars, block_start) should
                # contain NO training indices.
                purge_zone_start = max(0, block_start - purge_bars)
                for idx in range(purge_zone_start, block_start):
                    if idx in train_set:
                        purge_violations.append(
                            {
                                "train_idx": idx,
                                "val_block_start": int(block_start),
                                "distance": int(block_start - idx),
                            }
                        )

                # --- Embargo check ---
                # The zone (block_end, block_end + embargo_bars] should
                # contain NO training indices.
                embargo_zone_end = block_end + embargo_bars  # no upper clamp needed
                for idx in range(block_end + 1, embargo_zone_end + 1):
                    if idx in train_set:
                        embargo_violations.append(
                            {
                                "train_idx": idx,
                                "val_block_end": int(block_end),
                                "distance": int(idx - block_end),
                            }
                        )

            n_fold_violations = len(purge_violations) + len(embargo_violations)
            total_violations += n_fold_violations

            fold_details.append(
                {
                    "fold": fold_idx,
                    "n_purge_violations": len(purge_violations),
                    "n_embargo_violations": len(embargo_violations),
                    "purge_violations": purge_violations,
                    "embargo_violations": embargo_violations,
                }
            )

            if n_fold_violations > 0:
                logger.warning(
                    "Fold %d leakage: %d purge violations, %d embargo violations",
                    fold_idx,
                    len(purge_violations),
                    len(embargo_violations),
                )

        passed = total_violations == 0
        if passed:
            logger.info(
                "Fold leakage check passed for %d folds (purge=%d, embargo=%d)",
                len(fold_indices),
                purge_bars,
                embargo_bars,
            )

        return {
            "passed": passed,
            "n_violations": total_violations,
            "fold_details": fold_details,
        }

    @staticmethod
    def validate_coverage(
        oof_predictions: dict[str, OOFPrediction],
        original_index: pd.Index,
    ) -> dict[str, Any]:
        """
        Validate that OOF predictions cover all samples.

        Args:
            oof_predictions: Dict of OOF predictions by model
            original_index: Original DataFrame index

        Returns:
            Validation result dict with passed status and any issues
        """
        issues: list[dict[str, Any]] = []
        coverage_dict: dict[str, float] = {}
        validation: dict[str, Any] = {"passed": True, "issues": issues, "coverage": coverage_dict}

        for model_name, oof_pred in oof_predictions.items():
            # Check for NaN predictions
            nan_count = oof_pred.predictions[f"{model_name}_pred"].isna().sum()
            coverage = 1.0 - (nan_count / len(original_index))

            coverage_dict[model_name] = coverage

            if nan_count > 0:
                validation["passed"] = False
                issues.append(
                    {
                        "model": model_name,
                        "missing_samples": int(nan_count),
                        "coverage": coverage,
                    }
                )

        return validation

    @staticmethod
    def analyze_prediction_correlation(
        stacking_df: pd.DataFrame,
        model_names: list[str],
    ) -> pd.DataFrame:
        """
        Analyze correlation between model predictions.

        Low correlation = good diversity for ensemble.

        Args:
            stacking_df: Stacking dataset DataFrame
            model_names: List of model names

        Returns:
            DataFrame with correlation analysis
        """
        pred_cols = [f"{model}_pred" for model in model_names]
        pred_df = stacking_df[pred_cols]

        # Compute correlation matrix
        corr_matrix = pred_df.corr()

        # Summarize pairwise correlations
        summary = []
        for i, model_i in enumerate(model_names):
            for j, model_j in enumerate(model_names):
                if i < j:
                    corr = corr_matrix.loc[f"{model_i}_pred", f"{model_j}_pred"]
                    summary.append(
                        {
                            "model_1": model_i,
                            "model_2": model_j,
                            "correlation": corr,
                            "diversity_grade": _grade_diversity(corr),
                        }
                    )

        return pd.DataFrame(summary)


def _grade_diversity(corr: float) -> str:
    """Grade ensemble diversity based on prediction correlation."""
    if corr < 0.3:
        return "Excellent (highly diverse)"
    elif corr < 0.5:
        return "Good"
    elif corr < 0.7:
        return "Moderate"
    elif corr < 0.85:
        return "Low"
    else:
        return "Poor (models too similar)"


def _contiguous_blocks(sorted_indices: np.ndarray) -> list[tuple[int, int]]:
    """
    Identify contiguous blocks in a sorted array of integer indices.

    Returns a list of (start, end) tuples where *end* is the last index
    in the block (inclusive).

    Args:
        sorted_indices: 1-D array of sorted integer indices.

    Returns:
        List of (block_start, block_end) inclusive tuples.
    """
    if len(sorted_indices) == 0:
        return []

    blocks: list[tuple[int, int]] = []
    block_start = int(sorted_indices[0])
    prev = block_start

    for idx in sorted_indices[1:]:
        idx_int = int(idx)
        if idx_int != prev + 1:
            blocks.append((block_start, prev))
            block_start = idx_int
        prev = idx_int

    blocks.append((block_start, prev))
    return blocks


__all__ = [
    "OOFValidator",
    "_grade_diversity",  # Exposed for backward compatibility
]
