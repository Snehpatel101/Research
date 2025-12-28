"""
OOF prediction validation and quality checks.

Validates coverage, correlation, and ensemble diversity.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from src.cross_validation.oof_core import OOFPrediction

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION
# =============================================================================

class OOFValidator:
    """Validate OOF prediction quality and coverage."""

    @staticmethod
    def validate_coverage(
        oof_predictions: Dict[str, OOFPrediction],
        original_index: pd.Index,
    ) -> Dict[str, Any]:
        """
        Validate that OOF predictions cover all samples.

        Args:
            oof_predictions: Dict of OOF predictions by model
            original_index: Original DataFrame index

        Returns:
            Validation result dict with passed status and any issues
        """
        validation = {"passed": True, "issues": [], "coverage": {}}

        for model_name, oof_pred in oof_predictions.items():
            # Check for NaN predictions
            nan_count = oof_pred.predictions[f"{model_name}_pred"].isna().sum()
            coverage = 1.0 - (nan_count / len(original_index))

            validation["coverage"][model_name] = coverage

            if nan_count > 0:
                validation["passed"] = False
                validation["issues"].append({
                    "model": model_name,
                    "missing_samples": int(nan_count),
                    "coverage": coverage,
                })

        return validation

    @staticmethod
    def analyze_prediction_correlation(
        stacking_df: pd.DataFrame,
        model_names: List[str],
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
                    summary.append({
                        "model_1": model_i,
                        "model_2": model_j,
                        "correlation": corr,
                        "diversity_grade": _grade_diversity(corr),
                    })

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


__all__ = [
    "OOFValidator",
    "_grade_diversity",  # Exposed for backward compatibility
]
