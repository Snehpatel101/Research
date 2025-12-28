"""
Stacking dataset construction from OOF predictions.

Builds meta-learner training data by combining OOF predictions
from multiple base models with derived features.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.cross_validation.oof_core import OOFPrediction

logger = logging.getLogger(__name__)


# =============================================================================
# STACKING DATASET
# =============================================================================

@dataclass
class StackingDataset:
    """
    Dataset for training ensemble meta-learner.

    Contains OOF predictions from all base models plus true labels.

    Attributes:
        data: DataFrame with all model predictions and derived features
        model_names: List of base model names
        horizon: Label horizon
        metadata: Additional metadata
    """
    data: pd.DataFrame
    model_names: List[str]
    horizon: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def n_models(self) -> int:
        return len(self.model_names)

    def get_features(self) -> pd.DataFrame:
        """Get feature columns for meta-learner."""
        # Exclude y_true and datetime columns
        feature_cols = [c for c in self.data.columns if c not in ("y_true", "datetime")]
        return self.data[feature_cols]

    def get_labels(self) -> pd.Series:
        """Get true labels."""
        return self.data["y_true"]


# =============================================================================
# STACKING BUILDER
# =============================================================================

class StackingDatasetBuilder:
    """
    Build stacking datasets from OOF predictions.

    Combines predictions from multiple models and adds derived features
    for meta-learner training.
    """

    def build_stacking_dataset(
        self,
        oof_predictions: Dict[str, OOFPrediction],
        y_true: pd.Series,
        horizon: int,
        add_derived_features: bool = True,
    ) -> StackingDataset:
        """
        Build stacking dataset from OOF predictions.

        Creates a DataFrame with:
        - model1_prob_short, model1_prob_neutral, model1_prob_long
        - model2_prob_short, model2_prob_neutral, model2_prob_long
        - Derived features (confidence, agreement, entropy)
        - y_true (label)

        Args:
            oof_predictions: Dict of OOF predictions by model
            y_true: True labels
            horizon: Label horizon (for metadata)
            add_derived_features: Whether to add derived features

        Returns:
            StackingDataset for meta-learner training
        """
        model_names = list(oof_predictions.keys())

        # Start with first model's predictions
        first_model = model_names[0]
        stacking_df = oof_predictions[first_model].predictions.copy()

        # Add other models' predictions
        for model_name in model_names[1:]:
            oof_pred = oof_predictions[model_name].predictions
            # Add all columns except datetime (already present)
            for col in oof_pred.columns:
                if col != "datetime":
                    stacking_df[col] = oof_pred[col]

        # Add true labels
        stacking_df["y_true"] = y_true.values

        # Add derived features for meta-learner
        if add_derived_features:
            stacking_df = self._add_stacking_features(stacking_df, model_names)

        # Compute metadata
        metadata = {
            "horizon": horizon,
            "n_models": len(model_names),
            "model_names": model_names,
            "n_samples": len(stacking_df),
            "coverage": {m: oof_predictions[m].coverage for m in model_names},
        }

        return StackingDataset(
            data=stacking_df,
            model_names=model_names,
            horizon=horizon,
            metadata=metadata,
        )

    def _add_stacking_features(
        self,
        df: pd.DataFrame,
        model_names: List[str],
    ) -> pd.DataFrame:
        """Add derived features for meta-learner."""
        df = df.copy()

        # Model predictions (argmax)
        pred_cols = []
        for model in model_names:
            prob_cols = [
                f"{model}_prob_short",
                f"{model}_prob_neutral",
                f"{model}_prob_long"
            ]
            # Prediction already exists, but ensure it's -1, 0, 1 format
            pred_col = f"{model}_pred"
            pred_cols.append(pred_col)

        # Agreement features
        df["models_agree"] = (df[pred_cols].nunique(axis=1) == 1).astype(int)
        df["agreement_count"] = df[pred_cols].apply(
            lambda x: x.value_counts().max() if len(x.dropna()) > 0 else 0,
            axis=1
        )

        # Average confidence
        conf_cols = [f"{model}_confidence" for model in model_names]
        df["avg_confidence"] = df[conf_cols].mean(axis=1)
        df["min_confidence"] = df[conf_cols].min(axis=1)
        df["max_confidence"] = df[conf_cols].max(axis=1)

        # Prediction entropy (uncertainty) per model
        for model in model_names:
            prob_cols = [
                f"{model}_prob_short",
                f"{model}_prob_neutral",
                f"{model}_prob_long"
            ]
            probs = df[prob_cols].values
            # Entropy: -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            df[f"{model}_entropy"] = entropy

        # Average entropy
        entropy_cols = [f"{model}_entropy" for model in model_names]
        df["avg_entropy"] = df[entropy_cols].mean(axis=1)

        # Disagreement measure (std of predictions across models)
        df["prediction_std"] = df[pred_cols].std(axis=1)

        return df


__all__ = [
    "StackingDataset",
    "StackingDatasetBuilder",
]
