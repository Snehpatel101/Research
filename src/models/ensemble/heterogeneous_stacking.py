"""
HeterogeneousStackingBuilder - Build stacking datasets from heterogeneous model OOF predictions.

Handles alignment when mixing:
- Tabular models (XGBoost, LightGBM): 100% coverage, offset=0
- Sequence models (LSTM, TCN): ~98% coverage, offset=seq_len-1
- Multi-stream models (PatchTST): ~97% coverage, offset=seq_len-1

Uses PHASE_2 OOFAligner for alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.core import PipelineConfig, MODEL_DATA_RANKS
from src.core.interfaces import OOFResult, OOFPredictionProtocol
from src.adapters import OOFAligner, AlignedOOFResult

# TYPE_CHECKING import to break circular dependency:
# cross_validation/__init__ -> cv_feature_selection -> oof_generator -> oof_core
# -> models.base -> models/__init__ -> models.ensemble -> heterogeneous_stacking
# -> oof_core (CIRCULAR)
#
# The OOFPredictionProtocol from src.core.interfaces provides runtime type checking
# capability without the import cycle. The TYPE_CHECKING block provides full type
# hints for static analysis (mypy, pyright).
if TYPE_CHECKING:
    from src.cross_validation.oof_core import OOFPrediction
else:
    # At runtime, use the Protocol for isinstance checks if needed
    OOFPrediction = OOFPredictionProtocol

logger = logging.getLogger(__name__)


@dataclass
class StackingFeatures:
    """Features for meta-learner training."""

    X: np.ndarray  # (n_samples, n_features)
    y: np.ndarray  # (n_samples,)
    feature_names: List[str]
    model_names: List[str]
    n_samples: int
    n_features: int
    coverage: float
    alignment_offset: int
    # Derived feature columns
    has_mean_confidence: bool = True
    has_prediction_agreement: bool = True
    has_prediction_entropy: bool = True

    @property
    def n_base_models(self) -> int:
        return len(self.model_names)

    @property
    def n_prob_features(self) -> int:
        """Number of probability features (n_models * n_classes)."""
        return self.n_base_models * 3  # Assuming 3 classes

    @property
    def n_derived_features(self) -> int:
        """Number of derived features."""
        n = 0
        if self.has_mean_confidence:
            n += 1
        if self.has_prediction_agreement:
            n += 1
        if self.has_prediction_entropy:
            n += 1
        return n


class HeterogeneousStackingBuilder:
    """
    Build stacking datasets from heterogeneous model OOF predictions.

    Handles the alignment challenge when stacking:
    - Tabular models: Full coverage (N samples)
    - Sequence models: Partial coverage (N - seq_len + 1 samples)
    - Multi-stream models: Similar to sequence

    Uses PHASE_2 OOFAligner for alignment.

    Usage:
        from src.models.ensemble import HeterogeneousStackingBuilder

        builder = HeterogeneousStackingBuilder()
        features = builder.build(
            oof_predictions={
                "xgboost": xgb_oof,    # N samples
                "lstm": lstm_oof,       # N-59 samples
                "patchtst": patchtst_oof,  # N-59 samples
            },
            y_true=labels,
        )

        # features.X is ready for meta-learner training
        # features.coverage shows how much data was retained
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        add_derived_features: bool = True,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize builder.

        Args:
            config: Optional PipelineConfig
            add_derived_features: Whether to add mean_confidence, agreement, entropy
            class_names: Names for classes (default: ["short", "neutral", "long"])
        """
        self.config = config
        self.add_derived_features = add_derived_features
        self.class_names = class_names or ["short", "neutral", "long"]

        # PHASE_2 aligner
        self._aligner = OOFAligner()

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "HeterogeneousStackingBuilder":
        """Create builder from PipelineConfig."""
        return cls(config=config)

    def build(
        self,
        oof_predictions: Dict[str, OOFPrediction],
        y_true: Union[np.ndarray, pd.Series],
    ) -> StackingFeatures:
        """
        Build stacking features from OOF predictions.

        Args:
            oof_predictions: Dict mapping model_name -> OOFPrediction
            y_true: True labels (full length, will be aligned)

        Returns:
            StackingFeatures ready for meta-learner training
        """
        if not oof_predictions:
            raise ValueError("No OOF predictions provided")

        y_true = np.asarray(y_true)

        logger.info(f"Building stacking features from {len(oof_predictions)} models")

        # Step 1: Convert OOFPrediction to OOFResult for alignment
        oof_results = self._convert_to_oof_results(oof_predictions)

        # Step 2: Align using PHASE_2 OOFAligner
        aligned = self._aligner.align(oof_results)

        logger.info(
            f"  Alignment: offset={self._compute_alignment_offset(aligned)}, "
            f"samples={aligned.n_common}, "
            f"coverage={self._compute_overall_coverage(aligned):.2%}"
        )

        # Step 3: Build probability feature matrix
        prob_features, feature_names = self._extract_probability_features(aligned)

        # Step 4: Add derived features if requested
        if self.add_derived_features:
            derived, derived_names = self._compute_derived_features(aligned)
            prob_features = np.hstack([prob_features, derived])
            feature_names.extend(derived_names)

        # Step 5: Align labels
        y_aligned = self._align_labels(y_true, aligned)

        # Validate shapes
        if prob_features.shape[0] != len(y_aligned):
            raise ValueError(
                f"Shape mismatch: features={prob_features.shape[0]}, "
                f"labels={len(y_aligned)}"
            )

        logger.info(f"  Built features: {prob_features.shape}")

        alignment_offset = self._compute_alignment_offset(aligned)
        coverage = self._compute_overall_coverage(aligned)

        return StackingFeatures(
            X=prob_features,
            y=y_aligned,
            feature_names=feature_names,
            model_names=aligned.model_names,
            n_samples=prob_features.shape[0],
            n_features=prob_features.shape[1],
            coverage=coverage,
            alignment_offset=alignment_offset,
            has_mean_confidence=self.add_derived_features,
            has_prediction_agreement=self.add_derived_features,
            has_prediction_entropy=self.add_derived_features,
        )

    def _convert_to_oof_results(
        self,
        oof_predictions: Dict[str, OOFPrediction],
    ) -> List[OOFResult]:
        """
        Convert OOFPrediction objects to OOFResult objects for OOFAligner.

        Args:
            oof_predictions: Dict mapping model_name -> OOFPrediction

        Returns:
            List of OOFResult objects
        """
        oof_results = []

        for model_name, oof_pred in oof_predictions.items():
            # Extract probabilities from the OOFPrediction
            probs = oof_pred.get_probabilities()
            preds = oof_pred.get_class_predictions()

            # Determine indices
            if oof_pred.original_indices is not None:
                indices = oof_pred.original_indices
            else:
                # Use sequential indices starting from alignment_offset
                offset = oof_pred.alignment_offset
                n_samples = len(probs)
                indices = np.arange(offset, offset + n_samples)

            # Create fold_ids (if not available, use zeros)
            fold_ids = np.zeros(len(probs), dtype=np.int32)

            oof_result = OOFResult(
                predictions=preds.astype(np.int64),
                probabilities=probs,
                indices=indices,
                fold_ids=fold_ids,
                model_name=model_name,
                coverage=oof_pred.coverage,
            )
            oof_results.append(oof_result)

        return oof_results

    def _compute_alignment_offset(self, aligned: AlignedOOFResult) -> int:
        """Compute alignment offset from common indices."""
        if len(aligned.common_indices) > 0:
            return int(aligned.common_indices.min())
        return 0

    def _compute_overall_coverage(self, aligned: AlignedOOFResult) -> float:
        """Compute overall coverage as minimum across models."""
        if not aligned.coverage:
            return 1.0
        return min(aligned.coverage.values())

    def _extract_probability_features(
        self,
        aligned: AlignedOOFResult,
    ) -> tuple:
        """Extract probability features from aligned OOF."""
        feature_names = []
        features_list = []

        for model_name in aligned.model_names:
            # Get aligned probabilities for this model
            probs = aligned.get_model_probabilities(model_name)

            if probs is None:
                logger.warning(f"No probabilities for {model_name}, using zeros")
                probs = np.zeros((aligned.n_common, len(self.class_names)))

            features_list.append(probs)

            for class_name in self.class_names:
                feature_names.append(f"{model_name}_prob_{class_name}")

        prob_features = np.hstack(features_list)

        return prob_features, feature_names

    def _compute_derived_features(
        self,
        aligned: AlignedOOFResult,
    ) -> tuple:
        """Compute derived ensemble diversity features."""
        n_samples = aligned.n_common
        n_models = len(aligned.model_names)
        n_classes = len(self.class_names)

        # Stack all model probabilities: (n_models, n_samples, n_classes)
        all_probs = []
        for model_name in aligned.model_names:
            probs = aligned.get_model_probabilities(model_name)
            if probs is not None:
                all_probs.append(probs)

        if not all_probs:
            # Return zeros if no probabilities available
            return (
                np.zeros((n_samples, 3)),
                ["mean_confidence", "prediction_agreement", "prediction_entropy"],
            )

        stacked = np.stack(all_probs, axis=0)  # (n_models, n_samples, n_classes)

        derived = {}

        # 1. Mean confidence: average max probability across models
        max_probs = np.nanmax(stacked, axis=2)  # (n_models, n_samples)
        derived["mean_confidence"] = np.nanmean(max_probs, axis=0)  # (n_samples,)

        # 2. Prediction agreement: fraction of models agreeing on predicted class
        predictions = np.argmax(stacked, axis=2)  # (n_models, n_samples)
        agreement = np.zeros(n_samples)
        for i in range(n_samples):
            votes = predictions[:, i]
            # Filter out any invalid votes (NaN handling)
            valid_votes = votes[~np.isnan(stacked[:, i, 0])]
            if len(valid_votes) > 0:
                unique, counts = np.unique(valid_votes, return_counts=True)
                agreement[i] = np.max(counts) / len(valid_votes)
            else:
                agreement[i] = np.nan
        derived["prediction_agreement"] = agreement

        # 3. Prediction entropy: entropy of averaged probabilities
        mean_probs = np.nanmean(stacked, axis=0)  # (n_samples, n_classes)
        # Clip to avoid log(0)
        mean_probs = np.clip(mean_probs, 1e-10, 1.0)
        # Normalize to ensure sum to 1
        mean_probs = mean_probs / mean_probs.sum(axis=1, keepdims=True)
        entropy = -np.sum(mean_probs * np.log(mean_probs), axis=1)
        derived["prediction_entropy"] = entropy

        # Stack into array
        derived_array = np.column_stack(
            [
                derived["mean_confidence"],
                derived["prediction_agreement"],
                derived["prediction_entropy"],
            ]
        )

        derived_names = ["mean_confidence", "prediction_agreement", "prediction_entropy"]

        return derived_array, derived_names

    def _align_labels(
        self,
        y_true: np.ndarray,
        aligned: AlignedOOFResult,
    ) -> np.ndarray:
        """Align labels to match aligned OOF predictions."""
        # Use the common indices to extract aligned labels
        common_indices = aligned.common_indices

        if len(common_indices) == 0:
            raise ValueError("No common indices for label alignment")

        # Validate indices are within bounds
        max_idx = common_indices.max()
        if max_idx >= len(y_true):
            raise ValueError(
                f"Label alignment error: max_idx={max_idx}, y_true length={len(y_true)}"
            )

        return y_true[common_indices]

    def get_model_coverage_report(
        self,
        oof_predictions: Dict[str, OOFPrediction],
    ) -> Dict[str, Any]:
        """
        Get detailed coverage report for each model.

        Args:
            oof_predictions: Dict mapping model_name -> OOFPrediction

        Returns:
            Dict with coverage statistics per model
        """
        report: Dict[str, Any] = {
            "models": {},
            "summary": {},
        }

        max_samples = 0
        min_samples = float("inf")

        for model_name, oof in oof_predictions.items():
            # Get number of predictions
            n_predictions = len(oof.predictions)

            # Get total samples if available
            n_total = oof.n_total_samples or n_predictions

            offset = oof.alignment_offset
            coverage = oof.coverage

            max_samples = max(max_samples, n_total)
            min_samples = min(min_samples, n_predictions)

            # Get data rank from constants
            data_rank = MODEL_DATA_RANKS.get(model_name.lower(), 2)

            # Determine model type based on data rank
            if data_rank == 2:
                model_type = "tabular"
            elif data_rank == 3:
                model_type = "sequence"
            else:
                model_type = "multi_stream"

            report["models"][model_name] = {
                "n_predictions": n_predictions,
                "n_total_samples": n_total,
                "alignment_offset": offset,
                "coverage": coverage,
                "data_rank": data_rank,
                "model_type": model_type,
                "sequence_length": oof.sequence_length,
            }

        report["summary"] = {
            "n_models": len(oof_predictions),
            "max_samples": int(max_samples) if max_samples != float("inf") else 0,
            "min_samples": int(min_samples) if min_samples != float("inf") else 0,
            "common_samples": int(min_samples) if min_samples != float("inf") else 0,
            "max_offset": int(max_samples - min_samples)
            if max_samples != float("inf") and min_samples != float("inf")
            else 0,
        }

        return report


def build_stacking_features(
    oof_predictions: Dict[str, OOFPrediction],
    y_true: np.ndarray,
    config: Optional[PipelineConfig] = None,
    add_derived_features: bool = True,
) -> StackingFeatures:
    """
    Convenience function to build stacking features.

    Args:
        oof_predictions: Dict mapping model_name -> OOFPrediction
        y_true: True labels
        config: Optional PipelineConfig
        add_derived_features: Whether to add derived features

    Returns:
        StackingFeatures ready for meta-learner
    """
    builder = HeterogeneousStackingBuilder(
        config=config,
        add_derived_features=add_derived_features,
    )
    return builder.build(oof_predictions, y_true)


__all__ = [
    "HeterogeneousStackingBuilder",
    "StackingFeatures",
    "build_stacking_features",
]
