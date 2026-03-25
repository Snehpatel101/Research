"""
Stacking dataset construction from OOF predictions.

Builds meta-learner training data by combining OOF predictions
from multiple base models with derived features.

Handles NaN values from sequence models that cannot predict samples
at the beginning of segments due to lookback requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .oof_alignment import (
    OOFAlignmentResult,
    OOFAlignmentValidator,
)
from .oof_core import OOFPrediction
from .timestamp_alignment import (
    align_predictions_on_datetime,
    get_datetime_alignment_report,
    validate_datetime_alignment,
)

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
    model_names: list[str]
    horizon: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def n_models(self) -> int:
        return len(self.model_names)

    @property
    def n_original_samples(self) -> int:
        """Original sample count before NaN removal."""
        return int(self.metadata.get("n_original_samples", len(self.data)))

    @property
    def n_dropped_samples(self) -> int:
        """Number of samples dropped due to NaN values."""
        return int(self.metadata.get("n_dropped_samples", 0))

    def get_features(self) -> pd.DataFrame:
        """Get feature columns for meta-learner."""
        # Exclude y_true and datetime columns
        feature_cols = [c for c in self.data.columns if c not in ("y_true", "datetime")]
        return self.data[feature_cols]

    def get_labels(self) -> pd.Series:
        """Get true labels."""
        return self.data["y_true"]


# =============================================================================
# NAN HANDLING UTILITIES
# =============================================================================


def find_valid_samples_mask(
    oof_predictions: dict[str, OOFPrediction],
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Find samples with valid predictions across all models.

    For sequence models (LSTM, GRU, TCN, etc.), samples at the beginning
    of each segment cannot have predictions due to lookback requirements.
    This function identifies samples with complete predictions from all models.

    Args:
        oof_predictions: Dict of OOF predictions by model name

    Returns:
        Tuple of:
            - valid_mask: Boolean array where True = sample has predictions from all models
            - nan_counts: Dict of NaN count per model (for logging)
    """
    if not oof_predictions:
        raise ValueError("oof_predictions cannot be empty")

    # Get sample count from first model
    first_model = next(iter(oof_predictions.keys()))
    n_samples = len(oof_predictions[first_model].predictions)

    # Start with all True, then AND with each model's valid mask
    valid_mask = np.ones(n_samples, dtype=bool)
    nan_counts: dict[str, int] = {}

    for model_name, oof_pred in oof_predictions.items():
        # Check the prediction column for NaN
        pred_col = f"{model_name}_pred"
        model_preds = oof_pred.predictions[pred_col].values
        model_valid = ~np.isnan(model_preds)

        nan_count = int((~model_valid).sum())
        nan_counts[model_name] = nan_count

        # Update overall mask
        valid_mask = valid_mask & model_valid

    return valid_mask, nan_counts


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
        oof_predictions: dict[str, OOFPrediction],
        y_true: pd.Series,
        horizon: int,
        add_derived_features: bool = True,
        drop_nan_samples: bool = True,
    ) -> StackingDataset:
        """
        Build stacking dataset from OOF predictions.

        Creates a DataFrame with:
        - model1_prob_short, model1_prob_neutral, model1_prob_long
        - model2_prob_short, model2_prob_neutral, model2_prob_long
        - Derived features (confidence, agreement, entropy)
        - y_true (label)

        Handles NaN values from sequence models by either dropping affected
        samples (recommended) or keeping them for downstream handling.

        Args:
            oof_predictions: Dict of OOF predictions by model
            y_true: True labels
            horizon: Label horizon (for metadata)
            add_derived_features: Whether to add derived features
            drop_nan_samples: If True, drop samples with any NaN predictions.
                This is REQUIRED when sequence models are included, as they
                cannot predict samples at the beginning of segments due to
                lookback requirements. Default True.

        Returns:
            StackingDataset for meta-learner training

        Raises:
            ValueError: If drop_nan_samples=False but NaN values exist
                (explicit handling required)
        """
        model_names = list(oof_predictions.keys())
        n_original = len(y_true)

        is_aligned, alignment_issues = validate_datetime_alignment(oof_predictions, strict=False)
        if not is_aligned:
            logger.warning(
                f"Datetime alignment issues detected ({len(alignment_issues)} issues). "
                "Aligning predictions on common timestamps..."
            )
            alignment_report = get_datetime_alignment_report(oof_predictions)
            logger.info(
                f"Alignment report: {alignment_report['common_coverage']:.1%} common coverage"
            )
            oof_predictions = align_predictions_on_datetime(oof_predictions, method="inner")

        # Find valid samples (no NaN in any model's predictions)
        valid_mask, nan_counts = find_valid_samples_mask(oof_predictions)
        n_valid = int(valid_mask.sum())
        n_dropped = n_original - n_valid

        # Log NaN statistics per model
        has_nans = any(count > 0 for count in nan_counts.values())
        if has_nans:
            logger.info(
                f"Stacking dataset: {n_valid}/{n_original} samples valid "
                f"({100 * n_valid / n_original:.1f}%)"
            )
            for model_name, count in nan_counts.items():
                if count > 0:
                    logger.info(
                        f"  {model_name}: {count} NaN samples " f"({100 * count / n_original:.1f}%)"
                    )

        # Handle NaN samples
        if n_dropped > 0 and not drop_nan_samples:
            raise ValueError(
                f"Found {n_dropped} samples with NaN predictions but drop_nan_samples=False. "
                f"Set drop_nan_samples=True to remove incomplete samples, or handle NaN "
                f"values manually before calling build_stacking_dataset."
            )

        # Start with first model's predictions
        first_model = model_names[0]
        stacking_df = oof_predictions[first_model].predictions.copy()

        # Add other models' predictions — only model-prefixed columns
        # (y_true and fold_id from the first model are authoritative)
        for model_name in model_names[1:]:
            oof_pred = oof_predictions[model_name].predictions
            prob_cols = [c for c in oof_pred.columns if c.startswith(model_name)]
            for col in prob_cols:
                stacking_df[col] = oof_pred[col]

        # Add true labels
        stacking_df["y_true"] = y_true.values

        # Drop NaN samples if requested (filter BEFORE adding derived features)
        if drop_nan_samples and n_dropped > 0:
            stacking_df = stacking_df.loc[valid_mask].reset_index(drop=True)
            logger.info(
                f"Dropped {n_dropped} samples with incomplete predictions. "
                f"Stacking dataset size: {len(stacking_df)}"
            )

        # Add derived features for meta-learner (after NaN removal)
        if add_derived_features:
            stacking_df = self._add_stacking_features(stacking_df, model_names)

        # Compute metadata with NaN handling info
        metadata = {
            "horizon": horizon,
            "n_models": len(model_names),
            "model_names": model_names,
            "n_samples": len(stacking_df),
            "n_original_samples": n_original,
            "n_dropped_samples": n_dropped,
            "nan_counts_per_model": nan_counts,
            "coverage": {m: oof_predictions[m].coverage for m in model_names},
            "effective_coverage": n_valid / n_original if n_original > 0 else 0.0,
        }

        if n_dropped > 0:
            logger.info(
                f"Stacking dataset built: {len(stacking_df)} samples "
                f"(effective coverage: {metadata['effective_coverage']:.1%})"
            )

        return StackingDataset(
            data=stacking_df,
            model_names=model_names,
            horizon=horizon,
            metadata=metadata,
        )

    def _add_stacking_features(
        self,
        df: pd.DataFrame,
        model_names: list[str],
    ) -> pd.DataFrame:
        """Add derived features for meta-learner.

        Standardized to 3 derived features (consistent with alignment.py,
        heterogeneous_stacking.py, and ensemble_bundle.py):
        - mean_confidence: average max probability across models
        - prediction_agreement: fraction of models agreeing on predicted class
        - prediction_entropy: entropy of averaged probabilities
        """
        df = df.copy()

        # Collect per-model probability arrays: (n_samples, n_classes) each
        all_probs = []
        for model in model_names:
            prob_cols = [f"{model}_prob_short", f"{model}_prob_neutral", f"{model}_prob_long"]
            probs = df[prob_cols].values
            all_probs.append(probs)

        # Stack: (n_models, n_samples, n_classes)
        stacked = np.stack(all_probs, axis=0)

        # 1. Mean confidence: average max probability across models
        max_probs = np.nanmax(stacked, axis=2)  # (n_models, n_samples)
        df["mean_confidence"] = np.nanmean(max_probs, axis=0)  # (n_samples,)

        # 2. Prediction agreement: fraction of models agreeing on predicted class
        predictions = np.argmax(stacked, axis=2)  # (n_models, n_samples)
        n_samples = len(df)
        agreement = np.zeros(n_samples)
        for i in range(n_samples):
            votes = predictions[:, i]
            valid_votes = votes[~np.isnan(stacked[:, i, 0])]
            if len(valid_votes) > 0:
                unique, counts = np.unique(valid_votes, return_counts=True)
                agreement[i] = np.max(counts) / len(valid_votes)
            else:
                agreement[i] = np.nan
        df["prediction_agreement"] = agreement

        # 3. Prediction entropy: entropy of averaged probabilities
        mean_probs = np.nanmean(stacked, axis=0)  # (n_samples, n_classes)
        mean_probs = np.clip(mean_probs, 1e-10, 1.0)
        mean_probs = mean_probs / mean_probs.sum(axis=1, keepdims=True)
        df["prediction_entropy"] = -np.sum(mean_probs * np.log(mean_probs), axis=1)

        return df


# =============================================================================
# HETEROGENEOUS STACKING BUILDER (Phase 4 SNwH)
# =============================================================================


class HeterogeneousStackingBuilder:
    """
    Build stacking datasets from heterogeneous OOF predictions.

    Handles alignment between tabular and sequence models which have
    different coverage ranges due to sequence lookback requirements.

    Unlike StackingDatasetBuilder which assumes all models cover the same
    samples, this builder:
    1. Validates alignment using OOFAlignmentValidator
    2. Computes the common sample range across all models
    3. Extracts aligned portions of OOF predictions
    4. Aligns labels and weights to the common range

    Example:
        >>> builder = HeterogeneousStackingBuilder()
        >>> X_stack, y_aligned, weights_aligned = builder.build_stacking_dataset(
        ...     oof_results={"xgboost": xgb_oof, "lstm": lstm_oof},
        ...     y=y_true,
        ...     sample_weights=sample_weights,
        ... )
    """

    def __init__(self, purge_bars: int = 60, embargo_bars: int = 1440) -> None:
        """
        Initialize HeterogeneousStackingBuilder.

        Args:
            purge_bars: Number of bars for purging (stored for metadata)
            embargo_bars: Number of bars for embargo (stored for metadata)
        """
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.validator = OOFAlignmentValidator()
        self._alignment_result: OOFAlignmentResult | None = None

    def build_stacking_dataset(
        self,
        oof_results: dict[str, OOFPrediction],
        y: pd.Series | np.ndarray,
        sample_weights: pd.Series | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Build aligned stacking dataset from heterogeneous OOF predictions.

        Aligns OOF predictions from different model types (tabular, sequence)
        to a common sample range where all models have valid predictions.

        Args:
            oof_results: Dictionary mapping model_name -> OOFPrediction
            y: True labels (full dataset)
            sample_weights: Optional sample weights (full dataset)

        Returns:
            Tuple of (X_stack, y_aligned, weights_aligned) where:
                - X_stack: Stacked probability features (n_common, n_models * 3)
                - y_aligned: Labels aligned to common range (n_common,)
                - weights_aligned: Weights aligned to common range, or None

        Raises:
            ValueError: If alignment fails or no common range exists
        """
        if not oof_results:
            raise ValueError("oof_results cannot be empty")

        # Convert y to numpy if needed
        y_array = y.values if isinstance(y, pd.Series) else y
        n_total = len(y_array)

        # Step 1: Register all models' coverage with the validator
        self.validator = OOFAlignmentValidator()  # Reset for fresh build

        for model_name, oof_pred in oof_results.items():
            # Use alignment metadata from OOFPrediction if available
            self.validator.register_oof(
                model_name=model_name,
                oof_indices=oof_pred.original_indices,
                n_total_samples=oof_pred.n_total_samples or n_total,
                sequence_length=oof_pred.sequence_length,
            )

        # Step 2: Validate alignment
        self._alignment_result = self.validator.validate()

        if not self._alignment_result.is_aligned:
            # Check if we have a common range despite issues
            if self._alignment_result.n_common_samples == 0:
                raise ValueError(
                    f"Cannot build stacking dataset: no common sample range. "
                    f"Issues: {self._alignment_result.issues}"
                )
            else:
                # Log warning but proceed
                logger.warning(
                    f"Alignment issues detected but proceeding with "
                    f"{self._alignment_result.n_common_samples} common samples. "
                    f"Issues: {self._alignment_result.issues}"
                )

        # Step 3: Extract aligned OOF probabilities
        common_start = self._alignment_result.common_start_idx
        common_end = self._alignment_result.common_end_idx
        n_common = self._alignment_result.n_common_samples

        logger.info(
            f"Building heterogeneous stacking dataset: "
            f"common_range=[{common_start}:{common_end}], "
            f"n_common={n_common}/{n_total} samples "
            f"({100 * n_common / n_total:.1f}% coverage)"
        )

        # Build stacking features matrix
        model_names = list(oof_results.keys())
        n_models = len(model_names)
        X_stack = np.zeros((n_common, n_models * 3))  # 3 classes per model

        for i, model_name in enumerate(model_names):
            oof_pred = oof_results[model_name]

            # Get aligned probabilities using the common range
            try:
                aligned_probs = oof_pred.get_aligned_probabilities(common_start, n_common)
            except ValueError as e:
                # Fallback: extract from full predictions array
                logger.warning(f"Using fallback alignment for {model_name}: {e}")
                probs = oof_pred.get_probabilities()

                # Compute offset based on model's alignment
                offset = self._alignment_result.offsets.get(model_name, 0)
                aligned_probs = probs[offset : offset + n_common]

            # Store in stacking matrix (3 probability columns per model)
            X_stack[:, i * 3 : (i + 1) * 3] = aligned_probs

        # Step 4: Align labels to common range
        y_aligned = y_array[common_start:common_end]

        # Step 5: Align weights to common range (if provided)
        weights_aligned: np.ndarray | None = None
        if sample_weights is not None:
            w_array = (
                sample_weights.values if isinstance(sample_weights, pd.Series) else sample_weights
            )
            weights_aligned = w_array[common_start:common_end]

        # Log summary
        logger.info(
            f"Heterogeneous stacking dataset built: "
            f"{n_common} samples, {n_models} models, "
            f"X_stack shape={X_stack.shape}"
        )

        return X_stack, y_aligned, weights_aligned

    def get_alignment_summary(self) -> dict[str, Any]:
        """
        Get summary of the last alignment operation.

        Returns:
            Dictionary with alignment details:
                - common_start: Start index of common range
                - common_end: End index of common range
                - n_common: Number of common samples
                - model_coverages: Coverage info per model
                - issues: Any alignment issues detected
        """
        if self._alignment_result is None:
            return {"error": "No alignment performed yet. Call build_stacking_dataset first."}

        return {
            "common_start": self._alignment_result.common_start_idx,
            "common_end": self._alignment_result.common_end_idx,
            "n_common": self._alignment_result.n_common_samples,
            "is_aligned": self._alignment_result.is_aligned,
            "sample_counts": self._alignment_result.sample_counts,
            "offsets": self._alignment_result.offsets,
            "issues": self._alignment_result.issues,
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
        }


__all__ = [
    "StackingDataset",
    "StackingDatasetBuilder",
    "HeterogeneousStackingBuilder",
    "find_valid_samples_mask",
]
