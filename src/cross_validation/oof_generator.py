"""
Out-of-Fold (OOF) Prediction Generator.

Generates truly out-of-sample predictions where each sample is predicted
by a model that never saw that sample during training. These OOF predictions
become training data for Phase 4 ensemble stacking.

Why OOF predictions matter:
- In-sample predictions are overconfident (overfitting)
- OOF predictions reflect realistic model performance
- Meta-learner trains on honest prediction quality
- Better generalization to new data

This module provides a unified interface that delegates to specialized
sub-modules for different aspects of OOF generation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.cross_validation.purged_kfold import PurgedKFold
from src.models.registry import ModelRegistry

# Import from specialized modules
from src.cross_validation.oof_core import (
    OOFPrediction,
    CoreOOFGenerator,
)
from src.cross_validation.oof_sequence import (
    SequenceOOFGenerator,
    DEFAULT_SEQUENCE_LENGTH,
)
from src.cross_validation.oof_stacking import (
    StackingDataset,
    StackingDatasetBuilder,
)
from src.cross_validation.oof_validation import OOFValidator, _grade_diversity
from src.cross_validation.oof_io import OOFDatasetIO

logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED OOF GENERATOR
# =============================================================================

class OOFGenerator:
    """
    Generate out-of-fold predictions for stacking.

    Each sample gets a prediction from a model trained without
    seeing that sample. This prevents overfitting in the meta-learner.

    This class provides a unified interface that delegates to:
    - CoreOOFGenerator: Tabular model OOF generation
    - SequenceOOFGenerator: Sequence model OOF generation
    - StackingDatasetBuilder: Stacking dataset construction
    - OOFValidator: Coverage and correlation validation
    - OOFDatasetIO: Save/load operations

    Example:
        >>> oof_gen = OOFGenerator(cv)
        >>> model_configs = {"xgboost": {"max_depth": 6}}
        >>> oof_predictions = oof_gen.generate_oof_predictions(X, y, model_configs)
        >>> stacking_ds = oof_gen.build_stacking_dataset(oof_predictions, y, horizon=20)
    """

    def __init__(self, cv: PurgedKFold) -> None:
        """
        Initialize OOFGenerator.

        Args:
            cv: PurgedKFold cross-validator
        """
        self.cv = cv
        self._core_generator = CoreOOFGenerator(cv)
        self._sequence_generator = SequenceOOFGenerator(cv)
        self._stacking_builder = StackingDatasetBuilder()
        self._validator = OOFValidator()
        self._io = OOFDatasetIO()

    def generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_configs: Dict[str, Dict[str, Any]],
        sample_weights: Optional[pd.Series] = None,
        feature_subset: Optional[List[str]] = None,
        calibrate: bool = False,
        calibration_method: str = "auto",
        label_end_times: Optional[pd.Series] = None,
    ) -> Dict[str, OOFPrediction]:
        """
        Generate OOF predictions for all models.

        Args:
            X: Feature DataFrame
            y: Labels
            model_configs: Dict mapping model_name to hyperparameters
            sample_weights: Optional quality weights
            feature_subset: Optional subset of features to use
            calibrate: Whether to apply probability calibration to OOF predictions
            calibration_method: Calibration method ("auto", "isotonic", "sigmoid")
            label_end_times: Optional Series of datetime when each label is resolved.
                If provided, enables proper purging of overlapping labels in CV.

        Returns:
            Dict mapping model_name to OOFPrediction

        Note:
            Calibration is leakage-safe because OOF predictions are already
            out-of-sample (each prediction is from a model that never saw
            that sample). The calibrator learns the mapping between OOF
            probability outputs and actual outcomes.
        """
        oof_results: Dict[str, OOFPrediction] = {}

        # Apply feature subset if specified
        if feature_subset:
            X = X[feature_subset]

        for model_name, config in model_configs.items():
            logger.info(f"Generating OOF predictions for {model_name}...")

            oof_pred = self._generate_single_model_oof(
                X=X,
                y=y,
                model_name=model_name,
                config=config,
                sample_weights=sample_weights,
                label_end_times=label_end_times,
            )
            oof_results[model_name] = oof_pred

            logger.info(
                f"  {model_name}: {oof_pred.predictions.shape[0]} predictions, "
                f"coverage={oof_pred.coverage:.2%}"
            )

        # Apply calibration if requested (leakage-safe: OOF predictions are out-of-sample)
        if calibrate:
            oof_results = self._core_generator.calibrate_oof_predictions(
                oof_results, y, calibration_method
            )

        return oof_results

    def _generate_single_model_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        config: Dict[str, Any],
        sample_weights: Optional[pd.Series] = None,
        label_end_times: Optional[pd.Series] = None,
    ) -> OOFPrediction:
        """Generate OOF predictions for a single model."""
        # Check if model requires sequences
        try:
            model_info = ModelRegistry.get_model_info(model_name)
            requires_sequences = model_info.get("requires_sequences", False)
        except ValueError:
            requires_sequences = False

        # Route to appropriate generator
        if requires_sequences:
            seq_len = config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)
            return self._sequence_generator.generate_sequence_oof(
                X=X,
                y=y,
                model_name=model_name,
                config=config,
                seq_len=seq_len,
                sample_weights=sample_weights,
                label_end_times=label_end_times,
            )
        else:
            return self._core_generator.generate_tabular_oof(
                X=X,
                y=y,
                model_name=model_name,
                config=config,
                sample_weights=sample_weights,
                label_end_times=label_end_times,
            )

    def validate_oof_coverage(
        self,
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
        return self._validator.validate_coverage(oof_predictions, original_index)

    def build_stacking_dataset(
        self,
        oof_predictions: Dict[str, OOFPrediction],
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
        """
        return self._stacking_builder.build_stacking_dataset(
            oof_predictions, y_true, horizon, add_derived_features, drop_nan_samples
        )

    def save_stacking_dataset(
        self,
        stacking_ds: StackingDataset,
        output_dir: Path,
    ) -> Path:
        """
        Save stacking dataset to parquet.

        Args:
            stacking_ds: StackingDataset to save
            output_dir: Output directory

        Returns:
            Path to saved parquet file
        """
        return self._io.save_stacking_dataset(stacking_ds, output_dir)


# =============================================================================
# UTILITIES
# =============================================================================

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
    return OOFValidator.analyze_prediction_correlation(stacking_df, model_names)


__all__ = [
    "OOFPrediction",
    "StackingDataset",
    "OOFGenerator",
    "analyze_prediction_correlation",
    "_grade_diversity",  # Re-exported for backward compatibility
]
