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
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from src.models.registry import ModelRegistry

# Import from specialized modules
from .oof_cache import OOFCache, compute_data_hash
from .oof_core import (
    CoreOOFGenerator,
    OOFPrediction,
)
from .oof_io import OOFDatasetIO
from .oof_sequence import (
    DEFAULT_SEQUENCE_LENGTH,
    SequenceOOFGenerator,
)
from .oof_stacking import (
    StackingDataset,
    StackingDatasetBuilder,
)
from .oof_validation import OOFValidator, _grade_diversity
from .purged_kfold import PurgedKFold

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
    - OOFCache: Caching of OOF predictions (optional)

    Example:
        >>> oof_gen = OOFGenerator(cv)
        >>> model_configs = {"xgboost": {"max_depth": 6}}
        >>> oof_predictions = oof_gen.generate_oof_predictions(X, y, model_configs)
        >>> stacking_ds = oof_gen.build_stacking_dataset(oof_predictions, y, horizon=20)

    Example with caching:
        >>> oof_gen = OOFGenerator(cv, cache_dir=Path("cache/oof"))
        >>> oof_predictions = oof_gen.generate_oof_predictions(
        ...     X, y, model_configs, use_cache=True
        ... )
    """

    def __init__(
        self,
        cv: PurgedKFold,
        cache_dir: Path | None = None,
        n_classes: int = 3,
    ) -> None:
        """
        Initialize OOFGenerator.

        Args:
            cv: PurgedKFold cross-validator
            cache_dir: Directory for OOF prediction caching. If None, caching disabled.
            n_classes: Number of output classes (2 for binary mode, 3 default).
        """
        self.cv = cv
        self.n_classes = n_classes
        self._core_generator = CoreOOFGenerator(cv)
        self._sequence_generator = SequenceOOFGenerator(cv)
        self._stacking_builder = StackingDatasetBuilder()
        self._validator = OOFValidator()
        self._io = OOFDatasetIO()

        # Initialize cache if directory provided
        self._cache: OOFCache | None = None
        if cache_dir is not None:
            self._cache = OOFCache(cache_dir)
            logger.info(f"OOF caching enabled: {cache_dir}")

    def generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_configs: dict[str, dict[str, Any]],
        sample_weights: pd.Series | None = None,
        feature_subset: list[str] | None = None,
        calibrate: bool = False,
        calibration_method: str = "auto",
        label_end_times: pd.Series | None = None,
        use_cache: bool = False,
    ) -> dict[str, OOFPrediction]:
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
            use_cache: Whether to use cached OOF predictions if available.
                Requires cache_dir to be set in __init__.

        Returns:
            Dict mapping model_name to OOFPrediction

        Note:
            Calibration is leakage-safe because OOF predictions are already
            out-of-sample (each prediction is from a model that never saw
            that sample). The calibrator learns the mapping between OOF
            probability outputs and actual outcomes.
        """
        oof_results: dict[str, OOFPrediction] = {}

        # Apply feature subset if specified
        if feature_subset:
            X = X[feature_subset]

        # Compute data hash for caching (once for all models)
        data_hash: str | None = None
        if use_cache and self._cache is not None:
            data_hash = compute_data_hash(X, y)
            logger.debug(f"Data hash for caching: {data_hash}")

        # Get CV config for cache key
        cv_config = {
            "n_splits": self.cv.config.n_splits,
            "purge_bars": self.cv.config.purge_bars,
            "embargo_bars": self.cv.config.embargo_bars,
        }

        for model_name, config in model_configs.items():
            # Try cache first if enabled
            if use_cache and self._cache is not None and data_hash is not None:
                cache_key = self._cache.compute_cache_key(
                    model_name=model_name,
                    model_config=config,
                    data_hash=data_hash,
                    cv_config=cv_config,
                )

                if self._cache.has_oof(cache_key):
                    logger.info(f"Loading cached OOF predictions for {model_name}...")
                    cached = self._cache.get_oof(cache_key)
                    oof_pred = OOFPrediction(
                        model_name=model_name,
                        predictions=cached["predictions"],
                        fold_info=cached["fold_info"],
                        coverage=cached["metadata"].coverage,
                    )
                    oof_results[model_name] = oof_pred
                    logger.info(
                        f"  {model_name} (cached): {oof_pred.predictions.shape[0]} predictions, "
                        f"coverage={oof_pred.coverage:.2%}"
                    )
                    continue

            # Generate OOF predictions
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

            # Store in cache if enabled
            if use_cache and self._cache is not None and data_hash is not None:
                cache_key = self._cache.compute_cache_key(
                    model_name=model_name,
                    model_config=config,
                    data_hash=data_hash,
                    cv_config=cv_config,
                )
                self._cache.put_oof(
                    cache_key=cache_key,
                    predictions=oof_pred.predictions,
                    fold_info=oof_pred.fold_info,
                    model_name=model_name,
                    model_config=config,
                    data_hash=data_hash,
                    cv_config=cv_config,
                    coverage=oof_pred.coverage,
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
        config: dict[str, Any],
        sample_weights: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
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
                n_classes=self.n_classes,
            )
        else:
            return self._core_generator.generate_tabular_oof(
                X=X,
                y=y,
                model_name=model_name,
                config=config,
                sample_weights=sample_weights,
                label_end_times=label_end_times,
                n_classes=self.n_classes,
            )

    def validate_oof_coverage(
        self,
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
        return self._validator.validate_coverage(oof_predictions, original_index)

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
    return OOFValidator.analyze_prediction_correlation(stacking_df, model_names)


__all__ = [
    "OOFPrediction",
    "StackingDataset",
    "OOFGenerator",
    "analyze_prediction_correlation",
    "_grade_diversity",  # Re-exported for backward compatibility
]
