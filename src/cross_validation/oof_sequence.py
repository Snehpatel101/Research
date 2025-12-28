"""
Sequence model OOF (Out-of-Fold) prediction generation.

Handles specialized logic for generating out-of-sample predictions
for sequence models (LSTM, GRU, TCN, Transformer).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.cross_validation.purged_kfold import PurgedKFold
from src.cross_validation.fold_scaling import FoldAwareScaler, get_scaling_method_for_model
from src.cross_validation.sequence_cv import SequenceCVBuilder
from src.cross_validation.oof_core import OOFPrediction
from src.models.registry import ModelRegistry
from src.models.base import PredictionOutput

logger = logging.getLogger(__name__)

# Default sequence length for sequence models
DEFAULT_SEQUENCE_LENGTH = 60

# Coverage validation thresholds
COVERAGE_WARNING_THRESHOLD = 0.05  # Warn if coverage is >5% below expected


# =============================================================================
# SEQUENCE OOF GENERATOR
# =============================================================================

class SequenceOOFGenerator:
    """
    OOF prediction generator for sequence models (LSTM, GRU, TCN, etc.).

    Handles 3D sequence construction with proper boundary detection
    and fold-aware scaling.
    """

    def __init__(self, cv: PurgedKFold) -> None:
        """
        Initialize SequenceOOFGenerator.

        Args:
            cv: PurgedKFold cross-validator
        """
        self.cv = cv

    def generate_sequence_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        config: Dict[str, Any],
        seq_len: int,
        sample_weights: Optional[pd.Series] = None,
        label_end_times: Optional[pd.Series] = None,
        symbol_column: Optional[str] = "symbol",
    ) -> OOFPrediction:
        """
        Generate OOF predictions for a sequence model (LSTM, GRU, TCN, etc.).

        This method properly handles 3D sequence construction for each CV fold:
        1. Builds sequences from fold indices using SequenceCVBuilder
        2. Respects boundaries (symbol changes or time gaps) - no cross-boundary sequences
        3. Maps predictions back to original sample indices

        Args:
            X: Feature DataFrame (with DatetimeIndex recommended for gap detection)
            y: Label Series
            model_name: Name of the sequence model
            config: Model configuration
            seq_len: Sequence length
            sample_weights: Optional sample weights
            label_end_times: Optional label end times for purging
            symbol_column: Column name for symbol isolation (None to use datetime gaps)

        Returns:
            OOFPrediction with mapped predictions

        Note:
            Coverage < 100% is EXPECTED for sequence models due to lookback requirements.
            Each segment (separated by symbol boundaries or time gaps) loses seq_len
            samples at the start. Expected coverage ≈ 1 - (n_segments * seq_len / n_samples).
            Warnings only appear if coverage is significantly below expected (>5% below).
        """
        n_samples = len(X)
        n_classes = 3  # short, neutral, long

        # Initialize OOF storage at original sample indices
        oof_probs = np.full((n_samples, n_classes), np.nan)
        oof_preds = np.full(n_samples, np.nan)
        oof_confidence = np.full(n_samples, np.nan)
        fold_info: List[Dict[str, Any]] = []

        # Create sequence builder with symbol awareness
        # Check if symbol column exists
        actual_symbol_col = symbol_column if symbol_column in X.columns else None

        seq_builder = SequenceCVBuilder(
            X=X,
            y=y,
            seq_len=seq_len,
            weights=sample_weights,
            symbol_column=actual_symbol_col,
        )

        # Determine scaling method for sequence model
        scaling_method = get_scaling_method_for_model(model_name)
        fold_scaler = FoldAwareScaler(method=scaling_method)

        # Log boundary detection method
        logger.info(
            f"Generating sequence OOF for {model_name} (seq_len={seq_len}, "
            f"boundary_detection={seq_builder._boundary_detection_method})"
        )

        # Generate predictions fold by fold
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv.split(X, y, label_end_times=label_end_times)
        ):
            # Build 3D sequences for this fold
            # allow_lookback_outside=True: sequence lookback can include data outside fold
            # but TARGET must be in fold
            train_result = seq_builder.build_fold_sequences(
                train_idx, allow_lookback_outside=True
            )
            val_result = seq_builder.build_fold_sequences(
                val_idx, allow_lookback_outside=True
            )

            if train_result.n_sequences == 0 or val_result.n_sequences == 0:
                logger.warning(
                    f"  Fold {fold_idx + 1}: Skipping - insufficient sequences "
                    f"(train={train_result.n_sequences}, val={val_result.n_sequences})"
                )
                continue

            logger.debug(
                f"  Fold {fold_idx + 1}: train_seq={train_result.n_sequences} "
                f"(from {len(train_idx)}), val_seq={val_result.n_sequences} "
                f"(from {len(val_idx)})"
            )

            # FOLD-AWARE SCALING on the 3D sequences
            # Reshape to 2D for scaling, then back to 3D
            train_shape = train_result.X_sequences.shape  # (n_train, seq_len, features)
            val_shape = val_result.X_sequences.shape  # (n_val, seq_len, features)

            # Flatten: (n_samples * seq_len, features)
            X_train_flat = train_result.X_sequences.reshape(-1, train_shape[2])
            X_val_flat = val_result.X_sequences.reshape(-1, val_shape[2])

            scaling_result = fold_scaler.fit_transform_fold(X_train_flat, X_val_flat)

            # Reshape back to 3D
            X_train_scaled = scaling_result.X_train_scaled.reshape(train_shape)
            X_val_scaled = scaling_result.X_val_scaled.reshape(val_shape)

            # Create and train sequence model
            model = ModelRegistry.create(model_name, config=config)

            training_metrics = model.fit(
                X_train=X_train_scaled,
                y_train=train_result.y,
                X_val=X_val_scaled,
                y_val=val_result.y,
                sample_weights=train_result.weights,
            )

            # Generate predictions for validation sequences
            prediction_output: PredictionOutput = model.predict(X_val_scaled)

            # Map predictions back to original indices
            for seq_idx, original_idx in enumerate(val_result.target_indices):
                oof_probs[original_idx] = prediction_output.class_probabilities[seq_idx]
                oof_preds[original_idx] = prediction_output.class_predictions[seq_idx]
                oof_confidence[original_idx] = prediction_output.confidence[seq_idx]

            # Track fold info
            fold_info.append({
                "fold": fold_idx,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_sequences": train_result.n_sequences,
                "val_sequences": val_result.n_sequences,
                "val_accuracy": training_metrics.val_accuracy,
                "val_f1": training_metrics.val_f1,
            })

        # Validate coverage (expected to be < 100% for sequence models due to lookback)
        coverage = float((~np.isnan(oof_preds)).mean())
        n_missing = int(np.isnan(oof_preds).sum())

        # Calculate expected coverage based on sequence length and boundaries
        # Each segment (symbol or gap-separated region) loses seq_len samples at start
        n_boundaries = (
            len(seq_builder._symbol_boundaries)
            if seq_builder._symbol_boundaries is not None
            else 0
        )
        n_segments = n_boundaries + 1  # boundaries divide data into segments
        expected_missing = n_segments * seq_len
        expected_coverage = max(0.0, 1.0 - (expected_missing / n_samples))

        # Only warn if coverage is significantly below expected
        coverage_shortfall = expected_coverage - coverage

        if coverage_shortfall > COVERAGE_WARNING_THRESHOLD:
            logger.warning(
                f"{model_name}: Coverage {coverage:.2%} is UNEXPECTEDLY LOW "
                f"(expected ~{expected_coverage:.1%} for seq_len={seq_len}, {n_segments} segments). "
                f"Missing {n_missing} samples ({coverage_shortfall:.1%} below expected). "
                f"Investigate: possible data issues or excessive gaps."
            )
        else:
            logger.info(
                f"{model_name}: Coverage {coverage:.2%} ({n_missing} samples missing) - "
                f"EXPECTED for seq_len={seq_len} with {n_segments} segments. "
                f"Expected coverage: ~{expected_coverage:.1%}, actual is within normal range."
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


__all__ = [
    "SequenceOOFGenerator",
    "DEFAULT_SEQUENCE_LENGTH",
]
