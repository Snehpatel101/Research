"""
Sequence model OOF (Out-of-Fold) prediction generation.

Handles specialized logic for generating out-of-sample predictions
for sequence models (LSTM, GRU, TCN, Transformer).
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.models.base import PredictionResult
from src.models.registry import ModelRegistry

from .fold_scaling import FoldAwareScaler, get_scaling_method_for_model
from .oof_core import OOFPrediction
from .purged_kfold import PurgedKFold
from .sequence_cv import SequenceCVBuilder

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
        config: dict[str, Any],
        seq_len: int,
        sample_weights: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
        symbol_column: str | None = "symbol",
        strict_validation: bool = True,  # Phase 4 SNwH: strict coverage validation
        n_classes: int = 3,
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
            strict_validation: If True, raise error on coverage issues (default True).
                             Set to False to proceed with warning for heterogeneous stacking.
            n_classes: Number of output classes (default 3: short/neutral/long).
                      Set to 2 for binary mode.

        Returns:
            OOFPrediction with mapped predictions and alignment metadata

        Note:
            Coverage < 100% is EXPECTED for sequence models due to lookback requirements.
            Each segment (separated by symbol boundaries or time gaps) loses seq_len
            samples at the start. Expected coverage ≈ 1 - (n_segments * seq_len / n_samples).
            Warnings only appear if coverage is significantly below expected (>5% below).

        Raises:
            ValueError: If strict_validation=True and coverage is unacceptably low
        """
        n_samples = len(X)

        # Initialize OOF storage at original sample indices
        oof_probs = np.full((n_samples, n_classes), np.nan)
        oof_preds = np.full(n_samples, np.nan)
        oof_confidence = np.full(n_samples, np.nan)
        oof_fold_ids = np.full(n_samples, -1, dtype=int)
        fold_info: list[dict[str, Any]] = []

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

        # Chunk size for memory-efficient validation prediction.
        # Each chunk materialises at most chunk_size * seq_len * n_features floats.
        val_chunk_size = 5000

        # Generate predictions fold by fold
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv.split(X, y, label_end_times=label_end_times)
        ):
            # -----------------------------------------------------------------
            # STEP 1: Scale raw 2D data at fold level (fit on train rows only)
            # -----------------------------------------------------------------
            # This avoids building a giant 3D array, flattening, scaling, and
            # reshaping.  Instead we scale the compact 2D data ONCE, then build
            # 3D windows from the already-scaled values.
            raw_X = seq_builder._X  # (n_samples, n_features)
            X_train_raw = raw_X[train_idx]

            # For transformation we scale ALL rows so that lookback windows
            # that reach into data outside the fold are also correctly scaled.
            # .copy() is required because fit_transform_fold scales in-place —
            # without it, raw_X (which is seq_builder._X) gets permanently
            # modified, corrupting data for subsequent folds.
            scaling_result = fold_scaler.fit_transform_fold(X_train_raw, raw_X.copy())

            # scaled_builder shares boundaries/labels but uses scaled features
            scaled_builder = seq_builder.with_scaled_data(scaling_result.X_val_scaled)

            # -----------------------------------------------------------------
            # STEP 2: Build train sequences (all-at-once — needed for model.fit)
            # -----------------------------------------------------------------
            train_result = scaled_builder.build_fold_sequences(
                train_idx, allow_lookback_outside=True
            )

            if train_result.n_sequences == 0:
                logger.warning(
                    f"  Fold {fold_idx + 1}: Skipping - no train sequences "
                    f"(from {len(train_idx)} samples)"
                )
                continue

            # -----------------------------------------------------------------
            # STEP 3: Build a small validation sample for model.fit's val_*
            # arguments (needed for early stopping / metric logging).
            # We take the first chunk only; this avoids materialising every
            # validation sequence just to pass to .fit().
            # -----------------------------------------------------------------
            first_val_chunk = next(
                scaled_builder.build_fold_sequences_chunked(
                    val_idx,
                    chunk_size=val_chunk_size,
                    allow_lookback_outside=True,
                ),
                None,
            )

            if first_val_chunk is None or first_val_chunk.n_sequences == 0:
                logger.warning(
                    f"  Fold {fold_idx + 1}: Skipping - no val sequences "
                    f"(from {len(val_idx)} samples)"
                )
                continue

            logger.debug(
                f"  Fold {fold_idx + 1}: train_seq={train_result.n_sequences} "
                f"(from {len(train_idx)}), val_idx_count={len(val_idx)}"
            )

            # -----------------------------------------------------------------
            # STEP 4: Train the model
            # -----------------------------------------------------------------
            model = ModelRegistry.create(model_name, config=config)

            training_metrics = model.fit(
                X_train=train_result.X_sequences,
                y_train=train_result.y,
                X_val=first_val_chunk.X_sequences,
                y_val=first_val_chunk.y,
                sample_weights=train_result.weights,
            )

            # -----------------------------------------------------------------
            # STEP 5: Predict on validation sequences IN CHUNKS
            # -----------------------------------------------------------------
            val_sequences_total = 0

            for chunk_idx, val_chunk in enumerate(
                scaled_builder.build_fold_sequences_chunked(
                    val_idx,
                    chunk_size=val_chunk_size,
                    allow_lookback_outside=True,
                )
            ):
                if val_chunk.n_sequences == 0:
                    continue

                prediction_output: PredictionResult = model.predict(val_chunk.X_sequences)

                # Map predictions back to original indices
                for seq_idx, original_idx in enumerate(val_chunk.target_indices):
                    oof_probs[original_idx] = prediction_output.class_probabilities[seq_idx]
                    oof_preds[original_idx] = prediction_output.class_predictions[seq_idx]
                    oof_confidence[original_idx] = prediction_output.confidence[seq_idx]
                    oof_fold_ids[original_idx] = fold_idx

                val_sequences_total += val_chunk.n_sequences
                logger.debug(
                    f"  Fold {fold_idx + 1}: predicted val chunk {chunk_idx + 1} "
                    f"({val_chunk.n_sequences} seqs, running total: {val_sequences_total})"
                )

            logger.info(
                f"  Fold {fold_idx + 1}: completed — "
                f"train_seq={train_result.n_sequences}, "
                f"val_seq={val_sequences_total}"
            )

            # Track fold info
            fold_info.append(
                {
                    "fold": fold_idx,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "train_sequences": train_result.n_sequences,
                    "val_sequences": val_sequences_total,
                    "val_accuracy": training_metrics.val_accuracy,
                    "val_f1": training_metrics.val_f1,
                }
            )

            # Free memory between folds to prevent OOM on large datasets
            # train_result holds 3D sequences (~18 GB per fold for 1.6M rows)
            del model, train_result, first_val_chunk
            del scaling_result, scaled_builder, X_train_raw
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Validate coverage (expected to be < 100% for sequence models due to lookback)
        coverage = float((~np.isnan(oof_preds)).mean())
        n_missing = int(np.isnan(oof_preds).sum())

        # Calculate expected coverage based on sequence length and boundaries
        # Each segment (symbol or gap-separated region) loses seq_len samples at start
        n_boundaries = (
            len(seq_builder._symbol_boundaries) if seq_builder._symbol_boundaries is not None else 0
        )
        n_segments = n_boundaries + 1  # boundaries divide data into segments
        expected_missing = n_segments * seq_len
        expected_coverage = max(0.0, 1.0 - (expected_missing / n_samples))

        # Only warn if coverage is significantly below expected
        coverage_shortfall = expected_coverage - coverage

        if coverage_shortfall > COVERAGE_WARNING_THRESHOLD:
            # Phase 4 SNwH: Strict validation for heterogeneous stacking
            if strict_validation:
                raise ValueError(
                    f"{model_name}: OOF coverage {coverage:.1%} is unacceptably low "
                    f"(expected ~{expected_coverage:.1%}). "
                    f"Missing {n_missing} samples. "
                    f"This will cause stacking alignment issues. "
                    f"Set strict_validation=False to proceed with warning."
                )
            else:
                logger.warning(
                    f"{model_name}: Coverage {coverage:.2%} is UNEXPECTEDLY LOW "
                    f"(expected ~{expected_coverage:.1%} for seq_len={seq_len}, {n_segments} segments). "
                    f"Missing {n_missing} samples ({coverage_shortfall:.1%} below expected). "
                    f"Proceeding anyway (strict_validation=False)."
                )
        else:
            logger.info(
                f"{model_name}: Coverage {coverage:.2%} ({n_missing} samples missing) - "
                f"EXPECTED for seq_len={seq_len} with {n_segments} segments. "
                f"Expected coverage: ~{expected_coverage:.1%}, actual is within normal range."
            )

        # Phase 4 SNwH: Store original indices for alignment
        valid_indices = np.where(~np.isnan(oof_preds))[0]

        # Build result DataFrame
        oof_df = pd.DataFrame(
            {
                "datetime": X.index if isinstance(X.index, pd.DatetimeIndex) else range(len(X)),
                "y_true": y.values,
                f"{model_name}_prob_short": oof_probs[:, 0],
                f"{model_name}_prob_neutral": oof_probs[:, 1],
                f"{model_name}_prob_long": oof_probs[:, 2],
                f"{model_name}_pred": oof_preds,
                f"{model_name}_confidence": oof_confidence,
                "fold_id": oof_fold_ids,
            }
        )

        return OOFPrediction(
            model_name=model_name,
            predictions=oof_df,
            fold_info=fold_info,
            coverage=coverage,
            # Phase 4 SNwH: Alignment metadata for heterogeneous stacking
            original_indices=valid_indices,
            sequence_length=seq_len,
            n_total_samples=n_samples,
        )


__all__ = [
    "SequenceOOFGenerator",
    "DEFAULT_SEQUENCE_LENGTH",
]
