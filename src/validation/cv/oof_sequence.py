"""
Sequence model OOF (Out-of-Fold) prediction generation.

Handles specialized logic for generating out-of-sample predictions
for sequence models (LSTM, GRU, TCN, Transformer).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.core.label_spans import LabelSpans
from src.models.base import PredictionResult
from src.models.registry import ModelRegistry

from .early_stopping_split import carve_early_stopping_split
from .fold_scaling import FoldAwareScaler, get_scaling_method_for_model
from .oof_core import OOFPrediction, build_oof_frame, held_out_fold_metrics
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
        label_spans: LabelSpans | None = None,
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
            label_spans: Optional label spans (bar positions) for purging
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

        # ONE backup of the raw feature array before the CV loop.
        # fit_transform_fold scales in-place, so without protection seq_builder._X
        # would be permanently modified after fold 1.  Previously we called
        # raw_X.copy() every fold (5x); now we copy once and restore via
        # np.copyto (in-place overwrite, no new allocation).
        raw_X_backup = seq_builder._X.copy()

        # Generate predictions fold by fold
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv.split(X, y, label_end_times=label_end_times, label_spans=label_spans)
        ):
            # Early stopping selects on a purged tail of the TRAIN rows —
            # never on the held-out fold these predictions are made for.
            es_split = carve_early_stopping_split(train_idx, self.cv.config.purge_bars)
            fit_idx, es_idx = es_split.fit_idx, es_split.es_idx

            # -----------------------------------------------------------------
            # STEP 1: Scale raw 2D data at fold level (fit on fit rows only)
            # -----------------------------------------------------------------
            # This avoids building a giant 3D array, flattening, scaling, and
            # reshaping.  Instead we scale the compact 2D data ONCE, then build
            # 3D windows from the already-scaled values.

            # Restore raw features from backup (in-place, no new allocation)
            np.copyto(seq_builder._X, raw_X_backup)
            raw_X = seq_builder._X  # (n_samples, n_features)
            X_train_raw = raw_X[fit_idx]

            # For transformation we scale ALL rows so that lookback windows
            # that reach into data outside the fold are also correctly scaled.
            # fit_transform_fold scales in-place — the backup/restore pattern
            # above protects seq_builder._X from permanent modification.
            scaling_result = fold_scaler.fit_transform_fold(X_train_raw, raw_X)

            # scaled_builder shares boundaries/labels but uses scaled features
            scaled_builder = seq_builder.with_scaled_data(scaling_result.X_val_scaled)

            # -----------------------------------------------------------------
            # STEP 2: Build fit + early-stopping sequences (needed for model.fit)
            # -----------------------------------------------------------------
            train_result = scaled_builder.build_fold_sequences(fit_idx, allow_lookback_outside=True)

            if train_result.n_sequences == 0:
                logger.warning(
                    f"  Fold {fold_idx + 1}: Skipping - no train sequences "
                    f"(from {len(fit_idx)} samples)"
                )
                continue

            es_result = scaled_builder.build_fold_sequences(es_idx, allow_lookback_outside=True)
            if es_result.n_sequences == 0:
                # Boundaries swallowed the tail: fall back to a fixed-length
                # fit validated on the fit sequences themselves.
                logger.warning(
                    f"  Fold {fold_idx + 1}: no early-stopping sequences from "
                    f"{len(es_idx)} tail rows; fitting fixed-length"
                )
                es_result = train_result

            logger.debug(
                f"  Fold {fold_idx + 1}: fit_seq={train_result.n_sequences}, "
                f"early_stop_seq={es_result.n_sequences}, held_out_rows={len(val_idx)}"
            )

            # -----------------------------------------------------------------
            # STEP 3: Train the model
            # -----------------------------------------------------------------
            model = ModelRegistry.create(model_name, config=config)

            model.fit(
                X_train=train_result.X_sequences,
                y_train=train_result.y,
                X_val=es_result.X_sequences,
                y_val=es_result.y,
                sample_weights=train_result.weights,
            )

            # -----------------------------------------------------------------
            # STEP 4: Predict the untouched held-out fold IN CHUNKS
            # -----------------------------------------------------------------
            val_sequences_total = 0
            fold_y: list[np.ndarray] = []
            fold_pred: list[np.ndarray] = []

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
                targets = np.asarray(val_chunk.target_indices)
                oof_probs[targets] = prediction_output.class_probabilities
                oof_preds[targets] = prediction_output.class_predictions
                oof_confidence[targets] = prediction_output.confidence
                oof_fold_ids[targets] = fold_idx
                fold_y.append(np.asarray(val_chunk.y))
                fold_pred.append(np.asarray(prediction_output.class_predictions))

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
                    "train_size": len(fit_idx),
                    "early_stopping_size": len(es_idx),
                    "early_stopping_held_out": es_split.held_out,
                    "val_size": len(val_idx),
                    "train_sequences": train_result.n_sequences,
                    "val_sequences": val_sequences_total,
                    **held_out_fold_metrics(
                        np.concatenate(fold_y) if fold_y else np.empty(0),
                        np.concatenate(fold_pred) if fold_pred else np.empty(0),
                    ),
                }
            )

            # Free memory between folds to prevent OOM on large datasets
            # train_result holds 3D sequences (~18 GB per fold for 1.6M rows)
            del model, train_result, es_result
            del scaling_result, scaled_builder, X_train_raw
            from src.models.device import release_gpu_memory

            release_gpu_memory()

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

        oof_df = build_oof_frame(
            model_name, X.index, y.values, oof_probs, oof_preds, oof_confidence, oof_fold_ids
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
