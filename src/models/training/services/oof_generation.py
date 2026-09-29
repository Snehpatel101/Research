"""Service for generating out-of-fold predictions."""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.label_spans import LabelSpans
from src.data.adapters import PreparedData
from src.models.base import PredictionResult
from src.models.registry import ModelRegistry
from src.validation.cv import OOFGenerator, OOFPrediction, PurgedKFold, PurgedKFoldConfig
from src.validation.cv.early_stopping_split import carve_early_stopping_split
from src.validation.cv.oof_core import (
    _get_prob_column_names,
    held_out_fold_metrics,
    reindex_oof_to_rows,
)
from src.validation.cv.oof_validation import OOFValidator

logger = logging.getLogger(__name__)


@dataclass
class OOFRequest:
    """Request to generate OOF predictions.

    Label spans for overlap purging come from ``prepared_data`` (its
    ``label_end_positions`` mapped through ``train_indices``, so windowed
    3D/4D samples use the span of their label bar). Pass ``label_spans`` to
    override them; either way they must cover every training sample.
    """

    model_name: str
    horizon: int
    prepared_data: PreparedData
    n_splits: int = 5
    purge_bars: int = 10
    embargo_bars: int = 5
    model_config: dict[str, Any] | None = None  # Model config (seq_length, hidden_size, etc.)
    n_classes: int = 3  # Number of output classes (2 for binary, 3 for short/neutral/long)
    label_spans: LabelSpans | None = None  # Override for prepared_data.label_spans("train")

    def train_label_spans(self) -> LabelSpans | None:
        """Label spans of the training samples (None = fixed purge only)."""
        if self.label_spans is not None:
            return self.label_spans
        return self.prepared_data.label_spans("train")


class OOFGenerationService:
    """
    Service for generating out-of-fold predictions.

    Responsibilities:
    - Create CV strategy
    - Generate OOF predictions via cross-validation
    - Return OOFPrediction object

    Does NOT:
    - Train the final model (handled by ModelTrainingService)
    - Align OOF predictions (handled by ensemble module)
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize OOF generation service.

        Args:
            cache_dir: Optional directory for caching OOF predictions
        """
        self._cache_dir = cache_dir

    def _create_generator(self, request: OOFRequest) -> OOFGenerator:
        """
        Create a fresh OOF generator from request parameters.

        A new generator is created per call to prevent CV config
        contamination when this service is shared across models.

        Args:
            request: OOF generation request

        Returns:
            Configured OOFGenerator instance
        """
        cv_config = PurgedKFoldConfig(
            n_splits=request.n_splits,
            purge_bars=request.purge_bars,
            embargo_bars=request.embargo_bars,
        )
        cv = PurgedKFold(cv_config)
        return OOFGenerator(cv, cache_dir=self._cache_dir, n_classes=request.n_classes)

    def _create_cv(self, request: OOFRequest) -> PurgedKFold:
        """Create a fresh PurgedKFold CV splitter from request parameters."""
        cv_config = PurgedKFoldConfig(
            n_splits=request.n_splits,
            purge_bars=request.purge_bars,
            embargo_bars=request.embargo_bars,
        )
        return PurgedKFold(cv_config)

    @staticmethod
    def _model_config(request: OOFRequest) -> dict[str, Any]:
        """Fold-model config: the run's class count is part of the problem definition."""
        return {**(request.model_config or {}), "n_classes": request.n_classes}

    def generate_oof(self, request: OOFRequest) -> OOFPrediction | None:
        """
        Generate out-of-fold predictions for a model.

        Routes to the appropriate OOF generation strategy based on data rank:
        - 2D/3D: Uses the standard OOFGenerator (flatten + tabular/sequence path)
        - 4D: Uses direct 4D OOF generation (samples are already windowed)

        On CUDA OOM the method frees GPU memory and retries once.  If the
        retry also fails it falls back to CPU so that ensemble stacking
        always gets the OOF predictions it needs — regardless of GPU size.

        Args:
            request: OOF generation request containing model name, horizon,
                    prepared data, and CV configuration

        Returns:
            OOFPrediction object or None if generation fails
        """
        from src.models.device import release_gpu_memory

        try:
            return self._generate_oof_inner(request)
        except RuntimeError as e:
            error_msg = str(e).lower()
            is_cuda_error = any(
                kw in error_msg for kw in ["out of memory", "cuda", "cublas", "cudnn", "nccl"]
            )
            if not is_cuda_error:
                logger.warning(f"Failed to generate OOF for {request.model_name}: {e}")
                return None
            # First CUDA error: free GPU memory and retry on same device
            logger.warning(
                f"CUDA error during OOF for {request.model_name} — "
                "freeing memory and retrying..."
            )
            release_gpu_memory()
            try:
                return self._generate_oof_inner(request)
            except RuntimeError as e2:
                error_msg2 = str(e2).lower()
                is_cuda_error2 = any(
                    kw in error_msg2 for kw in ["out of memory", "cuda", "cublas", "cudnn", "nccl"]
                )
                if not is_cuda_error2:
                    logger.warning(f"OOF retry failed for {request.model_name}: {e2}")
                    return None
                # Second CUDA error: fall back to CPU
                logger.warning(
                    f"CUDA error persists for {request.model_name} — "
                    "falling back to CPU for OOF generation"
                )
                import os

                prev = os.environ.get("CUDA_VISIBLE_DEVICES")
                try:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    release_gpu_memory()
                    return self._generate_oof_inner(request)
                except Exception as cpu_err:
                    logger.warning(
                        f"OOF CPU fallback also failed for {request.model_name}: {cpu_err}"
                    )
                    return None
                finally:
                    if prev is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = prev
        except Exception as e:
            logger.warning(f"Failed to generate OOF for {request.model_name}: {e}")
            return None

    def _generate_oof_inner(self, request: OOFRequest) -> OOFPrediction | None:
        """Core OOF generation logic (no error handling — called by generate_oof)."""
        prepared = request.prepared_data
        row_positions = prepared.train_indices

        # Sequence (3D) and multi-stream (4D) samples are already windows:
        # split them by sample index. Flattening them and re-windowing would
        # train OOF models on windows-of-windows (seq_len x the features).
        if prepared.data_rank in (3, 4):
            return reindex_oof_to_rows(self._generate_windowed_oof(request), row_positions)

        model_name = request.model_name
        X_train_2d = prepared.X_train
        spans = request.train_label_spans()

        X_train_df = pd.DataFrame(
            X_train_2d,
            columns=[f"f{i}" for i in range(X_train_2d.shape[1])],
        )
        y_train = pd.Series(prepared.y_train)
        weights = pd.Series(prepared.train_weights) if prepared.has_weights else None

        # Drop intermediate references — X_train_df/y_train hold the data
        del X_train_2d, prepared
        gc.collect()

        oof_generator = self._create_generator(request)

        oof_predictions = oof_generator.generate_oof_predictions(
            X=X_train_df,
            y=y_train,
            model_configs={model_name: self._model_config(request)},
            sample_weights=weights,
            label_spans=spans,
            use_cache=True,
        )

        # Post-training fold leakage verification (C4 audit fix)
        cv = self._create_cv(request)
        fold_indices = list(cv.split(X_train_df, y_train, label_spans=spans))
        leakage_result = OOFValidator.validate_fold_leakage(
            fold_indices=fold_indices,
            purge_bars=request.purge_bars,
            embargo_bars=request.embargo_bars,
        )
        if not leakage_result["passed"]:
            logger.error(
                "OOF fold leakage detected for %s: %d violations",
                model_name,
                leakage_result["n_violations"],
            )

        return reindex_oof_to_rows(oof_predictions.get(model_name), row_positions)

    def _generate_windowed_oof(self, request: OOFRequest) -> OOFPrediction | None:
        """
        Generate OOF predictions for windowed (3D sequence / 4D multi-stream) models.

        PreparedData X_train has shape (n_samples, seq_len, n_features) or
        (n_samples, n_timeframes, seq_len, n_features). Each sample is
        already a window ending at its label bar, so CV splits by sample
        index (no re-windowing needed).

        Args:
            request: OOF generation request

        Returns:
            OOFPrediction or None if generation fails
        """
        prepared = request.prepared_data
        model_name = request.model_name
        X_4d = prepared.X_train  # (n_samples, [n_timeframes,] seq_len, n_features)
        y = prepared.y_train

        n_samples = X_4d.shape[0]
        n_classes = request.n_classes

        logger.info(
            f"Generating windowed OOF predictions for {model_name} "
            f"(shape={X_4d.shape}, n_splits={request.n_splits})"
        )

        # Initialize OOF storage
        oof_probs = np.full((n_samples, n_classes), np.nan)
        oof_preds = np.full(n_samples, np.nan)
        oof_confidence = np.full(n_samples, np.nan)
        oof_fold_ids = np.full(n_samples, -1, dtype=int)
        fold_info: list[dict[str, Any]] = []

        # Create a dummy 2D DataFrame for PurgedKFold.split() index generation
        # (PurgedKFold only needs the length; label spans purge overlapping labels)
        X_dummy = pd.DataFrame({"dummy": np.zeros(n_samples)})
        y_series = pd.Series(y)

        cv = self._create_cv(request)

        # Collect fold indices for post-training leakage verification
        all_fold_indices: list[tuple[np.ndarray, np.ndarray]] = []

        splits = cv.split(X_dummy, y_series, label_spans=request.train_label_spans())
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.debug(f"  Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")
            all_fold_indices.append((train_idx, val_idx))

            # Early stopping selects on a purged tail of the TRAIN samples —
            # never on the held-out fold these predictions are made for.
            es_split = carve_early_stopping_split(train_idx, request.purge_bars)
            fit_idx, es_idx = es_split.fit_idx, es_split.es_idx

            # Slice windowed arrays directly by sample index. Fancy indexing
            # already returns new arrays, so .copy() is redundant and the
            # in-place scaling below cannot touch prepared.X_train.
            X_fit_fold = X_4d[fit_idx]
            X_es_fold = X_4d[es_idx]
            X_val_fold = X_4d[val_idx]

            # Per-fold robust scaling fit on the fit samples only
            n_feat = X_fit_fold.shape[-1]
            fit_2d = X_fit_fold.reshape(-1, n_feat)
            median = np.median(fit_2d, axis=0).astype(np.float32)
            q75 = np.percentile(fit_2d, 75, axis=0).astype(np.float32)
            q25 = np.percentile(fit_2d, 25, axis=0).astype(np.float32)
            iqr = np.where((q75 - q25) > 1e-8, q75 - q25, np.float32(1.0))
            for block in (X_fit_fold, X_es_fold, X_val_fold):
                block -= median
                block /= iqr
            del fit_2d, median, q75, q25, iqr

            w_fit = None
            if prepared.train_weights is not None:
                w_fit = prepared.train_weights[fit_idx]

            model = ModelRegistry.create(model_name, config=self._model_config(request))
            model.fit(
                X_train=X_fit_fold,
                y_train=y[fit_idx],
                X_val=X_es_fold,
                y_val=y[es_idx],
                sample_weights=w_fit,
            )

            # Predict the untouched held-out fold
            prediction_output: PredictionResult = model.predict(X_val_fold)

            # Store OOF predictions at original indices
            oof_probs[val_idx] = prediction_output.class_probabilities
            oof_preds[val_idx] = prediction_output.class_predictions
            oof_confidence[val_idx] = prediction_output.confidence
            oof_fold_ids[val_idx] = fold_idx

            fold_info.append(
                {
                    "fold": fold_idx,
                    "train_size": len(fit_idx),
                    "early_stopping_size": len(es_idx),
                    "early_stopping_held_out": es_split.held_out,
                    "val_size": len(val_idx),
                    **held_out_fold_metrics(y[val_idx], prediction_output.class_predictions),
                }
            )

            # Free fold model and data to prevent memory accumulation
            del model, X_fit_fold, X_es_fold, X_val_fold, prediction_output
            gc.collect()

        # Post-training fold leakage verification (C4 audit fix)
        leakage_result = OOFValidator.validate_fold_leakage(
            fold_indices=all_fold_indices,
            purge_bars=request.purge_bars,
            embargo_bars=request.embargo_bars,
        )
        if not leakage_result["passed"]:
            logger.error(
                "Windowed OOF fold leakage detected for %s: %d violations",
                model_name,
                leakage_result["n_violations"],
            )

        # Validate coverage
        coverage = float((~np.isnan(oof_preds)).mean())
        if coverage < 1.0:
            logger.warning(
                f"{model_name}: windowed OOF coverage {coverage:.2%}. "
                f"{int(np.isnan(oof_preds).sum())} samples missing predictions."
            )

        # Build result DataFrame with dynamic probability columns
        prob_col_names = _get_prob_column_names(model_name, n_classes)
        oof_data: dict[str, Any] = {
            "datetime": range(n_samples),
            "y_true": y,
        }
        for i, col_name in enumerate(prob_col_names):
            oof_data[col_name] = oof_probs[:, i]
        oof_data[f"{model_name}_pred"] = oof_preds
        oof_data[f"{model_name}_confidence"] = oof_confidence
        oof_data["fold_id"] = oof_fold_ids
        oof_df = pd.DataFrame(oof_data)

        valid_indices = np.where(~np.isnan(oof_preds))[0]

        return OOFPrediction(
            model_name=model_name,
            predictions=oof_df,
            fold_info=fold_info,
            coverage=coverage,
            original_indices=valid_indices,
        )
