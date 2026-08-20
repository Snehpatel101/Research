"""Service for generating out-of-fold predictions."""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.adapters import PreparedData
from src.models.base import PredictionResult
from src.models.registry import ModelRegistry
from src.validation.cv import OOFGenerator, OOFPrediction, PurgedKFold, PurgedKFoldConfig
from src.validation.cv.oof_core import _get_prob_column_names
from src.validation.cv.oof_validation import OOFValidator

logger = logging.getLogger(__name__)


@dataclass
class OOFRequest:
    """Request to generate OOF predictions."""

    model_name: str
    horizon: int
    prepared_data: PreparedData
    n_splits: int = 5
    purge_bars: int = 10
    embargo_bars: int = 5
    fold_models: list[Any] | None = None  # Pre-trained fold models for 4D caching
    model_config: dict[str, Any] | None = None  # Model config (seq_length, hidden_size, etc.)
    n_classes: int = 3  # Number of output classes (2 for binary, 3 for short/neutral/long)


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

    def _flatten_to_2d(self, X: np.ndarray, data_rank: int) -> np.ndarray:
        """
        Flatten multi-dimensional data to 2D for OOF generation.

        Args:
            X: Input array of any dimensionality
            data_rank: Rank of the data (2, 3, or 4)

        Returns:
            2D array of shape (n_samples, n_features)
        """
        if data_rank > 2:
            return X.reshape(X.shape[0], -1)
        return X

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

        # Route 4D models to dedicated 4D OOF path
        if prepared.data_rank == 4:
            return self._generate_4d_oof(request)

        model_name = request.model_name

        # Flatten to 2D for OOF generation (handles 3D→2D)
        X_train_2d = self._flatten_to_2d(prepared.X_train, prepared.data_rank)

        X_train_df = pd.DataFrame(
            X_train_2d,
            columns=[f"f{i}" for i in range(X_train_2d.shape[1])],
        )
        y_train = pd.Series(prepared.y_train)

        # Drop intermediate references — X_train_df/y_train hold the data
        del X_train_2d, prepared
        gc.collect()

        oof_generator = self._create_generator(request)

        oof_predictions = oof_generator.generate_oof_predictions(
            X=X_train_df,
            y=y_train,
            model_configs={model_name: request.model_config or {}},
            use_cache=True,
        )

        # Post-training fold leakage verification (C4 audit fix)
        cv = self._create_cv(request)
        fold_indices = list(cv.split(X_train_df, y_train))
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

        return oof_predictions.get(model_name)

    def _generate_4d_oof(self, request: OOFRequest) -> OOFPrediction | None:
        """
        Generate OOF predictions for 4D (multi-stream) models.

        4D models (PatchTST, iTransformer, TFT) have PreparedData with
        X_train of shape (n_samples, n_timeframes, seq_len, n_features).
        Each sample is already a windowed multi-timeframe tensor, so we
        split by sample index for CV (no re-windowing needed).

        Args:
            request: OOF generation request

        Returns:
            OOFPrediction or None if generation fails
        """
        prepared = request.prepared_data
        model_name = request.model_name
        X_4d = prepared.X_train  # (n_samples, n_timeframes, seq_len, n_features)
        y = prepared.y_train

        n_samples = X_4d.shape[0]
        n_classes = request.n_classes

        logger.info(
            f"Generating 4D OOF predictions for {model_name} "
            f"(shape={X_4d.shape}, n_splits={request.n_splits})"
        )

        # Initialize OOF storage
        oof_probs = np.full((n_samples, n_classes), np.nan)
        oof_preds = np.full(n_samples, np.nan)
        oof_confidence = np.full(n_samples, np.nan)
        oof_fold_ids = np.full(n_samples, -1, dtype=int)
        fold_info: list[dict[str, Any]] = []

        # Create a dummy 2D DataFrame for PurgedKFold.split() index generation
        # (PurgedKFold only needs the length and optionally label_end_times)
        X_dummy = pd.DataFrame({"dummy": np.zeros(n_samples)})
        y_series = pd.Series(y)

        cv = self._create_cv(request)

        # Collect fold indices for post-training leakage verification
        all_fold_indices: list[tuple[np.ndarray, np.ndarray]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_dummy, y_series)):
            logger.debug(f"  Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")
            all_fold_indices.append((train_idx, val_idx))

            # Slice 4D arrays directly by sample index
            # Fancy indexing already returns a new array (copy); .copy() is redundant
            X_train_fold = X_4d[train_idx]
            X_val_fold = X_4d[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # Per-fold scaling: reshape 4D→2D, scale in-place, reshape back
            orig_train_shape = X_train_fold.shape
            orig_val_shape = X_val_fold.shape
            X_train_2d = X_train_fold.reshape(-1, orig_train_shape[-1])
            X_val_2d = X_val_fold.reshape(-1, orig_val_shape[-1])

            median = np.median(X_train_2d, axis=0).astype(np.float32)
            q75 = np.percentile(X_train_2d, 75, axis=0).astype(np.float32)
            q25 = np.percentile(X_train_2d, 25, axis=0).astype(np.float32)
            iqr = np.where((q75 - q25) > 1e-8, q75 - q25, np.float32(1.0))

            X_train_2d -= median
            X_train_2d /= iqr
            X_val_2d -= median
            X_val_2d /= iqr

            X_train_fold = X_train_2d.reshape(orig_train_shape)
            X_val_fold = X_val_2d.reshape(orig_val_shape)
            del X_train_2d, X_val_2d, median, q75, q25, iqr

            # Handle sample weights
            w_train = None
            if prepared.train_weights is not None:
                w_train = prepared.train_weights[train_idx]

            if request.fold_models and fold_idx < len(request.fold_models):
                # Use pre-trained fold model (cached from training CV)
                model = request.fold_models[fold_idx]
                training_metrics = None
                logger.debug(f"  Using cached fold model for fold {fold_idx + 1}")
            else:
                # Fallback: train from scratch (no cached models available)
                model = ModelRegistry.create(model_name, config=request.model_config or {})
                training_metrics = model.fit(
                    X_train=X_train_fold,
                    y_train=y_train_fold,
                    X_val=X_val_fold,
                    y_val=y_val_fold,
                    sample_weights=w_train,
                )

            # Generate predictions for validation fold
            prediction_output: PredictionResult = model.predict(X_val_fold)

            # Store OOF predictions at original indices
            oof_probs[val_idx] = prediction_output.class_probabilities
            oof_preds[val_idx] = prediction_output.class_predictions
            oof_confidence[val_idx] = prediction_output.confidence
            oof_fold_ids[val_idx] = fold_idx

            fold_info.append(
                {
                    "fold": fold_idx,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "val_accuracy": training_metrics.val_accuracy if training_metrics else None,
                    "val_f1": training_metrics.val_f1 if training_metrics else None,
                }
            )

            # Free fold model and data to prevent memory accumulation
            if not (request.fold_models and fold_idx < len(request.fold_models)):
                del model  # Only delete if we created it (not cached fold models)
            del X_train_fold, X_val_fold, y_train_fold, y_val_fold, prediction_output
            gc.collect()

        # Post-training fold leakage verification (C4 audit fix)
        leakage_result = OOFValidator.validate_fold_leakage(
            fold_indices=all_fold_indices,
            purge_bars=request.purge_bars,
            embargo_bars=request.embargo_bars,
        )
        if not leakage_result["passed"]:
            logger.error(
                "4D OOF fold leakage detected for %s: %d violations",
                model_name,
                leakage_result["n_violations"],
            )

        # Validate coverage
        coverage = float((~np.isnan(oof_preds)).mean())
        if coverage < 1.0:
            logger.warning(
                f"{model_name}: 4D OOF coverage {coverage:.2%}. "
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
