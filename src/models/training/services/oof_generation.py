"""Service for generating out-of-fold predictions."""

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
        self._oof_generator: OOFGenerator | None = None

    def _ensure_generator(self, request: OOFRequest) -> OOFGenerator:
        """
        Ensure OOF generator is initialized with appropriate CV strategy.

        Args:
            request: OOF generation request

        Returns:
            Configured OOFGenerator instance
        """
        if self._oof_generator is None:
            cv_config = PurgedKFoldConfig(
                n_splits=request.n_splits,
                purge_bars=request.purge_bars,
                embargo_bars=request.embargo_bars,
            )
            cv = PurgedKFold(cv_config)
            self._oof_generator = OOFGenerator(cv, cache_dir=self._cache_dir)
        return self._oof_generator

    def _ensure_cv(self, request: OOFRequest) -> PurgedKFold:
        """Get or create a PurgedKFold CV splitter."""
        self._ensure_generator(request)
        assert self._oof_generator is not None
        return self._oof_generator.cv

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

        Args:
            request: OOF generation request containing model name, horizon,
                    prepared data, and CV configuration

        Returns:
            OOFPrediction object or None if generation fails
        """
        try:
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

            # Ensure generator is initialized
            oof_generator = self._ensure_generator(request)

            oof_predictions = oof_generator.generate_oof_predictions(
                X=X_train_df,
                y=y_train,
                model_configs={model_name: {}},
                use_cache=True,
            )

            return oof_predictions.get(model_name)

        except Exception as e:
            logger.warning(f"Failed to generate OOF for {request.model_name}: {e}")
            return None

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
        n_classes = 3  # short, neutral, long

        logger.info(
            f"Generating 4D OOF predictions for {model_name} "
            f"(shape={X_4d.shape}, n_splits={request.n_splits})"
        )

        # Initialize OOF storage
        oof_probs = np.full((n_samples, n_classes), np.nan)
        oof_preds = np.full(n_samples, np.nan)
        oof_confidence = np.full(n_samples, np.nan)
        fold_info: list[dict[str, Any]] = []

        # Create a dummy 2D DataFrame for PurgedKFold.split() index generation
        # (PurgedKFold only needs the length and optionally label_end_times)
        X_dummy = pd.DataFrame({"dummy": np.zeros(n_samples)})
        y_series = pd.Series(y)

        cv = self._ensure_cv(request)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_dummy, y_series)):
            logger.debug(f"  Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")

            # Slice 4D arrays directly by sample index
            X_train_fold = X_4d[train_idx]
            X_val_fold = X_4d[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # Handle sample weights
            w_train = None
            if prepared.train_weights is not None:
                w_train = prepared.train_weights[train_idx]

            # TODO(perf): Cache fold models from training CV and reload here
            # instead of retraining from scratch. Requires saving each fold's
            # trained weights during _train_model_sequential and passing them
            # to this service. Would eliminate 5-6x overhead for 4D OOF generation.
            model = ModelRegistry.create(model_name, config={})

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

            fold_info.append(
                {
                    "fold": fold_idx,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "val_accuracy": training_metrics.val_accuracy,
                    "val_f1": training_metrics.val_f1,
                }
            )

        # Validate coverage
        coverage = float((~np.isnan(oof_preds)).mean())
        if coverage < 1.0:
            logger.warning(
                f"{model_name}: 4D OOF coverage {coverage:.2%}. "
                f"{int(np.isnan(oof_preds).sum())} samples missing predictions."
            )

        # Build result DataFrame
        oof_df = pd.DataFrame(
            {
                "datetime": range(n_samples),
                f"{model_name}_prob_short": oof_probs[:, 0],
                f"{model_name}_prob_neutral": oof_probs[:, 1],
                f"{model_name}_prob_long": oof_probs[:, 2],
                f"{model_name}_pred": oof_preds,
                f"{model_name}_confidence": oof_confidence,
            }
        )

        valid_indices = np.where(~np.isnan(oof_preds))[0]

        return OOFPrediction(
            model_name=model_name,
            predictions=oof_df,
            fold_info=fold_info,
            coverage=coverage,
            original_indices=valid_indices,
        )

    def set_cv_strategy(self, cv: PurgedKFold) -> None:
        """
        Set a custom CV strategy for OOF generation.

        Args:
            cv: PurgedKFold instance to use for cross-validation
        """
        self._oof_generator = OOFGenerator(cv, cache_dir=self._cache_dir)
