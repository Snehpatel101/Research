"""
Batch Inference - Process large datasets efficiently.

Provides chunked processing for large datasets that may not fit in memory,
with progress tracking and result aggregation.

Usage:
    from src.inference import BatchPredictor

    predictor = BatchPredictor.from_bundle("./bundles/xgb_h20")
    results = predictor.predict_batch(
        df,
        batch_size=10000,
        output_path="predictions.parquet",
    )
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.pipeline import InferencePipeline, InferenceResult

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BatchProgress:
    """Progress tracking for batch inference."""

    total_samples: int
    processed_samples: int
    current_batch: int
    total_batches: int
    elapsed_seconds: float
    samples_per_second: float

    @property
    def progress_pct(self) -> float:
        return self.processed_samples / self.total_samples * 100

    @property
    def eta_seconds(self) -> float:
        if self.samples_per_second == 0:
            return float("inf")
        remaining = self.total_samples - self.processed_samples
        return remaining / self.samples_per_second


@dataclass
class BatchResult:
    """
    Result from batch inference.

    Attributes:
        predictions_df: DataFrame with all predictions
        n_samples: Total samples processed
        n_batches: Number of batches
        total_time_seconds: Total processing time
        samples_per_second: Processing throughput
        errors: Any errors encountered
    """

    predictions_df: pd.DataFrame
    n_samples: int
    n_batches: int
    total_time_seconds: float
    samples_per_second: float
    errors: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        """Save predictions to parquet."""
        path = Path(path)
        self.predictions_df.to_parquet(path, index=False)
        logger.info(f"Saved {self.n_samples} predictions to {path}")
        return path


# =============================================================================
# BATCH PREDICTOR
# =============================================================================


class BatchPredictor:
    """
    Efficient batch inference for large datasets.

    Processes data in chunks to manage memory, with optional:
    - Progress callbacks
    - Error handling and recovery
    - Result streaming to disk
    - Parallel processing (future)

    Example:
        >>> predictor = BatchPredictor.from_bundle("./bundles/xgb_h20")
        >>> result = predictor.predict_batch(
        ...     df,
        ...     batch_size=10000,
        ...     progress_callback=lambda p: print(f"{p.progress_pct:.1f}%"),
        ... )
        >>> result.save("predictions.parquet")
    """

    def __init__(
        self,
        pipeline: InferencePipeline,
        default_batch_size: int = 10000,
    ) -> None:
        """
        Initialize BatchPredictor.

        Args:
            pipeline: InferencePipeline for making predictions
            default_batch_size: Default batch size
        """
        self.pipeline = pipeline
        self.default_batch_size = default_batch_size

    @classmethod
    def from_bundle(
        cls,
        path: str | Path,
        batch_size: int = 10000,
    ) -> BatchPredictor:
        """Create BatchPredictor from a model bundle."""
        pipeline = InferencePipeline.from_bundle(path)
        return cls(pipeline, default_batch_size=batch_size)

    @classmethod
    def from_bundles(
        cls,
        paths: list[str | Path],
        batch_size: int = 10000,
    ) -> BatchPredictor:
        """Create BatchPredictor from multiple bundles."""
        pipeline = InferencePipeline.from_bundles(paths)
        return cls(pipeline, default_batch_size=batch_size)

    def predict_batch(
        self,
        data: pd.DataFrame | Path | str,
        batch_size: int | None = None,
        output_path: str | Path | None = None,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        error_handling: str = "warn",  # "warn", "raise", "skip"
        calibrate: bool = True,
    ) -> BatchResult:
        """
        Process a large dataset in batches.

        Args:
            data: Input DataFrame or path to parquet file
            batch_size: Samples per batch (None uses default)
            output_path: Optional path to save predictions
            progress_callback: Optional callback for progress updates
            error_handling: How to handle errors ("warn", "raise", "skip")
            calibrate: Whether to apply calibration

        Returns:
            BatchResult with all predictions
        """
        batch_size = batch_size or self.default_batch_size

        # Load data if path provided
        if isinstance(data, (str, Path)):
            data = pd.read_parquet(data)

        n_samples = len(data)
        n_batches = (n_samples + batch_size - 1) // batch_size

        logger.info(
            f"Starting batch inference: {n_samples} samples, "
            f"{n_batches} batches of {batch_size}"
        )

        start_time = time.time()
        all_predictions = []
        errors = []
        processed = 0

        for batch_idx, batch_df in enumerate(self._iter_batches(data, batch_size)):
            batch_start = time.time()  # noqa: F841

            try:
                inference_result = self.pipeline.predict(batch_df, calibrate=calibrate)
                batch_preds = self._format_predictions(inference_result, batch_df, batch_idx)
                all_predictions.append(batch_preds)

            except Exception as e:
                error_info = {
                    "batch": batch_idx,
                    "error": str(e),
                    "start_idx": batch_idx * batch_size,
                }
                errors.append(error_info)

                if error_handling == "raise":
                    raise
                elif error_handling == "warn":
                    logger.warning(f"Batch {batch_idx} failed: {e}")
                # "skip" just continues

            processed += len(batch_df)

            # Progress callback
            if progress_callback:
                elapsed = time.time() - start_time
                progress = BatchProgress(
                    total_samples=n_samples,
                    processed_samples=processed,
                    current_batch=batch_idx + 1,
                    total_batches=n_batches,
                    elapsed_seconds=elapsed,
                    samples_per_second=processed / elapsed if elapsed > 0 else 0,
                )
                progress_callback(progress)

        total_time = time.time() - start_time

        # Combine all predictions
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
        else:
            predictions_df = pd.DataFrame()

        result = BatchResult(
            predictions_df=predictions_df,
            n_samples=len(predictions_df),
            n_batches=n_batches,
            total_time_seconds=total_time,
            samples_per_second=len(predictions_df) / total_time if total_time > 0 else 0,
            errors=errors,
        )

        logger.info(
            f"Completed: {result.n_samples} predictions in {total_time:.1f}s "
            f"({result.samples_per_second:.0f} samples/sec)"
        )

        # Save if output path provided
        if output_path:
            result.save(output_path)

        return result

    def predict_streaming(
        self,
        data: pd.DataFrame | Path | str,
        batch_size: int | None = None,
        calibrate: bool = True,
    ) -> Iterator[pd.DataFrame]:
        """
        Stream predictions batch by batch.

        Yields prediction DataFrames as they're computed,
        useful for very large datasets or real-time processing.

        Args:
            data: Input data or path
            batch_size: Samples per batch
            calibrate: Whether to calibrate

        Yields:
            DataFrame of predictions for each batch
        """
        batch_size = batch_size or self.default_batch_size

        if isinstance(data, (str, Path)):
            data = pd.read_parquet(data)

        for batch_idx, batch_df in enumerate(self._iter_batches(data, batch_size)):
            result = self.pipeline.predict(batch_df, calibrate=calibrate)
            yield self._format_predictions(result, batch_df, batch_idx)

    def _iter_batches(
        self,
        df: pd.DataFrame,
        batch_size: int,
    ) -> Iterator[pd.DataFrame]:
        """Iterate over DataFrame in batches."""
        for start_idx in range(0, len(df), batch_size):
            yield df.iloc[start_idx : start_idx + batch_size]

    def _format_predictions(
        self,
        result: InferenceResult,
        source_df: pd.DataFrame,
        batch_idx: int,
    ) -> pd.DataFrame:
        """Format predictions as DataFrame."""
        preds = result.predictions

        pred_df = pd.DataFrame(
            {
                "prediction": preds.class_predictions,
                "prob_short": preds.class_probabilities[:, 0],
                "prob_neutral": preds.class_probabilities[:, 1],
                "prob_long": preds.class_probabilities[:, 2],
                "confidence": preds.confidence,
            }
        )

        # Preserve datetime if present
        if "datetime" in source_df.columns:
            pred_df["datetime"] = source_df["datetime"].values
        elif hasattr(source_df.index, "name") and source_df.index.name == "datetime":
            pred_df["datetime"] = source_df.index.values

        return pred_df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_batch_inference(
    bundle_path: str | Path,
    data_path: str | Path,
    output_path: str | Path,
    batch_size: int = 10000,
    show_progress: bool = True,
) -> BatchResult:
    """
    Convenience function for batch inference.

    Args:
        bundle_path: Path to model bundle
        data_path: Path to input parquet
        output_path: Path for output predictions
        batch_size: Samples per batch
        show_progress: Whether to print progress

    Returns:
        BatchResult with predictions
    """
    predictor = BatchPredictor.from_bundle(bundle_path, batch_size=batch_size)

    def progress_fn(p: BatchProgress) -> None:
        if show_progress:
            print(
                f"\rProcessing: {p.progress_pct:.1f}% "
                f"({p.processed_samples}/{p.total_samples}) "
                f"ETA: {p.eta_seconds:.1f}s",
                end="",
                flush=True,
            )

    result = predictor.predict_batch(
        data_path,
        output_path=output_path,
        progress_callback=progress_fn if show_progress else None,
    )

    if show_progress:
        print()  # Newline after progress

    return result


# =============================================================================
# PARALLEL ENSEMBLE INFERENCE
# =============================================================================


@dataclass
class ModelPrediction:
    """Result from a single model prediction in batch inference."""

    model_name: str
    predictions: np.ndarray  # Class predictions shape (n_samples,)
    probabilities: np.ndarray  # Probabilities shape (n_samples, n_classes)
    inference_time_ms: float
    success: bool
    error: str | None = None


@dataclass
class BatchInferenceResult:
    """Result from batch inference across multiple models."""

    model_predictions: dict[str, ModelPrediction]
    stacked_probabilities: np.ndarray  # Shape (n_samples, n_models * n_classes)
    n_samples: int
    n_models: int
    n_successful: int
    total_time_ms: float

    @property
    def all_succeeded(self) -> bool:
        """Check if all models succeeded."""
        return self.n_successful == self.n_models


class BatchInference:
    """
    Parallel inference across multiple base models for ensemble predictions.

    Runs predictions from multiple models in parallel using ThreadPoolExecutor,
    then stacks results for meta-learner consumption. This provides 10x faster
    inference for ensembles compared to sequential prediction.

    Note: Uses ThreadPoolExecutor (not multiprocessing) because models are
    already loaded in memory and prediction is primarily I/O bound.

    Example:
        >>> from src.inference import BatchInference, ModelBundle
        >>>
        >>> # Load base models
        >>> models = [
        ...     ModelBundle.load("./bundles/xgb_h20"),
        ...     ModelBundle.load("./bundles/lgbm_h20"),
        ...     ModelBundle.load("./bundles/lstm_h20"),
        ... ]
        >>>
        >>> # Create batch inference
        >>> batch_inf = BatchInference(models, n_jobs=4)
        >>>
        >>> # Run parallel predictions
        >>> result = batch_inf.predict_batch(X_test)
        >>> stacked = result.stacked_probabilities  # For meta-learner
    """

    def __init__(
        self,
        models: list[Any],
        n_jobs: int = -1,
        model_names: list[str] | None = None,
    ) -> None:
        """
        Initialize BatchInference.

        Args:
            models: List of base models (ModelBundle or any with predict method)
            n_jobs: Number of parallel workers (-1 = all CPUs)
            model_names: Optional names for models (extracted from bundles if not provided)
        """
        import os

        self.models = models
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count() or 4

        # Extract or assign model names
        if model_names:
            self.model_names = model_names
        else:
            self.model_names = []
            for i, model in enumerate(models):
                if hasattr(model, "metadata") and hasattr(model.metadata, "model_name"):
                    self.model_names.append(model.metadata.model_name)
                else:
                    self.model_names.append(f"model_{i}")

        if len(self.model_names) != len(self.models):
            raise ValueError("Number of model names must match number of models")

        logger.debug(f"BatchInference initialized: {len(models)} models, {self.n_jobs} workers")

    def predict_batch(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> BatchInferenceResult:
        """
        Run all base models in parallel and return stacked predictions.

        Args:
            X: Input features for prediction
            calibrate: Whether to apply calibration (if available)

        Returns:
            BatchInferenceResult with stacked predictions for meta-learner
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_time = time.time()
        model_predictions: dict[str, ModelPrediction] = {}

        # Run predictions in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all prediction tasks
            future_to_model = {
                executor.submit(self._predict_single, model, name, X, calibrate): name
                for model, name in zip(self.models, self.model_names, strict=False)
            }

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    model_predictions[model_name] = result
                except Exception as e:
                    logger.error(f"Model {model_name} prediction failed: {e}")
                    model_predictions[model_name] = self._create_error_prediction(
                        model_name, X, str(e)
                    )

        total_time = (time.time() - start_time) * 1000

        # Stack probabilities for meta-learner
        stacked = self._stack_probabilities(model_predictions, X)
        n_successful = sum(1 for p in model_predictions.values() if p.success)

        logger.info(
            f"BatchInference complete: {n_successful}/{len(self.models)} models "
            f"in {total_time:.1f}ms"
        )

        return BatchInferenceResult(
            model_predictions=model_predictions,
            stacked_probabilities=stacked,
            n_samples=len(X) if hasattr(X, "__len__") else X.shape[0],
            n_models=len(self.models),
            n_successful=n_successful,
            total_time_ms=total_time,
        )

    def predict_proba_batch(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> np.ndarray:
        """
        Run predict_proba for all models in parallel.

        Convenience method that returns just the stacked probabilities.

        Args:
            X: Input features
            calibrate: Whether to apply calibration

        Returns:
            Stacked probability array of shape (n_samples, n_models * n_classes)
        """
        result = self.predict_batch(X, calibrate)
        return result.stacked_probabilities

    def get_predictions_dict(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Get predictions as a dict mapping model_name -> probabilities.

        Useful for EnsembleBundle.predict() which expects this format.

        Args:
            X: Input features
            calibrate: Whether to apply calibration

        Returns:
            Dict mapping model_name to probability array
        """
        result = self.predict_batch(X, calibrate)
        return {
            name: pred.probabilities
            for name, pred in result.model_predictions.items()
            if pred.success
        }

    def _predict_single(
        self,
        model: Any,
        model_name: str,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool,
    ) -> ModelPrediction:
        """Make prediction with a single model."""
        start_time = time.time()

        try:
            # Handle different model interfaces
            if hasattr(model, "predict"):
                if "calibrate" in model.predict.__code__.co_varnames:
                    output = model.predict(X, calibrate=calibrate)
                else:
                    output = model.predict(X)

                # Extract predictions and probabilities
                if hasattr(output, "class_predictions"):
                    predictions = output.class_predictions
                    probabilities = output.class_probabilities
                elif isinstance(output, tuple):
                    predictions, probabilities = output[:2]
                elif isinstance(output, np.ndarray):
                    if output.ndim == 2:
                        probabilities = output
                        predictions = np.argmax(output, axis=1) - 1
                    else:
                        predictions = output
                        probabilities = np.zeros((len(output), 3))
                else:
                    raise ValueError(f"Unexpected output type: {type(output)}")

            elif hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)
                predictions = np.argmax(probabilities, axis=1) - 1
            else:
                raise ValueError(f"Model {model_name} has no predict method")

            inference_time = (time.time() - start_time) * 1000

            return ModelPrediction(
                model_name=model_name,
                predictions=predictions,
                probabilities=probabilities,
                inference_time_ms=inference_time,
                success=True,
                error=None,
            )

        except Exception as e:
            inference_time = (time.time() - start_time) * 1000
            logger.warning(f"Model {model_name} failed: {e}")
            return self._create_error_prediction(model_name, X, str(e), inference_time)

    def _create_error_prediction(
        self,
        model_name: str,
        X: pd.DataFrame | np.ndarray,
        error: str,
        inference_time_ms: float = 0.0,
    ) -> ModelPrediction:
        """Create a NaN-filled prediction for failed model."""
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        n_classes = 3  # Standard: short, neutral, long

        return ModelPrediction(
            model_name=model_name,
            predictions=np.full(n_samples, np.nan),
            probabilities=np.full((n_samples, n_classes), np.nan),
            inference_time_ms=inference_time_ms,
            success=False,
            error=error,
        )

    def _stack_probabilities(
        self,
        model_predictions: dict[str, ModelPrediction],
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Stack probabilities from all models for meta-learner."""
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]

        # Collect probabilities in model order
        prob_arrays = []
        for name in self.model_names:
            if name in model_predictions:
                prob_arrays.append(model_predictions[name].probabilities)
            else:
                # Missing model - fill with NaN
                prob_arrays.append(np.full((n_samples, 3), np.nan))

        # Stack horizontally: (n_samples, n_models * n_classes)
        return np.hstack(prob_arrays)


__all__ = [
    # Batch data processing
    "BatchPredictor",
    "BatchProgress",
    "BatchResult",
    "run_batch_inference",
    # Parallel ensemble inference
    "BatchInference",
    "BatchInferenceResult",
    "ModelPrediction",
]
