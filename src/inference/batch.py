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
            batch_start = time.time()

            try:
                result = self.pipeline.predict(batch_df, calibrate=calibrate)
                batch_preds = self._format_predictions(result, batch_df, batch_idx)
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


__all__ = [
    "BatchPredictor",
    "BatchProgress",
    "BatchResult",
    "run_batch_inference",
]
