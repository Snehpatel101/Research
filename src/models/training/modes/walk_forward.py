"""
Walk-forward training mode.

Implements walk-forward validation training that respects temporal ordering.
More realistic than k-fold for trading applications as it simulates how
models would be deployed in production.

Reference: Pardo (2008) "The Evaluation and Optimization of Trading Strategies"
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.core.container import TimeSeriesDataContainer

from src.core.constants import DEFAULT_BATCH_SIZE, DEFAULT_MAX_EPOCHS
from src.core.contracts import get_model_contract
from src.core.types import DataRank
from src.models.base import PredictionResult
from src.models.registry import ModelRegistry
from src.validation.cv.fold_scaling import FoldAwareScaler, get_scaling_method_for_model
from src.validation.cv.walk_forward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardResult,
    WindowMetrics,
)

from ..config import ExperimentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class WalkForwardTrainerConfig:
    """
    Configuration for walk-forward training.

    Attributes:
        n_windows: Number of walk-forward windows
        window_type: "expanding" or "rolling"
        min_train_pct: Minimum training data percentage (first window)
        test_pct: Percentage of data per test window
        gap_bars: Gap between train and test (label leakage prevention)
        embargo_bars: Post-test embargo (serial correlation prevention)
    """

    n_windows: int = 5
    window_type: str = "expanding"
    min_train_pct: float = 0.4
    test_pct: float = 0.1
    gap_bars: int = 0
    embargo_bars: int = 0

    def to_walk_forward_config(self) -> WalkForwardConfig:
        """Convert to WalkForwardConfig for the evaluator."""
        return WalkForwardConfig(
            n_windows=self.n_windows,
            window_type=self.window_type,
            min_train_pct=self.min_train_pct,
            test_pct=self.test_pct,
            gap_bars=self.gap_bars,
            embargo_bars=self.embargo_bars,
        )


@dataclass
class WalkForwardTrainingResult:
    """
    Results from walk-forward training.

    Attributes:
        model_name: Name of the trained model
        horizon: Label horizon used
        window_results: List of WalkForwardResult per model
        aggregated_metrics: Aggregated metrics across all windows
        predictions_df: DataFrame with all OOF predictions
        model_paths: Paths to saved models (one per window if save_per_window)
        total_time: Total training time in seconds
    """

    model_name: str
    horizon: int
    window_results: list[WalkForwardResult] = field(default_factory=list)
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_paths: list[Path] = field(default_factory=list)
    total_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "aggregated_metrics": self.aggregated_metrics,
            "n_windows": len(self.window_results),
            "total_time": self.total_time,
            "model_paths": [str(p) for p in self.model_paths],
        }


# =============================================================================
# METRICS
# =============================================================================


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    # sklearn accepts 0, 1, or "warn" but type stubs are outdated
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),  # pyright: ignore[reportArgumentType]
        "precision": float(
            precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )  # pyright: ignore[reportArgumentType]
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )  # pyright: ignore[reportArgumentType]
        ),
    }


# =============================================================================
# WALK-FORWARD TRAINER
# =============================================================================


class WalkForwardTrainer:
    """
    Walk-forward validation trainer.

    Trains models using expanding or rolling window approach. This is more
    realistic than k-fold for trading as it simulates how models would be
    deployed in production - always training on past data and testing on
    future data.

    Walk-forward windows:
        Expanding: |---Train---|--Test--|  |----Train-----|--Test--|
        Rolling:   |---Train---|--Test--|    |---Train---|--Test--|

    Example:
        >>> config = ExperimentConfig(symbol="MES", horizons=[20], models=["xgboost"])
        >>> trainer = WalkForwardTrainer(config)
        >>> results = trainer.run(container)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        wf_config: WalkForwardTrainerConfig | None = None,
    ) -> None:
        """
        Initialize walk-forward trainer.

        Args:
            config: Experiment configuration
            wf_config: Walk-forward specific configuration (optional)
        """
        self.config = config
        self.wf_config = wf_config or WalkForwardTrainerConfig()
        self.output_dir = config.output_dir / "walk_forward"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized WalkForwardTrainer")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Windows: {self.wf_config.n_windows}")
        logger.info(f"  Window type: {self.wf_config.window_type}")

    def run(
        self,
        container: TimeSeriesDataContainer | None = None,
        save_models: bool = True,
        save_predictions: bool = True,
    ) -> dict[str, Any]:
        """
        Run walk-forward training.

        Args:
            container: TimeSeriesDataContainer with data. If None, loads from config.
            save_models: Whether to save trained models for each window
            save_predictions: Whether to save prediction DataFrames

        Returns:
            Dict with keys:
                - model_results: Dict[model_name, WalkForwardTrainingResult]
                - summary: Aggregated summary metrics
                - output_path: Path to output directory
        """
        start_time = time.time()

        # Load container if not provided
        if container is None:
            from src.core.container import TimeSeriesDataContainer

            container = TimeSeriesDataContainer.from_parquet_dir(
                path=self.config.data_dir,
                horizon=self.config.horizons[0],  # Use first horizon
            )

        logger.info("=" * 60)
        logger.info("WALK-FORWARD TRAINING")
        logger.info("=" * 60)
        logger.info(f"Container: {container}")

        # Get model names - handle both string and ModelConfig types
        model_names = [m if isinstance(m, str) else m.name for m in self.config.models]
        logger.info(f"Models: {model_names}")

        # Build walk-forward config
        wf_eval_config = self.wf_config.to_walk_forward_config()

        # Run walk-forward for each model
        all_results: dict[str, WalkForwardTrainingResult] = {}

        for model_name in model_names:
            logger.info("-" * 60)
            logger.info(f"Running walk-forward for: {model_name}")
            logger.info("-" * 60)

            try:
                result = self._run_single_model(
                    container=container,
                    model_name=model_name,
                    wf_config=wf_eval_config,
                    save_models=save_models,
                )
                all_results[model_name] = result

                if save_predictions and not result.predictions_df.empty:
                    pred_path = self.output_dir / f"wf_preds_{model_name}.parquet"
                    result.predictions_df.to_parquet(pred_path, index=False)
                    logger.info(f"  Saved predictions to: {pred_path}")

            except Exception as e:
                logger.error(f"Failed walk-forward for {model_name}: {e}")
                raise

        # Build summary
        summary = self._build_summary(all_results)

        total_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info("WALK-FORWARD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")

        # Log summary
        self._log_summary(summary)

        return {
            "model_results": all_results,
            "summary": summary,
            "output_path": str(self.output_dir),
            "total_time": total_time,
        }

    def _run_single_model(
        self,
        container: TimeSeriesDataContainer,
        model_name: str,
        wf_config: WalkForwardConfig,
        save_models: bool = True,
    ) -> WalkForwardTrainingResult:
        """
        Run walk-forward for a single model.

        Args:
            container: Data container
            model_name: Name of model to train
            wf_config: Walk-forward configuration
            save_models: Whether to save models

        Returns:
            WalkForwardTrainingResult
        """
        model_start = time.time()

        # Get training data (as DataFrames since return_df=True)
        X_result, y_result, weights_result = container.get_sklearn_arrays("train", return_df=True)
        # Cast to pandas types since we know return_df=True returns DataFrames/Series
        X = cast(pd.DataFrame, X_result)
        y = cast(pd.Series, y_result)
        weights = cast(pd.Series, weights_result)

        # Get label end times for purging
        label_end_times = container.get_label_end_times("train")

        n_samples = len(X)
        n_classes = len(np.unique(y))

        # Initialize prediction storage
        all_preds = np.full(n_samples, np.nan)
        all_probs = np.full((n_samples, n_classes), np.nan)
        all_confidence = np.full(n_samples, np.nan)

        # Create evaluator
        evaluator = WalkForwardEvaluator(wf_config)
        window_metrics: list[WindowMetrics] = []
        model_paths: list[Path] = []

        # Get scaling method for this model
        scaling_method = get_scaling_method_for_model(model_name)

        logger.info(f"  Running {wf_config.n_windows} windows ({wf_config.window_type})")

        # Pre-extract numpy values to avoid DataFrame.iloc copies (which duplicate
        # the full DataFrame structure). For TCN with 13,620 flattened columns,
        # this avoids ~80GB of unnecessary DataFrame overhead per window.
        X_np = X.values.astype(np.float32, copy=False)

        for window_idx, (train_idx, test_idx) in enumerate(
            evaluator.split(X, y, label_end_times=label_end_times)
        ):
            window_start = time.time()

            logger.debug(
                f"    Window {window_idx + 1}: train={len(train_idx)}, test={len(test_idx)}"
            )

            # Extract window data as numpy arrays (avoids DataFrame copy overhead)
            X_train_raw = X_np[train_idx]
            X_test_raw = X_np[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Fold-aware scaling
            scaler = FoldAwareScaler(method=scaling_method)
            scaling_result = scaler.fit_transform_fold(X_train_raw, X_test_raw)
            X_train_scaled = scaling_result.X_train_scaled
            X_test_scaled = scaling_result.X_val_scaled

            # Free raw arrays immediately — they are no longer needed after scaling.
            # For TCN (13,620 cols), this frees ~47GB before model training starts.
            del X_train_raw, X_test_raw, scaling_result

            # Reshape/sequence data for sequential models (LSTM, GRU, etc.)
            # Sequential models need 3D: (n_samples, seq_len, n_features)
            contract = get_model_contract(model_name)
            if contract.input_rank == DataRank.SEQUENCE_3D:
                seq_len = contract.sequence_length
                n_flat = X_train_scaled.shape[1]
                if n_flat % seq_len == 0:
                    # Data was pre-sequenced then flattened — reshape back
                    n_features = n_flat // seq_len
                    X_train_scaled = X_train_scaled.reshape(-1, seq_len, n_features)
                    X_test_scaled = X_test_scaled.reshape(-1, seq_len, n_features)
                    logger.debug(
                        f"    Reshaped to 3D: ({X_train_scaled.shape[0]}, {seq_len}, {n_features})"
                    )
                else:
                    # Data is raw 2D features — create sequences via sliding window
                    n_features = n_flat
                    X_train_scaled = self._create_sequences(X_train_scaled, seq_len)
                    X_test_scaled = self._create_sequences(X_test_scaled, seq_len)
                    # Trim labels/weights to match shortened arrays
                    n_train_new = X_train_scaled.shape[0]
                    n_test_new = X_test_scaled.shape[0]
                    y_train = y_train.iloc[-n_train_new:]
                    y_test = y_test.iloc[-n_test_new:]
                    # Also update train/test indices for proper metric alignment
                    train_idx = train_idx[-n_train_new:]
                    test_idx = test_idx[-n_test_new:]
                    if weights is not None:
                        w_train = None  # Will recompute below
                    logger.debug(
                        f"    Created sequences: ({X_train_scaled.shape[0]}, {seq_len}, {n_features})"
                    )

            elif contract.input_rank == DataRank.MULTI_TF_4D:
                # 4D multi-stream models (PatchTST, iTransformer): reshape from flattened 2D
                nd_shape = container.metadata.get("original_nd_shape")
                if nd_shape is not None:
                    # Reconstruct from stored original shape: (n_timeframes, seq_len, n_features)
                    X_train_scaled = X_train_scaled.reshape(-1, *nd_shape)
                    X_test_scaled = X_test_scaled.reshape(-1, *nd_shape)
                    logger.debug(f"    Reshaped to 4D: {X_train_scaled.shape}")
                else:
                    # Fallback: try to infer shape from contract
                    seq_len = contract.sequence_length
                    n_tf = container.metadata.get(
                        "n_timeframes", len(getattr(contract, "mtf_timeframes", ("5min",)))
                    )
                    n_flat = X_train_scaled.shape[1]
                    if n_flat % (n_tf * seq_len) == 0:
                        n_features = n_flat // (n_tf * seq_len)
                        X_train_scaled = X_train_scaled.reshape(-1, n_tf, seq_len, n_features)
                        X_test_scaled = X_test_scaled.reshape(-1, n_tf, seq_len, n_features)
                        logger.debug(f"    Inferred 4D reshape: {X_train_scaled.shape}")
                    else:
                        raise ValueError(
                            f"Cannot reconstruct 4D shape for {model_name}: "
                            f"n_flat={n_flat}, n_tf={n_tf}, seq_len={seq_len}"
                        )
            # Handle sample weights
            w_train = None
            if weights is not None:
                w_train = weights.iloc[train_idx].values

            # Create and train model with training config
            _model_config = {}
            if hasattr(self, "_pipeline_config") and self._pipeline_config is not None:
                _model_config["max_epochs"] = getattr(
                    self._pipeline_config, "max_epochs", DEFAULT_MAX_EPOCHS
                )
                _model_config["batch_size"] = getattr(
                    self._pipeline_config, "batch_size", DEFAULT_BATCH_SIZE
                )
                _model_config["early_stopping_patience"] = getattr(
                    self._pipeline_config, "early_stopping_patience", 10
                )
            model = ModelRegistry.create(model_name, config=_model_config)
            model.fit(
                X_train=X_train_scaled,
                y_train=y_train.values,
                X_val=X_test_scaled,
                y_val=y_test.values,
                sample_weights=w_train,
                config=_model_config,
            )

            # Free training arrays immediately — only test data needed for prediction.
            # For TCN (13,620 flattened cols), X_train_scaled alone is ~47GB.
            del X_train_scaled, y_train, w_train

            # Generate predictions
            prediction_output: PredictionResult = model.predict(X_test_scaled)

            # Store predictions
            all_preds[test_idx] = prediction_output.class_predictions
            if prediction_output.class_probabilities is not None:
                all_probs[test_idx, : prediction_output.class_probabilities.shape[1]] = (
                    prediction_output.class_probabilities
                )
            all_confidence[test_idx] = prediction_output.confidence

            # Compute metrics
            metrics = compute_classification_metrics(
                y_test.values, prediction_output.class_predictions
            )
            window_time = time.time() - window_start

            # Build window metrics
            has_datetime = isinstance(X.index, pd.DatetimeIndex)
            window_metric = WindowMetrics(
                window=window_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                train_start_idx=int(train_idx[0]),
                train_end_idx=int(train_idx[-1]),
                test_start_idx=int(test_idx[0]),
                test_end_idx=int(test_idx[-1]),
                train_start_time=X.index[train_idx[0]] if has_datetime else None,
                train_end_time=X.index[train_idx[-1]] if has_datetime else None,
                test_start_time=X.index[test_idx[0]] if has_datetime else None,
                test_end_time=X.index[test_idx[-1]] if has_datetime else None,
                accuracy=metrics["accuracy"],
                f1=metrics["f1"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                training_time=window_time,
            )
            window_metrics.append(window_metric)

            # Save model if requested
            if save_models:
                model_path = self.output_dir / f"{model_name}_window{window_idx}.pkl"
                model.save(model_path)
                model_paths.append(model_path)

            logger.info(
                f"    Window {window_idx + 1}: acc={metrics['accuracy']:.3f}, "
                f"f1={metrics['f1']:.3f}, time={window_time:.1f}s"
            )

            # Free memory between walk-forward windows to prevent OOM
            # on large datasets (1M+ rows with 3D/4D reshaping).
            # X_train_scaled, y_train, w_train already freed above after model.fit().
            # Move neural model to CPU before deletion (frees GPU VRAM)
            if hasattr(model, "cpu"):
                model.cpu()
            del model, X_test_scaled, y_test
            del prediction_output, scaler
            gc.collect()
            try:
                import torch

                if hasattr(torch, "_dynamo"):
                    torch._dynamo.reset()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Build predictions DataFrame
        predictions_df = pd.DataFrame(
            {
                "datetime": X.index if isinstance(X.index, pd.DatetimeIndex) else range(len(X)),
                f"{model_name}_pred": all_preds,
                f"{model_name}_confidence": all_confidence,
                "y_true": y.values,
            }
        )

        # Add probability columns
        for i in range(n_classes):
            predictions_df[f"{model_name}_prob_class{i}"] = all_probs[:, i]

        total_time = time.time() - model_start

        # Build WalkForwardResult for compatibility
        wf_result = WalkForwardResult(
            model_name=model_name,
            horizon=container.horizon,
            window_metrics=window_metrics,
            predictions=predictions_df,
            config=wf_config,
            total_time=total_time,
        )

        # Build aggregated metrics
        aggregated = {
            "mean_accuracy": float(np.mean([m.accuracy for m in window_metrics])),
            "std_accuracy": float(np.std([m.accuracy for m in window_metrics])),
            "mean_f1": float(np.mean([m.f1 for m in window_metrics])),
            "std_f1": float(np.std([m.f1 for m in window_metrics])),
            "mean_precision": float(np.mean([m.precision for m in window_metrics])),
            "mean_recall": float(np.mean([m.recall for m in window_metrics])),
        }

        return WalkForwardTrainingResult(
            model_name=model_name,
            horizon=container.horizon,
            window_results=[wf_result],
            aggregated_metrics=aggregated,
            predictions_df=predictions_df,
            model_paths=model_paths,
            total_time=total_time,
        )

    @staticmethod
    def _create_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
        """
        Create 3D sequences from 2D data using a sliding window.

        Converts (n_rows, n_features) -> (n_rows - seq_len + 1, seq_len, n_features).
        The label for each sequence corresponds to the LAST row in the window.

        Args:
            X: 2D array of shape (n_rows, n_features)
            seq_len: Sequence length (number of time steps per window)

        Returns:
            3D array of shape (n_sequences, seq_len, n_features)
        """
        if len(X) < seq_len:
            return np.empty((0, seq_len, X.shape[1]), dtype=X.dtype)

        windows = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
        # sliding_window_view gives (n_windows, n_features, seq_len)
        # transpose to (n_windows, seq_len, n_features)
        return windows.transpose(0, 2, 1).copy()

    def _build_summary(self, results: dict[str, WalkForwardTrainingResult]) -> dict[str, Any]:
        """Build summary from all model results."""
        models_summary: dict[str, dict[str, float]] = {}

        for model_name, result in results.items():
            models_summary[model_name] = {
                "mean_accuracy": result.aggregated_metrics.get("mean_accuracy", 0),
                "std_accuracy": result.aggregated_metrics.get("std_accuracy", 0),
                "mean_f1": result.aggregated_metrics.get("mean_f1", 0),
                "total_time": result.total_time,
            }

        summary: dict[str, Any] = {
            "n_models": len(results),
            "n_windows": self.wf_config.n_windows,
            "window_type": self.wf_config.window_type,
            "models": models_summary,
        }

        return summary

    def _log_summary(self, summary: dict[str, Any]) -> None:
        """Log summary to console."""
        logger.info("Walk-Forward Summary:")
        logger.info(f"  Windows: {summary['n_windows']} ({summary['window_type']})")

        for model_name, metrics in summary["models"].items():
            logger.info(
                f"  {model_name}: "
                f"acc={metrics['mean_accuracy']:.3f} (+/- {metrics['std_accuracy']:.3f}), "
                f"f1={metrics['mean_f1']:.3f}, "
                f"time={metrics['total_time']:.1f}s"
            )
