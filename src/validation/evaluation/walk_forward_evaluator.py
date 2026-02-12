"""
Walk-forward evaluator.

Implements walk-forward validation with expanding or rolling windows for
realistic simulation of model deployment over time.

Reference: Pardo (2008) "The Evaluation and Optimization of Trading Strategies"
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.validation.cv.walk_forward import (
    WalkForwardConfig,
    WindowMetrics,
)
from src.validation.cv.walk_forward import (
    WalkForwardEvaluator as WalkForwardSplitter,
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardEvaluatorConfig:
    """
    Configuration for walk-forward evaluation.

    Attributes:
        n_windows: Number of walk-forward windows
        window_type: "expanding" or "rolling"
        min_train_pct: Minimum training data percentage
        test_pct: Percentage per test window
        embargo_bars: Post-test embargo bars
        gap_bars: Gap between train and test
        output_dir: Directory for evaluation outputs
        save_predictions: Whether to save predictions to disk
    """

    n_windows: int = 5
    window_type: str = "expanding"
    min_train_pct: float = 0.4
    test_pct: float = 0.1
    embargo_bars: int = 60
    gap_bars: int = 60
    output_dir: str = "experiments/evaluation/walk_forward"
    save_predictions: bool = True


class WalkForwardEvaluator:
    """
    Walk-forward validation evaluator.

    Evaluates models using expanding or sliding window walk-forward validation
    that simulates realistic deployment scenarios where models only see past data.

    This evaluator:
    1. Splits data into sequential train/test windows
    2. Trains model on each window's training data
    3. Evaluates on out-of-sample future data
    4. Collects performance metrics over time

    Example:
        >>> config = {"n_windows": 5, "window_type": "expanding"}
        >>> evaluator = WalkForwardEvaluator(config)
        >>> result = evaluator.run(X, y, model)
        >>> print(f"Mean accuracy: {result['aggregate_metrics']['accuracy']:.3f}")
    """

    def __init__(self, config: dict[str, Any] | WalkForwardEvaluatorConfig):
        """
        Initialize walk-forward evaluator.

        Args:
            config: Evaluation configuration (dict or WalkForwardEvaluatorConfig)
        """
        if isinstance(config, WalkForwardEvaluatorConfig):
            self.config = config
        else:
            self.config = WalkForwardEvaluatorConfig(
                n_windows=config.get("n_windows", 5),
                window_type=config.get("window_type", "expanding"),
                min_train_pct=config.get("min_train_pct", 0.4),
                test_pct=config.get("test_pct", 0.1),
                embargo_bars=config.get("embargo_bars", 60),
                gap_bars=config.get("gap_bars", 60),
                output_dir=config.get("output_dir", "experiments/evaluation/walk_forward"),
                save_predictions=config.get("save_predictions", True),
            )

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create walk-forward splitter
        wf_config = WalkForwardConfig(
            n_windows=self.config.n_windows,
            window_type=self.config.window_type,
            min_train_pct=self.config.min_train_pct,
            test_pct=self.config.test_pct,
            embargo_bars=self.config.embargo_bars,
            gap_bars=self.config.gap_bars,
        )
        self.wf = WalkForwardSplitter(wf_config)

        logger.info("Initialized WalkForwardEvaluator")
        logger.info(f"WF config: {self.config.n_windows} windows, type={self.config.window_type}")
        logger.info(f"Output directory: {self.output_dir}")

    def run(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        model: Any = None,
        label_end_times: pd.Series | None = None,
        sample_weights: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Run walk-forward evaluation.

        Args:
            X: Feature DataFrame
            y: Target labels
            model: Model with fit() and predict() methods
            label_end_times: Optional label end times for purging
            sample_weights: Optional sample weights for training

        Returns:
            Dict with keys:
                - window_metrics: Per-window performance metrics
                - aggregate_metrics: Aggregated metrics across all windows
                - predictions: Time-series predictions DataFrame
                - total_time: Evaluation time in seconds

        Raises:
            ValueError: If required arguments are missing
        """
        logger.info("Starting walk-forward evaluation")
        start_time = time.time()

        # Validate inputs
        if X is None or y is None:
            raise ValueError("X and y are required for walk-forward evaluation")

        if model is None:
            raise ValueError("model is required for walk-forward evaluation")

        # Initialize predictions array
        n_samples = len(X)
        predictions = np.full(n_samples, np.nan)
        probabilities = np.full((n_samples, 3), np.nan)

        # Collect window results
        window_results = []

        for window_idx, (train_idx, test_idx) in enumerate(
            self.wf.split(X, y, label_end_times=label_end_times)
        ):
            window_start = time.time()
            logger.info(
                f"Evaluating window {window_idx + 1}/{self.config.n_windows}: "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Get sample weights if provided
            sw_train = sample_weights.iloc[train_idx] if sample_weights is not None else None

            try:
                # Fit model
                if sw_train is not None:
                    try:
                        model.fit(X_train, y_train, sample_weight=sw_train)
                    except TypeError:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Get probabilities if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                    if proba.shape[1] <= probabilities.shape[1]:
                        probabilities[test_idx, : proba.shape[1]] = proba
                else:
                    proba = None

                # Store predictions
                predictions[test_idx] = y_pred

                # Compute metrics
                accuracy = float(np.mean(y_test.values == y_pred))

                # Precision, recall, F1 for binary classification
                y_test_arr = y_test.values
                if len(np.unique(y_test_arr)) <= 2:
                    tp = np.sum((y_pred == 1) & (y_test_arr == 1))
                    fp = np.sum((y_pred == 1) & (y_test_arr == 0))
                    fn = np.sum((y_pred == 0) & (y_test_arr == 1))

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0
                        else 0.0
                    )
                else:
                    precision, recall, f1 = 0.0, 0.0, 0.0

                training_time = time.time() - window_start

                # Create window metrics
                window_metric = WindowMetrics(
                    window=window_idx,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    train_start_idx=int(train_idx[0]),
                    train_end_idx=int(train_idx[-1]),
                    test_start_idx=int(test_idx[0]),
                    test_end_idx=int(test_idx[-1]),
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    training_time=training_time,
                )

                # Add timestamps if available
                if isinstance(X.index, pd.DatetimeIndex):
                    window_metric.train_start_time = X.index[train_idx[0]]
                    window_metric.train_end_time = X.index[train_idx[-1]]
                    window_metric.test_start_time = X.index[test_idx[0]]
                    window_metric.test_end_time = X.index[test_idx[-1]]

                window_results.append(window_metric)

                logger.info(
                    f"Window {window_idx + 1}: accuracy={accuracy:.4f}, "
                    f"f1={f1:.4f}, time={training_time:.2f}s"
                )

            except Exception as e:
                logger.warning(f"Window {window_idx} failed: {e}")
                continue

        if not window_results:
            raise ValueError("All walk-forward windows failed - check model and data compatibility")

        # Aggregate metrics
        aggregate_metrics = {
            "accuracy": float(np.mean([w.accuracy for w in window_results])),
            "accuracy_std": float(np.std([w.accuracy for w in window_results])),
            "precision": float(np.mean([w.precision for w in window_results])),
            "recall": float(np.mean([w.recall for w in window_results])),
            "f1": float(np.mean([w.f1 for w in window_results])),
            "f1_std": float(np.std([w.f1 for w in window_results])),
        }

        # Compute performance degradation (if enough windows)
        if len(window_results) >= 3:
            first_half = window_results[: len(window_results) // 2]
            second_half = window_results[len(window_results) // 2 :]

            first_avg = np.mean([w.accuracy for w in first_half])
            second_avg = np.mean([w.accuracy for w in second_half])

            if first_avg > 0:
                degradation = (first_avg - second_avg) / first_avg
                aggregate_metrics["performance_degradation"] = float(degradation)

        # Create predictions DataFrame
        pred_df = pd.DataFrame(
            {
                "prediction": predictions,
                "actual": y.values,
                "prob_class_0": probabilities[:, 0],
                "prob_class_1": probabilities[:, 1] if probabilities.shape[1] > 1 else np.nan,
                "prob_class_2": probabilities[:, 2] if probabilities.shape[1] > 2 else np.nan,
            },
            index=X.index,
        )

        # Save predictions if configured
        if self.config.save_predictions:
            pred_path = self.output_dir / "wf_predictions.parquet"
            pred_df.to_parquet(pred_path)
            logger.info(f"Saved predictions to {pred_path}")

        total_time = time.time() - start_time
        logger.info(f"Walk-forward evaluation completed in {total_time:.2f}s")
        logger.info(
            f"Mean accuracy: {aggregate_metrics['accuracy']:.4f} ± "
            f"{aggregate_metrics['accuracy_std']:.4f}"
        )

        # Check for performance degradation
        if "performance_degradation" in aggregate_metrics:
            deg = aggregate_metrics["performance_degradation"]
            if deg > 0.1:
                logger.warning(f"Performance degradation detected: {deg:.1%}")

        return {
            "window_metrics": [
                {
                    "window": w.window,
                    "train_size": w.train_size,
                    "test_size": w.test_size,
                    "train_start_idx": w.train_start_idx,
                    "train_end_idx": w.train_end_idx,
                    "test_start_idx": w.test_start_idx,
                    "test_end_idx": w.test_end_idx,
                    "accuracy": w.accuracy,
                    "precision": w.precision,
                    "recall": w.recall,
                    "f1": w.f1,
                    "training_time": w.training_time,
                }
                for w in window_results
            ],
            "aggregate_metrics": aggregate_metrics,
            "predictions": pred_df,
            "total_time": total_time,
            "n_windows": len(window_results),
            "window_type": self.config.window_type,
        }

    def get_window_info(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get detailed information about each walk-forward window.

        Args:
            X: Feature DataFrame
            y: Labels (optional)
            label_end_times: Label end times (optional)

        Returns:
            List of dicts with window information
        """
        return self.wf.get_window_info(X, y, label_end_times)

    def validate_coverage(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Validate walk-forward coverage statistics.

        Args:
            X: Feature DataFrame
            y: Labels (optional)
            label_end_times: Label end times (optional)

        Returns:
            Dict with coverage statistics
        """
        return self.wf.validate_coverage(X, y, label_end_times)


__all__ = ["WalkForwardEvaluator", "WalkForwardEvaluatorConfig"]
