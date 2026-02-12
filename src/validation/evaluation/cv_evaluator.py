"""
Cross-validation evaluator.

Implements purged k-fold cross-validation for time series models with
proper handling of label leakage through purging and embargo.

Reference: López de Prado (2018) "Advances in Financial Machine Learning"
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

logger = logging.getLogger(__name__)


@dataclass
class CVFoldResult:
    """Results from a single CV fold."""

    fold: int
    train_size: int
    test_size: int
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    training_time: float = 0.0
    predictions: np.ndarray | None = None
    probabilities: np.ndarray | None = None


@dataclass
class CVEvaluatorConfig:
    """
    Configuration for CV evaluation.

    Attributes:
        n_splits: Number of CV folds
        purge_bars: Bars to purge before test set
        embargo_bars: Bars to embargo after test set
        min_train_size: Minimum training set fraction
        output_dir: Directory for evaluation outputs
        save_oof: Whether to save out-of-fold predictions
    """

    n_splits: int = 5
    purge_bars: int = 60
    embargo_bars: int = 1440
    min_train_size: float = 0.3
    output_dir: str = "experiments/evaluation/cv"
    save_oof: bool = True


class CVEvaluator:
    """
    Cross-validation evaluator using PurgedKFold.

    Evaluates trained models using time-series aware cross-validation
    that prevents label leakage through purging and embargo.

    This evaluator:
    1. Splits data using PurgedKFold
    2. Trains and evaluates model on each fold
    3. Collects out-of-fold predictions
    4. Aggregates metrics across folds

    Example:
        >>> config = {"n_splits": 5, "purge_bars": 60}
        >>> evaluator = CVEvaluator(config)
        >>> result = evaluator.run(X, y, model)
        >>> print(f"Mean accuracy: {result['aggregate_metrics']['accuracy']:.3f}")
    """

    def __init__(self, config: dict[str, Any] | CVEvaluatorConfig):
        """
        Initialize CV evaluator.

        Args:
            config: Evaluation configuration (dict or CVEvaluatorConfig)
        """
        if isinstance(config, CVEvaluatorConfig):
            self.config = config
        else:
            self.config = CVEvaluatorConfig(
                n_splits=config.get("n_splits", 5),
                purge_bars=config.get("purge_bars", 60),
                embargo_bars=config.get("embargo_bars", 1440),
                min_train_size=config.get("min_train_size", 0.3),
                output_dir=config.get("output_dir", "experiments/evaluation/cv"),
                save_oof=config.get("save_oof", True),
            )

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create PurgedKFold splitter
        kfold_config = PurgedKFoldConfig(
            n_splits=self.config.n_splits,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
            min_train_size=self.config.min_train_size,
        )
        self.cv = PurgedKFold(kfold_config)

        logger.info("Initialized CVEvaluator")
        logger.info(
            f"CV config: {self.config.n_splits} folds, purge={self.config.purge_bars}, embargo={self.config.embargo_bars}"
        )
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
        Run cross-validation evaluation.

        Args:
            X: Feature DataFrame
            y: Target labels
            model: Model with fit() and predict() methods
            label_end_times: Optional label end times for purging
            sample_weights: Optional sample weights for training

        Returns:
            Dict with keys:
                - fold_metrics: Per-fold performance metrics
                - aggregate_metrics: Aggregated CV metrics
                - oof_predictions: Out-of-fold predictions DataFrame
                - total_time: Evaluation time in seconds

        Raises:
            ValueError: If required arguments are missing
        """
        logger.info("Starting cross-validation evaluation")
        start_time = time.time()

        # Validate inputs
        if X is None or y is None:
            raise ValueError("X and y are required for CV evaluation")

        if model is None:
            raise ValueError("model is required for CV evaluation")

        # Initialize OOF arrays
        n_samples = len(X)
        oof_predictions = np.full(n_samples, np.nan)
        oof_probabilities = np.full((n_samples, 3), np.nan)  # 3-class: buy/hold/sell

        # Collect fold results
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            self.cv.split(X, y, label_end_times=label_end_times)
        ):
            fold_start = time.time()
            logger.info(
                f"Evaluating fold {fold_idx + 1}/{self.config.n_splits}: train={len(train_idx)}, test={len(test_idx)}"
            )

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Get sample weights if provided
            sw_train = sample_weights.iloc[train_idx] if sample_weights is not None else None

            try:
                # Fit model
                if sw_train is not None and hasattr(model, "fit"):
                    # Try to pass sample weights
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
                    # Store OOF probabilities
                    if proba.shape[1] <= oof_probabilities.shape[1]:
                        oof_probabilities[test_idx, : proba.shape[1]] = proba
                else:
                    proba = None

                # Store OOF predictions
                oof_predictions[test_idx] = y_pred

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

                training_time = time.time() - fold_start

                fold_result = CVFoldResult(
                    fold=fold_idx,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    training_time=training_time,
                    predictions=y_pred,
                    probabilities=proba,
                )
                fold_results.append(fold_result)

                logger.info(
                    f"Fold {fold_idx + 1}: accuracy={accuracy:.4f}, f1={f1:.4f}, time={training_time:.2f}s"
                )

            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")
                continue

        if not fold_results:
            raise ValueError("All CV folds failed - check model and data compatibility")

        # Aggregate metrics
        aggregate_metrics = {
            "accuracy": float(np.mean([f.accuracy for f in fold_results])),
            "accuracy_std": float(np.std([f.accuracy for f in fold_results])),
            "precision": float(np.mean([f.precision for f in fold_results])),
            "recall": float(np.mean([f.recall for f in fold_results])),
            "f1": float(np.mean([f.f1 for f in fold_results])),
            "f1_std": float(np.std([f.f1 for f in fold_results])),
        }

        # Compute overall OOF accuracy
        valid_mask = ~np.isnan(oof_predictions)
        if valid_mask.sum() > 0:
            oof_accuracy = float(np.mean(y.values[valid_mask] == oof_predictions[valid_mask]))
            aggregate_metrics["oof_accuracy"] = oof_accuracy

        # Create OOF DataFrame
        oof_df = pd.DataFrame(
            {
                "prediction": oof_predictions,
                "actual": y.values,
                "prob_class_0": oof_probabilities[:, 0],
                "prob_class_1": (
                    oof_probabilities[:, 1] if oof_probabilities.shape[1] > 1 else np.nan
                ),
                "prob_class_2": (
                    oof_probabilities[:, 2] if oof_probabilities.shape[1] > 2 else np.nan
                ),
            },
            index=X.index,
        )

        # Save OOF if configured
        if self.config.save_oof:
            oof_path = self.output_dir / "oof_predictions.parquet"
            oof_df.to_parquet(oof_path)
            logger.info(f"Saved OOF predictions to {oof_path}")

        total_time = time.time() - start_time
        logger.info(f"CV evaluation completed in {total_time:.2f}s")
        logger.info(
            f"Mean accuracy: {aggregate_metrics['accuracy']:.4f} ± {aggregate_metrics['accuracy_std']:.4f}"
        )

        return {
            "fold_metrics": [
                {
                    "fold": f.fold,
                    "train_size": f.train_size,
                    "test_size": f.test_size,
                    "accuracy": f.accuracy,
                    "precision": f.precision,
                    "recall": f.recall,
                    "f1": f.f1,
                    "training_time": f.training_time,
                }
                for f in fold_results
            ],
            "aggregate_metrics": aggregate_metrics,
            "oof_predictions": oof_df,
            "total_time": total_time,
            "n_folds": len(fold_results),
        }

    def get_fold_info(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Get detailed information about each CV fold.

        Args:
            X: Feature DataFrame

        Returns:
            List of dicts with fold information
        """
        return self.cv.get_fold_info(X)

    def validate_coverage(self, X: pd.DataFrame) -> dict[str, Any]:
        """
        Validate CV coverage statistics.

        Args:
            X: Feature DataFrame

        Returns:
            Dict with coverage statistics
        """
        return self.cv.validate_coverage(X)


__all__ = ["CVEvaluator", "CVEvaluatorConfig", "CVFoldResult"]
