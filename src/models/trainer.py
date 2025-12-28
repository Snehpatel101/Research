"""
Trainer - Orchestrates model training workflow.

The Trainer class handles the complete training pipeline:
1. Load and prepare data from TimeSeriesDataContainer
2. Apply model-specific preprocessing
3. Train model with early stopping
4. Evaluate on validation set
5. Save artifacts (model, metrics, predictions)

Example:
    >>> from src.models.trainer import Trainer
    >>> from src.models.config import TrainerConfig
    >>> from src.phase1.stages.datasets.container import TimeSeriesDataContainer
    ...
    >>> config = TrainerConfig(model_name="xgboost", horizon=20)
    >>> container = TimeSeriesDataContainer.from_parquet_dir(
    ...     "data/splits/scaled", horizon=20
    ... )
    ...
    >>> trainer = Trainer(config)
    >>> results = trainer.run(container)
    >>> print(results["evaluation_metrics"]["val_f1"])
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import PredictionOutput, TrainingMetrics
from .calibration import CalibrationConfig, ProbabilityCalibrator
from .config import TrainerConfig, save_config_json
from .data_preparation import prepare_test_data, prepare_training_data
from .metrics import compute_classification_metrics, compute_trading_metrics
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates model training and evaluation.

    Handles the complete training workflow including data preparation,
    model training, evaluation, and artifact saving.

    Attributes:
        config: TrainerConfig with training settings
        model: Instantiated model from registry
        run_id: Unique identifier for this training run
        output_path: Path to output directory

    Example:
        >>> config = TrainerConfig(model_name="xgboost", horizon=20)
        >>> trainer = Trainer(config)
        >>> results = trainer.run(container)
    """

    def __init__(self, config: TrainerConfig) -> None:
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.run_id = self._generate_run_id()
        self.output_path = config.output_dir / self.run_id

        # Create model from registry
        self.model = ModelRegistry.create(
            config.model_name,
            config=config.model_config,
        )

        logger.info(
            f"Initialized Trainer: model={config.model_name}, "
            f"horizon={config.horizon}, run_id={self.run_id}"
        )

    def _generate_run_id(self) -> str:
        """
        Generate unique run identifier with collision prevention.

        Format: {model}_{horizon}_{timestamp_with_ms}_{random_suffix}
        Example: xgboost_h20_20251228_143025_789456_a3f9

        Milliseconds + random suffix ensure uniqueness even for parallel runs.
        """
        import secrets
        # Include milliseconds (%f) for sub-second precision
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Add 4-character random suffix for collision prevention
        random_suffix = secrets.token_hex(2)  # 2 bytes = 4 hex chars
        return f"{self.config.model_name}_h{self.config.horizon}_{timestamp}_{random_suffix}"

    def _setup_output_dir(self) -> None:
        """Create output directory structure."""
        dirs = [
            self.output_path / "config",
            self.output_path / "checkpoints",
            self.output_path / "predictions",
            self.output_path / "metrics",
            self.output_path / "plots",
            self.output_path / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Created output directory: {self.output_path}")

    def run(
        self,
        container: "TimeSeriesDataContainer",
        skip_save: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute complete training pipeline.

        Workflow:
        1. Setup output directories
        2. Load and prepare data
        3. Apply model-specific preprocessing
        4. Train model
        5. Evaluate on validation set
        6. Save artifacts

        Args:
            container: TimeSeriesDataContainer with train/val data
            skip_save: If True, skip saving artifacts (for testing)

        Returns:
            Dict with training results including:
            - run_id: Run identifier
            - training_metrics: TrainingMetrics from training
            - evaluation_metrics: Validation set metrics
            - output_path: Path to outputs
        """
        start_time = time.time()

        # Setup
        self._setup_output_dir()
        self._save_config()

        # Load data
        logger.info("Loading data from container...")
        X_train, y_train, w_train, X_val, y_val = prepare_training_data(
            container,
            requires_sequences=self.model.requires_sequences,
            sequence_length=self.config.sequence_length,
        )

        # Extract label_end_times for overlapping label purging (used by ensemble models)
        label_end_times = container.get_label_end_times("train")
        if label_end_times is not None:
            logger.info(
                f"Label end times available for purging overlapping labels "
                f"(prevents leakage in stacking/blending ensembles)"
            )

        # Log data shapes
        logger.info(
            f"Data shapes: "
            f"X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_val={X_val.shape}, y_val={y_val.shape}"
        )

        # Train model (pass label_end_times for ensemble models with internal CV)
        logger.info(f"Training {self.config.model_name}...")
        fit_kwargs = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "sample_weights": w_train,
            "config": self.config.model_config,
        }

        # Add label_end_times if model supports it (ensemble models with internal CV)
        # Non-ensemble models ignore this parameter (not in their fit() signature)
        if self.model.model_family == "ensemble" and label_end_times is not None:
            fit_kwargs["label_end_times"] = label_end_times

        training_metrics = self.model.fit(**fit_kwargs)

        # Evaluate
        logger.info("Evaluating on validation set...")
        val_predictions = self.model.predict(X_val)
        eval_metrics = compute_classification_metrics(
            y_true=y_val,
            y_pred=val_predictions.class_predictions,
            y_proba=val_predictions.class_probabilities,
        )

        # Add trading metrics
        eval_metrics["trading"] = compute_trading_metrics(
            y_true=y_val,
            y_pred=val_predictions.class_predictions,
        )

        # Test set evaluation (one-shot generalization estimate)
        test_metrics = None
        test_predictions = None
        if self.config.evaluate_test_set:
            test_metrics, test_predictions = self._evaluate_test_set(container)

        # Probability calibration (leakage-safe: fits on held-out val set)
        self.calibrator = None
        if self.config.use_calibration:
            logger.info("Applying probability calibration...")
            cal_config = CalibrationConfig(method=self.config.calibration_method)
            self.calibrator = ProbabilityCalibrator(cal_config)
            calibration_metrics = self.calibrator.fit(
                y_true=y_val,
                probabilities=val_predictions.class_probabilities,
            )
            eval_metrics["calibration"] = calibration_metrics.to_dict()

        # Save artifacts
        if not skip_save:
            self._save_artifacts(
                training_metrics, eval_metrics, val_predictions,
                test_metrics=test_metrics, test_predictions=test_predictions
            )
            self._save_model()
            if self.calibrator is not None:
                self._save_calibrator()

        total_time = time.time() - start_time

        results = {
            "run_id": self.run_id,
            "model_name": self.config.model_name,
            "horizon": self.config.horizon,
            "training_metrics": training_metrics.to_dict(),
            "evaluation_metrics": eval_metrics,
            "test_metrics": test_metrics,
            "output_path": str(self.output_path),
            "total_time_seconds": total_time,
            "val_predictions": val_predictions.class_predictions,
            "val_true": y_val,
        }

        logger.info(
            f"Training complete: "
            f"val_f1={eval_metrics['macro_f1']:.4f}, "
            f"val_accuracy={eval_metrics['accuracy']:.4f}, "
            f"time={total_time:.1f}s"
        )

        return results

    def _evaluate_test_set(
        self,
        container: "TimeSeriesDataContainer",
    ) -> tuple[Optional[Dict[str, Any]], Optional[PredictionOutput]]:
        """
        Evaluate model on test set with warnings about one-shot evaluation.

        Args:
            container: TimeSeriesDataContainer with test split

        Returns:
            Tuple of (test_metrics, test_predictions)
        """
        logger.warning("=" * 70)
        logger.warning("⚠️  TEST SET EVALUATION - ONE-SHOT GENERALIZATION ESTIMATE")
        logger.warning("=" * 70)
        logger.warning(
            "You are evaluating on the TEST SET. This is your final, "
            "one-shot generalization estimate."
        )
        logger.warning(
            "DO NOT iterate on these results. If you do, you're overfitting to test."
        )
        logger.warning(
            "Iterate on VALIDATION metrics during development, "
            "then evaluate test ONCE when ready."
        )
        logger.warning("=" * 70)

        # Load test data
        X_test, y_test, _ = prepare_test_data(
            container,
            requires_sequences=self.model.requires_sequences,
            sequence_length=self.config.sequence_length,
        )
        logger.info(f"Test set size: {X_test.shape}")

        # Evaluate on test set
        test_predictions = self.model.predict(X_test)
        test_metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=test_predictions.class_predictions,
            y_proba=test_predictions.class_probabilities,
        )
        test_metrics["trading"] = compute_trading_metrics(
            y_true=y_test,
            y_pred=test_predictions.class_predictions,
        )

        logger.warning("=" * 70)
        logger.warning("⚠️  TEST SET RESULTS (DO NOT ITERATE ON THESE)")
        logger.warning("=" * 70)
        logger.warning(
            f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
            f"Test F1: {test_metrics['macro_f1']:.4f}"
        )
        logger.warning(
            "If test results are disappointing: DO NOT tune and re-evaluate. "
            "Move on to the next experiment."
        )
        logger.warning("=" * 70)

        return test_metrics, test_predictions

    def _save_config(self) -> None:
        """Save training configuration."""
        config_path = self.output_path / "config" / "training_config.json"
        save_config_json(self.config.to_dict(), config_path)

        model_config_path = self.output_path / "config" / "model_config.json"
        save_config_json(self.config.model_config, model_config_path)

    def _save_artifacts(
        self,
        training_metrics: TrainingMetrics,
        eval_metrics: Dict[str, Any],
        predictions: PredictionOutput,
        test_metrics: Optional[Dict[str, Any]] = None,
        test_predictions: Optional[PredictionOutput] = None,
    ) -> None:
        """Save training artifacts."""
        # Training metrics
        metrics_path = self.output_path / "metrics" / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(training_metrics.to_dict(), f, indent=2)

        # Evaluation metrics (validation)
        eval_path = self.output_path / "metrics" / "evaluation_metrics.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)

        # Validation predictions
        pred_path = self.output_path / "predictions" / "val_predictions.npz"
        np.savez(
            pred_path,
            class_predictions=predictions.class_predictions,
            class_probabilities=predictions.class_probabilities,
            confidence=predictions.confidence,
        )

        # Test set metrics and predictions (if evaluated)
        if test_metrics is not None:
            test_metrics_path = self.output_path / "metrics" / "test_metrics.json"
            with open(test_metrics_path, "w") as f:
                json.dump(test_metrics, f, indent=2)
            logger.info(f"Saved test metrics to {test_metrics_path}")

        if test_predictions is not None:
            test_pred_path = self.output_path / "predictions" / "test_predictions.npz"
            np.savez(
                test_pred_path,
                class_predictions=test_predictions.class_predictions,
                class_probabilities=test_predictions.class_probabilities,
                confidence=test_predictions.confidence,
            )
            logger.info(f"Saved test predictions to {test_pred_path}")

        # Feature importance (if available)
        importance = self.model.get_feature_importance()
        if importance:
            importance_path = self.output_path / "metrics" / "feature_importance.json"
            with open(importance_path, "w") as f:
                json.dump(importance, f, indent=2)

        logger.debug(f"Saved artifacts to {self.output_path}")

    def _save_model(self) -> None:
        """Save trained model."""
        model_path = self.output_path / "checkpoints" / "best_model"
        self.model.save(model_path)
        logger.info(f"Saved model to {model_path}")

    def _save_calibrator(self) -> None:
        """Save probability calibrator."""
        if self.calibrator is None:
            return
        calibrator_path = self.output_path / "checkpoints" / "calibrator.pkl"
        self.calibrator.save(calibrator_path)
        logger.info(f"Saved calibrator to {calibrator_path}")


# Re-export for backward compatibility
from .metrics import compute_classification_metrics, compute_trading_metrics
from .training_utils import evaluate_model, train_model

__all__ = [
    "Trainer",
    "TrainerConfig",
    "compute_classification_metrics",
    "compute_trading_metrics",
    "train_model",
    "evaluate_model",
]
