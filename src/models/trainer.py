"""
Trainer - Orchestrates model training workflow.

The Trainer class handles the complete training pipeline:
1. Load and prepare data from TimeSeriesDataContainer
2. Apply per-model feature selection (for tabular/classical models)
3. Apply model-specific preprocessing
4. Train model with early stopping
5. Evaluate on validation set
6. Save artifacts (model, metrics, predictions, feature selection)

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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .base import PredictionOutput, TrainingMetrics
from .calibration import CalibrationConfig, ProbabilityCalibrator
from .config import TrainerConfig, save_config_json
from .data_preparation import prepare_test_data, prepare_training_data
from .feature_selection import FeatureSelectionConfig, FeatureSelectionManager
from .metrics import compute_classification_metrics, compute_trading_metrics
from .registry import ModelRegistry

if TYPE_CHECKING:
    from src.phase1.stages.datasets.container import TimeSeriesDataContainer

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

        # Initialize feature selection manager based on model family
        self.feature_selector: FeatureSelectionManager | None = None
        self._setup_feature_selection()

        logger.info(
            f"Initialized Trainer: model={config.model_name}, "
            f"horizon={config.horizon}, run_id={self.run_id}, "
            f"feature_selection={self._is_feature_selection_enabled()}"
        )

    def _setup_feature_selection(self) -> None:
        """Initialize feature selection manager based on model family and config."""
        if not self.config.use_feature_selection:
            self.feature_selector = FeatureSelectionManager.disabled()
            return

        # Sequence models don't use external feature selection
        if self.model.requires_sequences:
            logger.info(
                f"Feature selection disabled for {self.config.model_name}: "
                "sequence models handle selection internally"
            )
            self.feature_selector = FeatureSelectionManager.disabled()
            return

        # Create feature selection config based on model family
        fs_config = FeatureSelectionConfig.from_model_family(
            model_family=self.model.model_family,
            override={
                "n_features": self.config.feature_selection_n_features,
                "method": self.config.feature_selection_method,
                "random_state": self.config.random_seed,
            },
        )

        # Override n_features if explicitly set to 0 (use family default)
        if self.config.feature_selection_n_features == 0:
            from .feature_selection.config import ModelFamilyDefaults

            defaults = ModelFamilyDefaults.get_defaults(self.model.model_family)
            fs_config.n_features = defaults.get("n_features", 50)

        self.feature_selector = FeatureSelectionManager(config=fs_config)

    def _is_feature_selection_enabled(self) -> bool:
        """Check if feature selection is enabled for this trainer."""
        return self.feature_selector is not None and self.feature_selector.is_enabled

    def _is_heterogeneous_ensemble(self) -> bool:
        """Check if this is a heterogeneous stacking ensemble requiring both 2D and 3D data."""
        if self.model.model_family != "ensemble":
            return False
        if not hasattr(self.model, "ensemble_type"):
            return False
        # Only stacking supports heterogeneous (voting/blending require same shape)
        if self.model.ensemble_type != "stacking":
            return False
        # Check if base models have mixed input requirements
        from .ensemble.validator import is_heterogeneous_ensemble

        base_models = self.config.model_config.get("base_model_names", [])
        return is_heterogeneous_ensemble(base_models)

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
        container: TimeSeriesDataContainer,
        skip_save: bool = False,
    ) -> dict[str, Any]:
        """
        Execute complete training pipeline.

        Workflow:
        1. Setup output directories
        2. Load and prepare data
        3. Run feature selection (for tabular/classical models)
        4. Apply model-specific preprocessing
        5. Train model
        6. Evaluate on validation set
        7. Save artifacts (including feature selection)

        Args:
            container: TimeSeriesDataContainer with train/val data
            skip_save: If True, skip saving artifacts (for testing)

        Returns:
            Dict with training results including:
            - run_id: Run identifier
            - training_metrics: TrainingMetrics from training
            - evaluation_metrics: Validation set metrics
            - output_path: Path to outputs
            - feature_selection: Feature selection results (if enabled)
        """
        start_time = time.time()

        # Setup
        self._setup_output_dir()
        self._save_config()

        # Load data - get DataFrames for feature selection
        logger.info("Loading data from container...")

        # For feature selection, we need the feature names
        feature_names = container.feature_columns

        # Get raw training data (as DataFrames for feature selection)
        X_train_df, y_train_series, w_train_series = container.get_sklearn_arrays(
            "train", return_df=True
        )
        X_val_df, y_val_series, _ = container.get_sklearn_arrays("val", return_df=True)

        # Extract label_end_times for overlapping label purging
        label_end_times = container.get_label_end_times("train")
        if label_end_times is not None:
            logger.info(
                "Label end times available for purging overlapping labels "
                "(prevents leakage in stacking/blending ensembles)"
            )

        # Run feature selection (for tabular/classical models only)
        feature_selection_result = None
        if self._is_feature_selection_enabled():
            feature_selection_result = self._run_feature_selection(
                X_train_df=X_train_df,
                y_train=y_train_series,
                w_train=w_train_series,
                label_end_times=label_end_times,
            )

            # Apply feature selection to training data
            X_train_df = self.feature_selector.apply_selection_df(X_train_df)
            X_val_df = self.feature_selector.apply_selection_df(X_val_df)

            logger.info(
                f"Applied feature selection: {feature_selection_result.n_features_selected} features "
                f"(from {feature_selection_result.n_features_original})"
            )

        # Convert to numpy arrays and apply model-specific preprocessing
        # Track sequence data for heterogeneous stacking ensembles
        X_train_seq = None
        X_val_seq = None

        if self._is_heterogeneous_ensemble():
            # Heterogeneous stacking: load BOTH tabular and sequence data
            logger.info(
                "Heterogeneous stacking detected: preparing both tabular and sequence data"
            )

            # Tabular data (already loaded as DataFrames)
            X_train = X_train_df.values
            y_train = y_train_series.values
            w_train = w_train_series.values
            X_val = X_val_df.values
            y_val = y_val_series.values

            # Sequence data for sequence-based base models
            X_train_seq, _, _, X_val_seq, _ = prepare_training_data(
                container,
                requires_sequences=True,
                sequence_length=self.config.sequence_length,
            )

            logger.info(
                f"Heterogeneous data prepared: "
                f"tabular={X_train.shape}, sequence={X_train_seq.shape}"
            )

        elif self.model.requires_sequences:
            # Pure sequence model
            X_train, y_train, w_train, X_val, y_val = prepare_training_data(
                container,
                requires_sequences=True,
                sequence_length=self.config.sequence_length,
            )
        else:
            # Pure tabular model
            X_train = X_train_df.values
            y_train = y_train_series.values
            w_train = w_train_series.values
            X_val = X_val_df.values
            y_val = y_val_series.values

        # Log data shapes
        logger.info(
            f"Data shapes: "
            f"X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_val={X_val.shape}, y_val={y_val.shape}"
        )

        # Set feature names on model (for interpretability)
        if hasattr(self.model, "set_feature_names") and not self.model.requires_sequences:
            if self._is_feature_selection_enabled() and self.feature_selector.is_fitted:
                self.model.set_feature_names(self.feature_selector.selected_features)
            else:
                self.model.set_feature_names(feature_names)

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

        # Add sequence data if heterogeneous stacking ensemble
        if self._is_heterogeneous_ensemble() and X_train_seq is not None:
            fit_kwargs["X_train_seq"] = X_train_seq
            fit_kwargs["X_val_seq"] = X_val_seq
            logger.info("Passing sequence data to heterogeneous stacking ensemble")

        training_metrics = self.model.fit(**fit_kwargs)

        # Evaluate
        logger.info("Evaluating on validation set...")
        # For heterogeneous stacking, pass both tabular and sequence data
        if self._is_heterogeneous_ensemble() and X_val_seq is not None:
            val_predictions = self.model.predict(X_val, X_seq=X_val_seq)
        else:
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

        # Add feature selection info to eval metrics
        if feature_selection_result is not None:
            eval_metrics["feature_selection"] = {
                "n_features_original": feature_selection_result.n_features_original,
                "n_features_selected": feature_selection_result.n_features_selected,
                "reduction_ratio": feature_selection_result.reduction_ratio,
                "method": feature_selection_result.selection_method,
            }

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
                training_metrics,
                eval_metrics,
                val_predictions,
                test_metrics=test_metrics,
                test_predictions=test_predictions,
            )
            self._save_model()
            self._save_feature_selection()
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
            "feature_selection": (
                self.feature_selector.get_feature_report()
                if self._is_feature_selection_enabled()
                else None
            ),
        }

        logger.info(
            f"Training complete: "
            f"val_f1={eval_metrics['macro_f1']:.4f}, "
            f"val_accuracy={eval_metrics['accuracy']:.4f}, "
            f"time={total_time:.1f}s"
        )

        return results

    def _run_feature_selection(
        self,
        X_train_df: pd.DataFrame,
        y_train: pd.Series,
        w_train: pd.Series | None,
        label_end_times: pd.Series | None,
    ):
        """
        Run feature selection on training data.

        Uses walk-forward selection to prevent lookahead bias.
        """
        from .feature_selection import PersistedFeatureSelection

        logger.info(
            f"Running feature selection: method={self.config.feature_selection_method}, "
            f"n_features={self.config.feature_selection_n_features}"
        )

        # Run walk-forward feature selection
        result = self.feature_selector.select_features(
            X=X_train_df,
            y=y_train,
            sample_weights=w_train,
            n_splits=self.config.feature_selection_cv_splits,
            purge_bars=self.config.horizon * 3,  # Purge based on horizon
            embargo_bars=1440,  # ~5 days at 5-min resolution
            label_end_times=label_end_times,
        )

        return result

    def _evaluate_test_set(
        self,
        container: TimeSeriesDataContainer,
    ) -> tuple[dict[str, Any] | None, PredictionOutput | None]:
        """
        Evaluate model on test set with warnings about one-shot evaluation.

        Applies the same feature selection used during training to test data.

        Args:
            container: TimeSeriesDataContainer with test split

        Returns:
            Tuple of (test_metrics, test_predictions)
        """
        logger.warning("=" * 70)
        logger.warning("TEST SET EVALUATION - ONE-SHOT GENERALIZATION ESTIMATE")
        logger.warning("=" * 70)
        logger.warning(
            "You are evaluating on the TEST SET. This is your final, "
            "one-shot generalization estimate."
        )
        logger.warning("DO NOT iterate on these results. If you do, you're overfitting to test.")
        logger.warning(
            "Iterate on VALIDATION metrics during development, "
            "then evaluate test ONCE when ready."
        )
        logger.warning("=" * 70)

        # Load test data
        # Track sequence data for heterogeneous stacking ensembles
        X_test_seq = None

        if self._is_heterogeneous_ensemble():
            # Heterogeneous stacking: load BOTH tabular and sequence test data
            logger.info(
                "Heterogeneous stacking: preparing both tabular and sequence test data"
            )

            # Tabular test data
            X_test_df, y_test_series, _ = container.get_sklearn_arrays("test", return_df=True)

            # Apply feature selection if enabled
            if self._is_feature_selection_enabled() and self.feature_selector.is_fitted:
                X_test_df = self.feature_selector.apply_selection_df(X_test_df)
                logger.debug(
                    f"Applied feature selection to test set: {X_test_df.shape[1]} features"
                )

            X_test = X_test_df.values
            y_test = y_test_series.values

            # Sequence test data for sequence-based base models
            X_test_seq, _, _ = prepare_test_data(
                container,
                requires_sequences=True,
                sequence_length=self.config.sequence_length,
            )

            logger.info(
                f"Heterogeneous test data prepared: "
                f"tabular={X_test.shape}, sequence={X_test_seq.shape}"
            )

        elif self.model.requires_sequences:
            # Sequence models: load sequences directly
            X_test, y_test, _ = prepare_test_data(
                container,
                requires_sequences=True,
                sequence_length=self.config.sequence_length,
            )
        else:
            # Tabular models: load DataFrame and apply feature selection
            X_test_df, y_test_series, _ = container.get_sklearn_arrays("test", return_df=True)

            # Apply feature selection if enabled
            if self._is_feature_selection_enabled() and self.feature_selector.is_fitted:
                X_test_df = self.feature_selector.apply_selection_df(X_test_df)
                logger.debug(
                    f"Applied feature selection to test set: {X_test_df.shape[1]} features"
                )

            X_test = X_test_df.values
            y_test = y_test_series.values

        logger.info(f"Test set size: {X_test.shape}")

        # Evaluate on test set
        # For heterogeneous stacking, pass both tabular and sequence data
        if self._is_heterogeneous_ensemble() and X_test_seq is not None:
            test_predictions = self.model.predict(X_test, X_seq=X_test_seq)
        else:
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
        logger.warning("TEST SET RESULTS (DO NOT ITERATE ON THESE)")
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
        eval_metrics: dict[str, Any],
        predictions: PredictionOutput,
        test_metrics: dict[str, Any] | None = None,
        test_predictions: PredictionOutput | None = None,
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

    def _save_feature_selection(self) -> None:
        """Save feature selection result with model artifacts."""
        if not self._is_feature_selection_enabled():
            return
        if not self.feature_selector.is_fitted:
            return

        fs_path = self.output_path / "config" / "feature_selection.json"
        self.feature_selector.save(fs_path)
        logger.info(f"Saved feature selection to {fs_path}")

    def _save_calibrator(self) -> None:
        """Save probability calibrator."""
        if self.calibrator is None:
            return
        calibrator_path = self.output_path / "checkpoints" / "calibrator.pkl"
        self.calibrator.save(calibrator_path)
        logger.info(f"Saved calibrator to {calibrator_path}")


# Re-export for backward compatibility
from .training_utils import evaluate_model, train_model

__all__ = [
    "Trainer",
    "TrainerConfig",
    "compute_classification_metrics",
    "compute_trading_metrics",
    "train_model",
    "evaluate_model",
]
