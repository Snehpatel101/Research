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

import logging
import secrets
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..base import PredictionOutput, TrainingMetrics
from ..calibration import CalibrationConfig, ProbabilityCalibrator
from ..config import TrainerConfig
from ..data_preparation import prepare_training_data
from ..feature_selection import FeatureSelectionManager
from ..metrics import compute_classification_metrics, compute_trading_metrics
from ..registry import ModelRegistry
from .artifacts import TrainerArtifactsMixin
from .evaluation import INVALID_LABEL_SENTINEL, TrainerEvaluationMixin, _validate_labels
from .features import TrainerFeaturesMixin

if TYPE_CHECKING:
    from src.phase1.stages.datasets.container import TimeSeriesDataContainer

logger = logging.getLogger(__name__)


class Trainer(TrainerFeaturesMixin, TrainerEvaluationMixin, TrainerArtifactsMixin):
    """
    Orchestrates model training and evaluation.

    Handles the complete training workflow including data preparation,
    model training, evaluation, and artifact saving.

    Inherits from:
    - TrainerFeaturesMixin: Feature selection and feature set resolution
    - TrainerEvaluationMixin: Test set evaluation
    - TrainerArtifactsMixin: Artifact saving (configs, metrics, models)

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

        # Feature set columns (set during run() for per-model filtering)
        self._feature_set_columns: list[str] | None = None

        # Calibrator (set during run() if calibration is enabled)
        self.calibrator: ProbabilityCalibrator | None = None

        # Validate feature_set against model recommendations (MOD-005)
        self._validate_feature_set()

        logger.info(
            f"Initialized Trainer: model={config.model_name}, "
            f"horizon={config.horizon}, run_id={self.run_id}, "
            f"feature_selection={self._is_feature_selection_enabled()}"
        )

    def _generate_run_id(self) -> str:
        """
        Generate unique run identifier with collision prevention.

        Format: {model}_{horizon}_{timestamp_with_ms}_{random_suffix}
        Example: xgboost_h20_20251228_143025_789456_a3f9

        Milliseconds + random suffix ensure uniqueness even for parallel runs.
        """
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
        from ..ensemble.validator import is_heterogeneous_ensemble

        base_models = self.config.model_config.get("base_model_names", [])
        return is_heterogeneous_ensemble(base_models)

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

        # LEAKAGE PREVENTION: Validate no invalid labels (-99) in training data
        # The container should filter these by default, but this is a defensive check
        _validate_labels(y_train_series, "training labels")
        _validate_labels(y_val_series, "validation labels")

        # Extract label_end_times for overlapping label purging
        label_end_times = container.get_label_end_times("train")
        if label_end_times is not None:
            logger.info(
                "Label end times available for purging overlapping labels "
                "(prevents leakage in stacking/blending ensembles)"
            )

        # Apply per-model feature set filtering (if specified)
        # This filters features BEFORE MDA-based feature selection
        feature_set_columns = self._resolve_feature_set_columns(X_train_df)
        self._feature_set_columns = feature_set_columns  # Store for test set evaluation
        if feature_set_columns is not None:
            X_train_df = self._apply_feature_set_filter(X_train_df, feature_set_columns)
            X_val_df = self._apply_feature_set_filter(X_val_df, feature_set_columns)
            # Update feature_names to reflect filtered set
            feature_names = list(X_train_df.columns)
            logger.info(
                f"Feature set filter applied: {len(feature_names)} features "
                f"(from original {len(container.feature_columns)})"
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
            if self.feature_selector is not None:
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
            # Use np.asarray for type-safe conversion (handles both DataFrame and ndarray)
            X_train = np.asarray(X_train_df)
            y_train = np.asarray(y_train_series)
            w_train = np.asarray(w_train_series)
            X_val = np.asarray(X_val_df)
            y_val = np.asarray(y_val_series)

            # Get feature columns for sequence models
            # Use the first sequence model's optimal feature set
            seq_feature_columns = None
            base_model_names = self.config.model_config.get("base_model_names", [])
            for model_name in base_model_names:
                model_info = ModelRegistry.get_model_info(model_name)
                if model_info and model_info.get("requires_sequences", False):
                    seq_feature_columns = self._get_sequence_model_feature_columns(
                        model_name, container
                    )
                    break  # Use first sequence model's feature set

            # Sequence data for sequence-based base models (with feature filtering)
            X_train_seq, y_train_seq, _, X_val_seq, y_val_seq = prepare_training_data(
                container,
                requires_sequences=True,
                sequence_length=self.config.sequence_length,
                feature_columns=seq_feature_columns,
            )

            # MOD-009 FIX: Alignment validation for heterogeneous ensemble data flow
            # Sequence data has fewer samples due to windowing (loses seq_len-1)
            tabular_train_samples = X_train.shape[0]
            seq_train_samples = X_train_seq.shape[0]
            expected_offset = tabular_train_samples - seq_train_samples

            logger.info(
                f"Heterogeneous data prepared: "
                f"tabular={X_train.shape}, sequence={X_train_seq.shape}"
            )
            logger.info(
                f"MOD-009: Data alignment check - "
                f"tabular_train={tabular_train_samples}, seq_train={seq_train_samples}, "
                f"offset={expected_offset} (expected: seq_len-1={self.config.sequence_length - 1})"
            )

            # Validate alignment consistency
            if expected_offset < 0:
                raise ValueError(
                    f"MOD-009: Invalid data alignment - sequence data ({seq_train_samples}) "
                    f"has more samples than tabular data ({tabular_train_samples}). "
                    f"This indicates a data preparation error."
                )

            # Validate labels are consistent between tabular and sequence views
            # After accounting for offset, labels should match
            y_train_trimmed = y_train[expected_offset:] if expected_offset > 0 else y_train
            if len(y_train_trimmed) != len(y_train_seq):
                logger.warning(
                    f"MOD-009: Label length mismatch after offset adjustment - "
                    f"y_train_trimmed={len(y_train_trimmed)}, y_train_seq={len(y_train_seq)}. "
                    f"Proceeding but results may be misaligned."
                )
            elif not np.array_equal(y_train_trimmed, y_train_seq):
                logger.warning(
                    f"MOD-009: Labels differ between tabular and sequence views after alignment. "
                    f"This may indicate different data sources or processing paths."
                )

        elif self.model.requires_sequences:
            # Pure sequence model - get model-specific feature columns
            seq_feature_columns = self._get_sequence_model_feature_columns(
                self.config.model_name, container
            )
            X_train, y_train, w_train, X_val, y_val = prepare_training_data(
                container,
                requires_sequences=True,
                sequence_length=self.config.sequence_length,
                feature_columns=seq_feature_columns,
            )
        else:
            # Pure tabular model
            # Use np.asarray for type-safe conversion (handles both DataFrame and ndarray)
            X_train = np.asarray(X_train_df)
            y_train = np.asarray(y_train_series)
            w_train = np.asarray(w_train_series)
            X_val = np.asarray(X_val_df)
            y_val = np.asarray(y_val_series)

        # Log data shapes
        logger.info(
            f"Data shapes: "
            f"X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_val={X_val.shape}, y_val={y_val.shape}"
        )

        # Set feature names on model (for interpretability)
        if hasattr(self.model, "set_feature_names") and not self.model.requires_sequences:
            if (
                self._is_feature_selection_enabled()
                and self.feature_selector is not None
                and self.feature_selector.is_fitted
            ):
                self.model.set_feature_names(self.feature_selector.selected_features)  # type: ignore[union-attr]
            else:
                self.model.set_feature_names(feature_names)  # type: ignore[union-attr]

        # Train model (pass label_end_times for ensemble models with internal CV)
        logger.info(f"Training {self.config.model_name}...")
        fit_kwargs: dict[str, Any] = {
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
            # Heterogeneous stacking models accept X_seq as keyword argument
            val_predictions = self.model.predict(X_val, X_seq=X_val_seq)  # type: ignore[call-arg]
        else:
            val_predictions = self.model.predict(X_val)

        # MOD-005 FIX: Align y_val with predictions (may be trimmed for heterogeneous ensembles)
        # Also trim X_val if passthrough is enabled since it will be concatenated with OOF preds
        y_val_aligned = y_val
        X_val_aligned = X_val
        if len(val_predictions.class_predictions) < len(y_val):
            offset = len(y_val) - len(val_predictions.class_predictions)
            y_val_aligned = y_val[offset:]
            X_val_aligned = X_val[offset:]
            logger.info(
                f"Aligned validation data: trimmed {offset} samples from start "
                f"(y_val: {len(y_val)} -> {len(y_val_aligned)}, "
                f"X_val: {X_val.shape[0]} -> {X_val_aligned.shape[0]}) "
                f"to match prediction count {len(val_predictions.class_predictions)}"
            )

        eval_metrics = compute_classification_metrics(
            y_true=y_val_aligned,
            y_pred=val_predictions.class_predictions,
            y_proba=val_predictions.class_probabilities,
        )

        # Add trading metrics
        eval_metrics["trading"] = compute_trading_metrics(
            y_true=y_val_aligned,
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
            # Cast calibration_method to Literal type for CalibrationConfig
            calibration_method = self.config.calibration_method
            if calibration_method not in ("isotonic", "sigmoid", "auto"):
                calibration_method = "auto"
            cal_config = CalibrationConfig(method=calibration_method)  # type: ignore[arg-type]
            self.calibrator = ProbabilityCalibrator(cal_config)
            calibration_metrics = self.calibrator.fit(
                y_true=y_val_aligned,
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
            "val_true": y_val_aligned,
            "feature_selection": (
                self.feature_selector.get_feature_report()
                if self._is_feature_selection_enabled() and self.feature_selector is not None
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
