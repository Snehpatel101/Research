# src/models/training/services/ensemble_service.py
"""
Service for building ensembles from OOF predictions.

Delegates to EnsembleOrchestrator for the actual ensemble building,
providing a simpler interface for the UnifiedTrainingOrchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core import OOFResult
from src.data.adapters import AlignedOOFResult, OOFAligner
from src.validation.cv import OOFPrediction, StackingDataset

if TYPE_CHECKING:
    from src.core import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class EnsembleRequest:
    """Request to build an ensemble."""

    oof_predictions: dict[str, OOFPrediction]
    config: PipelineConfig
    df: pd.DataFrame | None = None  # For label extraction


@dataclass
class EnsembleServiceResult:
    """Result from ensemble building."""

    aligned_oof: AlignedOOFResult | None
    stacking_dataset: StackingDataset | None
    ensemble_metrics: dict[str, Any]
    meta_learner: Any | None = None
    training_time_seconds: float = 0.0


class EnsembleService:
    """
    Service for building ensembles from OOF predictions.

    Responsibilities:
    - Align OOF predictions from heterogeneous models
    - Build stacking datasets
    - Train meta-learners
    - Return structured results

    Does NOT:
    - Generate OOF predictions (handled by OOFGenerationService)
    - Train base models (handled by ModelTrainingService)
    """

    def __init__(self) -> None:
        """Initialize EnsembleService."""
        self._aligner = OOFAligner()

    def build_ensemble(
        self,
        request: EnsembleRequest,
    ) -> EnsembleServiceResult:
        """
        Build ensemble from OOF predictions.

        Args:
            request: EnsembleRequest with OOF predictions and config

        Returns:
            EnsembleServiceResult with aligned OOF and ensemble metrics
        """
        import time

        start_time = time.time()

        oof_predictions = request.oof_predictions
        config = request.config

        if not oof_predictions:
            logger.warning("No OOF predictions available for ensemble")
            return EnsembleServiceResult(
                aligned_oof=None,
                stacking_dataset=None,
                ensemble_metrics={},
            )

        if len(oof_predictions) < 2:
            logger.warning("Need at least 2 models for ensemble")
            return EnsembleServiceResult(
                aligned_oof=None,
                stacking_dataset=None,
                ensemble_metrics={},
            )

        logger.info(f"Building ensemble from {len(oof_predictions)} models...")

        # Convert OOFPrediction to OOFResult for alignment
        oof_results = self._convert_to_oof_results(oof_predictions)

        # Align OOF predictions
        try:
            aligned = self._aligner.align(oof_results, strategy="intersection")
        except ValueError as e:
            logger.error(f"Failed to align OOF predictions: {e}")
            return EnsembleServiceResult(
                aligned_oof=None,
                stacking_dataset=None,
                ensemble_metrics={"error": str(e)},
            )

        logger.info(f"Aligned {len(aligned.model_names)} models")
        logger.info(f"Valid samples: {aligned.n_common}")

        # Extract aligned labels
        y_aligned = self._extract_aligned_labels(oof_predictions, aligned)

        if y_aligned is None:
            logger.warning("Could not extract aligned labels for meta-learner")
            return EnsembleServiceResult(
                aligned_oof=aligned,
                stacking_dataset=None,
                ensemble_metrics={"n_common": aligned.n_common},
            )

        # Build stacking dataset
        stacking_features = aligned.stacking_features
        stacking_df = pd.DataFrame(
            stacking_features,
            columns=aligned.get_feature_names(),
        )
        stacking_df["y_true"] = y_aligned

        stacking_dataset = StackingDataset(
            data=stacking_df,
            model_names=aligned.model_names,
            horizon=config.horizons[0] if config.horizons else 20,
            metadata={
                "n_common": aligned.n_common,
                "coverage": aligned.coverage,
            },
        )

        logger.info(f"Stacking dataset: {stacking_dataset.n_samples} samples")

        # Train meta-learner
        meta_learner, ensemble_metrics = self._train_meta_learner(stacking_dataset, config)

        training_time = time.time() - start_time

        return EnsembleServiceResult(
            aligned_oof=aligned,
            stacking_dataset=stacking_dataset,
            ensemble_metrics=ensemble_metrics,
            meta_learner=meta_learner,
            training_time_seconds=training_time,
        )

    def _convert_to_oof_results(
        self,
        oof_predictions: dict[str, OOFPrediction],
    ) -> list[OOFResult]:
        """Convert OOFPrediction dict to list of OOFResult for OOFAligner."""
        oof_results: list[OOFResult] = []

        for model_name, oof_pred in oof_predictions.items():
            probs = oof_pred.get_probabilities()
            preds = oof_pred.get_class_predictions()

            n_samples = len(probs)
            indices = np.arange(n_samples)
            fold_ids = np.zeros(n_samples, dtype=int)

            oof_result = OOFResult(
                predictions=preds.astype(int),
                probabilities=probs,
                indices=indices,
                fold_ids=fold_ids,
                model_name=model_name,
                coverage=oof_pred.coverage,
            )
            oof_results.append(oof_result)

        return oof_results

    def _extract_aligned_labels(
        self,
        oof_predictions: dict[str, OOFPrediction],
        aligned: AlignedOOFResult,
    ) -> np.ndarray | None:
        """Extract aligned labels from OOF predictions."""
        for _key, oof_pred in oof_predictions.items():
            if "y_true" in oof_pred.predictions.columns:
                y_full = oof_pred.predictions["y_true"].values
                # Align to common indices
                if aligned.common_indices is not None:
                    valid_indices = aligned.common_indices[aligned.common_indices < len(y_full)]
                    return np.asarray(y_full[valid_indices])
        return None

    def _train_meta_learner(
        self,
        stacking_dataset: StackingDataset,
        config: PipelineConfig,
    ) -> tuple[Any, dict[str, Any]]:
        """Train meta-learner on stacking dataset."""
        import time

        try:
            from src.models import Trainer, TrainerConfig

            start = time.time()

            X_stack = stacking_dataset.get_features()
            y_stack = stacking_dataset.get_labels()

            # Split into train/val
            n_samples = len(X_stack)
            n_train = int(n_samples * 0.8)

            X_train = X_stack.iloc[:n_train]
            X_val = X_stack.iloc[n_train:]
            y_train = y_stack.iloc[:n_train]
            y_val = y_stack.iloc[n_train:]

            meta_config = TrainerConfig(
                model_name=config.meta_learner,
                horizon=stacking_dataset.horizon,
            )

            from src.core.container import TimeSeriesDataContainer

            # Build train and val DataFrames for container
            train_df = X_train.copy()
            train_df[f"label_h{stacking_dataset.horizon}"] = y_train.values
            train_df[f"sample_weight_h{stacking_dataset.horizon}"] = np.ones(len(y_train))

            val_df = X_val.copy()
            val_df[f"label_h{stacking_dataset.horizon}"] = y_val.values

            container = TimeSeriesDataContainer.from_dataframes(
                train_df=train_df,
                val_df=val_df,
                test_df=None,
                horizon=stacking_dataset.horizon,
                feature_columns=list(X_train.columns),
            )

            trainer = Trainer(meta_config)
            results = trainer.run(container)

            training_time = time.time() - start

            metrics = {
                "val_f1": results["evaluation_metrics"].get("val_f1", 0),
                "val_accuracy": results["evaluation_metrics"].get("val_accuracy", 0),
                "training_time": training_time,
            }

            logger.info(
                f"Meta-learner ({config.meta_learner}) trained: " f"val_f1={metrics['val_f1']:.4f}"
            )

            return trainer, metrics

        except Exception as e:
            logger.error(f"Failed to train meta-learner: {e}")
            return None, {"error": str(e)}


__all__ = ["EnsembleService", "EnsembleRequest", "EnsembleServiceResult"]
