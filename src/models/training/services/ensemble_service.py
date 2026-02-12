# src/models/training/services/ensemble_service.py
"""
Service for building ensembles from OOF predictions.

Delegates to EnsembleOrchestrator for the actual ensemble building,
providing a simpler interface for the UnifiedTrainingOrchestrator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core import OOFResult
from src.data.adapters import AlignedOOFResult, OOFAligner
from src.models.ensemble.diversity import DiversityAnalyzer, DiversityMetrics
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
    diversity_metrics: DiversityMetrics | None = None


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
        self._diversity_analyzer = DiversityAnalyzer(
            min_diversity_threshold=0.3,
            correlation_threshold=0.8,
            n_classes=3,
        )

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

        start_time = time.time()

        oof_predictions = request.oof_predictions
        config = request.config

        if not oof_predictions:
            logger.error("No OOF predictions available for ensemble")
            return EnsembleServiceResult(
                aligned_oof=None,
                stacking_dataset=None,
                ensemble_metrics={"error": "no_oof_predictions"},
            )

        if len(oof_predictions) < 2:
            logger.error(
                f"Need at least 2 models for ensemble, got {len(oof_predictions)}: "
                f"{list(oof_predictions.keys())}"
            )
            return EnsembleServiceResult(
                aligned_oof=None,
                stacking_dataset=None,
                ensemble_metrics={"error": "insufficient_models", "n_models": len(oof_predictions)},
            )

        # Validate OOF prediction quality
        for model_name, oof_pred in oof_predictions.items():
            probs = oof_pred.get_probabilities()
            if np.all(probs == 0):
                logger.error(f"OOF predictions for {model_name} are all zeros")
                return EnsembleServiceResult(
                    aligned_oof=None,
                    stacking_dataset=None,
                    ensemble_metrics={"error": f"all_zero_predictions:{model_name}"},
                )
            if np.any(np.isnan(probs)):
                nan_count = int(np.isnan(probs).sum())
                logger.warning(f"OOF predictions for {model_name} contain {nan_count} NaN values")

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

        # Perform diversity analysis if ensemble building is enabled
        diversity_metrics = None
        if config.build_ensemble:
            diversity_metrics = self._analyze_diversity(
                oof_predictions=oof_predictions,
                aligned=aligned,
                y_aligned=y_aligned,
            )

        if y_aligned is None:
            logger.warning("Could not extract aligned labels for meta-learner")
            return EnsembleServiceResult(
                aligned_oof=aligned,
                stacking_dataset=None,
                ensemble_metrics={"n_common": aligned.n_common},
                diversity_metrics=diversity_metrics,
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
            diversity_metrics=diversity_metrics,
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

            # Use original_indices for proper alignment (critical for
            # sequence models that produce fewer samples than tabular)
            if oof_pred.original_indices is not None:
                indices = oof_pred.original_indices
            else:
                indices = np.arange(n_samples)

            # TODO: Fold provenance not tracked in OOFPrediction — all assigned fold 0
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
        """Train meta-learner directly on stacking features.

        Uses the meta-learner's fit() method directly rather than routing
        through Trainer/TimeSeriesDataContainer, which is designed for
        OHLCV time-series data and incompatible with OOF stacking features.
        """
        try:
            from sklearn.metrics import accuracy_score, f1_score

            from src.models.ensemble import (
                CalibratedMetaLearner,
                MLPMetaLearner,
                RidgeMetaLearner,
                XGBoostMeta,
            )

            start = time.time()

            X_stack = stacking_dataset.get_features()
            y_stack = stacking_dataset.get_labels()

            # Time-based split into train/val (preserves temporal ordering)
            n_samples = len(X_stack)
            n_train = int(n_samples * 0.8)

            X_train = X_stack.iloc[:n_train].values
            X_val = X_stack.iloc[n_train:].values
            y_train = y_stack.iloc[:n_train].values
            y_val = y_stack.iloc[n_train:].values

            # Create meta-learner directly
            meta_learner_map: dict[str, type] = {
                "ridge_meta": RidgeMetaLearner,
                "mlp_meta": MLPMetaLearner,
                "xgboost_meta": XGBoostMeta,
                "calibrated_meta": CalibratedMetaLearner,
            }

            meta_learner_name = config.meta_learner
            if meta_learner_name not in meta_learner_map:
                raise ValueError(
                    f"Unknown meta_learner: {meta_learner_name}. "
                    f"Available: {list(meta_learner_map.keys())}"
                )

            meta_learner = meta_learner_map[meta_learner_name]()

            # Train meta-learner directly
            training_metrics = meta_learner.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            # Evaluate on validation set
            output = meta_learner.predict(X_val)
            val_accuracy = float(accuracy_score(y_val, output.class_predictions))
            val_f1 = float(
                f1_score(y_val, output.class_predictions, average="macro", zero_division=0)
            )

            training_time = time.time() - start

            metrics = {
                "val_f1": val_f1,
                "val_accuracy": val_accuracy,
                "train_loss": training_metrics.train_loss,
                "val_loss": training_metrics.val_loss,
                "training_time": training_time,
            }

            logger.info(f"Meta-learner ({meta_learner_name}) trained: val_f1={val_f1:.4f}")

            return meta_learner, metrics

        except Exception as e:
            logger.error(f"Failed to train meta-learner: {e}")
            return None, {"error": str(e)}

    def _analyze_diversity(
        self,
        oof_predictions: dict[str, OOFPrediction],
        aligned: AlignedOOFResult,
        y_aligned: np.ndarray | None,
    ) -> DiversityMetrics | None:
        """
        Analyze ensemble diversity and log warnings if diversity is low.

        Args:
            oof_predictions: OOF predictions from base models
            aligned: Aligned OOF result
            y_aligned: Aligned ground truth labels

        Returns:
            DiversityMetrics containing diversity analysis
        """
        try:
            # Extract class predictions for each model
            base_predictions: dict[str, np.ndarray] = {}
            base_probabilities: dict[str, np.ndarray] = {}

            for model_name in aligned.model_names:
                if model_name in oof_predictions:
                    oof_pred = oof_predictions[model_name]

                    # Get class predictions (aligned to common indices)
                    preds = oof_pred.get_class_predictions()
                    if aligned.common_indices is not None:
                        valid_indices = aligned.common_indices[aligned.common_indices < len(preds)]
                        preds = preds[valid_indices]
                    base_predictions[model_name] = preds

                    # Get probabilities if available
                    probs = oof_pred.get_probabilities()
                    if aligned.common_indices is not None:
                        valid_indices = aligned.common_indices[aligned.common_indices < len(probs)]
                        probs = probs[valid_indices]
                    base_probabilities[model_name] = probs

            # Run diversity analysis
            diversity_metrics = self._diversity_analyzer.analyze(
                base_predictions=base_predictions,
                base_probabilities=base_probabilities,
                y_true=y_aligned,
            )

            # Log diversity results
            logger.info("Diversity analysis complete:")
            logger.info(f"  - Diversity score: {diversity_metrics.diversity_score:.3f}")
            logger.info(f"  - Pairwise correlation: {diversity_metrics.pairwise_correlation:.3f}")
            logger.info(f"  - Disagreement rate: {diversity_metrics.disagreement:.3f}")
            logger.info(f"  - Q-statistic: {diversity_metrics.q_statistic:.3f}")

            # Warn if diversity is low
            if diversity_metrics.diversity_score < self._diversity_analyzer.min_diversity_threshold:
                logger.warning(
                    f"Low ensemble diversity detected: "
                    f"score={diversity_metrics.diversity_score:.3f} < "
                    f"threshold={self._diversity_analyzer.min_diversity_threshold:.3f}"
                )
                logger.warning("Consider using more diverse model families or architectures")

            # Log recommendations if any
            if diversity_metrics.recommendations:
                logger.warning("Diversity recommendations:")
                for rec in diversity_metrics.recommendations:
                    logger.warning(f"  - {rec}")

            return diversity_metrics

        except Exception as e:
            logger.error(f"Failed to analyze diversity: {e}")
            return None


__all__ = ["EnsembleService", "EnsembleRequest", "EnsembleServiceResult"]
