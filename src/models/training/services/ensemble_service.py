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
        target_horizon = config.horizons[0] if config.horizons else None
        y_aligned = self._extract_aligned_labels(
            oof_predictions, aligned, df=request.df, target_horizon=target_horizon
        )

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

        # Safety check: y_aligned must match stacking feature rows
        if len(y_aligned) != len(stacking_df):
            logger.warning(
                f"Label/feature length mismatch: y_aligned={len(y_aligned)}, "
                f"stacking_features={len(stacking_df)}. Truncating to minimum."
            )
            min_len = min(len(y_aligned), len(stacking_df))
            stacking_df = stacking_df.iloc[:min_len].copy()
            y_aligned = y_aligned[:min_len]

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

            n_total = len(probs)

            # Use original_indices for proper alignment (critical for
            # sequence models that produce fewer samples than tabular)
            if oof_pred.original_indices is not None:
                indices = oof_pred.original_indices
                # Filter to only valid rows — the OOF DataFrame contains ALL
                # samples (including NaN for sequence models), but original_indices
                # marks which rows have actual predictions.
                probs = probs[indices]
                preds = preds[indices]
            else:
                indices = np.arange(n_total)

            n_samples = len(indices)

            # Extract fold provenance from OOF DataFrame if available
            if "fold_id" in oof_pred.predictions.columns:
                fold_ids_full = oof_pred.predictions["fold_id"].values
                if oof_pred.original_indices is not None:
                    fold_ids = fold_ids_full[oof_pred.original_indices].astype(int)
                else:
                    fold_ids = fold_ids_full.astype(int)
            else:
                # Fallback for legacy OOF predictions without fold_id
                fold_ids = np.zeros(n_samples, dtype=int)

            # Safe cast: NaN floats cannot be cast to int directly
            safe_preds = np.where(np.isnan(preds), 0, preds).astype(int)

            oof_result = OOFResult(
                predictions=safe_preds,
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
        df: pd.DataFrame | None = None,
        target_horizon: int | None = None,
    ) -> np.ndarray | None:
        """Extract aligned labels from OOF predictions or source DataFrame.

        First tries OOF predictions y_true column. If the OOF y_true length
        doesn't cover all common_indices (e.g. walk-forward mode), falls back
        to extracting labels from the source DataFrame using positional indexing.

        Args:
            oof_predictions: OOF predictions from base models
            aligned: Aligned OOF result with common indices
            df: Optional source DataFrame for label extraction
            target_horizon: Target prediction horizon for label column selection
        """
        common_indices = aligned.common_indices
        if common_indices is None:
            return None

        # Strategy 1: Try extracting from OOF predictions
        for _key, oof_pred in oof_predictions.items():
            if "y_true" in oof_pred.predictions.columns:
                y_full = oof_pred.predictions["y_true"].values
                # Only use if y_full covers all common indices
                if len(y_full) > common_indices.max():
                    valid_indices = common_indices[common_indices < len(y_full)]
                    if len(valid_indices) == aligned.n_common:
                        return np.asarray(y_full[valid_indices])

        # Strategy 2: Fall back to source DataFrame (handles walk-forward mode)
        if df is not None:
            # Find label column(s) in source df
            label_cols = [c for c in df.columns if c.startswith("label_h")]
            if label_cols:
                # Select the label column matching the target horizon
                if target_horizon is not None:
                    target_col = f"label_h{target_horizon}"
                    if target_col in df.columns:
                        label_col = target_col
                    else:
                        label_col = label_cols[0]
                        logger.warning(
                            f"Target label column '{target_col}' not found, using '{label_col}'"
                        )
                else:
                    label_col = label_cols[0]
                y_source = df[label_col].values
                valid_mask = common_indices < len(y_source)
                valid_indices = common_indices[valid_mask]
                if len(valid_indices) == aligned.n_common:
                    return np.asarray(y_source[valid_indices])
                elif len(valid_indices) > 0:
                    logger.warning(
                        f"Partial label coverage: {len(valid_indices)}/{aligned.n_common} "
                        f"samples have labels from df['{label_col}']"
                    )
                    # Return labels only for valid indices; caller must handle mismatch
                    return np.asarray(y_source[valid_indices])

        logger.warning("Could not extract aligned labels for meta-learner")
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

            # Drop rows with NaN values (common when heterogeneous models
            # have different coverage, e.g. sequence models produce NaN
            # probabilities for early indices lost to windowing)
            nan_mask = X_stack.isna().any(axis=1) | y_stack.isna()
            n_nan = int(nan_mask.sum())
            if n_nan > 0:
                logger.warning(
                    f"Dropping {n_nan}/{len(X_stack)} NaN rows from stacking dataset "
                    f"({n_nan / len(X_stack) * 100:.1f}% of samples)"
                )
                X_stack = X_stack[~nan_mask].reset_index(drop=True)
                y_stack = y_stack[~nan_mask].reset_index(drop=True)

            if len(X_stack) < 10:
                raise ValueError(
                    f"Insufficient samples after NaN removal: {len(X_stack)} " f"(need at least 10)"
                )

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


def to_ensemble_result(
    service_result: EnsembleServiceResult,
    config: PipelineConfig,
) -> Any:
    """Bridge EnsembleServiceResult to EnsembleResult for EnsembleBundle.

    The EnsembleService produces an ``EnsembleServiceResult`` while
    ``EnsembleBundle.from_ensemble_result()`` expects the orchestrator's
    ``EnsembleResult``.  This function converts between the two so that
    a bundle can be created directly after ensemble training.

    Args:
        service_result: Result from ``EnsembleService.build_ensemble()``.
        config: PipelineConfig used for training.

    Returns:
        An ``EnsembleResult`` suitable for ``EnsembleBundle.from_ensemble_result()``.
    """
    from src.models.ensemble.orchestrator import EnsembleResult

    aligned = service_result.aligned_oof
    model_names: list[str] = []
    if aligned is not None:
        model_names = list(aligned.model_names)

    metrics: dict[str, float] = {}
    for key, value in service_result.ensemble_metrics.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    meta_learner_name = config.meta_learner or "unknown"

    coverage = 1.0
    alignment_offset = 0
    if aligned is not None:
        coverage = getattr(aligned, "coverage", 1.0)
        alignment_offset = getattr(aligned, "alignment_offset", 0)

    return EnsembleResult(
        ensemble_name=f"{meta_learner_name}_ensemble",
        meta_learner_name=meta_learner_name,
        base_model_names=model_names,
        metrics=metrics,
        stacking_dataset=service_result.stacking_dataset,
        aligned_oof=aligned,
        training_time_seconds=service_result.training_time_seconds,
        n_base_models=len(model_names),
        coverage=coverage,
        alignment_offset=alignment_offset,
    )


__all__ = [
    "EnsembleService",
    "EnsembleRequest",
    "EnsembleServiceResult",
    "to_ensemble_result",
]
