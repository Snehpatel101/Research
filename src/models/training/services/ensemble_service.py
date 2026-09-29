# src/models/training/services/ensemble_service.py
"""
Service for building ensembles from OOF predictions.

Delegates to EnsembleOrchestrator for the actual ensemble building,
providing a simpler interface for the UnifiedTrainingOrchestrator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
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

# Trailing share of the aligned OOF rows held out to evaluate the meta-learner
META_HOLDOUT_FRACTION = 0.2
# Trailing share of the meta-train rows used for early stopping (when the
# meta-learner early-stops) — kept apart from the holdout it is scored on
META_EARLY_STOPPING_FRACTION = 0.15
# Fewest meta-train rows worth keeping a purge gap for
MIN_META_TRAIN_ROWS = 20


def split_temporal_tail(
    rows: np.ndarray,
    fraction: float,
    purge_bars: int,
    min_head: int = MIN_META_TRAIN_ROWS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Split positions into a head and a trailing tail with a purge gap.

    ``rows`` are the source-bar indices of time-ordered samples. The tail is
    the last ``fraction`` of the samples; the head keeps only samples whose
    bar lies more than ``purge_bars`` bars before the tail's first bar, so
    no head label window reaches into the tail.

    Returns:
        (head_positions, tail_positions, purge_used). When the purge would
        leave fewer than ``min_head`` head samples it is dropped (purge_used=0)
        and a warning is logged.
    """
    rows = np.asarray(rows)
    n = len(rows)
    n_tail = max(1, int(round(n * fraction)))
    tail = np.arange(n - n_tail, n)
    candidates = np.arange(n - n_tail)
    head = candidates[rows[candidates] < rows[tail[0]] - purge_bars]
    if purge_bars > 0 and len(head) < min(min_head, len(candidates)):
        logger.warning(
            "Purge gap of %d bars would leave %d head rows (< %d); splitting without it",
            purge_bars,
            len(head),
            min_head,
        )
        return candidates, tail, 0
    return head, tail, purge_bars


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
    # Each base model scored on the meta-learner's holdout rows, from its
    # OOF probabilities with the same metric code (model name -> metrics)
    base_model_holdout_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    # Out-of-sample signals of the stacking system on its purged holdout:
    # columns row (source bar position), prediction (trading label),
    # confidence — what the backtest of the deployed ensemble replays
    holdout_predictions: pd.DataFrame | None = None


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

    # Diversity thresholds (single definition — the analyzer is built
    # per-request in _analyze_diversity with the run's n_classes)
    MIN_DIVERSITY_THRESHOLD = 0.3
    CORRELATION_THRESHOLD = 0.8

    def __init__(self) -> None:
        """Initialize EnsembleService."""

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
            # Aligner must match the run's class count (binary mode uses 2)
            aligner = OOFAligner(n_classes=getattr(config, "n_classes", 3))
            aligned = aligner.align(oof_results, strategy="intersection")
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
                n_classes=getattr(config, "n_classes", 3),
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
        row_indices = (
            np.asarray(aligned.common_indices)[: len(stacking_df)]
            if aligned.common_indices is not None
            else np.arange(len(stacking_df))
        )

        stacking_dataset = StackingDataset(
            data=stacking_df,
            model_names=aligned.model_names,
            horizon=config.horizons[0] if config.horizons else 20,
            metadata={
                "n_common": aligned.n_common,
                "coverage": aligned.coverage,
                "row_indices": row_indices,
            },
        )

        logger.info(f"Stacking dataset: {stacking_dataset.n_samples} samples")

        # Train meta-learner
        meta_learner, ensemble_metrics, base_metrics, holdout_predictions = (
            self._train_meta_learner(stacking_dataset, config)
        )

        training_time = time.time() - start_time

        return EnsembleServiceResult(
            aligned_oof=aligned,
            stacking_dataset=stacking_dataset,
            ensemble_metrics=ensemble_metrics,
            meta_learner=meta_learner,
            training_time_seconds=training_time,
            diversity_metrics=diversity_metrics,
            base_model_holdout_metrics=base_metrics,
            holdout_predictions=holdout_predictions,
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

            # Schema contract: the predictions DataFrame must be full-length
            # (n_total_samples rows, NaN-padded). A compact frame here would
            # be silently mis-indexed below.
            if oof_pred.n_total_samples is not None and n_total != oof_pred.n_total_samples:
                raise ValueError(
                    f"OOFPrediction schema violation for '{model_name}': predictions "
                    f"DataFrame has {n_total} rows but n_total_samples="
                    f"{oof_pred.n_total_samples}. Producers must emit full-length "
                    f"NaN-padded frames with original_indices marking valid rows."
                )

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
    ) -> tuple[Any, dict[str, Any], dict[str, dict[str, float]], pd.DataFrame | None]:
        """Evaluate the meta-learner on a purged temporal holdout, then refit it.

        1. The trailing ``META_HOLDOUT_FRACTION`` of the aligned OOF rows is
           the holdout; meta-train rows within ``config.purge_bars`` bars of
           it are dropped so no meta-train label overlaps the holdout.
        2. A meta-learner that early-stops selects its stopping point on a
           purged tail of the meta-train rows, never on the holdout.
        3. The meta-learner AND every base model are scored on the identical
           holdout rows from their class probabilities (same metric code).
        4. The deployed meta-learner is refit on ALL aligned OOF rows (with
           the iteration count chosen in step 2 fixed), so the most recent
           rows inform the deployed model. The reported metrics are the
           step-3 holdout metrics of the evaluation fit.

        Uses the meta-learner's fit() directly rather than routing through
        Trainer/TimeSeriesDataContainer, which is designed for OHLCV
        time-series data and incompatible with OOF stacking features.

        Returns:
            (deployed meta-learner, ensemble metrics, base-model holdout
            metrics, holdout predictions). The holdout predictions are the
            evaluation fit's signals on the holdout rows (``row`` = source bar,
            ``prediction`` = trading label, ``confidence``): out-of-sample
            for the meta-learner and built from out-of-fold base predictions.
        """
        try:
            from src.models.ensemble import get_meta_learner
            from src.models.metrics import compute_probability_metrics

            start = time.time()
            n_classes = int(getattr(config, "n_classes", 3) or 3)
            purge_bars = int(getattr(config, "purge_bars", 0) or 0)

            X_stack = stacking_dataset.get_features()
            y_stack = stacking_dataset.get_labels()
            rows = np.asarray(stacking_dataset.metadata.get("row_indices", np.arange(len(X_stack))))

            # Drop rows with NaN values (common when heterogeneous models
            # have different coverage, e.g. sequence models produce NaN
            # probabilities for early indices lost to windowing)
            nan_mask = (X_stack.isna().any(axis=1) | y_stack.isna()).to_numpy()
            n_nan = int(nan_mask.sum())
            if n_nan > 0:
                logger.warning(
                    f"Dropping {n_nan}/{len(X_stack)} NaN rows from stacking dataset "
                    f"({n_nan / len(X_stack) * 100:.1f}% of samples)"
                )
            # Time order (by source bar) so the holdout is the most recent rows
            keep = np.flatnonzero(~nan_mask)
            keep = keep[np.argsort(rows[keep], kind="stable")]
            X = X_stack.to_numpy()[keep]
            y = y_stack.to_numpy()[keep].astype(int)
            rows = rows[keep]

            if len(X) < 10:
                raise ValueError(
                    f"Insufficient samples after NaN removal: {len(X)} (need at least 10)"
                )

            # 1. Purged temporal holdout
            train_pos, holdout_pos, holdout_purge = split_temporal_tail(
                rows, META_HOLDOUT_FRACTION, purge_bars
            )

            # 2. Evaluation fit (early stopping on a purged meta-train tail)
            meta_learner = get_meta_learner(config.meta_learner, n_classes=n_classes)
            fit_pos, es_pos = train_pos, holdout_pos
            early_stops = bool(getattr(meta_learner, "uses_early_stopping", False))
            if early_stops:
                head, tail, _ = split_temporal_tail(
                    rows[train_pos], META_EARLY_STOPPING_FRACTION, purge_bars
                )
                fit_pos, es_pos = train_pos[head], train_pos[tail]
            # Non-early-stopping learners only report on X_val; for them it
            # is the holdout, which never influences the fit.
            training_metrics = meta_learner.fit(
                X_train=X[fit_pos],
                y_train=y[fit_pos],
                X_val=X[es_pos],
                y_val=y[es_pos],
            )

            # 3. Uniform holdout metrics: meta-learner and every base model
            X_hold, y_hold = X[holdout_pos], y[holdout_pos]
            holdout_output = meta_learner.predict(X_hold)
            holdout_metrics = compute_probability_metrics(
                y_hold, holdout_output.class_probabilities, n_classes
            )
            holdout_predictions = pd.DataFrame(
                {
                    "row": rows[holdout_pos],
                    "prediction": np.asarray(holdout_output.class_predictions).astype(int),
                    "confidence": np.asarray(holdout_output.class_probabilities).max(axis=1),
                }
            )
            base_metrics = {
                name: compute_probability_metrics(
                    y_hold, X_hold[:, i * n_classes : (i + 1) * n_classes], n_classes
                )
                for i, name in enumerate(stacking_dataset.model_names)
            }

            # 4. Refit on every aligned OOF row for deployment
            refit_overrides: dict[str, Any] = meta_learner.refit_config() if early_stops else {}
            deployed = get_meta_learner(config.meta_learner, n_classes=n_classes, **refit_overrides)
            deployed.fit(X_train=X, y_train=y, X_val=X_hold, y_val=y_hold)

            training_time = time.time() - start
            n_holdout = holdout_metrics.pop("n_samples")
            metrics: dict[str, Any] = {
                # Legacy keys (factory summary, bundle score) = holdout values
                "val_f1": holdout_metrics["macro_f1"],
                "val_accuracy": holdout_metrics["accuracy"],
                "val_loss": holdout_metrics["log_loss"],
                "train_loss": training_metrics.train_loss,
                **holdout_metrics,
                "n_meta_train": len(fit_pos),
                "n_meta_early_stopping": len(es_pos) if early_stops else 0,
                "n_holdout": int(n_holdout),
                "holdout_purge_bars": holdout_purge,
                "n_refit": len(X),
                "training_time": training_time,
            }

            logger.info(
                f"Meta-learner ({config.meta_learner}) holdout ({int(n_holdout)} rows): "
                f"macro_f1={holdout_metrics['macro_f1']:.4f}, "
                f"log_loss={holdout_metrics['log_loss']:.4f}; refit on {len(X)} rows"
            )
            for name, bm in base_metrics.items():
                logger.info(
                    f"  base {name} on same holdout: macro_f1={bm['macro_f1']:.4f}, "
                    f"log_loss={bm['log_loss']:.4f}"
                )

            return deployed, metrics, base_metrics, holdout_predictions

        except Exception as e:
            logger.error(f"Failed to train meta-learner: {e}")
            return None, {"error": str(e)}, {}, None

    def _analyze_diversity(
        self,
        oof_predictions: dict[str, OOFPrediction],
        aligned: AlignedOOFResult,
        y_aligned: np.ndarray | None,
        n_classes: int = 3,
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

            # Analyzer is built here (its only use site) with the run's
            # class count — binary mode uses 2.
            analyzer = DiversityAnalyzer(
                min_diversity_threshold=self.MIN_DIVERSITY_THRESHOLD,
                correlation_threshold=self.CORRELATION_THRESHOLD,
                n_classes=n_classes,
            )
            diversity_metrics = analyzer.analyze(
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
            if diversity_metrics.diversity_score < analyzer.min_diversity_threshold:
                logger.warning(
                    f"Low ensemble diversity detected: "
                    f"score={diversity_metrics.diversity_score:.3f} < "
                    f"threshold={analyzer.min_diversity_threshold:.3f}"
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


__all__ = [
    "EnsembleService",
    "EnsembleRequest",
    "EnsembleServiceResult",
]
