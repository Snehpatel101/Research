"""
Core OOF (Out-of-Fold) prediction generation.

Handles the main logic for generating out-of-sample predictions
for tabular (non-sequence) models.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.core.label_spans import LabelSpans
from src.models.base import PredictionResult
from src.models.calibration import CalibrationConfig, ProbabilityCalibrator
from src.models.registry import ModelRegistry

from .early_stopping_split import carve_early_stopping_split
from .fold_scaling import FoldAwareScaler, get_scaling_method_for_model
from .purged_kfold import PurgedKFold

logger = logging.getLogger(__name__)


def get_prob_column_names(model_name: str, n_classes: int) -> list[str]:
    """Return probability column names based on n_classes.

    n_classes=3: prob_short, prob_neutral, prob_long (backward compatible)
    n_classes=2: prob_0, prob_1
    Otherwise:   prob_0, prob_1, ..., prob_{n-1}
    """
    if n_classes == 3:
        return [
            f"{model_name}_prob_short",
            f"{model_name}_prob_neutral",
            f"{model_name}_prob_long",
        ]
    return [f"{model_name}_prob_{i}" for i in range(n_classes)]


def build_oof_frame(
    model_name: str,
    index: pd.Index,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    fold_ids: np.ndarray,
) -> pd.DataFrame:
    """The OOFPrediction frame: one row per sample, NaN where not predicted.

    Every OOF producer (tabular, sequence, windowed, per-fold feature selection,
    walk-forward) emits this schema. Columns: ``datetime`` (``index`` when it is
    a DatetimeIndex, else row positions), ``y_true``, one ``{model}_prob_*``
    column per class (``get_prob_column_names``), ``{model}_pred``,
    ``{model}_confidence``, ``fold_id`` (fold / window of each prediction, -1
    where none).

    Args:
        index: The samples' index (bar times when available).
        y_true: Labels, one per sample.
    """
    n_classes = probabilities.shape[1]
    oof_data: dict[str, Any] = {
        "datetime": index if isinstance(index, pd.DatetimeIndex) else range(len(index)),
        "y_true": np.asarray(y_true),
    }
    for i, col_name in enumerate(get_prob_column_names(model_name, n_classes)):
        oof_data[col_name] = probabilities[:, i]
    oof_data[f"{model_name}_pred"] = predictions
    oof_data[f"{model_name}_confidence"] = confidence
    oof_data["fold_id"] = fold_ids
    return pd.DataFrame(oof_data)


def held_out_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy and macro-F1 of a fold model on its held-out fold."""
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"val_accuracy": float("nan"), "val_f1": float("nan")}
    return {
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# =============================================================================
# DATA CLASSES
# =============================================================================


class OOFPrediction:
    """
    Out-of-fold predictions for a single model.

    Attributes:
        model_name: Name of the model
        predictions: DataFrame with OOF predictions
        fold_info: Per-fold training information
        coverage: Fraction of samples with predictions
        original_indices: Indices that have valid predictions (for alignment)
        sequence_length: Sequence length if sequence model
        n_total_samples: Total samples in source data
    """

    def __init__(
        self,
        model_name: str,
        predictions: pd.DataFrame,
        fold_info: list[dict[str, Any]],
        coverage: float = 1.0,
        # Phase 4 SNwH: Alignment metadata for heterogeneous stacking
        original_indices: np.ndarray | None = None,
        sequence_length: int | None = None,
        n_total_samples: int | None = None,
    ):
        # Schema contract: when n_total_samples is declared, the predictions
        # frame must be FULL-LENGTH (NaN-padded), with original_indices
        # marking the valid rows. Enforced here — at the root — so a compact
        # frame fails at the offending producer, not deep inside a consumer.
        if n_total_samples is not None and len(predictions) != n_total_samples:
            raise ValueError(
                f"OOFPrediction schema violation for '{model_name}': predictions "
                f"DataFrame has {len(predictions)} rows but n_total_samples="
                f"{n_total_samples}. Producers must emit full-length NaN-padded "
                f"frames with original_indices marking valid rows."
            )
        self.model_name = model_name
        self.predictions = predictions
        self.fold_info = fold_info
        self.coverage = coverage
        # Phase 4 alignment metadata
        self.original_indices = original_indices
        self.sequence_length = sequence_length
        self.n_total_samples = n_total_samples

    @property
    def n_valid(self) -> int:
        """Number of samples with valid predictions."""
        if self.original_indices is not None:
            return len(self.original_indices)
        return len(self.predictions)

    @property
    def alignment_offset(self) -> int:
        """
        Offset from start of dataset to first valid prediction.

        For tabular models: 0
        For sequence models: seq_len - 1
        """
        if self.original_indices is not None and len(self.original_indices) > 0:
            return int(self.original_indices.min())
        if self.sequence_length:
            return self.sequence_length - 1
        return 0

    def get_probabilities(self) -> np.ndarray:
        """
        Get probability matrix (n_samples, n_classes).

        Column-dynamic: uses the canonical short/neutral/long columns when
        present (3-class), otherwise any {model}_prob_* columns in frame
        order (e.g. prob_0/prob_1 in binary mode).
        """
        canonical = get_prob_column_names(self.model_name, 3)
        if all(c in self.predictions.columns for c in canonical):
            cols: list[str] = canonical
        else:
            prefix = f"{self.model_name}_prob_"
            cols = [c for c in self.predictions.columns if c.startswith(prefix)]
            if not cols:
                raise KeyError(
                    f"No probability columns found for model '{self.model_name}' "
                    f"(expected columns starting with '{prefix}')"
                )
        result: np.ndarray = self.predictions[cols].values
        return result

    def get_class_predictions(self) -> np.ndarray:
        """Get predicted classes (-1, 0, 1)."""
        result: np.ndarray = self.predictions[f"{self.model_name}_pred"].values
        return result


def reindex_oof_to_rows(
    oof: OOFPrediction | None, row_positions: np.ndarray | None
) -> OOFPrediction | None:
    """
    Re-index OOF predictions from sample order to source-DataFrame rows.

    Samples of different input ranks cover different bars (a 3D window's
    sample 0 is the bar at row seq_len - 1). Stacking ensembles and the
    backtest align models on ``original_indices``, so every model must
    report predictions at the rows its labels come from.

    Args:
        oof: OOF predictions indexed by sample (0..n_samples-1).
        row_positions: Row of the source DataFrame for each sample
            (``PreparedData.train_indices``); None keeps sample order.
    """
    if oof is None or row_positions is None:
        return oof
    row_positions = np.asarray(row_positions, dtype=np.int64)
    local = oof.predictions
    if len(local) != len(row_positions):
        raise ValueError(
            f"OOF for {oof.model_name} has {len(local)} rows but "
            f"{len(row_positions)} sample row positions"
        )

    n_rows = int(row_positions.max()) + 1
    columns: dict[str, Any] = {"datetime": np.arange(n_rows)}
    for col in local.columns:
        if col == "datetime":
            continue
        # Rows without a sample: fold_id -1, everything else NaN
        full = np.full(n_rows, -1, dtype=np.int64) if col == "fold_id" else np.full(n_rows, np.nan)
        full[row_positions] = local[col].to_numpy()
        columns[col] = full
    valid_local = (
        oof.original_indices
        if oof.original_indices is not None
        else np.flatnonzero(local.filter(like="_prob_").notna().all(axis=1).to_numpy())
    )
    return OOFPrediction(
        model_name=oof.model_name,
        predictions=pd.DataFrame(columns),
        fold_info=oof.fold_info,
        coverage=oof.coverage,
        original_indices=row_positions[valid_local],
        sequence_length=oof.sequence_length,
        n_total_samples=n_rows,
    )


def merge_oof_predictions(parts: list[OOFPrediction]) -> OOFPrediction:
    """
    Merge OOF predictions that cover disjoint source rows into one model's OOF.

    Used for regime-routed models: each regime's model predicts its own
    regime's rows. All parts must be in source-row coordinates
    (see :func:`reindex_oof_to_rows`) and share one model name.
    """
    if not parts:
        raise ValueError("merge_oof_predictions needs at least one OOFPrediction")
    n_rows = max(len(p.predictions) for p in parts)
    columns: dict[str, Any] = {"datetime": np.arange(n_rows)}
    for part in parts:
        rows = part.original_indices
        if rows is None:
            raise ValueError(f"OOF part for {part.model_name} has no original_indices")
        for col in part.predictions.columns:
            if col == "datetime":
                continue
            if col not in columns:
                columns[col] = (
                    np.full(n_rows, -1, dtype=np.int64)
                    if col == "fold_id"
                    else np.full(n_rows, np.nan)
                )
            columns[col][rows] = part.predictions[col].to_numpy()[rows]

    rows = np.unique(np.concatenate([p.original_indices for p in parts]))
    n_samples = sum(len(p.original_indices) / p.coverage for p in parts if p.coverage > 0)
    return OOFPrediction(
        model_name=parts[0].model_name,
        predictions=pd.DataFrame(columns),
        fold_info=[info for p in parts for info in p.fold_info],
        coverage=len(rows) / n_samples if n_samples else 0.0,
        original_indices=rows,
        sequence_length=parts[0].sequence_length,
        n_total_samples=n_rows,
    )


# =============================================================================
# CORE OOF GENERATOR
# =============================================================================


class CoreOOFGenerator:
    """
    Core OOF prediction generator for tabular (non-sequence) models.

    Handles fold-aware scaling, training, and prediction generation.
    """

    def __init__(self, cv: PurgedKFold) -> None:
        """
        Initialize CoreOOFGenerator.

        Args:
            cv: PurgedKFold cross-validator
        """
        self.cv = cv

    def generate_tabular_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        config: dict[str, Any],
        sample_weights: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
        label_spans: LabelSpans | None = None,
        n_classes: int = 3,
    ) -> OOFPrediction:
        """
        Generate OOF predictions for a tabular model.

        Args:
            X: Feature DataFrame
            y: Labels
            model_name: Name of the model
            config: Model hyperparameters
            sample_weights: Optional quality weights
            label_end_times: Optional Series of datetime when each label is resolved.
                If provided, enables proper purging of overlapping labels in CV.
            n_classes: Number of output classes (default 3: short/neutral/long).
                      Set to 2 for binary mode.

        Returns:
            OOFPrediction with predictions and fold info
        """
        n_samples = len(X)

        # Initialize OOF storage
        oof_probs = np.full((n_samples, n_classes), np.nan)
        oof_preds = np.full(n_samples, np.nan)
        oof_confidence = np.full(n_samples, np.nan)
        oof_fold_ids = np.full(n_samples, -1, dtype=int)
        fold_info: list[dict[str, Any]] = []

        # Determine scaling method based on model requirements
        scaling_method = get_scaling_method_for_model(model_name)
        fold_scaler = FoldAwareScaler(method=scaling_method)

        # Generate predictions fold by fold (with label_end_times for overlapping label purge)
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv.split(X, y, label_end_times=label_end_times, label_spans=label_spans)
        ):
            # Early stopping selects on a purged tail of the TRAIN rows —
            # never on the held-out fold these predictions are made for.
            es_split = carve_early_stopping_split(train_idx, self.cv.config.purge_bars)
            fit_idx, es_idx = es_split.fit_idx, es_split.es_idx
            logger.debug(
                f"  Fold {fold_idx + 1}: fit={len(fit_idx)}, early_stop={len(es_idx)}, "
                f"held_out={len(val_idx)}"
            )

            # FOLD-AWARE SCALING: fit on the fit rows only, transform the
            # early-stopping and held-out rows with the same statistics.
            X_fit_raw = X.iloc[fit_idx].values
            scaling_result = fold_scaler.fit_transform_fold(
                X_fit_raw, np.vstack([X.iloc[es_idx].values, X.iloc[val_idx].values])
            )
            X_fit_scaled = scaling_result.X_train_scaled
            X_es_scaled = scaling_result.X_val_scaled[: len(es_idx)]
            X_val_scaled = scaling_result.X_val_scaled[len(es_idx) :]

            w_fit = sample_weights.iloc[fit_idx].values if sample_weights is not None else None

            model = ModelRegistry.create(model_name, config=config)
            model.fit(
                X_train=X_fit_scaled,
                y_train=y.iloc[fit_idx].values,
                X_val=X_es_scaled,
                y_val=y.iloc[es_idx].values,
                sample_weights=w_fit,
            )

            # Predict the untouched held-out fold
            prediction_output: PredictionResult = model.predict(X_val_scaled)

            # Store OOF predictions
            oof_probs[val_idx] = prediction_output.class_probabilities
            oof_preds[val_idx] = prediction_output.class_predictions
            oof_confidence[val_idx] = prediction_output.confidence
            oof_fold_ids[val_idx] = fold_idx

            fold_info.append(
                {
                    "fold": fold_idx,
                    "train_size": len(fit_idx),
                    "early_stopping_size": len(es_idx),
                    "early_stopping_held_out": es_split.held_out,
                    "val_size": len(val_idx),
                    **held_out_fold_metrics(
                        y.iloc[val_idx].values, prediction_output.class_predictions
                    ),
                }
            )

            # Free memory between folds
            del model, X_fit_scaled, X_es_scaled, X_val_scaled, X_fit_raw
            del scaling_result, prediction_output
            gc.collect()

        # Validate coverage
        coverage = float((~np.isnan(oof_preds)).mean())
        if coverage < 1.0:
            logger.warning(
                f"{model_name}: Only {coverage:.2%} coverage. "
                f"{int(np.isnan(oof_preds).sum())} samples missing predictions."
            )

        oof_df = build_oof_frame(
            model_name, X.index, y.values, oof_probs, oof_preds, oof_confidence, oof_fold_ids
        )

        return OOFPrediction(
            model_name=model_name,
            predictions=oof_df,
            fold_info=fold_info,
            coverage=coverage,
        )

    def calibrate_oof_predictions(
        self,
        oof_results: dict[str, OOFPrediction],
        y_true: pd.Series,
        calibration_method: str = "auto",
    ) -> dict[str, OOFPrediction]:
        """
        Apply probability calibration to OOF predictions.

        This is leakage-safe because OOF predictions are truly out-of-sample:
        each prediction was made by a model that never saw that sample during
        training. The calibrator learns the probability mapping from these
        honest predictions.

        Args:
            oof_results: Dict of OOF predictions by model
            y_true: True labels
            calibration_method: Calibration method

        Returns:
            Dict of calibrated OOF predictions
        """
        logger.info("Applying probability calibration to OOF predictions...")

        y_array = y_true.values

        for model_name, oof_pred in oof_results.items():
            # Get probability columns dynamically (supports binary and 3-class)
            prob_cols = [
                c for c in oof_pred.predictions.columns if c.startswith(f"{model_name}_prob_")
            ]
            if not prob_cols:
                logger.warning(f"  {model_name}: No probability columns found")
                continue
            probs = oof_pred.predictions[prob_cols].values

            # Handle NaN predictions (keep them as-is)
            valid_mask = ~np.isnan(probs[:, 0])
            if valid_mask.sum() == 0:
                logger.warning(f"  {model_name}: No valid predictions to calibrate")
                continue

            valid_probs = probs[valid_mask]
            valid_y = y_array[valid_mask]

            # Fit and apply calibrator
            # NOTE: We do NOT report before/after improvement metrics here because
            # the calibrator is fit and evaluated on the same OOF data. Reporting
            # "improvement" on in-sample data would be self-referential and misleading.
            # The calibration mapping is still valid (OOF predictions are truly
            # out-of-sample w.r.t. model training), but improvement should only be
            # measured on a held-out set.
            from typing import Literal, cast

            cal_method = cast(Literal["isotonic", "sigmoid", "auto"], calibration_method)
            cal_config = CalibrationConfig(method=cal_method)
            calibrator = ProbabilityCalibrator(cal_config)
            calibrator.fit(valid_y, valid_probs)

            calibrated_probs = calibrator.calibrate(valid_probs)

            # Update predictions DataFrame
            oof_pred.predictions.loc[valid_mask, prob_cols] = calibrated_probs

            # Update confidence based on calibrated probabilities
            oof_pred.predictions.loc[valid_mask, f"{model_name}_confidence"] = calibrated_probs.max(
                axis=1
            )

            logger.info(
                f"  {model_name}: Calibration applied ({calibration_method} method, "
                f"{valid_mask.sum()} samples)"
            )

        return oof_results


__all__ = [
    "OOFPrediction",
    "CoreOOFGenerator",
]
