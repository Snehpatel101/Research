"""
MetaLabelingBundle - primary direction model + meta-model bet filter.

Meta-labeling (Lopez de Prado, AFML ch. 3) trains a primary model to call the
side and a meta-model to predict P(the primary's bet pays off). The meta-model
is trained only on bars where the primary takes a side (prediction != neutral)
and a trade is taken only when that probability clears a threshold; bars where
the primary predicts neutral are never traded.

Meta features (``build_meta_features``) are the primary's own scaled model
input plus its uncalibrated class probabilities and confidence — built by the
same function at training and at serving time (train/serve parity).

Layout on disk:
    path/
        meta_labeling_metadata.json
        primary_bundle/     (ModelBundle)
        meta_model.pkl      (fitted estimator with predict_proba)

Usage:
    bundle = MetaLabelingBundle.load("./bundles/meta_labeling_xgboost_logistic_h5")
    result = bundle.predict_from_raw(raw_df)   # neutral where the filter says no
    meta = bundle.predict_meta(raw_df)         # directions, P(bet pays off), trade mask
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.utils.json_utils import NumpyEncoder
from src.core.utils.safe_pickle import safe_pickle_dump, safe_pickle_load
from src.inference.bundle import ModelBundle
from src.models.base import PredictionResult

logger = logging.getLogger(__name__)

META_LABELING_BUNDLE_VERSION = "3.0.0"
# Version of the meta-feature layout; bundles with another layout were trained
# on different inputs and cannot be served by this code.
META_FEATURE_VERSION = 2
META_LABELING_METADATA_FILE = "meta_labeling_metadata.json"
PRIMARY_BUNDLE_DIR = "primary_bundle"
META_MODEL_FILE = "meta_model.pkl"

# Class label meaning "no trade" in both 3-class {-1,0,1} and binary {0,1} labels.
# A primary "side" is any other prediction: -1/+1 (short/long) in 3-class mode,
# 1 ("a barrier will be hit") in binary mode, whose labels carry no direction.
NEUTRAL_LABEL = 0


def primary_sides(class_predictions: np.ndarray) -> np.ndarray:
    """Mask of bars where the primary takes a side (prediction != neutral)."""
    return np.asarray(class_predictions) != NEUTRAL_LABEL


class ConstantBetFilter:
    """Meta-model for a primary whose bets cannot train a classifier.

    When the primary's sided OOF bets are all wins, all losses, or absent,
    there is nothing to discriminate; the honest filter is the observed win
    rate (1.0 with no bets, so the filter passes whatever the primary does).
    """

    def __init__(self, win_rate: float) -> None:
        self.win_rate = float(win_rate)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ConstantBetFilter:  # noqa: ARG002
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full(len(X), self.win_rate)
        return np.column_stack([1.0 - p, p])


def build_meta_features(model_input: np.ndarray, primary_probabilities: np.ndarray) -> np.ndarray:
    """Meta-model input: primary model input, primary class probabilities, confidence.

    Args:
        model_input: The primary's scaled input in any rank (flattened per sample).
        primary_probabilities: The primary's UNCALIBRATED class probabilities
            (n_samples, n_classes) — out-of-fold at training time, the served
            model's at inference time.

    Returns:
        float32 array (n_samples, n_inputs + n_classes + 1).
    """
    n = len(model_input)
    probs = np.asarray(primary_probabilities, dtype=np.float32)
    if probs.ndim != 2 or len(probs) != n:
        raise ValueError(
            f"primary_probabilities must be (n_samples={n}, n_classes), got {probs.shape}"
        )
    inputs = np.asarray(model_input, dtype=np.float32)
    # Explicit width: reshape(0, -1) is ambiguous when the primary bets on no rows.
    n_inputs = int(np.prod(inputs.shape[1:], dtype=np.int64))
    return np.hstack(
        [
            inputs.reshape(n, n_inputs),
            probs,
            probs.max(axis=1, keepdims=True),
        ]
    )


@dataclass
class MetaLabelingPrediction:
    """Result of a meta-labeling prediction.

    Attributes:
        directions: Primary model class predictions (n_samples,).
        direction_probabilities: Primary model class probabilities (n_samples, n_classes).
        meta_probabilities: P(the primary's bet pays off) from the meta-model
            (n_samples,); only meaningful where the primary takes a side.
        positions: Signed position sizes = direction * meta_probability on
            traded bars, 0 elsewhere (n_samples,).
        trade_mask: Primary takes a side AND meta_probability >= threshold.
        threshold: The threshold used for filtering.
        timestamps: Bar timestamp of every row.
    """

    directions: np.ndarray
    direction_probabilities: np.ndarray
    meta_probabilities: np.ndarray
    positions: np.ndarray
    trade_mask: np.ndarray
    threshold: float
    timestamps: pd.DatetimeIndex
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        """Number of trades passing the threshold filter."""
        return int(self.trade_mask.sum())


class MetaLabelingBundle:
    """Primary ModelBundle + meta estimator + threshold.

    Satisfies the InferenceBundle protocol.
    """

    def __init__(
        self,
        primary_bundle: ModelBundle,
        meta_model: Any,
        threshold: float,
        meta_model_name: str,
        horizon: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            primary_bundle: Bundle producing directional predictions.
            meta_model: Fitted binary classifier (class 1 = the primary's bet
                paid off) with ``predict_proba``, trained on
                ``build_meta_features`` of the primary's sided bars.
            threshold: Minimum P(correct) to take a trade.
            meta_model_name: Meta-model family name (e.g. "logistic").
            horizon: Prediction horizon in bars.
            metrics: Training metrics recorded for the deploy manifest.
        """
        self.primary_bundle = primary_bundle
        self.meta_model = meta_model
        self.threshold = threshold
        self.meta_model_name = meta_model_name
        self.horizon = horizon
        self.metrics = metrics or {}

    @property
    def model_name(self) -> str:
        return f"meta_labeling_{self.primary_bundle.metadata.model_name}_{self.meta_model_name}"

    # -----------------------------------------------------------------
    # Meta-labeling prediction
    # -----------------------------------------------------------------

    def meta_probability(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """P(the primary's bet pays off) for unscaled primary input X (as accepted by predict)."""
        return self._score(X, calibrate=False)[1]

    def _score(
        self, X: pd.DataFrame | np.ndarray, calibrate: bool
    ) -> tuple[PredictionResult, np.ndarray]:
        """Primary prediction (optionally calibrated) and meta P(bet pays off)."""
        model_input = self.primary_bundle.model_input(X)
        # Meta features use the primary's UNCALIBRATED probabilities, as in training
        raw = self.primary_bundle.model.predict(model_input)
        meta_input = build_meta_features(model_input, raw.class_probabilities)
        p_win = self.meta_model.predict_proba(meta_input)[:, 1]
        primary = self.primary_bundle.predict(X, calibrate=True) if calibrate else raw
        return primary, p_win

    def trade_mask(self, directions: np.ndarray, p_win: np.ndarray) -> np.ndarray:
        """Bars that are traded: the primary takes a side and the meta filter accepts."""
        return primary_sides(directions) & (p_win >= self.threshold)

    def predict_meta(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> MetaLabelingPrediction:
        """Directions, P(bet pays off), positions and trade mask from raw OHLCV."""
        X, timestamps = self.primary_bundle.raw_to_input(raw_df, skip_cleaning=skip_cleaning)
        primary, p_win = self._score(X, calibrate=calibrate)
        trade = self.trade_mask(primary.class_predictions, p_win)
        return MetaLabelingPrediction(
            directions=primary.class_predictions,
            direction_probabilities=primary.class_probabilities,
            meta_probabilities=p_win,
            positions=np.where(trade, primary.class_predictions.astype(np.float64) * p_win, 0.0),
            trade_mask=trade,
            threshold=self.threshold,
            timestamps=timestamps,
            metadata={"primary_model": self.primary_bundle.metadata.model_name},
        )

    # -----------------------------------------------------------------
    # InferenceBundle protocol
    # -----------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray, calibrate: bool = True) -> PredictionResult:
        """Primary predictions with filtered-out bars set to neutral."""
        return self._apply_filter(*self._score(X, calibrate=calibrate))

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> PredictionResult:
        """Primary predictions from raw OHLCV, neutral where the meta filter rejects."""
        X, timestamps = self.primary_bundle.raw_to_input(raw_df, skip_cleaning=skip_cleaning)
        result = self.predict(X, calibrate=calibrate)
        result.metadata["timestamps"] = timestamps
        flags = self.event_flags(raw_df, timestamps, skip_cleaning=skip_cleaning)
        if flags is not None:
            result.metadata["is_event"] = flags
        return result

    def event_flags(
        self,
        raw_df: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
        skip_cleaning: bool = False,
    ) -> np.ndarray | None:
        """Event-bar flags of the primary model (see ``ModelBundle.event_flags``)."""
        return self.primary_bundle.event_flags(raw_df, timestamps, skip_cleaning=skip_cleaning)

    def _apply_filter(self, primary: PredictionResult, p_win: np.ndarray) -> PredictionResult:
        trade = self.trade_mask(primary.class_predictions, p_win)
        return PredictionResult(
            class_predictions=np.where(trade, primary.class_predictions, NEUTRAL_LABEL),
            class_probabilities=primary.class_probabilities,
            confidence=primary.confidence,
            metadata={**primary.metadata, "meta_probability": p_win, "trade_mask": trade},
        )

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save metadata, the primary bundle, and the meta estimator."""
        path = Path(path)
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"Bundle already exists at {path}. Use overwrite=True.")
            shutil.rmtree(path)
        path.mkdir(parents=True)

        metadata = {
            "version": META_LABELING_BUNDLE_VERSION,
            "meta_feature_version": META_FEATURE_VERSION,
            "model_name": self.model_name,
            "horizon": self.horizon,
            "threshold": self.threshold,
            "primary_model_name": self.primary_bundle.metadata.model_name,
            "meta_model_name": self.meta_model_name,
            "metrics": self.metrics,
        }
        with open(path / META_LABELING_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
        self.primary_bundle.save(path / PRIMARY_BUNDLE_DIR)
        safe_pickle_dump(self.meta_model, path / META_MODEL_FILE)
        logger.info(f"Saved MetaLabelingBundle ({self.model_name}, threshold={self.threshold})")
        return path

    @classmethod
    def load(cls, path: str | Path) -> MetaLabelingBundle:
        """Load a bundle saved with :meth:`save`."""
        path = Path(path)
        metadata_path = path / META_LABELING_METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing {META_LABELING_METADATA_FILE} in {path}")
        with open(metadata_path) as f:
            metadata = json.load(f)
        feature_version = metadata.get("meta_feature_version")
        if feature_version != META_FEATURE_VERSION:
            raise ValueError(
                f"Meta-labeling bundle at {path} was trained with meta-feature layout "
                f"{feature_version!r}; this code builds layout {META_FEATURE_VERSION}. "
                "Retrain it — serving it would feed the meta-model different inputs."
            )
        return cls(
            primary_bundle=ModelBundle.load(path / PRIMARY_BUNDLE_DIR),
            meta_model=safe_pickle_load(path / META_MODEL_FILE),
            threshold=metadata["threshold"],
            meta_model_name=metadata["meta_model_name"],
            horizon=metadata["horizon"],
            metrics=metadata.get("metrics", {}),
        )

    def __repr__(self) -> str:
        return f"MetaLabelingBundle({self.model_name!r}, threshold={self.threshold:.3f})"


__all__ = [
    "build_meta_features",
    "primary_sides",
    "META_FEATURE_VERSION",
    "MetaLabelingBundle",
    "MetaLabelingPrediction",
    "META_LABELING_BUNDLE_VERSION",
    "META_LABELING_METADATA_FILE",
]
