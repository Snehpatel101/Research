"""
MetaLabelingBundle - primary direction model + meta-model bet filter.

Meta-labeling (Lopez de Prado, AFML ch. 3) trains a primary model to call the
direction and a meta-model to predict P(primary is correct). A trade is taken
only when that probability clears a threshold.

The meta-model was trained on the primary model's own (scaled) inputs, so the
bundle stores the primary ModelBundle plus the fitted meta estimator and scores
both from the same model input.

Layout on disk:
    path/
        meta_labeling_metadata.json
        primary_bundle/     (ModelBundle)
        meta_model.pkl      (fitted estimator with predict_proba)

Usage:
    bundle = MetaLabelingBundle.load("./bundles/meta_labeling_xgboost_logistic_h5")
    result = bundle.predict_from_raw(raw_df)   # neutral where the filter says no
    meta = bundle.predict_meta(raw_df)         # directions, P(correct), trade mask
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

META_LABELING_BUNDLE_VERSION = "2.0.0"
META_LABELING_METADATA_FILE = "meta_labeling_metadata.json"
PRIMARY_BUNDLE_DIR = "primary_bundle"
META_MODEL_FILE = "meta_model.pkl"

# Class label meaning "no trade" in both 3-class {-1,0,1} and binary {0,1} labels
NEUTRAL_LABEL = 0


@dataclass
class MetaLabelingPrediction:
    """Result of a meta-labeling prediction.

    Attributes:
        directions: Primary model class predictions (n_samples,).
        direction_probabilities: Primary model class probabilities (n_samples, n_classes).
        meta_probabilities: P(primary_is_correct) from the meta-model (n_samples,).
        positions: Signed position sizes = direction * meta_probability (n_samples,).
        trade_mask: Boolean mask where meta_probability >= threshold (n_samples,).
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

    @property
    def trade_ratio(self) -> float:
        """Fraction of samples passing the threshold filter."""
        return float(self.trade_mask.mean()) if len(self.trade_mask) else 0.0


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
            meta_model: Fitted binary classifier (class 1 = primary correct)
                with ``predict_proba``, trained on the primary's model input.
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
        """P(primary is correct) for unscaled primary input X (as accepted by predict)."""
        model_input = self.primary_bundle.model_input(X)
        return self.meta_model.predict_proba(model_input.reshape(len(model_input), -1))[:, 1]

    def predict_meta(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> MetaLabelingPrediction:
        """Directions, P(correct), positions and trade mask from raw OHLCV."""
        X, timestamps = self.primary_bundle.raw_to_input(raw_df, skip_cleaning=skip_cleaning)
        primary = self.primary_bundle.predict(X, calibrate=calibrate)
        p_correct = self.meta_probability(X)
        return MetaLabelingPrediction(
            directions=primary.class_predictions,
            direction_probabilities=primary.class_probabilities,
            meta_probabilities=p_correct,
            positions=primary.class_predictions.astype(np.float64) * p_correct,
            trade_mask=p_correct >= self.threshold,
            threshold=self.threshold,
            timestamps=timestamps,
            metadata={"primary_model": self.primary_bundle.metadata.model_name},
        )

    # -----------------------------------------------------------------
    # InferenceBundle protocol
    # -----------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray, calibrate: bool = True) -> PredictionResult:
        """Primary predictions with filtered-out bars set to neutral."""
        primary = self.primary_bundle.predict(X, calibrate=calibrate)
        return self._apply_filter(primary, self.meta_probability(X))

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
        return result

    def _apply_filter(self, primary: PredictionResult, p_correct: np.ndarray) -> PredictionResult:
        trade = p_correct >= self.threshold
        return PredictionResult(
            class_predictions=np.where(trade, primary.class_predictions, NEUTRAL_LABEL),
            class_probabilities=primary.class_probabilities,
            confidence=primary.confidence,
            metadata={**primary.metadata, "meta_probability": p_correct, "trade_mask": trade},
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
    "MetaLabelingBundle",
    "MetaLabelingPrediction",
    "META_LABELING_BUNDLE_VERSION",
    "META_LABELING_METADATA_FILE",
]
