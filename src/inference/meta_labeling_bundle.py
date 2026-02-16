"""
MetaLabelingBundle - Primary model + meta-model for confidence-weighted inference.

Combines a primary directional model with a meta-model that predicts
P(primary_is_correct). Uses threshold filtering to only trade when the
meta-model is sufficiently confident.

Implements the InferenceBundle protocol from src.core.protocols.

Usage:
    bundle = MetaLabelingBundle(
        primary_bundle=primary_model_bundle,
        meta_bundle=meta_model_bundle,
        threshold=0.55,
    )
    # Standard prediction (primary only)
    result = bundle.predict_from_raw(raw_df)

    # Meta-labeling prediction with confidence filtering
    meta_result = bundle.predict_meta(raw_df)
    print(meta_result.trade_mask)  # bool mask of filtered trades

    bundle.save("./bundles/meta_xgb_h20")
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

from src.inference.bundle import ModelBundle
from src.models.base import PredictionResult

logger = logging.getLogger(__name__)

META_LABELING_BUNDLE_VERSION = "1.0.0"
META_LABELING_METADATA_FILE = "meta_labeling_metadata.json"
PRIMARY_BUNDLE_DIR = "primary_bundle"
META_BUNDLE_DIR = "meta_bundle"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MetaLabelingPrediction:
    """Result of a meta-labeling prediction.

    Attributes:
        directions: Primary model class predictions (n_samples,).
        direction_probabilities: Primary model class probabilities (n_samples, n_classes).
        meta_probabilities: P(primary_is_correct) from meta-model (n_samples,).
        positions: Signed position sizes = direction * meta_probability (n_samples,).
        trade_mask: Boolean mask where meta_probability >= threshold (n_samples,).
        threshold: The threshold used for filtering.
    """

    directions: np.ndarray
    direction_probabilities: np.ndarray
    meta_probabilities: np.ndarray
    positions: np.ndarray
    trade_mask: np.ndarray
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        """Number of trades passing the threshold filter."""
        return int(self.trade_mask.sum())

    @property
    def trade_ratio(self) -> float:
        """Fraction of samples passing the threshold filter."""
        if len(self.trade_mask) == 0:
            return 0.0
        return float(self.trade_mask.mean())


@dataclass
class MetaLabelingBundleMetadata:
    """Metadata for meta-labeling bundle."""

    version: str
    threshold: float
    primary_model_name: str
    meta_model_name: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "threshold": self.threshold,
            "primary_model_name": self.primary_model_name,
            "meta_model_name": self.meta_model_name,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetaLabelingBundleMetadata:
        return cls(
            version=data["version"],
            threshold=data["threshold"],
            primary_model_name=data.get("primary_model_name", "unknown"),
            meta_model_name=data.get("meta_model_name", "unknown"),
            extra=data.get("extra", {}),
        )


# =============================================================================
# META LABELING BUNDLE
# =============================================================================


class MetaLabelingBundle:
    """Primary model + meta-model for confidence-weighted predictions.

    The primary bundle produces direction predictions. The meta bundle
    produces P(primary_is_correct). Together they enable:
    - Threshold filtering: only trade when meta-model confidence >= threshold
    - Position sizing: scale position by meta-model confidence

    Satisfies the InferenceBundle protocol (predict / predict_from_raw / load).
    """

    def __init__(
        self,
        primary_bundle: ModelBundle,
        meta_bundle: ModelBundle,
        threshold: float = 0.5,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            primary_bundle: Bundle producing directional predictions.
            meta_bundle: Bundle producing P(primary_is_correct) as binary
                classifier (class 1 = correct).
            threshold: Minimum meta-probability to execute a trade.
            extra: Additional metadata.
        """
        self.primary_bundle = primary_bundle
        self.meta_bundle = meta_bundle
        self.threshold = threshold
        self.ml_metadata = MetaLabelingBundleMetadata(
            version=META_LABELING_BUNDLE_VERSION,
            threshold=threshold,
            primary_model_name=primary_bundle.metadata.model_name,
            meta_model_name=meta_bundle.metadata.model_name,
            extra=extra or {},
        )

    # -----------------------------------------------------------------
    # Meta-labeling prediction
    # -----------------------------------------------------------------

    def predict_meta(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> MetaLabelingPrediction:
        """Full meta-labeling prediction from raw OHLCV data.

        1. Primary model produces directions.
        2. Meta model produces P(primary_is_correct).
        3. Positions = direction * meta_probability.
        4. Trade mask = meta_probability >= threshold.

        Args:
            raw_df: DataFrame with raw OHLCV data.
            calibrate: Whether to apply calibration.
            skip_cleaning: If True, skip resampling step.

        Returns:
            MetaLabelingPrediction with directions, meta-probabilities,
            positions, and trade mask.
        """
        # Primary model predictions
        primary_result = self.primary_bundle.predict_from_raw(
            raw_df, calibrate=calibrate, skip_cleaning=skip_cleaning
        )
        directions = primary_result.class_predictions
        direction_probs = primary_result.class_probabilities

        # Meta model predictions
        meta_result = self.meta_bundle.predict_from_raw(
            raw_df, calibrate=calibrate, skip_cleaning=skip_cleaning
        )
        # P(correct) = probability of class 1 in binary meta-classifier
        meta_probs = meta_result.class_probabilities
        if meta_probs.ndim == 2 and meta_probs.shape[1] >= 2:
            p_correct = meta_probs[:, 1]
        else:
            p_correct = meta_result.confidence

        # Align lengths (meta and primary may differ for sequence models)
        min_len = min(len(directions), len(p_correct))
        directions = directions[-min_len:]
        direction_probs = direction_probs[-min_len:]
        p_correct = p_correct[-min_len:]

        # Compute positions: direction sign * meta-probability
        # Map class predictions to signed direction (-1, 0, +1)
        signed_dir = directions.astype(np.float64)
        positions = signed_dir * p_correct

        # Threshold filter
        trade_mask = p_correct >= self.threshold

        return MetaLabelingPrediction(
            directions=directions,
            direction_probabilities=direction_probs,
            meta_probabilities=p_correct,
            positions=positions,
            trade_mask=trade_mask,
            threshold=self.threshold,
            metadata={
                "primary_model": self.ml_metadata.primary_model_name,
                "meta_model": self.ml_metadata.meta_model_name,
                "n_total": min_len,
                "n_trades": int(trade_mask.sum()),
            },
        )

    # -----------------------------------------------------------------
    # InferenceBundle protocol
    # -----------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> PredictionResult:
        """Predict using the primary model only.

        For meta-labeling with confidence, use predict_meta() instead.

        Args:
            X: Input features.
            calibrate: Whether to apply calibration.

        Returns:
            PredictionResult from the primary bundle.
        """
        return self.primary_bundle.predict(X, calibrate=calibrate)

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> PredictionResult:
        """Standard prediction from raw OHLCV (primary model only).

        For meta-labeling with confidence, use predict_meta() instead.

        Args:
            raw_df: DataFrame with raw OHLCV data.
            calibrate: Whether to apply calibration.
            skip_cleaning: If True, skip resampling step.

        Returns:
            PredictionResult from the primary bundle.
        """
        return self.primary_bundle.predict_from_raw(
            raw_df, calibrate=calibrate, skip_cleaning=skip_cleaning
        )

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save meta-labeling bundle to disk.

        Structure:
            path/
                meta_labeling_metadata.json
                primary_bundle/    (ModelBundle)
                meta_bundle/       (ModelBundle)

        Args:
            path: Directory path.
            overwrite: If True, overwrite existing.

        Returns:
            Path to saved bundle.
        """
        path = Path(path)

        if path.exists():
            if overwrite:
                shutil.rmtree(path)
            else:
                raise FileExistsError(
                    f"Bundle already exists at {path}. Use overwrite=True to replace."
                )

        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(path / META_LABELING_METADATA_FILE, "w") as f:
            json.dump(self.ml_metadata.to_dict(), f, indent=2)

        # Save inner bundles
        self.primary_bundle.save(path / PRIMARY_BUNDLE_DIR, overwrite=overwrite)
        self.meta_bundle.save(path / META_BUNDLE_DIR, overwrite=overwrite)

        logger.info(
            "Saved MetaLabelingBundle (primary=%s, meta=%s, threshold=%.3f) to %s",
            self.ml_metadata.primary_model_name,
            self.ml_metadata.meta_model_name,
            self.threshold,
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> MetaLabelingBundle:
        """Load meta-labeling bundle from disk.

        Args:
            path: Path to bundle directory.

        Returns:
            Loaded MetaLabelingBundle.

        Raises:
            FileNotFoundError: If bundle or components are missing.
        """
        path = Path(path)

        if not path.is_dir():
            raise FileNotFoundError(f"MetaLabelingBundle not found at {path}")

        # Load metadata
        meta_path = path / META_LABELING_METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {META_LABELING_METADATA_FILE} in {path}")

        with open(meta_path) as f:
            ml_meta = MetaLabelingBundleMetadata.from_dict(json.load(f))

        # Load inner bundles
        primary_bundle = ModelBundle.load(path / PRIMARY_BUNDLE_DIR)
        meta_bundle = ModelBundle.load(path / META_BUNDLE_DIR)

        bundle = cls(
            primary_bundle=primary_bundle,
            meta_bundle=meta_bundle,
            threshold=ml_meta.threshold,
            extra=ml_meta.extra,
        )
        bundle.ml_metadata = ml_meta

        logger.info(
            "Loaded MetaLabelingBundle (primary=%s, meta=%s, threshold=%.3f) from %s",
            ml_meta.primary_model_name,
            ml_meta.meta_model_name,
            ml_meta.threshold,
            path,
        )
        return bundle

    def __repr__(self) -> str:
        return (
            f"MetaLabelingBundle(primary={self.ml_metadata.primary_model_name!r}, "
            f"meta={self.ml_metadata.meta_model_name!r}, "
            f"threshold={self.threshold:.3f})"
        )


__all__ = [
    "MetaLabelingBundle",
    "MetaLabelingPrediction",
    "MetaLabelingBundleMetadata",
    "META_LABELING_BUNDLE_VERSION",
]
