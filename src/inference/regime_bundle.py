"""
RegimeBundle - per-bar regime routing for regime-aware models.

Regime-aware training fits one model per market regime (same model family,
same features, different training subsets). At inference every bar is routed
to the model of *its* regime, using the exact RegimeDetector configuration the
training run used — the same routing that produced the model's regime-routed
OOF predictions.

Layout on disk:
    path/
        regime_bundle_metadata.json
        regimes/<regime>/        (one ModelBundle per regime)

Usage:
    bundle = RegimeBundle.load("./bundles/xgboost_h5")
    result = bundle.predict_from_raw(raw_ohlcv_df)
    result.metadata["regimes"]  # regime label used for each prediction row
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.constants import OHLCV_COLUMNS
from src.core.utils.json_utils import NumpyEncoder
from src.inference.bundle import ModelBundle
from src.inference.preprocessing_graph import PreprocessingGraph
from src.models.base import PredictionResult

logger = logging.getLogger(__name__)

REGIME_BUNDLE_VERSION = "2.0.0"
REGIME_BUNDLE_METADATA_FILE = "regime_bundle_metadata.json"
REGIME_BUNDLES_DIR = "regimes"


class RegimeBundle:
    """Routes each bar to the ModelBundle trained on that bar's regime.

    Satisfies the InferenceBundle protocol.
    """

    def __init__(
        self,
        regime_bundles: dict[str, ModelBundle],
        detector_config: dict[str, Any],
        default_regime: str,
        model_name: str,
        horizon: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            regime_bundles: regime label -> ModelBundle trained on that regime.
            detector_config: ``asdict(RegimeDetectorConfig)`` used in training.
            default_regime: Regime whose model serves bars of a regime that had
                too few training samples to get its own model.
            model_name: Base model name (e.g. "xgboost").
            horizon: Prediction horizon in bars.
            metrics: Training metrics recorded for the deploy manifest.
        """
        if default_regime not in regime_bundles:
            raise ValueError(f"default_regime '{default_regime}' has no bundle")
        self.regime_bundles = regime_bundles
        self.detector_config = dict(detector_config)
        self.default_regime = default_regime
        self.model_name = model_name
        self.horizon = horizon
        self.metrics = metrics or {}

    # -----------------------------------------------------------------
    # Regime detection
    # -----------------------------------------------------------------

    def detect_regimes(self, raw_df: pd.DataFrame, skip_cleaning: bool = False) -> pd.Series:
        """Regime label of every bar (at the training bar timeframe)."""
        from src.models.training.regime_detector import RegimeDetector

        graph = next(iter(self.regime_bundles.values())).preprocessing_graph
        bars = PreprocessingGraph._to_datetime_column(raw_df)
        if graph is not None and not skip_cleaning:
            bars = graph._resample_to_bar_timeframe(bars)
        bars = bars.set_index("datetime")[list(OHLCV_COLUMNS)]
        return RegimeDetector(**self.detector_config).detect(bars).regimes

    # -----------------------------------------------------------------
    # InferenceBundle protocol
    # -----------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray, calibrate: bool = True) -> PredictionResult:
        """Predict pre-computed features with the default regime's model.

        Pre-computed features carry no OHLCV to detect regimes from; use
        :meth:`predict_from_raw` for regime routing.
        """
        return self.regime_bundles[self.default_regime].predict(X, calibrate=calibrate)

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> PredictionResult:
        """Predict every bar with the model of that bar's regime."""
        regimes = self.detect_regimes(raw_df, skip_cleaning=skip_cleaning)
        outputs = {
            regime: bundle.predict_from_raw(
                raw_df, calibrate=calibrate, skip_cleaning=skip_cleaning
            )
            for regime, bundle in self.regime_bundles.items()
        }

        # All regime models share one architecture, so their timestamps match;
        # intersect anyway to be robust to warmup differences.
        stamps = [pd.DatetimeIndex(o.metadata["timestamps"]) for o in outputs.values()]
        common = stamps[0]
        for ts in stamps[1:]:
            common = common.intersection(ts)

        bar_regimes = regimes.reindex(common).fillna(self.default_regime).to_numpy()
        routed = np.where(
            np.isin(bar_regimes, list(self.regime_bundles)), bar_regimes, self.default_regime
        )

        n_classes = next(iter(outputs.values())).class_probabilities.shape[1]
        probabilities = np.empty((len(common), n_classes))
        predictions = np.empty(len(common), dtype=np.int64)
        for regime, output in outputs.items():
            rows = routed == regime
            if not rows.any():
                continue
            src = pd.DatetimeIndex(output.metadata["timestamps"]).get_indexer(common[rows])
            probabilities[rows] = output.class_probabilities[src]
            predictions[rows] = output.class_predictions[src]

        metadata: dict[str, Any] = {"timestamps": common, "regimes": routed}
        flags = next(iter(self.regime_bundles.values())).event_flags(
            raw_df, common, skip_cleaning=skip_cleaning
        )
        if flags is not None:
            metadata["is_event"] = flags
        return PredictionResult(
            class_predictions=predictions,
            class_probabilities=probabilities,
            confidence=probabilities.max(axis=1),
            metadata=metadata,
        )

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save the regime bundle (metadata + one ModelBundle per regime)."""
        path = Path(path)
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"Bundle already exists at {path}. Use overwrite=True.")
            shutil.rmtree(path)
        path.mkdir(parents=True)

        metadata = {
            "version": REGIME_BUNDLE_VERSION,
            "model_name": self.model_name,
            "horizon": self.horizon,
            "default_regime": self.default_regime,
            "detector_config": self.detector_config,
            "regimes": sorted(self.regime_bundles),
            "metrics": self.metrics,
        }
        with open(path / REGIME_BUNDLE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
        for regime, bundle in self.regime_bundles.items():
            bundle.save(path / REGIME_BUNDLES_DIR / regime)
        logger.info(f"Saved RegimeBundle ({self.model_name}, regimes={metadata['regimes']})")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RegimeBundle:
        """Load a regime bundle saved with :meth:`save`."""
        path = Path(path)
        metadata_path = path / REGIME_BUNDLE_METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing {REGIME_BUNDLE_METADATA_FILE} in {path}")
        with open(metadata_path) as f:
            metadata = json.load(f)
        return cls(
            regime_bundles={
                regime: ModelBundle.load(path / REGIME_BUNDLES_DIR / regime)
                for regime in metadata["regimes"]
            },
            detector_config=metadata["detector_config"],
            default_regime=metadata["default_regime"],
            model_name=metadata["model_name"],
            horizon=metadata["horizon"],
            metrics=metadata.get("metrics", {}),
        )

    def __repr__(self) -> str:
        return (
            f"RegimeBundle(model={self.model_name!r}, horizon={self.horizon}, "
            f"regimes={sorted(self.regime_bundles)}, default={self.default_regime!r})"
        )


__all__ = ["RegimeBundle", "REGIME_BUNDLE_VERSION", "REGIME_BUNDLE_METADATA_FILE"]
