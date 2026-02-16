"""
RegimeBundle - Per-regime model routing for inference.

Routes predictions through regime-specific ModelBundles based on the
current market regime detected by a RegimeDetector.

Implements the InferenceBundle protocol from src.core.protocols.

Usage:
    from src.inference import ModelBundle
    from src.inference.regime_detector import RegimeDetector
    from src.inference.regime_bundle import RegimeBundle

    detector = RegimeDetector()
    bundles = {
        "high_vol_trending": ModelBundle.load("./bundles/xgb_hvt"),
        "low_vol_ranging": ModelBundle.load("./bundles/lgbm_lvr"),
    }
    bundle = RegimeBundle(
        regime_detector=detector,
        regime_bundles=bundles,
        default_regime="low_vol_ranging",
    )
    result = bundle.predict_from_raw(raw_df)
    bundle.save("./bundles/regime_h20")
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.bundle import ModelBundle
from src.inference.regime_detector import RegimeDetector
from src.models.base import PredictionResult

logger = logging.getLogger(__name__)

REGIME_BUNDLE_VERSION = "1.0.0"
REGIME_BUNDLE_METADATA_FILE = "regime_bundle_metadata.json"
REGIME_DETECTOR_DIR = "regime_detector"
REGIME_BUNDLES_DIR = "regime_bundles"


# =============================================================================
# REGIME BUNDLE
# =============================================================================


class RegimeBundle:
    """Per-regime model routing bundle.

    Detects the current market regime from raw OHLCV data, then routes
    prediction to the corresponding regime-specific ModelBundle.

    Falls back to default_regime when detection fails or the detected
    regime has no dedicated bundle.

    Satisfies the InferenceBundle protocol.
    """

    def __init__(
        self,
        regime_detector: RegimeDetector,
        regime_bundles: dict[str, ModelBundle],
        default_regime: str = "low_vol_ranging",
    ) -> None:
        """
        Args:
            regime_detector: Detector that classifies market regime.
            regime_bundles: Mapping of regime_name -> ModelBundle.
            default_regime: Fallback regime if detection fails or regime
                is missing from regime_bundles.
        """
        self.regime_detector = regime_detector
        self.regime_bundles = regime_bundles
        self.default_regime = default_regime

    def _resolve_bundle(self, regime: str) -> ModelBundle:
        """Resolve regime to its ModelBundle, falling back if needed."""
        if regime in self.regime_bundles:
            return self.regime_bundles[regime]

        logger.warning(
            "No bundle for regime '%s', falling back to default '%s'",
            regime,
            self.default_regime,
        )
        if self.default_regime in self.regime_bundles:
            return self.regime_bundles[self.default_regime]

        # Last resort: use the first available bundle
        first_key = next(iter(self.regime_bundles))
        logger.warning(
            "Default regime '%s' also missing, using first available bundle '%s'",
            self.default_regime,
            first_key,
        )
        return self.regime_bundles[first_key]

    # -----------------------------------------------------------------
    # InferenceBundle protocol
    # -----------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> PredictionResult:
        """Predict using the default regime bundle.

        When calling predict() with pre-processed features there is no
        raw OHLCV data to detect a regime from, so the default regime
        bundle is used.

        Args:
            X: Input features (DataFrame or array).
            calibrate: Whether to apply calibration.

        Returns:
            PredictionResult from the default regime bundle.
        """
        bundle = self._resolve_bundle(self.default_regime)
        return bundle.predict(X, calibrate=calibrate)

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> PredictionResult:
        """Detect regime then route prediction to the correct bundle.

        Args:
            raw_df: DataFrame with raw OHLCV data.
            calibrate: Whether to apply calibration.
            skip_cleaning: If True, skip resampling step.

        Returns:
            PredictionResult from the regime-specific bundle.
        """
        regime = self.regime_detector.detect(raw_df)
        logger.info("Regime detected: %s", regime)

        bundle = self._resolve_bundle(regime)
        return bundle.predict_from_raw(raw_df, calibrate=calibrate, skip_cleaning=skip_cleaning)

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save regime bundle to disk.

        Structure:
            path/
                regime_bundle_metadata.json
                regime_detector/
                regime_bundles/
                    <regime_name>/   (each a ModelBundle)

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
        meta: dict[str, Any] = {
            "version": REGIME_BUNDLE_VERSION,
            "default_regime": self.default_regime,
            "regime_names": list(self.regime_bundles.keys()),
        }
        with open(path / REGIME_BUNDLE_METADATA_FILE, "w") as f:
            json.dump(meta, f, indent=2)

        # Save detector
        self.regime_detector.save(path / REGIME_DETECTOR_DIR)

        # Save per-regime bundles
        bundles_dir = path / REGIME_BUNDLES_DIR
        bundles_dir.mkdir(parents=True, exist_ok=True)
        for regime_name, bundle in self.regime_bundles.items():
            bundle.save(bundles_dir / regime_name, overwrite=overwrite)

        logger.info(
            "Saved RegimeBundle (%d regimes, default=%s) to %s",
            len(self.regime_bundles),
            self.default_regime,
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> RegimeBundle:
        """Load regime bundle from disk.

        Args:
            path: Path to bundle directory.

        Returns:
            Loaded RegimeBundle.

        Raises:
            FileNotFoundError: If bundle or components are missing.
        """
        path = Path(path)

        if not path.is_dir():
            raise FileNotFoundError(f"RegimeBundle not found at {path}")

        # Load metadata
        meta_path = path / REGIME_BUNDLE_METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {REGIME_BUNDLE_METADATA_FILE} in {path}")

        with open(meta_path) as f:
            meta = json.load(f)

        default_regime = meta.get("default_regime", "low_vol_ranging")
        regime_names: list[str] = meta.get("regime_names", [])

        # Load detector
        detector = RegimeDetector.load(path / REGIME_DETECTOR_DIR)

        # Load per-regime bundles
        regime_bundles: dict[str, ModelBundle] = {}
        bundles_dir = path / REGIME_BUNDLES_DIR
        for regime_name in regime_names:
            regime_path = bundles_dir / regime_name
            if regime_path.is_dir():
                regime_bundles[regime_name] = ModelBundle.load(regime_path)
                logger.debug("Loaded bundle for regime '%s'", regime_name)
            else:
                logger.warning("Bundle for regime '%s' not found at %s", regime_name, regime_path)

        logger.info(
            "Loaded RegimeBundle (%d regimes, default=%s) from %s",
            len(regime_bundles),
            default_regime,
            path,
        )

        return cls(
            regime_detector=detector,
            regime_bundles=regime_bundles,
            default_regime=default_regime,
        )

    def __repr__(self) -> str:
        return (
            f"RegimeBundle(regimes={list(self.regime_bundles.keys())}, "
            f"default={self.default_regime!r})"
        )


__all__ = [
    "RegimeBundle",
    "REGIME_BUNDLE_VERSION",
]
