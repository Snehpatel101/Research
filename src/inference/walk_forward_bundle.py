"""
WalkForwardBundle - Inference bundle wrapping the latest walk-forward window.

When walk-forward training produces N windows, only the latest-window
ModelBundle is used for live inference. WalkForwardBundle wraps that
bundle and preserves walk-forward metadata (n_windows, boundaries, etc.).

Implements the InferenceBundle protocol from src.core.protocols.

Usage:
    bundle = WalkForwardBundle(
        latest_bundle=model_bundle,
        n_windows=5,
        window_type="expanding",
        boundaries=[(0, 100), (20, 120), ...],
        window_metrics={"sharpe": [0.8, 1.1, 0.9, 1.3, 1.0]},
    )
    bundle.save("./bundles/wf_xgb_h20")

    # Load and predict
    bundle = WalkForwardBundle.load("./bundles/wf_xgb_h20")
    result = bundle.predict(X_new)
    result = bundle.predict_from_raw(raw_df)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.bundle import ModelBundle
from src.models.base import PredictionResult

logger = logging.getLogger(__name__)

WALK_FORWARD_BUNDLE_VERSION = "1.0.0"
WALK_FORWARD_METADATA_FILE = "walk_forward_metadata.json"
WALK_FORWARD_INNER_BUNDLE_DIR = "latest_bundle"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class WalkForwardMetadata:
    """Metadata for walk-forward bundle."""

    version: str
    n_windows: int
    window_type: str
    boundaries: list[tuple[int, int]]
    window_metrics: dict[str, list[float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_windows": self.n_windows,
            "window_type": self.window_type,
            "boundaries": self.boundaries,
            "window_metrics": self.window_metrics,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WalkForwardMetadata:
        boundaries = [tuple(b) for b in data.get("boundaries", [])]
        return cls(
            version=data["version"],
            n_windows=data["n_windows"],
            window_type=data.get("window_type", "expanding"),
            boundaries=boundaries,
            window_metrics=data.get("window_metrics", {}),
            extra=data.get("extra", {}),
        )


# =============================================================================
# WALK FORWARD BUNDLE
# =============================================================================


class WalkForwardBundle:
    """Inference bundle wrapping the latest walk-forward window.

    Delegates prediction to the inner ModelBundle while preserving
    walk-forward training metadata.

    Satisfies the InferenceBundle protocol.
    """

    def __init__(
        self,
        latest_bundle: ModelBundle,
        n_windows: int = 1,
        window_type: str = "expanding",
        boundaries: list[tuple[int, int]] | None = None,
        window_metrics: dict[str, list[float]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.latest_bundle = latest_bundle
        self.wf_metadata = WalkForwardMetadata(
            version=WALK_FORWARD_BUNDLE_VERSION,
            n_windows=n_windows,
            window_type=window_type,
            boundaries=boundaries or [],
            window_metrics=window_metrics or {},
            extra=extra or {},
        )

    # -----------------------------------------------------------------
    # InferenceBundle protocol
    # -----------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> PredictionResult:
        """Delegate prediction to the latest-window bundle.

        Args:
            X: Input features (DataFrame or array).
            calibrate: Whether to apply calibration.

        Returns:
            PredictionResult from the inner bundle.
        """
        return self.latest_bundle.predict(X, calibrate=calibrate)

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> PredictionResult:
        """End-to-end prediction from raw OHLCV data.

        Args:
            raw_df: DataFrame with raw OHLCV data.
            calibrate: Whether to apply calibration.
            skip_cleaning: If True, skip resampling step.

        Returns:
            PredictionResult from the inner bundle.
        """
        return self.latest_bundle.predict_from_raw(
            raw_df, calibrate=calibrate, skip_cleaning=skip_cleaning
        )

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save walk-forward bundle to disk.

        Structure:
            path/
                walk_forward_metadata.json
                latest_bundle/    (inner ModelBundle)

        Args:
            path: Directory path.
            overwrite: If True, overwrite existing.

        Returns:
            Path to saved bundle.
        """
        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Bundle already exists at {path}. Use overwrite=True to replace."
            )

        path.mkdir(parents=True, exist_ok=True)

        # Save walk-forward metadata
        meta_path = path / WALK_FORWARD_METADATA_FILE
        with open(meta_path, "w") as f:
            json.dump(self.wf_metadata.to_dict(), f, indent=2)

        # Save inner bundle
        inner_path = path / WALK_FORWARD_INNER_BUNDLE_DIR
        self.latest_bundle.save(inner_path, overwrite=overwrite)

        logger.info(
            "Saved WalkForwardBundle (%d windows, %s) to %s",
            self.wf_metadata.n_windows,
            self.wf_metadata.window_type,
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> WalkForwardBundle:
        """Load walk-forward bundle from disk.

        Args:
            path: Path to bundle directory.

        Returns:
            Loaded WalkForwardBundle.

        Raises:
            FileNotFoundError: If bundle or metadata is missing.
        """
        path = Path(path)

        if not path.is_dir():
            raise FileNotFoundError(f"WalkForwardBundle not found at {path}")

        # Load walk-forward metadata
        meta_path = path / WALK_FORWARD_METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {WALK_FORWARD_METADATA_FILE} in {path}")

        with open(meta_path) as f:
            wf_meta = WalkForwardMetadata.from_dict(json.load(f))

        # Load inner bundle
        inner_path = path / WALK_FORWARD_INNER_BUNDLE_DIR
        latest_bundle = ModelBundle.load(inner_path)

        bundle = cls(
            latest_bundle=latest_bundle,
            n_windows=wf_meta.n_windows,
            window_type=wf_meta.window_type,
            boundaries=wf_meta.boundaries,
            window_metrics=wf_meta.window_metrics,
            extra=wf_meta.extra,
        )
        bundle.wf_metadata = wf_meta

        logger.info(
            "Loaded WalkForwardBundle (%d windows, %s) from %s",
            wf_meta.n_windows,
            wf_meta.window_type,
            path,
        )
        return bundle

    def __repr__(self) -> str:
        return (
            f"WalkForwardBundle(n_windows={self.wf_metadata.n_windows}, "
            f"window_type={self.wf_metadata.window_type!r}, "
            f"inner={self.latest_bundle!r})"
        )


__all__ = [
    "WalkForwardBundle",
    "WalkForwardMetadata",
    "WALK_FORWARD_BUNDLE_VERSION",
]
