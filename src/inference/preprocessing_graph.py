"""
PreprocessingGraph - Serializable preprocessing pipeline for train/serve parity.

The graph records exactly how training turned raw OHLCV into model features
and replays it at inference time:

1. Bar timeframe — raw bars finer than the training bar timeframe are
   resampled to it (e.g. 1min -> 5min); coarser raw bars are rejected.
2. Feature engineering — the FeatureEngineer spec captured at training time
   (``FeatureEngineer.to_spec()``) is rebuilt with ``FeatureEngineer.from_spec``
   and ``compute_features()`` runs the *same code* training ran. There is no
   second feature implementation to drift out of sync.
3. Column selection — only the trained feature columns are returned, and
   warmup rows with NaN in those columns are dropped.
4. Optional scaling with the fitted training scaler.

Usage:
    # During training (BundleBuilder)
    graph = PreprocessingGraph.from_feature_pipeline(feature_pipeline, feature_columns)
    graph.save(bundle_path / "preprocessing_graph.json")

    # During inference
    graph = PreprocessingGraph.load(bundle_path / "preprocessing_graph.json")
    features = graph.transform(raw_ohlcv_df, skip_scaling=True)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.constants import OHLCV_COLUMNS
from src.core.utils.safe_pickle import safe_pickle_dump, safe_pickle_load

logger = logging.getLogger(__name__)

# 2.0.0: graph delegates to FeatureEngineer.compute_features (single feature engine)
PREPROCESSING_GRAPH_VERSION = "2.0.0"

SCALER_FILE = "graph_scaler.pkl"


@dataclass
class PreprocessingGraphConfig:
    """Everything needed to rebuild training features from raw OHLCV."""

    version: str = PREPROCESSING_GRAPH_VERSION
    created_at: str = ""
    horizon: int = 20
    symbol: str = ""
    # Bar timeframe the model was trained on (features/labels live on these bars)
    bar_timeframe: str = "5min"
    # FeatureEngineer.to_spec() captured at training time
    feature_engineering: dict[str, Any] = field(default_factory=dict)
    # Trained feature columns, in model input order
    feature_columns: list[str] = field(default_factory=list)
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "horizon": self.horizon,
            "symbol": self.symbol,
            "bar_timeframe": self.bar_timeframe,
            "feature_engineering": dict(self.feature_engineering),
            "feature_columns": list(self.feature_columns),
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingGraphConfig:
        return cls(
            version=data.get("version", ""),
            created_at=data.get("created_at", ""),
            horizon=data.get("horizon", 20),
            symbol=data.get("symbol", ""),
            bar_timeframe=data.get("bar_timeframe", "5min"),
            feature_engineering=dict(data.get("feature_engineering") or {}),
            feature_columns=list(data.get("feature_columns") or []),
            config_hash=data.get("config_hash", ""),
        )

    def compute_hash(self) -> str:
        """Hash of the configuration (excluding timestamps) for validation."""
        hash_data = self.to_dict()
        hash_data.pop("created_at", None)
        hash_data.pop("config_hash", None)
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


class PreprocessingGraph:
    """
    Serializable raw-OHLCV -> model-features transform for train/serve parity.

    Attributes:
        config: PreprocessingGraphConfig with all settings
    """

    def __init__(self, config: PreprocessingGraphConfig) -> None:
        self.config = config
        self._scaler: Any = None

    @classmethod
    def from_feature_pipeline(
        cls,
        feature_pipeline: dict[str, Any],
        feature_columns: list[str] | None = None,
        symbol: str = "",
        horizon: int = 20,
    ) -> PreprocessingGraph:
        """
        Create from the feature pipeline recorded by MLFactory.

        Args:
            feature_pipeline: ``{"bar_timeframe": str, "engineer": FeatureEngineer.to_spec()}``
            feature_columns: Trained feature columns (model input order)
            symbol: Trading symbol
            horizon: Prediction horizon
        """
        config = PreprocessingGraphConfig(
            created_at=datetime.now().isoformat(),
            horizon=horizon,
            symbol=symbol,
            bar_timeframe=feature_pipeline["bar_timeframe"],
            feature_engineering=dict(feature_pipeline["engineer"]),
            feature_columns=list(feature_columns or []),
        )
        config.config_hash = config.compute_hash()
        return cls(config)

    def with_feature_columns(self, feature_columns: list[str]) -> PreprocessingGraph:
        """Copy of this graph restricted to a model's own feature columns."""
        config = PreprocessingGraphConfig.from_dict(self.config.to_dict())
        config.feature_columns = list(feature_columns)
        config.config_hash = config.compute_hash()
        graph = PreprocessingGraph(config)
        graph._scaler = self._scaler
        return graph

    def set_scaler(self, scaler: Any) -> None:
        """Set the fitted scaler used when transform(skip_scaling=False)."""
        self._scaler = scaler

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        raw_df: pd.DataFrame,
        skip_cleaning: bool = False,
        skip_scaling: bool = False,
    ) -> pd.DataFrame:
        """
        Turn raw OHLCV bars into the trained feature matrix.

        Args:
            raw_df: Raw OHLCV with a DatetimeIndex or ``datetime`` column.
            skip_cleaning: If True, assume bars are already at the training
                bar timeframe and skip resampling.
            skip_scaling: If True, return unscaled features.

        Returns:
            DataFrame indexed by bar timestamp with the trained feature
            columns (warmup rows with NaN features dropped).

        Raises:
            ValueError: On missing OHLCV columns, bars coarser than the
                training timeframe, or features the spec cannot reproduce.
        """
        if not self.config.feature_engineering:
            raise ValueError(
                f"Preprocessing graph version {self.config.version or 'unknown'} has no "
                "feature_engineering spec — it predates train/serve parity. "
                "Retrain the model to produce a deployable bundle."
            )

        df = self._to_datetime_column(raw_df)
        if not skip_cleaning:
            df = self._resample_to_bar_timeframe(df)

        from src.data.pipeline.stages.features.engineer import FeatureEngineer

        engineer = FeatureEngineer.from_spec(self.config.feature_engineering)
        df, _stats = engineer.compute_features(df)
        df = df.set_index("datetime")

        columns = self.config.feature_columns
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Preprocessing could not reproduce {len(missing)} trained feature "
                    f"columns (e.g. {missing[:5]}). Provide more history: MTF features "
                    f"need >= {engineer.mtf_min_rows} bars and wavelets need "
                    f">= {engineer.wavelet_window} bars."
                )
            df = df[columns]
        df = df.dropna()

        if not skip_scaling and self._scaler is not None:
            df = pd.DataFrame(
                self._scaler.transform(df.to_numpy()), index=df.index, columns=df.columns
            )
        return df

    @staticmethod
    def _to_datetime_column(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV input and return a frame with a sorted ``datetime`` column."""
        df = raw_df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        if "datetime" not in df.columns:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have a 'datetime' column or DatetimeIndex")
            df = df.rename_axis("datetime").reset_index()
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)

    def _resample_to_bar_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample raw bars to the training bar timeframe (same code as training)."""
        from src.core.common.timeframes import detect_timeframe, get_timeframe_minutes
        from src.data.pipeline.stages.clean.utils import resample_ohlcv

        bar_tf = self.config.bar_timeframe
        source_tf = detect_timeframe(df)
        if source_tf is None or source_tf == bar_tf:
            return df

        if get_timeframe_minutes(source_tf) > get_timeframe_minutes(bar_tf):
            raise ValueError(
                f"Raw bars are {source_tf} but the model was trained on {bar_tf} bars; "
                "inference needs bars at or finer than the training timeframe."
            )
        return resample_ohlcv(df[["datetime", *OHLCV_COLUMNS]], bar_tf, include_metadata=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save graph configuration (JSON) and scaler (if any) next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.config.config_hash = self.config.compute_hash()
        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        if self._scaler is not None:
            safe_pickle_dump(self._scaler, path.parent / SCALER_FILE)
        logger.info(f"Saved preprocessing graph to {path}")

    @classmethod
    def load(cls, path: Path) -> PreprocessingGraph:
        """Load a graph saved with :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing graph not found at {path}")
        with open(path) as f:
            config = PreprocessingGraphConfig.from_dict(json.load(f))

        expected_hash = config.compute_hash()
        if config.config_hash and config.config_hash != expected_hash:
            logger.warning(
                f"Preprocessing graph hash mismatch (expected {expected_hash}, got "
                f"{config.config_hash}); the graph file may have been modified."
            )

        graph = cls(config)
        scaler_path = path.parent / SCALER_FILE
        if scaler_path.exists():
            graph._scaler = safe_pickle_load(scaler_path)
        return graph

    def to_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingGraph:
        return cls(PreprocessingGraphConfig.from_dict(data))

    def validate(self) -> dict[str, Any]:
        """Report structural problems with the graph."""
        issues: list[str] = []
        if self.config.version != PREPROCESSING_GRAPH_VERSION:
            issues.append(
                f"Version mismatch: graph {self.config.version} vs "
                f"current {PREPROCESSING_GRAPH_VERSION}"
            )
        if not self.config.feature_engineering:
            issues.append("No feature_engineering spec (graph predates train/serve parity)")
        if not self.config.feature_columns:
            issues.append("No feature columns specified")
        expected_hash = self.config.compute_hash()
        if self.config.config_hash and self.config.config_hash != expected_hash:
            issues.append("Configuration hash mismatch")
        return {
            "valid": not issues,
            "issues": issues,
            "version": self.config.version,
            "horizon": self.config.horizon,
            "symbol": self.config.symbol,
            "bar_timeframe": self.config.bar_timeframe,
            "n_features": len(self.config.feature_columns),
            "has_scaler": self._scaler is not None,
        }

    def __repr__(self) -> str:
        return (
            f"PreprocessingGraph(version={self.config.version}, "
            f"symbol={self.config.symbol}, horizon={self.config.horizon}, "
            f"bar_timeframe={self.config.bar_timeframe}, "
            f"features={len(self.config.feature_columns)})"
        )


# Constants for bundle integration
PREPROCESSING_GRAPH_FILE = "preprocessing_graph.json"

__all__ = [
    "PreprocessingGraph",
    "PreprocessingGraphConfig",
    "PREPROCESSING_GRAPH_VERSION",
    "PREPROCESSING_GRAPH_FILE",
]
