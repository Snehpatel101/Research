"""
RegimeDetector - Market regime classification from raw OHLCV data.

Classifies market regimes using volatility percentile and ADX trend strength.
Used by RegimeBundle to route predictions to regime-specific models.

Usage:
    detector = RegimeDetector(config=RegimeDetectorConfig())
    regime = detector.detect(df)  # e.g. "high_vol_trending"
    detector.save("./detectors/regime_v1")

    # Load later
    detector = RegimeDetector.load("./detectors/regime_v1")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_DETECTOR_VERSION = "1.0.0"
REGIME_DETECTOR_FILE = "regime_detector.json"


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class RegimeDetectorConfig:
    """Configuration for regime detection.

    Attributes:
        volatility_window: Rolling window for realized volatility calculation.
        volatility_threshold: Percentile boundary between low/high volatility.
        adx_window: Window for ADX (Average Directional Index) calculation.
        adx_threshold: ADX boundary between ranging and trending.
        regime_names: Mapping of (vol_bucket, trend_bucket) to regime name.
    """

    volatility_window: int = 20
    volatility_threshold: float = 50.0
    adx_window: int = 14
    adx_threshold: float = 25.0
    default_regime: str = "low_vol_ranging"
    regime_names: dict[str, str] = field(
        default_factory=lambda: {
            "low_vol_ranging": "low_vol_ranging",
            "low_vol_trending": "low_vol_trending",
            "high_vol_ranging": "high_vol_ranging",
            "high_vol_trending": "high_vol_trending",
        }
    )


# =============================================================================
# REGIME DETECTOR
# =============================================================================


class RegimeDetector:
    """Detect market regime from raw OHLCV data.

    Combines volatility percentile and ADX trend strength into one of
    four regimes: low_vol_ranging, low_vol_trending, high_vol_ranging,
    high_vol_trending.

    Attributes:
        config: Detector configuration.
        version: Detector version string.
    """

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        self.config = config or RegimeDetectorConfig()
        self.version = REGIME_DETECTOR_VERSION

    def detect(self, df: pd.DataFrame) -> str:
        """Classify the current market regime.

        Uses the tail of the DataFrame (last row after rolling computations)
        to determine the regime.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume].
                Must have enough rows for rolling calculations.

        Returns:
            Regime name string (e.g. "high_vol_trending").
        """
        if len(df) < max(self.config.volatility_window, self.config.adx_window) + 1:
            logger.warning(
                "Insufficient data for regime detection (%d rows), "
                "returning default regime '%s'",
                len(df),
                self.config.default_regime,
            )
            return self.config.default_regime

        vol_bucket = self._classify_volatility(df)
        trend_bucket = self._classify_trend(df)

        key = f"{vol_bucket}_{trend_bucket}"
        regime = self.config.regime_names.get(key, self.config.default_regime)

        logger.debug("Regime detected: %s (vol=%s, trend=%s)", regime, vol_bucket, trend_bucket)
        return regime

    def detect_series(self, df: pd.DataFrame) -> pd.Series:
        """Classify regime for every row in the DataFrame.

        Returns a Series aligned to df.index with regime labels.
        Rows with insufficient history are labeled with the default regime.

        Args:
            df: OHLCV DataFrame.

        Returns:
            pd.Series of regime name strings.
        """
        close = df["close"]
        log_returns = np.log(close / close.shift(1))
        vol = log_returns.rolling(window=self.config.volatility_window).std() * np.sqrt(252)
        vol_pct = vol.rank(pct=True) * 100.0

        adx = self._compute_adx_series(df)

        regimes = pd.Series(self.config.default_regime, index=df.index)
        valid = vol_pct.notna() & adx.notna()

        for idx in df.index[valid]:
            vb = "low_vol" if vol_pct.loc[idx] < self.config.volatility_threshold else "high_vol"
            tb = "ranging" if adx.loc[idx] < self.config.adx_threshold else "trending"
            key = f"{vb}_{tb}"
            regimes.loc[idx] = self.config.regime_names.get(key, self.config.default_regime)

        return regimes

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _classify_volatility(self, df: pd.DataFrame) -> str:
        """Return 'low_vol' or 'high_vol' based on rolling volatility percentile."""
        close = df["close"]
        log_returns = np.log(close / close.shift(1))
        vol = log_returns.rolling(window=self.config.volatility_window).std() * np.sqrt(252)
        vol_pct = vol.rank(pct=True) * 100.0
        latest = vol_pct.iloc[-1]
        if np.isnan(latest):
            return "low_vol"
        return "low_vol" if latest < self.config.volatility_threshold else "high_vol"

    def _classify_trend(self, df: pd.DataFrame) -> str:
        """Return 'ranging' or 'trending' based on ADX value."""
        adx_series = self._compute_adx_series(df)
        latest = adx_series.iloc[-1]
        if np.isnan(latest):
            return "ranging"
        return "ranging" if latest < self.config.adx_threshold else "trending"

    def _compute_adx_series(self, df: pd.DataFrame) -> pd.Series:
        """Compute ADX indicator series."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        window = self.config.adx_window

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=window, min_periods=window).mean()
        plus_di = 100 * (plus_dm.ewm(span=window, min_periods=window).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=window, min_periods=window).mean() / atr)

        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx: pd.Series = dx.ewm(span=window, min_periods=window).mean()
        return adx

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save detector configuration to JSON.

        Args:
            path: Directory to save into.

        Returns:
            Path to the saved directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "version": self.version,
            "config": asdict(self.config),
        }
        with open(path / REGIME_DETECTOR_FILE, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved regime detector to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> RegimeDetector:
        """Load detector from a saved directory.

        Args:
            path: Directory containing regime_detector.json.

        Returns:
            Loaded RegimeDetector instance.

        Raises:
            FileNotFoundError: If the detector file is missing.
        """
        path = Path(path)
        detector_file = path / REGIME_DETECTOR_FILE

        if not detector_file.exists():
            raise FileNotFoundError(f"Regime detector not found at {detector_file}")

        with open(detector_file) as f:
            data = json.load(f)

        config = RegimeDetectorConfig(**data["config"])
        detector = cls(config=config)
        detector.version = data.get("version", REGIME_DETECTOR_VERSION)

        logger.info("Loaded regime detector from %s", path)
        return detector

    def __repr__(self) -> str:
        return (
            f"RegimeDetector(vol_window={self.config.volatility_window}, "
            f"adx_window={self.config.adx_window}, "
            f"regimes={list(self.config.regime_names.values())})"
        )


__all__ = [
    "RegimeDetector",
    "RegimeDetectorConfig",
    "REGIME_DETECTOR_VERSION",
]
