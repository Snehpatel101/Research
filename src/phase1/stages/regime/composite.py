"""
Composite regime detection combining multiple regime detectors.

This module provides a unified interface for combining volatility,
trend, and market structure regimes into composite regime features.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .structure import MarketStructureDetector
from .trend import TrendRegimeDetector
from .volatility import VolatilityRegimeDetector

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class CompositeRegimeResult:
    """Results from composite regime detection.

    Attributes:
        regimes: DataFrame with regime columns
        summaries: Dict of regime type to summary statistics
        detector_configs: Dict of detector configurations used
    """

    regimes: pd.DataFrame
    summaries: dict[str, dict[str, Any]]
    detector_configs: dict[str, dict[str, Any]]


class CompositeRegimeDetector:
    """
    Combine multiple regime detectors into a unified system.

    This class orchestrates multiple regime detectors and provides:
    1. Regime columns for features (volatility_regime, trend_regime, structure_regime)
    2. Composite regime labels combining multiple dimensions
    3. Summary statistics for each regime type

    Integration Options:
    - Regime as Feature: Add regime columns to feature DataFrame for model training
    - Regime as Filter: Use regime to select which model to apply
    - Regime-adaptive Barriers: Adjust triple-barrier parameters per regime

    Attributes:
        volatility_detector: Volatility regime detector
        trend_detector: Trend regime detector
        structure_detector: Market structure detector

    Example:
        >>> composite = CompositeRegimeDetector.from_config(REGIME_CONFIG)
        >>> result = composite.detect_all(df)
        >>> df_with_regimes = result.regimes
        >>> print(df_with_regimes[['volatility_regime', 'trend_regime']].head())
    """

    def __init__(
        self,
        volatility_detector: VolatilityRegimeDetector | None = None,
        trend_detector: TrendRegimeDetector | None = None,
        structure_detector: MarketStructureDetector | None = None,
    ):
        """
        Initialize composite regime detector.

        Args:
            volatility_detector: Volatility regime detector instance
            trend_detector: Trend regime detector instance
            structure_detector: Market structure detector instance
        """
        self.volatility_detector = volatility_detector
        self.trend_detector = trend_detector
        self.structure_detector = structure_detector

        # Track which detectors are enabled
        self._enabled_detectors: list[str] = []
        if volatility_detector:
            self._enabled_detectors.append("volatility")
        if trend_detector:
            self._enabled_detectors.append("trend")
        if structure_detector:
            self._enabled_detectors.append("structure")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CompositeRegimeDetector":
        """
        Create composite detector from configuration dict.

        Expected config structure:
        {
            'volatility': {
                'enabled': True,
                'atr_period': 14,
                'lookback': 100,
                'low_percentile': 25,
                'high_percentile': 75
            },
            'trend': {
                'enabled': True,
                'adx_period': 14,
                'sma_period': 50,
                'adx_threshold': 25
            },
            'structure': {
                'enabled': True,
                'lookback': 100,
                'min_lag': 2,
                'max_lag': 20,
                'mean_reverting_threshold': 0.4,
                'trending_threshold': 0.6
            }
        }

        Args:
            config: Configuration dictionary

        Returns:
            Configured CompositeRegimeDetector
        """
        volatility_detector = None
        trend_detector = None
        structure_detector = None

        # Volatility detector
        vol_config = config.get("volatility", {})
        if vol_config.get("enabled", True):
            volatility_detector = VolatilityRegimeDetector(
                atr_period=vol_config.get("atr_period", 14),
                lookback=vol_config.get("lookback", 100),
                low_percentile=vol_config.get("low_percentile", 25.0),
                high_percentile=vol_config.get("high_percentile", 75.0),
                atr_column=vol_config.get("atr_column"),
            )

        # Trend detector
        trend_config = config.get("trend", {})
        if trend_config.get("enabled", True):
            trend_detector = TrendRegimeDetector(
                adx_period=trend_config.get("adx_period", 14),
                sma_period=trend_config.get("sma_period", 50),
                adx_threshold=trend_config.get("adx_threshold", 25.0),
                adx_column=trend_config.get("adx_column"),
                sma_column=trend_config.get("sma_column"),
            )

        # Structure detector
        struct_config = config.get("structure", {})
        if struct_config.get("enabled", True):
            structure_detector = MarketStructureDetector(
                lookback=struct_config.get("lookback", 100),
                min_lag=struct_config.get("min_lag", 2),
                max_lag=struct_config.get("max_lag", 20),
                mean_reverting_threshold=struct_config.get("mean_reverting_threshold", 0.4),
                trending_threshold=struct_config.get("trending_threshold", 0.6),
            )

        return cls(
            volatility_detector=volatility_detector,
            trend_detector=trend_detector,
            structure_detector=structure_detector,
        )

    @classmethod
    def with_defaults(cls) -> "CompositeRegimeDetector":
        """
        Create composite detector with default settings.

        Returns:
            CompositeRegimeDetector with default configuration
        """
        return cls(
            volatility_detector=VolatilityRegimeDetector(),
            trend_detector=TrendRegimeDetector(),
            structure_detector=MarketStructureDetector(),
        )

    def get_required_columns(self) -> list[str]:
        """Get all required columns for enabled detectors."""
        required = set()

        if self.volatility_detector:
            required.update(self.volatility_detector.get_required_columns())
        if self.trend_detector:
            required.update(self.trend_detector.get_required_columns())
        if self.structure_detector:
            required.update(self.structure_detector.get_required_columns())

        return list(required)

    def detect_all(self, df: pd.DataFrame) -> CompositeRegimeResult:
        """
        Run all enabled regime detectors.

        ANTI-LOOKAHEAD: All regime columns are shifted by 1 bar before being
        returned. This ensures that the regime classification at bar N only
        uses information available up to bar N-1, preventing lookahead bias
        when regime features are used in model training.

        Args:
            df: DataFrame with OHLC data

        Returns:
            CompositeRegimeResult with regime columns (shifted by 1 bar) and summaries
        """
        regimes = pd.DataFrame(index=df.index)
        summaries: dict[str, dict[str, Any]] = {}
        configs: dict[str, dict[str, Any]] = {}

        # Volatility regime
        if self.volatility_detector:
            try:
                vol_regimes = self.volatility_detector.detect(df)
                regimes["volatility_regime"] = vol_regimes
                summaries["volatility"] = self.volatility_detector.get_regime_summary(vol_regimes)
                configs["volatility"] = {
                    "atr_period": self.volatility_detector.atr_period,
                    "lookback": self.volatility_detector.lookback,
                    "low_percentile": self.volatility_detector.low_percentile,
                    "high_percentile": self.volatility_detector.high_percentile,
                }
                logger.info(f"Volatility regime: {summaries['volatility']['distribution']}")
            except Exception as e:
                logger.warning(f"Failed to detect volatility regime: {e}")
                regimes["volatility_regime"] = np.nan

        # Trend regime
        if self.trend_detector:
            try:
                trend_regimes = self.trend_detector.detect(df)
                regimes["trend_regime"] = trend_regimes
                summaries["trend"] = self.trend_detector.get_regime_summary(trend_regimes)
                configs["trend"] = {
                    "adx_period": self.trend_detector.adx_period,
                    "sma_period": self.trend_detector.sma_period,
                    "adx_threshold": self.trend_detector.adx_threshold,
                }
                logger.info(f"Trend regime: {summaries['trend']['distribution']}")
            except Exception as e:
                logger.warning(f"Failed to detect trend regime: {e}")
                regimes["trend_regime"] = np.nan

        # Structure regime
        if self.structure_detector:
            try:
                struct_regimes = self.structure_detector.detect(df)
                regimes["structure_regime"] = struct_regimes
                summaries["structure"] = self.structure_detector.get_regime_summary(struct_regimes)
                configs["structure"] = {
                    "lookback": self.structure_detector.lookback,
                    "min_lag": self.structure_detector.min_lag,
                    "max_lag": self.structure_detector.max_lag,
                    "mean_reverting_threshold": self.structure_detector.mean_reverting_threshold,
                    "trending_threshold": self.structure_detector.trending_threshold,
                }
                logger.info(f"Structure regime: {summaries['structure']['distribution']}")
            except Exception as e:
                logger.warning(f"Failed to detect structure regime: {e}")
                regimes["structure_regime"] = np.nan

        # ANTI-LOOKAHEAD: Shift all regime columns by 1 bar
        # This ensures regime at bar N only uses data from bars 0..N-1
        regime_columns = [col for col in regimes.columns if col.endswith("_regime")]
        if regime_columns:
            logger.info(f"ANTI-LOOKAHEAD: Shifting {len(regime_columns)} regime columns by 1 bar")
            for col in regime_columns:
                regimes[col] = regimes[col].shift(1)
                # First row becomes NaN after shift - this is expected and correct
                # Downstream code should handle NaN appropriately (drop or fill)

        return CompositeRegimeResult(regimes=regimes, summaries=summaries, detector_configs=configs)

    def add_regime_columns(self, df: pd.DataFrame, encode_numeric: bool = False) -> pd.DataFrame:
        """
        Add regime columns to DataFrame in-place style (returns copy).

        ANTI-LOOKAHEAD: Regime columns are shifted by 1 bar internally
        (see detect_all). The first row of each regime column will be NaN.

        Args:
            df: Input DataFrame with OHLC data
            encode_numeric: If True, also add numeric encoded columns

        Returns:
            DataFrame with regime columns added (shifted by 1 bar)
        """
        result = self.detect_all(df)
        df_out = df.copy()

        # Add regime columns
        for col in result.regimes.columns:
            df_out[col] = result.regimes[col]

            # Add numeric encoding if requested
            if encode_numeric:
                numeric_col = f"{col}_encoded"
                df_out[numeric_col] = pd.Categorical(result.regimes[col]).codes
                # Handle NaN (-1 from codes)
                df_out.loc[result.regimes[col].isna(), numeric_col] = np.nan

        return df_out

    def create_composite_label(self, df: pd.DataFrame, separator: str = "_") -> pd.Series:
        """
        Create a composite regime label combining all regimes.

        Example output: "high_uptrend_trending"

        Args:
            df: DataFrame with OHLC data
            separator: String to join regime labels

        Returns:
            Series with composite regime labels
        """
        result = self.detect_all(df)

        # Combine non-null regime labels
        composite = pd.Series(index=df.index, dtype="object")

        for idx in df.index:
            parts = []
            for col in result.regimes.columns:
                val = result.regimes.loc[idx, col]
                if pd.notna(val):
                    parts.append(str(val))

            if parts:
                composite[idx] = separator.join(parts)
            else:
                composite[idx] = np.nan

        return composite


def add_regime_features_to_dataframe(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    feature_metadata: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Convenience function to add all regime features to a DataFrame.

    This function is designed to integrate with the existing feature
    engineering pipeline.

    ANTI-LOOKAHEAD: Regime columns are shifted by 1 bar to prevent lookahead
    bias. The regime at bar N reflects the regime computed using data from
    bars 0..N-1 only.

    Args:
        df: DataFrame with OHLC data
        config: Optional regime configuration dict
        feature_metadata: Optional dict to store feature descriptions

    Returns:
        DataFrame with regime columns added (shifted by 1 bar)
    """
    if config is None:
        detector = CompositeRegimeDetector.with_defaults()
    else:
        detector = CompositeRegimeDetector.from_config(config)

    df_out = detector.add_regime_columns(df, encode_numeric=True)

    # Update feature metadata if provided
    # Note: All regime columns are shifted by 1 bar to prevent lookahead
    if feature_metadata is not None:
        if "volatility_regime" in df_out.columns:
            feature_metadata["volatility_regime"] = (
                "Volatility regime (low/normal/high) based on ATR percentile (shifted by 1 bar)"
            )
            feature_metadata["volatility_regime_encoded"] = (
                "Volatility regime numeric encoding (shifted by 1 bar)"
            )

        if "trend_regime" in df_out.columns:
            feature_metadata["trend_regime"] = (
                "Trend regime (uptrend/downtrend/sideways) based on ADX + SMA (shifted by 1 bar)"
            )
            feature_metadata["trend_regime_encoded"] = (
                "Trend regime numeric encoding (shifted by 1 bar)"
            )

        if "structure_regime" in df_out.columns:
            feature_metadata["structure_regime"] = (
                "Market structure (mean_reverting/random/trending) based on Hurst (shifted by 1 bar)"
            )
            feature_metadata["structure_regime_encoded"] = (
                "Structure regime numeric encoding (shifted by 1 bar)"
            )

    return df_out


__all__ = [
    "CompositeRegimeDetector",
    "CompositeRegimeResult",
    "add_regime_features_to_dataframe",
]
