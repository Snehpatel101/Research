"""
Time Bar Builder.

Standard time-based OHLCV resampling (wrapper around existing resample_ohlcv).
This provides a unified interface with volume/dollar bar builders.

Usage:
    builder = TimeBarBuilder(target_timeframe='5min')
    bars = builder.build(df)
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BarBuilderRegistry, BaseBarBuilder

logger = logging.getLogger(__name__)


# Timeframe to pandas frequency mapping
TIMEFRAME_FREQ_MAP = {
    "1min": "1min",
    "5min": "5min",
    "10min": "10min",
    "15min": "15min",
    "20min": "20min",
    "30min": "30min",
    "45min": "45min",
    "60min": "60min",
    "1h": "60min",
}


@BarBuilderRegistry.register("time")
class TimeBarBuilder(BaseBarBuilder):
    """
    Build time-based bars (standard resampling).

    Wraps the standard pandas resample functionality to provide
    a consistent interface with alternative bar builders.

    Attributes:
        target_timeframe: Target bar timeframe (e.g., '5min', '15min')

    Example:
        >>> builder = TimeBarBuilder(target_timeframe='5min')
        >>> df_bars = builder.build(df_1min)
    """

    def __init__(
        self,
        target_timeframe: str = "5min",
    ) -> None:
        """
        Initialize time bar builder.

        Args:
            target_timeframe: Target timeframe for bars
        """
        if target_timeframe not in TIMEFRAME_FREQ_MAP:
            raise ValueError(
                f"Unsupported timeframe: {target_timeframe}. "
                f"Supported: {list(TIMEFRAME_FREQ_MAP.keys())}"
            )

        self.target_timeframe = target_timeframe

    @property
    def bar_type(self) -> str:
        return "time"

    def build(
        self,
        df: pd.DataFrame,
        symbol: str | None = None,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Build time bars from input OHLCV data.

        Args:
            df: DataFrame with datetime, open, high, low, close, volume
            symbol: Optional symbol name
            include_metadata: If True, add bar_type column

        Returns:
            DataFrame with time-resampled bars
        """
        self.validate_input(df)

        df = df.copy()

        # Ensure datetime is the index
        if "datetime" in df.columns:
            df = df.set_index("datetime")

        # Get pandas frequency
        freq = TIMEFRAME_FREQ_MAP[self.target_timeframe]

        # Define aggregation rules
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # Handle optional columns
        optional_cols = ["filled", "roll_event", "roll_window", "missing_bar"]
        for col in optional_cols:
            if col in df.columns:
                agg_rules[col] = "max"

        # Resample
        result = df.resample(freq, closed="left", label="left").agg(agg_rules)

        # Remove empty bars
        result = result.dropna(subset=["open", "close"])

        # Reset index to get datetime column back
        result = result.reset_index()

        # Add metadata
        if include_metadata and len(result) > 0:
            result["bar_type"] = self.bar_type
            result["timeframe"] = self.target_timeframe

        if symbol and len(result) > 0:
            result["symbol"] = symbol

        n_input = len(df)
        n_output = len(result)
        logger.debug(
            f"TimeBarBuilder: {n_input} -> {n_output} bars " f"(timeframe: {self.target_timeframe})"
        )

        return result


__all__ = ["TimeBarBuilder", "TIMEFRAME_FREQ_MAP"]
