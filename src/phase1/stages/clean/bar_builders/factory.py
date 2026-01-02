"""
Bar Builder Factory.

Unified interface for constructing different bar types.
Integrates with pipeline configuration.

Usage:
    from src.phase1.stages.clean.bar_builders import build_bars, BarConfig

    # Build with explicit parameters
    bars = build_bars(df, bar_type='volume', volume_threshold=100_000)

    # Build from config
    config = BarConfig(bar_type='dollar', dollar_threshold=10_000_000)
    bars = build_bars(df, config=config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .base import BarBuilderRegistry, BaseBarBuilder

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class BarConfig:
    """
    Configuration for bar construction.

    Attributes:
        bar_type: Type of bars ('time', 'volume', 'dollar')
        target_timeframe: For time bars, target timeframe (e.g., '5min')
        volume_threshold: For volume bars, volume threshold
        dollar_threshold: For dollar bars, dollar value threshold
        min_bars_per_output: Minimum input bars per output bar
        use_vwap: For dollar bars, use VWAP instead of close price
        include_metadata: Add bar_type column to output
    """

    bar_type: str = "time"
    target_timeframe: str = "5min"
    volume_threshold: float = 100_000
    dollar_threshold: float = 10_000_000
    min_bars_per_output: int = 1
    use_vwap: bool = False
    include_metadata: bool = True
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_types = BarBuilderRegistry.list_all() or ["time", "volume", "dollar"]
        if self.bar_type.lower() not in [t.lower() for t in valid_types]:
            raise ValueError(f"Unknown bar_type: {self.bar_type}. " f"Available: {valid_types}")

    def get_builder_kwargs(self) -> dict[str, Any]:
        """Get keyword arguments for the builder."""
        bar_type = self.bar_type.lower()

        if bar_type == "time":
            return {"target_timeframe": self.target_timeframe}
        elif bar_type == "volume":
            return {
                "volume_threshold": self.volume_threshold,
                "min_bars_per_output": self.min_bars_per_output,
            }
        elif bar_type == "dollar":
            return {
                "dollar_threshold": self.dollar_threshold,
                "min_bars_per_output": self.min_bars_per_output,
                "use_vwap": self.use_vwap,
            }
        else:
            return self.extra_params


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def build_bars(
    df: pd.DataFrame,
    bar_type: str | None = None,
    config: BarConfig | None = None,
    symbol: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Unified bar building interface.

    Constructs bars from OHLCV data using the specified method.
    Supports time, volume, and dollar bar types.

    Args:
        df: Input DataFrame with datetime, open, high, low, close, volume
        bar_type: Type of bars to build ('time', 'volume', 'dollar')
        config: BarConfig instance (alternative to bar_type + kwargs)
        symbol: Optional symbol name to include in output
        **kwargs: Additional arguments passed to builder

    Returns:
        DataFrame with constructed bars

    Examples:
        >>> # Build 5-minute time bars
        >>> bars = build_bars(df, bar_type='time', target_timeframe='5min')

        >>> # Build volume bars with 100k threshold
        >>> bars = build_bars(df, bar_type='volume', volume_threshold=100_000)

        >>> # Build dollar bars with $10M threshold
        >>> bars = build_bars(df, bar_type='dollar', dollar_threshold=10_000_000)

        >>> # Build from config
        >>> config = BarConfig(bar_type='volume', volume_threshold=50_000)
        >>> bars = build_bars(df, config=config)
    """
    # Determine configuration
    if config is not None:
        bar_type = config.bar_type
        builder_kwargs = config.get_builder_kwargs()
        include_metadata = config.include_metadata
    else:
        bar_type = bar_type or "time"
        builder_kwargs = kwargs
        include_metadata = kwargs.pop("include_metadata", True)

    # Get builder class
    builder_cls = BarBuilderRegistry.get(bar_type)

    # Create builder instance
    builder = builder_cls(**builder_kwargs)

    # Build bars
    result = builder.build(df, symbol=symbol, include_metadata=include_metadata)

    logger.info(f"Built {len(result)} {bar_type} bars from {len(df)} input bars")

    return result


def create_builder(
    bar_type: str = "time",
    **kwargs: Any,
) -> BaseBarBuilder:
    """
    Create a bar builder instance.

    Args:
        bar_type: Type of builder to create
        **kwargs: Builder-specific arguments

    Returns:
        BaseBarBuilder instance
    """
    builder_cls = BarBuilderRegistry.get(bar_type)
    return builder_cls(**kwargs)


def estimate_bar_count(
    df: pd.DataFrame,
    bar_type: str,
    threshold: float | None = None,
    target_timeframe: str | None = None,
) -> int:
    """
    Estimate the number of output bars without building.

    Args:
        df: Input DataFrame
        bar_type: Type of bars
        threshold: For volume/dollar bars
        target_timeframe: For time bars

    Returns:
        Estimated number of output bars
    """
    n_input = len(df)

    if bar_type == "time":
        # Estimate based on timeframe ratio
        tf_minutes = {
            "1min": 1,
            "5min": 5,
            "10min": 10,
            "15min": 15,
            "20min": 20,
            "30min": 30,
            "45min": 45,
            "60min": 60,
        }
        if target_timeframe:
            ratio = tf_minutes.get(target_timeframe, 5)
            return n_input // ratio
        return n_input // 5

    elif bar_type == "volume":
        if threshold and "volume" in df.columns:
            total_volume = df["volume"].sum()
            return max(1, int(total_volume / threshold))
        return n_input // 10

    elif bar_type == "dollar":
        if threshold and "volume" in df.columns and "close" in df.columns:
            total_dollars = (df["volume"] * df["close"]).sum()
            return max(1, int(total_dollars / threshold))
        return n_input // 10

    return n_input


__all__ = [
    "build_bars",
    "create_builder",
    "estimate_bar_count",
    "BarConfig",
]
