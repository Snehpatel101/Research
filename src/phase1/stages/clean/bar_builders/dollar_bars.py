"""
Dollar Bar Builder.

Constructs bars based on cumulative dollar value (price * volume) threshold.
Each bar closes when cumulative dollar value crosses the threshold.

Benefits:
- Economically meaningful sampling
- Accounts for both volume and price
- Fair representation of capital flow

Usage:
    builder = DollarBarBuilder(dollar_threshold=10_000_000)
    bars = builder.build(df)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseBarBuilder, BarBuilderRegistry, BarMetadata

logger = logging.getLogger(__name__)


@BarBuilderRegistry.register("dollar")
class DollarBarBuilder(BaseBarBuilder):
    """
    Build bars when cumulative dollar value crosses threshold.

    Dollar bars sample based on economic activity (price * volume),
    which provides a more meaningful measure of market interest
    than volume alone, especially for assets with varying prices.

    Dollar Value Calculation:
        dollar_value = close_price * volume

    Attributes:
        dollar_threshold: Dollar amount to trigger bar close
        min_bars_per_output: Minimum input bars per output (anti-noise)

    Example:
        >>> builder = DollarBarBuilder(dollar_threshold=10_000_000)
        >>> df_bars = builder.build(df_1min)
        >>> print(f"Created {len(df_bars)} dollar bars")
    """

    def __init__(
        self,
        dollar_threshold: float = 10_000_000,
        min_bars_per_output: int = 1,
        use_vwap: bool = False,
    ) -> None:
        """
        Initialize dollar bar builder.

        Args:
            dollar_threshold: Cumulative dollar value to trigger bar close
            min_bars_per_output: Minimum input bars before bar can close
            use_vwap: If True, use VWAP for price; else use close
        """
        if dollar_threshold <= 0:
            raise ValueError(f"dollar_threshold must be > 0, got {dollar_threshold}")
        if min_bars_per_output < 1:
            raise ValueError(f"min_bars_per_output must be >= 1")

        self.dollar_threshold = dollar_threshold
        self.min_bars_per_output = min_bars_per_output
        self.use_vwap = use_vwap

    @property
    def bar_type(self) -> str:
        return "dollar"

    def build(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Build dollar bars from input OHLCV data.

        Args:
            df: DataFrame with datetime, open, high, low, close, volume
            symbol: Optional symbol name
            include_metadata: If True, add bar_type column

        Returns:
            DataFrame with dollar bars
        """
        self.validate_input(df)

        df = df.copy().sort_values("datetime").reset_index(drop=True)
        n_rows = len(df)

        # Compute dollar value per bar
        if self.use_vwap:
            # Approximate VWAP as (high + low + close) / 3
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
        else:
            typical_price = df["close"]

        dollar_values = typical_price * df["volume"]

        # Track bar boundaries
        bar_boundaries = [0]
        cumulative_dollars = 0.0

        for i in range(n_rows):
            cumulative_dollars += dollar_values.iloc[i]

            bars_in_current = i - bar_boundaries[-1] + 1
            if cumulative_dollars >= self.dollar_threshold:
                if bars_in_current >= self.min_bars_per_output:
                    bar_boundaries.append(i + 1)
                    cumulative_dollars = 0.0
                # NOTE: If min_bars not met, cumulative_dollars continues to grow.
                # This ensures the bar closes only after both conditions are met:
                # (1) dollar threshold reached AND (2) minimum bars included.
                # This may result in bars with > dollar_threshold value.

        # Handle last incomplete bar
        if bar_boundaries[-1] < n_rows:
            bar_boundaries.append(n_rows)

        # Aggregate bars
        bars = []
        bars_per_output = []

        for i in range(len(bar_boundaries) - 1):
            start_idx = bar_boundaries[i]
            end_idx = bar_boundaries[i + 1]
            group = df.iloc[start_idx:end_idx]

            bar = self._aggregate_bar(group)

            # Add dollar value for this bar
            bar["dollar_value"] = dollar_values.iloc[start_idx:end_idx].sum()

            bars.append(bar)
            bars_per_output.append(end_idx - start_idx)

        result = pd.DataFrame(bars)

        # Add metadata
        if include_metadata and len(result) > 0:
            result["bar_type"] = self.bar_type
            result["threshold"] = self.dollar_threshold

        if symbol and len(result) > 0:
            result["symbol"] = symbol

        # Log statistics
        metadata = self._compute_metadata(
            df, result, bars_per_output, self.dollar_threshold
        )
        logger.debug(
            f"DollarBarBuilder: {metadata.n_input_bars} -> {metadata.n_output_bars} bars "
            f"(compression: {metadata.compression_ratio:.1f}x)"
        )

        return result


__all__ = ["DollarBarBuilder"]
