"""
Volume Bar Builder.

Constructs bars based on cumulative volume threshold rather than time.
Each bar closes when the cumulative volume crosses the threshold.

Benefits:
- Bars reflect actual market activity
- More bars during high volume periods
- Information arrival rate alignment

Usage:
    builder = VolumeBarBuilder(volume_threshold=100_000)
    bars = builder.build(df)
"""
from __future__ import annotations

import logging

import pandas as pd

from .base import BarBuilderRegistry, BaseBarBuilder

logger = logging.getLogger(__name__)


@BarBuilderRegistry.register("volume")
class VolumeBarBuilder(BaseBarBuilder):
    """
    Build bars when cumulative volume crosses threshold.

    Volume bars sample data based on trading activity rather than
    time. This produces more bars during active periods and fewer
    during quiet periods, better aligning with information flow.

    Attributes:
        volume_threshold: Volume amount to trigger bar close
        min_bars_per_output: Minimum input bars per output (anti-noise)

    Example:
        >>> builder = VolumeBarBuilder(volume_threshold=100_000)
        >>> df_bars = builder.build(df_1min)
        >>> print(f"Created {len(df_bars)} volume bars")
    """

    def __init__(
        self,
        volume_threshold: float = 100_000,
        min_bars_per_output: int = 1,
    ) -> None:
        """
        Initialize volume bar builder.

        Args:
            volume_threshold: Cumulative volume to trigger bar close
            min_bars_per_output: Minimum input bars before bar can close
        """
        if volume_threshold <= 0:
            raise ValueError(f"volume_threshold must be > 0, got {volume_threshold}")
        if min_bars_per_output < 1:
            raise ValueError("min_bars_per_output must be >= 1")

        self.volume_threshold = volume_threshold
        self.min_bars_per_output = min_bars_per_output

    @property
    def bar_type(self) -> str:
        return "volume"

    def build(
        self,
        df: pd.DataFrame,
        symbol: str | None = None,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Build volume bars from input OHLCV data.

        Args:
            df: DataFrame with datetime, open, high, low, close, volume
            symbol: Optional symbol name
            include_metadata: If True, add bar_type column

        Returns:
            DataFrame with volume bars
        """
        self.validate_input(df)

        df = df.copy().sort_values("datetime").reset_index(drop=True)
        n_rows = len(df)

        # Track bar boundaries
        bar_boundaries = [0]  # Start indices of each bar
        cumulative_volume = 0.0

        for i in range(n_rows):
            cumulative_volume += df.loc[i, "volume"]

            # Check if threshold crossed (and minimum bars met)
            bars_in_current = i - bar_boundaries[-1] + 1
            if cumulative_volume >= self.volume_threshold:
                if bars_in_current >= self.min_bars_per_output:
                    bar_boundaries.append(i + 1)
                    cumulative_volume = 0.0

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
            bars.append(bar)
            bars_per_output.append(end_idx - start_idx)

        result = pd.DataFrame(bars)

        # Add metadata
        if include_metadata and len(result) > 0:
            result["bar_type"] = self.bar_type
            result["threshold"] = self.volume_threshold

        if symbol and len(result) > 0:
            result["symbol"] = symbol

        # Log statistics
        metadata = self._compute_metadata(
            df, result, bars_per_output, self.volume_threshold
        )
        logger.debug(
            f"VolumeBarBuilder: {metadata.n_input_bars} -> {metadata.n_output_bars} bars "
            f"(compression: {metadata.compression_ratio:.1f}x)"
        )

        return result


__all__ = ["VolumeBarBuilder"]
