"""
Base classes for alternative bar construction.

Provides the abstract interface that all bar builders must implement.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# BAR BUILDER REGISTRY
# =============================================================================

class BarBuilderRegistry:
    """
    Registry for bar builder types.

    Allows registration and discovery of bar builder implementations.

    Example:
        >>> @BarBuilderRegistry.register("custom")
        ... class CustomBarBuilder(BaseBarBuilder):
        ...     pass
        >>> builder_cls = BarBuilderRegistry.get("custom")
    """

    _builders: dict[str, type[BaseBarBuilder]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a bar builder.

        Args:
            name: Name for the bar type

        Returns:
            Decorator function
        """
        def decorator(builder_cls: type[BaseBarBuilder]):
            cls._builders[name.lower()] = builder_cls
            logger.debug(f"Registered bar builder: {name}")
            return builder_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseBarBuilder]:
        """
        Get a registered builder class.

        Args:
            name: Bar type name

        Returns:
            Builder class

        Raises:
            ValueError: If bar type is not registered
        """
        name_lower = name.lower()
        if name_lower not in cls._builders:
            available = sorted(cls._builders.keys())
            raise ValueError(
                f"Unknown bar type: '{name}'. Available: {available}"
            )
        return cls._builders[name_lower]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered bar types."""
        return sorted(cls._builders.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a bar type is registered."""
        return name.lower() in cls._builders


# =============================================================================
# BASE BAR BUILDER
# =============================================================================

@dataclass
class BarMetadata:
    """Metadata about constructed bars."""
    bar_type: str
    threshold: float | None
    n_input_bars: int
    n_output_bars: int
    compression_ratio: float
    min_bars_per_output: int
    max_bars_per_output: int
    avg_bars_per_output: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bar_type": self.bar_type,
            "threshold": self.threshold,
            "n_input_bars": self.n_input_bars,
            "n_output_bars": self.n_output_bars,
            "compression_ratio": self.compression_ratio,
            "min_bars_per_output": self.min_bars_per_output,
            "max_bars_per_output": self.max_bars_per_output,
            "avg_bars_per_output": self.avg_bars_per_output,
        }


class BaseBarBuilder(ABC):
    """
    Abstract base class for bar construction.

    All bar builders must inherit from this class and implement
    the build() method. The output schema matches standard OHLCV
    for compatibility with downstream pipeline stages.

    Output Columns:
        datetime, open, high, low, close, volume, [symbol], [bar_type]

    Example:
        >>> class MyBarBuilder(BaseBarBuilder):
        ...     def build(self, df, symbol):
        ...         # Custom aggregation logic
        ...         return bars_df
    """

    @property
    @abstractmethod
    def bar_type(self) -> str:
        """Return the bar type name."""
        pass

    @abstractmethod
    def build(
        self,
        df: pd.DataFrame,
        symbol: str | None = None,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Build bars from input data.

        Args:
            df: DataFrame with OHLCV data (datetime, open, high, low, close, volume)
            symbol: Optional symbol name to include in output
            include_metadata: If True, add bar_type column

        Returns:
            DataFrame with aggregated OHLCV bars
        """
        pass

    def validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame has required columns.

        Args:
            df: Input DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        required = ["datetime", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"{self.__class__.__name__}: Missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        if len(df) == 0:
            raise ValueError(f"{self.__class__.__name__}: Input DataFrame is empty")

    def _aggregate_bar(self, group: pd.DataFrame) -> dict[str, Any]:
        """
        Aggregate OHLCV data for a single bar.

        Args:
            group: DataFrame slice for one bar

        Returns:
            Dict with aggregated OHLCV values
        """
        return {
            "datetime": group["datetime"].iloc[0],  # First timestamp
            "open": group["open"].iloc[0],
            "high": group["high"].max(),
            "low": group["low"].min(),
            "close": group["close"].iloc[-1],
            "volume": group["volume"].sum(),
        }

    def _compute_metadata(
        self,
        df_input: pd.DataFrame,
        df_output: pd.DataFrame,
        bars_per_output: list[int],
        threshold: float | None = None,
    ) -> BarMetadata:
        """
        Compute metadata about the bar construction.

        Args:
            df_input: Original input DataFrame
            df_output: Constructed bars DataFrame
            bars_per_output: Number of input bars per output bar
            threshold: Threshold used (if applicable)

        Returns:
            BarMetadata with statistics
        """
        return BarMetadata(
            bar_type=self.bar_type,
            threshold=threshold,
            n_input_bars=len(df_input),
            n_output_bars=len(df_output),
            compression_ratio=len(df_input) / max(len(df_output), 1),
            min_bars_per_output=min(bars_per_output) if bars_per_output else 0,
            max_bars_per_output=max(bars_per_output) if bars_per_output else 0,
            avg_bars_per_output=np.mean(bars_per_output) if bars_per_output else 0,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bar_type={self.bar_type})"


__all__ = [
    "BaseBarBuilder",
    "BarBuilderRegistry",
    "BarMetadata",
]
