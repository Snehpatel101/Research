"""
Alternative Bar Builders.

Provides non-time-based bar construction methods:
- Volume Bars: Aggregate by cumulative volume threshold
- Dollar Bars: Aggregate by cumulative dollar value threshold
- Time Bars: Standard time-based resampling (wrapper)

Usage:
    from src.phase1.stages.clean.bar_builders import (
        build_bars,
        VolumeBarBuilder,
        DollarBarBuilder,
    )

    # Build volume bars
    bars = build_bars(df, bar_type='volume', volume_threshold=100_000)

    # Build dollar bars
    bars = build_bars(df, bar_type='dollar', dollar_threshold=10_000_000)

    # Standard time bars
    bars = build_bars(df, bar_type='time', target_timeframe='5min')
"""
from src.phase1.stages.clean.bar_builders.base import (
    BarBuilderRegistry,
    BaseBarBuilder,
)
from src.phase1.stages.clean.bar_builders.dollar_bars import DollarBarBuilder
from src.phase1.stages.clean.bar_builders.factory import (
    BarConfig,
    build_bars,
    create_builder,
    estimate_bar_count,
)
from src.phase1.stages.clean.bar_builders.time_bars import TimeBarBuilder
from src.phase1.stages.clean.bar_builders.volume_bars import VolumeBarBuilder

__all__ = [
    # Base
    "BaseBarBuilder",
    "BarBuilderRegistry",
    # Builders
    "VolumeBarBuilder",
    "DollarBarBuilder",
    "TimeBarBuilder",
    # Factory
    "build_bars",
    "BarConfig",
    "create_builder",
    "estimate_bar_count",
]
