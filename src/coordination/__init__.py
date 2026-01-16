"""
Coordination utilities for multi-timeframe and cross-model data alignment.

This package provides utilities for:
- Temporal alignment between different timeframes
- MTF feature lag application for leakage prevention
- Sequence/tabular data offset computation
- Timestamp validation across datasets
- TimeframeCoordinator for multi-timeframe data loading and alignment

Usage:
    from src.coordination import (
        # Alignment utilities
        align_to_anchor,
        apply_mtf_lag,
        compute_sequence_offset,
        validate_timestamp_alignment,
        # TimeframeCoordinator
        TimeframeCoordinator,
        TimeframeData,
    )
"""

from src.coordination.alignment import (
    align_to_anchor,
    apply_mtf_lag,
    compute_sequence_offset,
    validate_timestamp_alignment,
)
from src.coordination.timeframe_coordinator import (
    TimeframeCoordinator,
    TimeframeData,
)

__all__ = [
    # Alignment utilities
    "align_to_anchor",
    "apply_mtf_lag",
    "compute_sequence_offset",
    "validate_timestamp_alignment",
    # TimeframeCoordinator
    "TimeframeCoordinator",
    "TimeframeData",
]
