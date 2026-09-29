"""
Coordination utilities for multi-timeframe and cross-model data alignment.

This package provides utilities for:
- Temporal alignment between different timeframes
- MTF feature lag application for leakage prevention
- Sequence/tabular data offset computation
- Timestamp validation across datasets

Usage:
    from src.core.coordination import (
        # Alignment utilities
        align_to_anchor,
        apply_mtf_lag,
        compute_sequence_offset,
        validate_timestamp_alignment,
    )
"""

from .alignment import (
    align_to_anchor,
    apply_mtf_lag,
    compute_sequence_offset,
    validate_timestamp_alignment,
)

__all__ = [
    # Alignment utilities
    "align_to_anchor",
    "apply_mtf_lag",
    "compute_sequence_offset",
    "validate_timestamp_alignment",
]
