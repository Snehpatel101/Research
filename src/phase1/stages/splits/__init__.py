"""
Stage 7: Time-Based Splitting with Purging and Embargo.

This module implements chronological train/val/test splits with leakage prevention.
"""
from .core import (
    INVALID_LABEL_SENTINEL,
    create_chronological_splits,
    validate_label_distribution,
    validate_no_overlap,
    validate_per_symbol_distribution,
)

__all__ = [
    'create_chronological_splits',
    'validate_no_overlap',
    'validate_per_symbol_distribution',
    'validate_label_distribution',
    'INVALID_LABEL_SENTINEL'
]
