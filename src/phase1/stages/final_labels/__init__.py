"""
Stage 6: Final Labels with Quality Scoring.

This module applies optimized triple-barrier labels using GA-optimized parameters
and computes quality scores and sample weights for each label.
"""

from .core import (
    add_forward_return_columns,
    apply_optimized_labels,
    assign_sample_weights,
    compute_quality_scores,
    generate_labeling_report,
)

__all__ = [
    'apply_optimized_labels',
    'compute_quality_scores',
    'assign_sample_weights',
    'add_forward_return_columns',
    'generate_labeling_report',
]
