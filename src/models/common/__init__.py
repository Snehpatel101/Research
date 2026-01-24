"""
Common utilities shared across all model implementations.

This package provides shared functionality for:
- Label mapping between trading signals and ML class indices
- Class weight computation for imbalanced classification
"""

from .class_weights import (
    compute_balanced_weights,
    compute_class_weight_dict,
    compute_sample_weights,
)
from .label_mapping import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    map_classes_to_labels,
    map_labels_to_classes,
)

__all__ = [
    # Label mapping
    "LABEL_TO_CLASS",
    "CLASS_TO_LABEL",
    "map_labels_to_classes",
    "map_classes_to_labels",
    # Class weights (Phase 8A)
    "compute_balanced_weights",
    "compute_sample_weights",
    "compute_class_weight_dict",
]
