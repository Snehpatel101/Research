"""
Common utilities shared across all model implementations.

This package provides shared functionality for:
- Label mapping between trading signals and ML class indices
"""

from .label_mapping import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    map_classes_to_labels,
    map_labels_to_classes,
)

__all__ = [
    "LABEL_TO_CLASS",
    "CLASS_TO_LABEL",
    "map_labels_to_classes",
    "map_classes_to_labels",
]
