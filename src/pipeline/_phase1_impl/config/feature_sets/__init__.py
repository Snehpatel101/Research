"""
Feature set definitions for modular model training.

Provides named feature sets that can be selected without code edits.

This package re-exports all feature set components for backward compatibility.
"""

from .core import FeatureSetDefinition, FEATURE_SET_ALIASES
from .definitions import FEATURE_SET_DEFINITIONS
from .validation import (
    get_feature_set_definitions,
    resolve_feature_set_name,
    resolve_feature_set_names,
    validate_feature_set_config,
    validate_feature_set_coverage,
    get_feature_set_columns,
)

__all__ = [
    # Core types
    "FeatureSetDefinition",
    # Core constants
    "FEATURE_SET_DEFINITIONS",
    "FEATURE_SET_ALIASES",
    # Validation functions
    "get_feature_set_definitions",
    "resolve_feature_set_name",
    "resolve_feature_set_names",
    "validate_feature_set_config",
    "validate_feature_set_coverage",
    "get_feature_set_columns",
]
