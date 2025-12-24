"""Phase 1 Utilities.

Provides feature selection and feature set resolution utilities.
"""
from src.phase1.utils.feature_selection import (
    FeatureSelectionResult,
    FEATURE_PRIORITY,
    DEFAULT_PRIORITY,
    select_features,
    get_feature_priority,
    identify_feature_columns,
    filter_low_variance,
    filter_correlated_features,
    save_feature_selection_report,
    apply_feature_selection,
)

from src.phase1.utils.feature_sets import (
    METADATA_COLUMNS,
    LABEL_PREFIXES,
    resolve_feature_set,
    build_feature_set_manifest,
    validate_feature_set_columns,
)

__all__ = [
    # feature_selection
    'FeatureSelectionResult',
    'FEATURE_PRIORITY',
    'DEFAULT_PRIORITY',
    'select_features',
    'get_feature_priority',
    'identify_feature_columns',
    'filter_low_variance',
    'filter_correlated_features',
    'save_feature_selection_report',
    'apply_feature_selection',
    # feature_sets
    'METADATA_COLUMNS',
    'LABEL_PREFIXES',
    'resolve_feature_set',
    'build_feature_set_manifest',
    'validate_feature_set_columns',
]
