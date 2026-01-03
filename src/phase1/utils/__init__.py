"""Phase 1 Utilities.

Provides feature selection and feature set resolution utilities.
"""

from src.phase1.utils.constants import LABEL_PREFIXES, METADATA_COLUMNS
from src.phase1.utils.feature_selection import (
    DEFAULT_PRIORITY,
    FEATURE_PRIORITY,
    FeatureSelectionResult,
    apply_feature_selection,
    filter_correlated_features,
    filter_low_variance,
    get_feature_priority,
    identify_feature_columns,
    save_feature_selection_report,
    select_features,
)
from src.phase1.utils.feature_sets import (
    build_feature_set_manifest,
    resolve_feature_set,
    validate_feature_set_columns,
)

__all__ = [
    # feature_selection
    "FeatureSelectionResult",
    "FEATURE_PRIORITY",
    "DEFAULT_PRIORITY",
    "select_features",
    "get_feature_priority",
    "identify_feature_columns",
    "filter_low_variance",
    "filter_correlated_features",
    "save_feature_selection_report",
    "apply_feature_selection",
    # feature_sets
    "METADATA_COLUMNS",
    "LABEL_PREFIXES",
    "resolve_feature_set",
    "build_feature_set_manifest",
    "validate_feature_set_columns",
]
