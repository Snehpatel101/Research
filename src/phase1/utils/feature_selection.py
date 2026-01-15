"""
Feature Selection Module for Ensemble Price Prediction Pipeline.

DEPRECATED: This module has been moved to src.feature_selection.
Please update your imports:

    # Old (deprecated):
    from src.phase1.utils.feature_selection import select_features, FEATURE_PRIORITY

    # New (preferred):
    from src.feature_selection import select_features, FEATURE_PRIORITY

This module will be removed in a future version.
"""

from __future__ import annotations

# Keep main() accessible directly
def main():
    """Run feature selection on the combined labeled data."""
    import logging

    import pandas as pd

    from src.feature_selection import save_feature_selection_report, select_features
    from src.phase1.config import FINAL_DATA_DIR, RESULTS_DIR

    logger = logging.getLogger(__name__)

    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"
    output_path = RESULTS_DIR / "feature_selection_report.json"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Run feature selection
    result = select_features(df, correlation_threshold=0.85, variance_threshold=0.01)

    # Save report
    save_feature_selection_report(result, output_path)

    return result


def __getattr__(name):
    """Lazy import to avoid circular dependency."""
    import warnings

    warnings.warn(
        "Importing from src.phase1.utils.feature_selection is deprecated. "
        "Use src.feature_selection instead. "
        "Example: from src.feature_selection import select_features",
        DeprecationWarning,
        stacklevel=2,
    )

    # Import from canonical location
    from src.feature_selection import (
        DEFAULT_PRIORITY,
        FEATURE_PRIORITY,
        FeatureSelectionResult,
        apply_feature_selection,
        build_correlation_groups,
        filter_correlated_features,
        filter_low_variance,
        get_feature_priority,
        identify_feature_columns,
        save_feature_selection_report,
        select_features,
        select_from_correlated_group,
    )

    _exports = {
        "DEFAULT_PRIORITY": DEFAULT_PRIORITY,
        "FEATURE_PRIORITY": FEATURE_PRIORITY,
        "FeatureSelectionResult": FeatureSelectionResult,
        "apply_feature_selection": apply_feature_selection,
        "build_correlation_groups": build_correlation_groups,
        "filter_correlated_features": filter_correlated_features,
        "filter_low_variance": filter_low_variance,
        "get_feature_priority": get_feature_priority,
        "identify_feature_columns": identify_feature_columns,
        "save_feature_selection_report": save_feature_selection_report,
        "select_features": select_features,
        "select_from_correlated_group": select_from_correlated_group,
    }

    if name in _exports:
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_PRIORITY",
    "FEATURE_PRIORITY",
    "FeatureSelectionResult",
    "apply_feature_selection",
    "build_correlation_groups",
    "filter_correlated_features",
    "filter_low_variance",
    "get_feature_priority",
    "identify_feature_columns",
    "main",
    "save_feature_selection_report",
    "select_features",
    "select_from_correlated_group",
]
