"""
Feature Selection Package for OHLCV Time-Series ML.

This package provides enhanced feature selection specifically designed for
OHLCV time-series data in ML trading applications.

Main Components:
    OHLCVFeatureSelector: Enhanced selector with stability, correlation, and regime support
    PurgedFeatureSelector: Selector integrated with PurgedKFold CV
    filter_ohlcv_features: Filter features by category (momentum, volatility, etc.)
    FeatureSelectionResult: Result container with importances, stability, clusters

Key Features:
    - Time-series aware MDA (walk-forward, no lookahead)
    - Stability scoring (ranking consistency across folds)
    - Correlation filtering (hierarchical clustering)
    - Regime-conditional importance (market state aware)
    - OHLCV category filtering (momentum, volatility, volume, etc.)
    - PurgedKFold integration (purge + embargo for label leakage prevention)

Example:
    >>> from src.feature_selection import OHLCVFeatureSelector, filter_ohlcv_features
    >>>
    >>> # Filter features by category
    >>> features = filter_ohlcv_features(all_features, exclude_categories=['mtf', 'temporal'])
    >>>
    >>> # Select optimal features
    >>> selector = OHLCVFeatureSelector(
    ...     n_splits=5,
    ...     min_stability_score=0.6,
    ...     correlation_threshold=0.85,
    ... )
    >>> result = selector.select_features(X, y, features)
    >>> print(f"Selected: {result.n_selected}/{result.n_original}")

With PurgedKFold:
    >>> from src.cross_validation import PurgedKFold, PurgedKFoldConfig
    >>> from src.feature_selection import PurgedFeatureSelector
    >>>
    >>> cv = PurgedKFold(PurgedKFoldConfig(n_splits=5, purge_bars=60, embargo_bars=1440))
    >>> selector = PurgedFeatureSelector(cv=cv)
    >>> result = selector.select_features(X, y, feature_names)

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""

from src.feature_selection.ohlcv_selector import (
    # Category utilities
    FEATURE_CATEGORIES,
    # Result classes
    FeatureSelectionResult,
    # Main class
    OHLCVFeatureSelector,
    StabilityMetrics,
    categorize_feature,
    # Factory
    create_ohlcv_selector,
    filter_ohlcv_features,
    get_feature_categories,
)
from src.feature_selection.purged_selector import (
    PurgedFeatureSelector,
    create_purged_selector,
)

__all__ = [
    # Main classes
    "OHLCVFeatureSelector",
    "PurgedFeatureSelector",
    # Result classes
    "FeatureSelectionResult",
    "StabilityMetrics",
    # Category utilities
    "FEATURE_CATEGORIES",
    "categorize_feature",
    "filter_ohlcv_features",
    "get_feature_categories",
    # Factories
    "create_ohlcv_selector",
    "create_purged_selector",
]
