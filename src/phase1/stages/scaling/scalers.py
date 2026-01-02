"""
Scaler Implementations and Utilities

This module provides scaler implementations and utility functions for
feature transformation.

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Extracted from feature_scaler.py
"""

import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .core import FeatureCategory, ScalerType

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def categorize_feature(feature_name: str) -> FeatureCategory:
    """
    Determine the category of a feature based on its name.

    Args:
        feature_name: Name of the feature

    Returns:
        FeatureCategory enum value
    """
    from .core import FEATURE_PATTERNS

    feature_lower = feature_name.lower()

    for category, patterns in FEATURE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in feature_lower or feature_lower.startswith(pattern.lower()):
                return category

    return FeatureCategory.UNKNOWN


def get_default_scaler_type(category: FeatureCategory) -> ScalerType:
    """Get the default scaler type for a feature category."""
    from .core import DEFAULT_SCALING_STRATEGY

    return DEFAULT_SCALING_STRATEGY.get(category, ScalerType.ROBUST)


def should_log_transform(feature_name: str, category: FeatureCategory) -> bool:
    """
    Determine if a feature should have log transform applied.

    Log transform is recommended for:
    - Price level features (SMA, EMA, etc.)
    - Volume features (but NOT OBV which can be negative)
    - Features with high positive skewness
    """
    # OBV can be negative (cumulative buying - selling volume), so never log-transform it
    if "obv" in feature_name.lower():
        return False

    if category in [FeatureCategory.PRICE_LEVEL, FeatureCategory.VOLUME]:
        # Check if it's a raw price/volume feature (not a ratio)
        if not any(x in feature_name.lower() for x in ["ratio", "pct", "zscore", "to_"]):
            return True
    return False


def create_scaler(
    scaler_type: ScalerType, robust_quantile_range: tuple[float, float] = (25.0, 75.0)
) -> RobustScaler | StandardScaler | MinMaxScaler | None:
    """Create a sklearn scaler instance based on type."""
    if scaler_type == ScalerType.ROBUST:
        return RobustScaler(quantile_range=robust_quantile_range)
    elif scaler_type == ScalerType.STANDARD:
        return StandardScaler()
    elif scaler_type == ScalerType.MINMAX:
        return MinMaxScaler(feature_range=(0, 1))
    else:
        return None


def compute_statistics(
    data: np.ndarray, scaled_data: np.ndarray, feature_name: str
) -> "ScalingStatistics":
    """Compute statistics for a feature before and after scaling."""
    from .core import ScalingStatistics

    # Handle NaN/Inf
    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())

    clean_data = data[~np.isnan(data) & ~np.isinf(data)]
    clean_scaled = scaled_data[~np.isnan(scaled_data) & ~np.isinf(scaled_data)]

    if len(clean_data) == 0:
        return ScalingStatistics(
            feature_name=feature_name,
            train_mean=np.nan,
            train_std=np.nan,
            train_min=np.nan,
            train_max=np.nan,
            train_median=np.nan,
            train_q25=np.nan,
            train_q75=np.nan,
            scaled_mean=np.nan,
            scaled_std=np.nan,
            scaled_min=np.nan,
            scaled_max=np.nan,
            nan_count=nan_count,
            inf_count=inf_count,
        )

    return ScalingStatistics(
        feature_name=feature_name,
        train_mean=float(np.mean(clean_data)),
        train_std=float(np.std(clean_data)),
        train_min=float(np.min(clean_data)),
        train_max=float(np.max(clean_data)),
        train_median=float(np.median(clean_data)),
        train_q25=float(np.percentile(clean_data, 25)),
        train_q75=float(np.percentile(clean_data, 75)),
        scaled_mean=float(np.mean(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
        scaled_std=float(np.std(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
        scaled_min=float(np.min(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
        scaled_max=float(np.max(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
        nan_count=nan_count,
        inf_count=inf_count,
    )
