"""
Feature Scaler Infrastructure for Phase 1/2 Pipeline - Train-Only Scaling

This package provides a train-only scaling infrastructure to prevent data leakage.
All scalers are fitted ONLY on training data, then applied to validation and test sets.

Key Features:
- Fits scalers exclusively on training data to prevent leakage
- Supports multiple scaler types (StandardScaler, RobustScaler, MinMaxScaler)
- Feature-type-aware scaling (different strategies per feature category)
- Outlier clipping to prevent extreme values from dominating
- Persists scaler parameters to disk for production inference
- Validates scaling correctness on val/test sets
- Integrates with stage8_validate.py

Usage:
    from src.data.pipeline.stages.scaling import FeatureScaler

    scaler = FeatureScaler(scaler_type='robust')
    train_scaled = scaler.fit_transform(train_df, feature_cols)

    # Transform val/test using training statistics
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Save for production
    scaler.save(Path('models/scaler.pkl'))

    # Load in production
    scaler = FeatureScaler.load(Path('models/scaler.pkl'))

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Refactored into modular package
"""

# Core classes and configuration
from .core import (
    DEFAULT_SCALING_STRATEGY,
    FEATURE_PATTERNS,
    FeatureCategory,
    FeatureScalingConfig,
    ScalerConfig,
    ScalerType,
    ScalingStatistics,
)

# Main scaler class
from .scaler import FeatureScaler

# Scaler implementations and utilities
from .scalers import (
    categorize_feature,
    compute_statistics,
    create_scaler,
    get_default_scaler_type,
    should_log_transform,
)

# Validation functions
from .validators import (
    validate_no_leakage,
    validate_scaling,
)

__all__ = [
    # Core
    "ScalerType",
    "ScalerConfig",
    "FeatureCategory",
    "FeatureScalingConfig",
    "ScalingStatistics",
    "FEATURE_PATTERNS",
    "DEFAULT_SCALING_STRATEGY",
    # Utilities
    "categorize_feature",
    "get_default_scaler_type",
    "should_log_transform",
    "create_scaler",
    "compute_statistics",
    # Main class
    "FeatureScaler",
    # Validation
    "validate_scaling",
    "validate_no_leakage",
]
