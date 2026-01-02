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
    from stages.feature_scaler import FeatureScaler, scale_splits

    # Simple usage with scale_splits convenience function
    train_scaled, val_scaled, test_scaled, scaler = scale_splits(
        train_df, val_df, test_df, feature_cols
    )

    # Or use FeatureScaler directly for more control
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
# Convenience functions
from .convenience import (
    scale_splits,
    scale_train_val_test,
)
from .core import (
    DEFAULT_SCALING_STRATEGY,
    FEATURE_PATTERNS,
    FeatureCategory,
    FeatureScalingConfig,
    ScalerConfig,
    ScalerType,
    ScalingReport,
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
    add_scaling_validation_to_stage8,
    validate_no_leakage,
    validate_scaling,
    validate_scaling_for_splits,
)

__all__ = [
    # Core
    "ScalerType",
    "ScalerConfig",
    "FeatureCategory",
    "FeatureScalingConfig",
    "ScalingStatistics",
    "ScalingReport",
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
    "validate_scaling_for_splits",
    "add_scaling_validation_to_stage8",
    # Convenience
    "scale_splits",
    "scale_train_val_test",
]
