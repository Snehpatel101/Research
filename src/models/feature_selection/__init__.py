"""
Feature Selection Integration for Model Training.

This module provides feature selection integration for the model training pipeline.
It wraps the existing WalkForwardFeatureSelector and provides:
- Per-model family feature selection configuration
- Integration with the training pipeline
- Persistence of selected features with model artifacts
- Application of feature selection at inference time

Usage:
    from src.models.feature_selection import FeatureSelectionManager

    # Create manager with config
    manager = FeatureSelectionManager(
        n_features=50,
        method="mda",
        model_family="boosting"
    )

    # Run feature selection
    result = manager.select_features(X_train, y_train, sample_weights)

    # Apply to data
    X_train_selected = manager.apply_selection(X_train)

    # Save/load with model artifacts
    manager.save(path)
    manager.load(path)
"""
from .config import FeatureSelectionConfig, ModelFamilyDefaults
from .manager import FeatureSelectionManager
from .result import PersistedFeatureSelection

__all__ = [
    "FeatureSelectionConfig",
    "FeatureSelectionManager",
    "ModelFamilyDefaults",
    "PersistedFeatureSelection",
]
