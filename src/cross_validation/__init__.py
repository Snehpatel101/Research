"""
Cross-validation package for Phase 3: Out-of-Sample Predictions.

This package provides time-series aware cross-validation with proper
purging and embargo to prevent information leakage. It generates
out-of-fold predictions for ensemble stacking in Phase 4.

Main components:
- PurgedKFold: Time-series CV with label-aware purging
- WalkForwardFeatureSelector: Walk-forward feature selection
- OOFGenerator: Out-of-fold prediction generator
- CrossValidationRunner: Orchestrates CV for all models/horizons

Usage:
    from src.cross_validation import (
        PurgedKFold,
        PurgedKFoldConfig,
        WalkForwardFeatureSelector,
        OOFGenerator,
        CrossValidationRunner,
        CVResult,
    )

    # Configure CV
    cv_config = PurgedKFoldConfig(n_splits=5, purge_bars=60, embargo_bars=1440)
    cv = PurgedKFold(cv_config)

    # Generate OOF predictions
    oof_gen = OOFGenerator(cv)
    oof_predictions = oof_gen.generate_oof_predictions(X, y, model_configs)
"""
from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig, ModelAwareCV
from src.cross_validation.feature_selector import WalkForwardFeatureSelector
from src.cross_validation.oof_generator import OOFGenerator, StackingDataset
from src.cross_validation.cv_runner import CrossValidationRunner, CVResult, FoldMetrics
from src.cross_validation.param_spaces import PARAM_SPACES

__all__ = [
    "PurgedKFold",
    "PurgedKFoldConfig",
    "ModelAwareCV",
    "WalkForwardFeatureSelector",
    "OOFGenerator",
    "StackingDataset",
    "CrossValidationRunner",
    "CVResult",
    "FoldMetrics",
    "PARAM_SPACES",
]
