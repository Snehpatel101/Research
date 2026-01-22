"""
DEPRECATED: Import from 'src.validation.cv' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.cross_validation import PurgedKFold, CombinatorialPurgedCV

    # New (preferred)
    from src.validation.cv import PurgedKFold, CombinatorialPurgedCV
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

# Names that will be lazily loaded from src.validation.cv
_CV_EXPORTS = [
    "PurgedKFold",
    "PurgedKFoldConfig",
    "ModelAwareCV",
    "WalkForwardFeatureSelector",
    "OOFGenerator",
    "OOFPrediction",
    "StackingDataset",
    "CrossValidationRunner",
    "CVResult",
    "FoldMetrics",
    "PARAM_SPACES",
    # CV Tuner
    "TimeSeriesOptunaTuner",
    # CV Feature Selection
    "run_cv_with_per_fold_feature_selection",
    "compute_feature_stability",
    # CV Stacking
    "validate_stacking_consistency",
    "build_stacking_datasets_from_cv_results",
    "analyze_cv_stability",
    # Walk-forward
    "WalkForwardConfig",
    "WalkForwardEvaluator",
    "WalkForwardResult",
    "WindowMetrics",
    "create_walk_forward_evaluator",
    # CPCV
    "CPCVConfig",
    "CombinatorialPurgedCV",
    "CPCVResult",
    "CPCVPathResult",
    "create_cpcv",
    # PBO
    "PBOConfig",
    "PBOResult",
    "compute_pbo",
    "compute_pbo_from_returns",
    "pbo_gate",
    "analyze_overfitting_risk",
    # Sequence CV
    "SequenceCVBuilder",
    "SequenceFoldResult",
    "build_sequences_for_cv_fold",
    "validate_sequence_cv_coverage",
    # Sequence OOF
    "SequenceOOFGenerator",
    # Stacking
    "StackingDatasetBuilder",
    "find_valid_samples_mask",
    # Timestamp Alignment
    "validate_datetime_alignment",
    "align_predictions_on_datetime",
    "get_datetime_alignment_report",
    # OOF Cache
    "OOFCache",
    "OOFCacheEntry",
    "compute_data_hash",
    # OOF Alignment
    "OOFAlignmentResult",
    "OOFAlignmentValidator",
    "compute_oof_coverage",
    "validate_oof_for_stacking",
    # CV Orchestrator (PHASE_3)
    "CVOrchestrator",
    "CVFoldResult",
    "CVSplitInfo",
    "get_cv_for_model",
    "create_cv_orchestrator",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _CV_EXPORTS:
        # Show deprecation warning
        warnings.warn(
            f"Importing '{name}' from 'src.cross_validation' is deprecated. "
            f"Use 'from src.validation.cv import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import from new location
        from src.validation import cv as cv_module
        return getattr(cv_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return list of available names."""
    return _CV_EXPORTS


__all__ = _CV_EXPORTS
