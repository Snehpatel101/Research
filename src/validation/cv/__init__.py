"""
Cross-validation package for Phase 3: Out-of-Sample Predictions.

Import paths:
    # New (preferred):
    from src.validation.cv import PurgedKFold, CombinatorialPurgedCV

    # Legacy (still works, deprecation warning):
    from src.cross_validation import PurgedKFold, CombinatorialPurgedCV

This package provides time-series aware cross-validation with proper
purging and embargo to prevent information leakage. It generates
out-of-fold predictions for ensemble stacking in Phase 4.

Main components:
- PurgedKFold: Time-series CV with label-aware purging
- WalkForwardFeatureSelector: Walk-forward feature selection
- OOFGenerator: Out-of-fold prediction generator
- CrossValidationRunner: Orchestrates CV for all models/horizons
"""

from .cpcv import (
    CombinatorialPurgedCV,
    CPCVConfig,
    CPCVPathResult,
    CPCVResult,
    create_cpcv,
)
from .oof_cache import (
    OOFCache,
    OOFCacheEntry,
    compute_data_hash,
)
from .cv_dataclasses import CVResult, FoldMetrics
from .cv_feature_selection import (
    compute_feature_stability,
    run_cv_with_per_fold_feature_selection,
)
from .cv_runner import CrossValidationRunner
from .cv_stacking import (
    analyze_cv_stability,
    build_stacking_datasets_from_cv_results,
    validate_stacking_consistency,
)
from .cv_tuner import TimeSeriesOptunaTuner
# WalkForwardFeatureSelector is in optimization.feature_selection
from src.optimization.feature_selection import WalkForwardFeatureSelector
from .oof_core import OOFPrediction
from .oof_generator import OOFGenerator, StackingDataset
from .oof_sequence import SequenceOOFGenerator
from .oof_stacking import (
    StackingDatasetBuilder,
    find_valid_samples_mask,
)
from .timestamp_alignment import (
    align_predictions_on_datetime,
    get_datetime_alignment_report,
    validate_datetime_alignment,
)
from .oof_alignment import (
    OOFAlignmentResult,
    OOFAlignmentValidator,
    compute_oof_coverage,
    validate_oof_for_stacking,
)
from .param_spaces import PARAM_SPACES
from .pbo import (
    PBOConfig,
    PBOResult,
    analyze_overfitting_risk,
    compute_pbo,
    compute_pbo_from_returns,
    pbo_gate,
)
from .purged_kfold import ModelAwareCV, PurgedKFold, PurgedKFoldConfig
from .sequence_cv import (
    SequenceCVBuilder,
    SequenceFoldResult,
    build_sequences_for_cv_fold,
    validate_sequence_cv_coverage,
)
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardResult,
    WindowMetrics,
    create_walk_forward_evaluator,
)
from .cv_orchestrator import (
    CVOrchestrator,
    CVFoldResult,
    CVSplitInfo,
    get_cv_for_model,
    create_cv_orchestrator,
)

__all__ = [
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
