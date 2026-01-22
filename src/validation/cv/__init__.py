"""
Validation CV - Re-export from src.cross_validation.

New import path:
    from src.validation.cv import PurgedKFold, CombinatorialPurgedCV

Legacy import path (still works):
    from src.cross_validation import PurgedKFold, CombinatorialPurgedCV
"""

from src.cross_validation import (
    # Purged KFold
    PurgedKFold,
    PurgedKFoldConfig,
    ModelAwareCV,
    # Walk-forward
    WalkForwardFeatureSelector,
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardResult,
    WindowMetrics,
    create_walk_forward_evaluator,
    # OOF Generation
    OOFGenerator,
    OOFPrediction,
    StackingDataset,
    # Cross-validation
    CrossValidationRunner,
    CVResult,
    FoldMetrics,
    PARAM_SPACES,
    TimeSeriesOptunaTuner,
    # CV Feature Selection
    run_cv_with_per_fold_feature_selection,
    compute_feature_stability,
    # CV Stacking
    validate_stacking_consistency,
    build_stacking_datasets_from_cv_results,
    analyze_cv_stability,
    # CPCV
    CPCVConfig,
    CombinatorialPurgedCV,
    CPCVResult,
    CPCVPathResult,
    create_cpcv,
    # PBO
    PBOConfig,
    PBOResult,
    compute_pbo,
    compute_pbo_from_returns,
    pbo_gate,
    analyze_overfitting_risk,
    # Sequence CV
    SequenceCVBuilder,
    SequenceFoldResult,
    build_sequences_for_cv_fold,
    validate_sequence_cv_coverage,
    SequenceOOFGenerator,
    # Stacking
    StackingDatasetBuilder,
    find_valid_samples_mask,
    # Timestamp Alignment
    validate_datetime_alignment,
    align_predictions_on_datetime,
    get_datetime_alignment_report,
    # OOF Cache
    OOFCache,
    OOFCacheEntry,
    compute_data_hash,
    # OOF Alignment
    OOFAlignmentResult,
    OOFAlignmentValidator,
    compute_oof_coverage,
    validate_oof_for_stacking,
    # CV Orchestrator
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
    "WalkForwardConfig",
    "WalkForwardEvaluator",
    "WalkForwardResult",
    "WindowMetrics",
    "create_walk_forward_evaluator",
    "OOFGenerator",
    "OOFPrediction",
    "StackingDataset",
    "CrossValidationRunner",
    "CVResult",
    "FoldMetrics",
    "PARAM_SPACES",
    "TimeSeriesOptunaTuner",
    "run_cv_with_per_fold_feature_selection",
    "compute_feature_stability",
    "validate_stacking_consistency",
    "build_stacking_datasets_from_cv_results",
    "analyze_cv_stability",
    "CPCVConfig",
    "CombinatorialPurgedCV",
    "CPCVResult",
    "CPCVPathResult",
    "create_cpcv",
    "PBOConfig",
    "PBOResult",
    "compute_pbo",
    "compute_pbo_from_returns",
    "pbo_gate",
    "analyze_overfitting_risk",
    "SequenceCVBuilder",
    "SequenceFoldResult",
    "build_sequences_for_cv_fold",
    "validate_sequence_cv_coverage",
    "SequenceOOFGenerator",
    "StackingDatasetBuilder",
    "find_valid_samples_mask",
    "validate_datetime_alignment",
    "align_predictions_on_datetime",
    "get_datetime_alignment_report",
    "OOFCache",
    "OOFCacheEntry",
    "compute_data_hash",
    "OOFAlignmentResult",
    "OOFAlignmentValidator",
    "compute_oof_coverage",
    "validate_oof_for_stacking",
    "CVOrchestrator",
    "CVFoldResult",
    "CVSplitInfo",
    "get_cv_for_model",
    "create_cv_orchestrator",
]
