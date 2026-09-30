"""
Cross-validation package for Phase 3: Out-of-Sample Predictions.

Import paths:
    # New (preferred):
    from src.validation.cv import PurgedKFold, CombinatorialPurgedCV

    # Legacy (still works, deprecation warning):
    from src.validation.cv import PurgedKFold, CombinatorialPurgedCV

This package provides time-series aware cross-validation with proper
purging and embargo to prevent information leakage. It generates
out-of-fold predictions for ensemble stacking in Phase 4.

Main components:
- PurgedKFold: Time-series CV with label-aware purging
- WalkForwardFeatureSelector: Walk-forward feature selection
- OOFGenerator: Out-of-fold prediction generator
- CrossValidationRunner: Orchestrates CV for all models/horizons
"""

# WalkForwardFeatureSelector is in optimization.feature_selection
from src.core.label_spans import (
    LabelSpans,
    average_uniqueness,
    label_end_column,
    label_end_positions,
    uniqueness_sample_weights,
)
from src.optimization.feature_selection import WalkForwardFeatureSelector

from .cpcv import (
    CombinatorialPurgedCV,
    CPCVConfig,
    CPCVPathResult,
    CPCVResult,
)
from .cv_dataclasses import CVResult, FoldMetrics
from .cv_feature_selection import run_cv_with_per_fold_feature_selection
from .cv_runner import CrossValidationRunner
from .cv_stacking import (
    analyze_cv_stability,
    build_stacking_datasets_from_cv_results,
    validate_stacking_consistency,
)
from .cv_tuner import TimeSeriesOptunaTuner
from .oof_cache import (
    OOFCache,
    OOFCacheEntry,
    compute_data_hash,
)
from .oof_core import OOFPrediction
from .oof_generator import OOFGenerator, StackingDataset
from .oof_sequence import SequenceOOFGenerator
from .oof_stacking import (
    StackingDatasetBuilder,
    find_valid_samples_mask,
)
from .param_spaces import PARAM_SPACES
from .pbo import (
    PBOConfig,
    PBOResult,
    compute_pbo,
    pbo_gate,
)
from .purged_kfold import (
    PurgedKFold,
    PurgedKFoldConfig,
)
from .sequence_cv import (
    SequenceCVBuilder,
    SequenceFoldResult,
)
from .timestamp_alignment import (
    align_predictions_on_datetime,
    get_datetime_alignment_report,
    validate_datetime_alignment,
)
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardResult,
    WindowMetrics,
)

__all__ = [
    # Label spans (purging + uniqueness weights)
    "LabelSpans",
    "average_uniqueness",
    "label_end_column",
    "label_end_positions",
    "uniqueness_sample_weights",
    "PurgedKFold",
    "PurgedKFoldConfig",
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
    # CV Stacking
    "validate_stacking_consistency",
    "build_stacking_datasets_from_cv_results",
    "analyze_cv_stability",
    # Walk-forward
    "WalkForwardConfig",
    "WalkForwardEvaluator",
    "WalkForwardResult",
    "WindowMetrics",
    # CPCV
    "CPCVConfig",
    "CombinatorialPurgedCV",
    "CPCVResult",
    "CPCVPathResult",
    # PBO
    "PBOConfig",
    "PBOResult",
    "compute_pbo",
    "pbo_gate",
    # Sequence CV
    "SequenceCVBuilder",
    "SequenceFoldResult",
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
]
