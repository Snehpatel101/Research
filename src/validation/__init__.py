"""
Validation utilities for ML pipeline.

Provides lookahead bias detection, data quality checks,
cross-validation integrity verification, and selection bias correction.
"""

from src.validation.deflated_sharpe import (
    DSRConfig,
    DSRResult,
    analyze_selection_bias,
    compute_deflated_sharpe,
    compute_dsr_from_optuna_study,
    dsr_gate,
)
from src.validation.lookahead_audit import (
    LookaheadAuditor,
    LookaheadAuditResult,
    ResampleConfig,
    audit_feature_lookahead,
    audit_mtf_alignment,
    validate_resample_config,
)

__all__ = [
    # Lookahead audit
    "LookaheadAuditor",
    "LookaheadAuditResult",
    "ResampleConfig",
    "validate_resample_config",
    "audit_feature_lookahead",
    "audit_mtf_alignment",
    # Deflated Sharpe Ratio
    "DSRConfig",
    "DSRResult",
    "compute_deflated_sharpe",
    "compute_dsr_from_optuna_study",
    "dsr_gate",
    "analyze_selection_bias",
]
