"""
Validation utilities for ML pipeline.

Provides lookahead bias detection, data quality checks,
cross-validation integrity verification, selection bias correction,
statistical model comparison, bootstrap confidence intervals,
and feature-label leakage detection.
"""

from src.validation.bootstrap import (
    BootstrapResult,
    bootstrap_accuracy,
    bootstrap_f1_score,
    bootstrap_max_drawdown,
    bootstrap_metric,
    bootstrap_multiple_metrics,
    bootstrap_sharpe_ratio,
    bootstrap_win_rate,
)
from src.validation.deflated_sharpe import (
    DSRConfig,
    DSRResult,
    analyze_selection_bias,
    compute_deflated_sharpe,
    compute_dsr_from_optuna_study,
    dsr_gate,
)
from src.validation.leakage_detection import (
    LeakageCheckResult,
    LeakageReport,
    check_feature_label_correlation,
    check_information_leakage,
    check_temporal_leakage,
    comprehensive_leakage_check,
)
from src.validation.lookahead_audit import (
    LookaheadAuditor,
    LookaheadAuditResult,
    ResampleConfig,
    audit_feature_lookahead,
    audit_mtf_alignment,
    validate_resample_config,
)
from src.validation.statistical_tests import (
    LossFunction,
    StatisticalTestResult,
    compare_models,
    diebold_mariano_test,
    paired_ttest,
    wilcoxon_test,
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
    # Statistical tests
    "LossFunction",
    "StatisticalTestResult",
    "diebold_mariano_test",
    "paired_ttest",
    "wilcoxon_test",
    "compare_models",
    # Bootstrap confidence intervals
    "BootstrapResult",
    "bootstrap_metric",
    "bootstrap_sharpe_ratio",
    "bootstrap_max_drawdown",
    "bootstrap_accuracy",
    "bootstrap_f1_score",
    "bootstrap_win_rate",
    "bootstrap_multiple_metrics",
    # Leakage detection
    "LeakageCheckResult",
    "LeakageReport",
    "check_feature_label_correlation",
    "check_temporal_leakage",
    "check_information_leakage",
    "comprehensive_leakage_check",
]
