"""
Validation utilities for ML pipeline.

Provides lookahead bias detection, data quality checks,
cross-validation integrity verification, selection bias correction,
statistical model comparison, bootstrap confidence intervals,
feature-label leakage detection, ensemble diversity analysis,
regime-conditional evaluation, production backtesting,
walk-forward validation, CPCV, PBO, and feature store.
"""

from src.inference.backtesting import (
    BacktestConfig,
    BacktestResult,
    Backtester,
    CostCalculator,
    EquityCurve,
    ExecutionModel,
    FixedFractional,
    KellyCriterion,
    PerformanceMetrics,
    Position,
    PositionSizingMethod,
    Trade,
    TransactionCosts,
    VolatilityTargeted,
    calculate_all_metrics,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
    run_backtest,
    run_walk_forward_backtest,
)
from src.validation.cv.walk_forward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardResult,
    WindowMetrics,
    create_walk_forward_evaluator,
)
from src.models.ensemble.diversity import (
    DiversityAnalyzer,
    DiversityMetrics,
    compute_average_disagreement,
    compute_average_double_fault,
    compute_average_kl_divergence,
    compute_average_q_statistic,
    compute_disagreement,
    compute_diversity_score,
    compute_double_fault,
    compute_entropy_of_votes,
    compute_kl_divergence,
    compute_kl_diversity_penalty,
    compute_pairwise_correlation,
    compute_pairwise_correlation_matrix,
    compute_q_statistic,
)
from src.models.regime_evaluation import (
    RegimeClassifier,
    RegimeEvaluationResult,
    RegimeEvaluator,
    RegimeMetrics,
    TimeOfDay,
    TrendRegime,
    VolatilityRegime,
    evaluate_regime_performance,
    get_regime_summary,
)
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

# Phase 5: CPCV (Combinatorial Purged Cross-Validation)
from src.validation.cv.cpcv import (
    CombinatorialPurgedCV,
    CPCVConfig,
    CPCVPathResult,
    CPCVResult,
    create_cpcv,
)

# Phase 5: PBO (Probability of Backtest Overfitting)
from src.validation.cv.pbo import (
    PBOConfig,
    PBOResult,
    analyze_overfitting_risk,
    compute_pbo,
    compute_pbo_from_returns,
    pbo_gate,
)

# Phase 5: Feature Store
from src.data.store import (
    CacheMetadata,
    DataSource,
    FeatureCache,
    FeatureIntegrityError,
    FeatureLineage,
    FeatureNotFoundError,
    FeatureStore,
    FeatureStoreError,
    LineageTracker,
    SemanticVersion,
    Transformation,
    TransformationType,
    VersionInfo,
    VersionManager,
    compute_config_hash,
    compute_dataframe_checksum,
    compute_file_checksum,
    compute_schema_hash,
)

# Phase 6: Meta-Labeling (Lopez de Prado AFML approach)
from src.data.pipeline.stages.meta_labeling import (
    # Primary classifier for high-recall direction prediction
    PrimaryClassifier,
    PrimaryModelConfig,
    RecallOptimizer,
    # Meta-label generation (1=correct, 0=incorrect)
    MetaLabelGenerator,
    MetaLabelingConfig,
    MetaLabelResult,
    # Bet sizing (Kelly, volatility-scaled, etc.)
    BetSizer,
    BetSizingMethod,
    KellyCriterion as MetaKellyCriterion,  # Renamed to avoid conflict with backtesting
    VolatilityScaler,
    # Pipeline integration
    run_meta_labeling,
    add_meta_labels_standalone,
)

# Phase 6: Alternative MetaLabeler from labeling module
from src.data.pipeline.stages.labeling import (
    MetaLabeler,
    BetSizeMethod,
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
    # Ensemble diversity (Phase 3)
    "DiversityAnalyzer",
    "DiversityMetrics",
    "compute_pairwise_correlation",
    "compute_pairwise_correlation_matrix",
    "compute_q_statistic",
    "compute_average_q_statistic",
    "compute_disagreement",
    "compute_average_disagreement",
    "compute_double_fault",
    "compute_average_double_fault",
    "compute_entropy_of_votes",
    "compute_kl_divergence",
    "compute_average_kl_divergence",
    "compute_diversity_score",
    "compute_kl_diversity_penalty",
    # Regime-conditional evaluation (Phase 3)
    "VolatilityRegime",
    "TrendRegime",
    "TimeOfDay",
    "RegimeMetrics",
    "RegimeEvaluationResult",
    "RegimeClassifier",
    "RegimeEvaluator",
    "evaluate_regime_performance",
    "get_regime_summary",
    # Production backtesting (Phase 4)
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "ExecutionModel",
    "Position",
    "Trade",
    "EquityCurve",
    "TransactionCosts",
    "CostCalculator",
    "PositionSizingMethod",
    "KellyCriterion",
    "FixedFractional",
    "VolatilityTargeted",
    "PerformanceMetrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    "calculate_win_rate",
    "calculate_all_metrics",
    "run_backtest",
    "run_walk_forward_backtest",
    # Walk-forward validation (Phase 4)
    "WalkForwardConfig",
    "WalkForwardEvaluator",
    "WalkForwardResult",
    "WindowMetrics",
    "create_walk_forward_evaluator",
    # CPCV - Combinatorial Purged Cross-Validation (Phase 5)
    "CombinatorialPurgedCV",
    "CPCVConfig",
    "CPCVPathResult",
    "CPCVResult",
    "create_cpcv",
    # PBO - Probability of Backtest Overfitting (Phase 5)
    "PBOConfig",
    "PBOResult",
    "compute_pbo",
    "compute_pbo_from_returns",
    "pbo_gate",
    "analyze_overfitting_risk",
    # Feature Store (Phase 5)
    "FeatureStore",
    "FeatureStoreError",
    "FeatureNotFoundError",
    "FeatureIntegrityError",
    "FeatureCache",
    "CacheMetadata",
    "LineageTracker",
    "FeatureLineage",
    "DataSource",
    "Transformation",
    "TransformationType",
    "SemanticVersion",
    "VersionInfo",
    "VersionManager",
    "compute_file_checksum",
    "compute_dataframe_checksum",
    "compute_schema_hash",
    "compute_config_hash",
    # Meta-Labeling - Primary Classifier (Phase 6)
    "PrimaryClassifier",
    "PrimaryModelConfig",
    "RecallOptimizer",
    # Meta-Labeling - Meta Label Generation (Phase 6)
    "MetaLabelGenerator",
    "MetaLabelingConfig",
    "MetaLabelResult",
    # Meta-Labeling - Bet Sizing (Phase 6)
    "BetSizer",
    "BetSizingMethod",
    "MetaKellyCriterion",
    "VolatilityScaler",
    # Meta-Labeling - Pipeline Integration (Phase 6)
    "run_meta_labeling",
    "add_meta_labels_standalone",
    # Meta-Labeling - Alternative Labeler (Phase 6)
    "MetaLabeler",
    "BetSizeMethod",
]
