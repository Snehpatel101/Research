from .charts import (
    generate_all_charts,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_rolling_sharpe,
    plot_trade_analysis,
)

# Financial report generation
from .financial_report import (
    FinancialReport,
    FinancialReportConfig,
    generate_financial_report,
    simulate_trades,
)
from .report_generator import generate_markdown_report, save_report
from .report_schema import (
    DatasetMetrics,
    EvaluationReport,
    ModelConfig,
    ModelMetrics,
    PipelineInfo,
    TrainingInfo,
)

__all__ = [
    # Evaluation report
    "EvaluationReport",
    "ModelMetrics",
    "DatasetMetrics",
    "TrainingInfo",
    "ModelConfig",
    "PipelineInfo",
    "generate_markdown_report",
    "save_report",
    # Financial report
    "FinancialReport",
    "FinancialReportConfig",
    "generate_financial_report",
    "simulate_trades",
    # Charts
    "generate_all_charts",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_rolling_sharpe",
    "plot_trade_analysis",
]
