"""
Phase 1 Comprehensive Report Generator

This module is a backward-compatible wrapper that re-exports from the
reporting/ package. All implementation has been moved to:

    src/stages/reporting/
        __init__.py   - Main Phase1ReportGenerator class and generate_phase1_report()
        charts.py     - Chart generation functions (matplotlib)
        formatters.py - HTML conversion and formatting utilities
        sections.py   - Markdown report section generators

For new code, prefer importing directly from the package:
    from src.stages.reporting import Phase1ReportGenerator, generate_phase1_report
"""
import logging
from pathlib import Path

# Re-export all public API from the reporting package
from src.stages.reporting import Phase1ReportGenerator, generate_phase1_report

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Explicit exports for backward compatibility
__all__ = [
    'Phase1ReportGenerator',
    'generate_phase1_report',
    'main',
]


def main():
    """Generate Phase 1 report with default configuration."""
    from src.config import FINAL_DATA_DIR, SPLITS_DIR, RESULTS_DIR

    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"
    validation_path = RESULTS_DIR / "validation_report.json"
    split_config_path = SPLITS_DIR / "split_config.json"
    backtest_dir = RESULTS_DIR / "baseline_backtest"
    output_dir = RESULTS_DIR / "reports"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    output_files = generate_phase1_report(
        data_path=data_path,
        output_dir=output_dir,
        validation_report_path=validation_path if validation_path.exists() else None,
        split_config_path=split_config_path if split_config_path.exists() else None,
        backtest_results_dir=backtest_dir if backtest_dir.exists() else None
    )

    logger.info("Phase 1 Report Generation Complete!")


if __name__ == "__main__":
    main()
