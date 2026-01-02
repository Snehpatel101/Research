"""
Stage 9: Report Generation.

Wrapper that imports from src.phase1.stages.reporting.run
"""

from src.phase1.stages.reporting.run import generate_report_content, run_generate_report

__all__ = ["run_generate_report", "generate_report_content"]
