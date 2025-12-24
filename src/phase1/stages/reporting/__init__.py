"""
Reporting module for Phase 1 pipeline reports.

Public API:
    - Phase1ReportGenerator: Main class for generating reports
    - generate_phase1_report: Convenience function for full report generation
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.horizon_config import LOOKBACK_HORIZONS

from .charts import generate_all_charts
from .formatters import save_html_report
from .sections import (
    generate_backtest_section,
    generate_data_health_section,
    generate_executive_summary,
    generate_feature_section,
    generate_footer,
    generate_header,
    generate_label_section,
    generate_quality_gates,
    generate_recommendations,
    generate_split_section,
)

logger = logging.getLogger(__name__)


class Phase1ReportGenerator:
    """Generate comprehensive Phase 1 summary reports."""

    def __init__(
        self,
        data_path: Path,
        validation_report_path: Optional[Path] = None,
        split_config_path: Optional[Path] = None,
        backtest_results_dir: Optional[Path] = None
    ):
        self.data_path = data_path
        self.validation_report_path = validation_report_path
        self.split_config_path = split_config_path
        self.backtest_results_dir = backtest_results_dir

        # Load data
        logger.info(f"Loading data from {data_path}")
        self.df = pd.read_parquet(data_path)
        logger.info(f"  Loaded {len(self.df):,} rows")

        # Load validation report
        self.validation_data = self._load_json(validation_report_path)

        # Load split config
        self.split_config = self._load_json(split_config_path)

        # Load backtest results
        self.backtest_results = self._load_backtest_results(backtest_results_dir)

        self.charts_dir: Optional[Path] = None

    def _load_json(self, path: Optional[Path]) -> Optional[Dict[str, Any]]:
        """Load JSON file if it exists."""
        if path and path.exists():
            logger.info(f"Loading {path}")
            with open(path) as f:
                return json.load(f)
        return None

    def _load_backtest_results(
        self,
        results_dir: Optional[Path]
    ) -> Dict[int, Dict[str, Any]]:
        """Load backtest results from directory."""
        results = {}
        if results_dir and results_dir.exists():
            for result_file in results_dir.glob("baseline_backtest_h*.json"):
                horizon = int(result_file.stem.split('_h')[1])
                logger.info(f"Loading backtest results from {result_file}")
                with open(result_file) as f:
                    results[horizon] = json.load(f)
        return results

    def identify_feature_columns(self) -> List[str]:
        """Identify feature columns by excluding known non-feature columns."""
        excluded = [
            'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'timeframe', 'session_id', 'missing_bar', 'roll_event', 'roll_window', 'filled'
        ]
        excluded_prefixes = (
            'label_', 'bars_to_hit_', 'mae_', 'mfe_', 'quality_', 'sample_weight_',
            'touch_type_', 'pain_to_gain_', 'time_weighted_dd_', 'fwd_return_',
            'fwd_return_log_', 'time_to_hit_'
        )
        feature_cols = [
            c for c in self.df.columns
            if c not in excluded
            and not any(c.startswith(prefix) for prefix in excluded_prefixes)
        ]
        return feature_cols

    def generate_charts(self, output_dir: Path) -> Path:
        """Generate all charts for the report."""
        self.charts_dir = output_dir / "charts"
        generate_all_charts(self.df, output_dir)
        return self.charts_dir

    def generate_markdown_report(self, output_path: Path) -> str:
        """Generate comprehensive markdown report."""
        logger.info(f"Generating markdown report: {output_path}")

        feature_cols = self.identify_feature_columns()
        horizons = LOOKBACK_HORIZONS

        # Build report sections
        sections = [
            generate_header(self.data_path),
            generate_executive_summary(self.df, feature_cols, self.validation_data),
            generate_data_health_section(self.validation_data, self.charts_dir),
            generate_feature_section(feature_cols, self.validation_data),
            generate_label_section(self.df, horizons, self.charts_dir),
            generate_split_section(self.split_config),
            generate_backtest_section(self.backtest_results, self.charts_dir),
            generate_quality_gates(
                self.df, self.validation_data, self.split_config, self.backtest_results
            ),
            generate_recommendations(self.validation_data, self.backtest_results),
            generate_footer(
                self.data_path, self.validation_report_path, self.split_config_path
            ),
        ]

        report = '\n\n'.join(sections)

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"  Markdown report saved to {output_path}")
        return report

    def generate_html_report(self, markdown_path: Path, output_path: Path) -> None:
        """Generate HTML version of the report."""
        save_html_report(markdown_path, output_path)

    def generate_json_export(self, output_path: Path) -> None:
        """Generate JSON export for programmatic access."""
        logger.info(f"Generating JSON export: {output_path}")

        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_file': str(self.data_path),
                'total_rows': int(len(self.df)),
                'total_columns': int(len(self.df.columns))
            },
            'features': {
                'total': len(self.identify_feature_columns()),
                'columns': self.identify_feature_columns()
            },
            'validation': self.validation_data,
            'splits': self.split_config,
            'backtest': self.backtest_results
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"  JSON export saved to {output_path}")


def generate_phase1_report(
    data_path: Path,
    output_dir: Path,
    validation_report_path: Optional[Path] = None,
    split_config_path: Optional[Path] = None,
    backtest_results_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Generate comprehensive Phase 1 summary report.

    Args:
        data_path: Path to combined labeled data
        output_dir: Output directory for reports
        validation_report_path: Optional path to validation JSON
        split_config_path: Optional path to split config JSON
        backtest_results_dir: Optional directory with backtest results

    Returns:
        Dictionary with paths to generated files
    """
    logger.info("=" * 70)
    logger.info("PHASE 1 REPORT GENERATION")
    logger.info("=" * 70)

    generator = Phase1ReportGenerator(
        data_path=data_path,
        validation_report_path=validation_report_path,
        split_config_path=split_config_path,
        backtest_results_dir=backtest_results_dir
    )

    # Generate charts
    generator.generate_charts(output_dir)

    # Generate markdown report
    md_path = output_dir / "phase1_summary.md"
    generator.generate_markdown_report(md_path)

    # Generate HTML report
    html_path = output_dir / "phase1_summary.html"
    generator.generate_html_report(md_path, html_path)

    # Generate JSON export
    json_path = output_dir / "phase1_summary.json"
    generator.generate_json_export(json_path)

    output_files = {
        'markdown': md_path,
        'html': html_path,
        'json': json_path,
        'charts_dir': generator.charts_dir
    }

    logger.info("\n" + "=" * 70)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nGenerated files:")
    logger.info(f"  Markdown: {md_path}")
    logger.info(f"  HTML:     {html_path}")
    logger.info(f"  JSON:     {json_path}")
    logger.info(f"  Charts:   {generator.charts_dir}")

    return output_files


__all__ = [
    'Phase1ReportGenerator',
    'generate_phase1_report',
]
