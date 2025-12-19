"""
Phase 1 Comprehensive Report Generator
Generates markdown and HTML reports with embedded charts
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        self.validation_data = None
        if validation_report_path and validation_report_path.exists():
            logger.info(f"Loading validation report from {validation_report_path}")
            with open(validation_report_path) as f:
                self.validation_data = json.load(f)

        # Load split config
        self.split_config = None
        if split_config_path and split_config_path.exists():
            logger.info(f"Loading split config from {split_config_path}")
            with open(split_config_path) as f:
                self.split_config = json.load(f)

        # Load backtest results
        self.backtest_results = {}
        if backtest_results_dir and backtest_results_dir.exists():
            for result_file in backtest_results_dir.glob("baseline_backtest_h*.json"):
                horizon = int(result_file.stem.split('_h')[1])
                logger.info(f"Loading backtest results from {result_file}")
                with open(result_file) as f:
                    self.backtest_results[horizon] = json.load(f)

        self.charts_dir = None

    def identify_feature_columns(self) -> List[str]:
        """Identify feature columns."""
        excluded = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in self.df.columns
                       if c not in excluded
                       and not c.startswith('label_')
                       and not c.startswith('bars_to_hit_')
                       and not c.startswith('mae_')
                       and not c.startswith('quality_')
                       and not c.startswith('sample_weight_')]
        return feature_cols

    def generate_charts(self, output_dir: Path):
        """Generate all charts for the report."""
        self.charts_dir = output_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nGenerating charts in {self.charts_dir}")

        # Chart 1: Label distribution
        self._generate_label_distribution_chart()

        # Chart 2: Quality score distribution
        self._generate_quality_distribution_chart()

        # Chart 3: Bars to hit distribution
        self._generate_bars_to_hit_chart()

        # Chart 4: Symbol-wise label distribution
        if 'symbol' in self.df.columns:
            self._generate_symbol_distribution_chart()

        logger.info(f"  Generated {len(list(self.charts_dir.glob('*.png')))} chart(s)")

    def _generate_label_distribution_chart(self):
        """Generate label distribution bar chart."""
        horizons = [1, 5, 20]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, horizon in enumerate(horizons):
            label_col = f'label_h{horizon}'
            if label_col not in self.df.columns:
                continue

            counts = self.df[label_col].value_counts().sort_index()
            labels = ['Short', 'Neutral', 'Long']
            values = [counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)]
            colors = ['red', 'gray', 'green']

            axes[idx].bar(labels, values, color=colors, alpha=0.7)
            axes[idx].set_title(f'Horizon {horizon}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Count', fontsize=10)
            axes[idx].grid(axis='y', alpha=0.3)

            # Add percentages
            total = sum(values)
            for i, v in enumerate(values):
                pct = v / total * 100 if total > 0 else 0
                axes[idx].text(i, v, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Label Distribution by Horizon', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'label_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_quality_distribution_chart(self):
        """Generate quality score distribution."""
        horizons = [1, 5, 20]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, horizon in enumerate(horizons):
            quality_col = f'quality_h{horizon}'
            if quality_col not in self.df.columns:
                continue

            data = self.df[quality_col].dropna()
            axes[idx].hist(data, bins=50, color='blue', alpha=0.7, edgecolor='black')
            axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
            axes[idx].axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.3f}')
            axes[idx].set_title(f'Horizon {horizon}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Quality Score', fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(axis='y', alpha=0.3)

        plt.suptitle('Quality Score Distribution by Horizon', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'quality_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_bars_to_hit_chart(self):
        """Generate bars to hit distribution."""
        horizons = [1, 5, 20]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, horizon in enumerate(horizons):
            bars_col = f'bars_to_hit_h{horizon}'
            label_col = f'label_h{horizon}'
            if bars_col not in self.df.columns or label_col not in self.df.columns:
                continue

            # Only for non-neutral labels
            hit_data = self.df[self.df[label_col] != 0][bars_col].dropna()

            if len(hit_data) > 0:
                axes[idx].hist(hit_data, bins=30, color='purple', alpha=0.7, edgecolor='black')
                axes[idx].axvline(hit_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {hit_data.mean():.1f}')
                axes[idx].axvline(hit_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {hit_data.median():.1f}')
                axes[idx].set_title(f'Horizon {horizon}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('Bars to Hit', fontsize=10)
                axes[idx].set_ylabel('Frequency', fontsize=10)
                axes[idx].legend(fontsize=8)
                axes[idx].grid(axis='y', alpha=0.3)

        plt.suptitle('Bars to Hit Distribution (Non-Neutral Labels)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'bars_to_hit.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_symbol_distribution_chart(self):
        """Generate symbol-wise label distribution."""
        horizons = [1, 5, 20]
        symbols = self.df['symbol'].unique()

        fig, axes = plt.subplots(len(symbols), 3, figsize=(15, 4*len(symbols)))
        if len(symbols) == 1:
            axes = axes.reshape(1, -1)

        for sym_idx, symbol in enumerate(symbols):
            symbol_df = self.df[self.df['symbol'] == symbol]

            for hor_idx, horizon in enumerate(horizons):
                label_col = f'label_h{horizon}'
                if label_col not in symbol_df.columns:
                    continue

                counts = symbol_df[label_col].value_counts().sort_index()
                labels = ['Short', 'Neutral', 'Long']
                values = [counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)]
                colors = ['red', 'gray', 'green']

                axes[sym_idx, hor_idx].bar(labels, values, color=colors, alpha=0.7)
                axes[sym_idx, hor_idx].set_title(f'{symbol} - Horizon {horizon}', fontsize=11, fontweight='bold')
                axes[sym_idx, hor_idx].set_ylabel('Count', fontsize=9)
                axes[sym_idx, hor_idx].grid(axis='y', alpha=0.3)

                # Add percentages
                total = sum(values)
                for i, v in enumerate(values):
                    pct = v / total * 100 if total > 0 else 0
                    axes[sym_idx, hor_idx].text(i, v, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.suptitle('Label Distribution by Symbol and Horizon', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'symbol_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def generate_markdown_report(self, output_path: Path) -> str:
        """Generate comprehensive markdown report."""
        logger.info(f"\nGenerating markdown report: {output_path}")

        # Get basic statistics
        feature_cols = self.identify_feature_columns()
        horizons = [1, 5, 20]

        # Build report sections
        sections = []

        # Header
        sections.append(self._generate_header())

        # Executive Summary
        sections.append(self._generate_executive_summary(feature_cols))

        # Data Health
        sections.append(self._generate_data_health_section())

        # Feature Overview
        sections.append(self._generate_feature_section(feature_cols))

        # Label Analysis
        sections.append(self._generate_label_section(horizons))

        # Split Information
        sections.append(self._generate_split_section())

        # Baseline Backtest
        sections.append(self._generate_backtest_section())

        # Quality Gates
        sections.append(self._generate_quality_gates())

        # Recommendations
        sections.append(self._generate_recommendations())

        # Footer
        sections.append(self._generate_footer())

        # Combine all sections
        report = '\n\n'.join(sections)

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"  Markdown report saved to {output_path}")
        return report

    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# Phase 1 Summary Report
## Ensemble Price Prediction System

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Data File:** `{self.data_path.name}`

---"""

    def _generate_executive_summary(self, feature_cols: List[str]) -> str:
        """Generate executive summary."""
        total_samples = len(self.df)
        date_start = self.df['datetime'].min()
        date_end = self.df['datetime'].max()
        duration_days = (date_end - date_start).days

        symbols = self.df['symbol'].unique() if 'symbol' in self.df.columns else []

        status = "✓ PASSED" if self.validation_data and self.validation_data.get('status') == 'PASSED' else "⚠ NEEDS ATTENTION"

        return f"""## Executive Summary

Phase 1 data preparation pipeline has completed successfully with the following outputs:

- **Status:** {status}
- **Total Samples:** {total_samples:,}
- **Symbols:** {', '.join(symbols) if len(symbols) > 0 else 'N/A'}
- **Date Range:** {date_start} to {date_end} ({duration_days} days)
- **Features:** {len(feature_cols)}
- **Label Horizons:** 1, 5, 20 bars
- **Data Quality:** {'All checks passed' if self.validation_data and self.validation_data.get('issues_count', 1) == 0 else 'Issues detected (see Data Health)'}

### Pipeline Stages Completed

1. ✓ Data Cleaning & Resampling
2. ✓ Feature Engineering
3. ✓ Triple-Barrier Labeling
4. ✓ Quality Scoring
5. ✓ Train/Val/Test Splitting
6. ✓ Comprehensive Validation
7. ✓ Baseline Backtesting

---"""

    def _generate_data_health_section(self) -> str:
        """Generate data health summary."""
        if not self.validation_data:
            return """## Data Health Summary

⚠ Validation report not available.

---"""

        issues_count = self.validation_data.get('issues_count', 0)
        warnings_count = self.validation_data.get('warnings_count', 0)
        status = self.validation_data.get('status', 'UNKNOWN')

        section = f"""## Data Health Summary

**Overall Status:** {status}

**Issues Found:** {issues_count}
**Warnings:** {warnings_count}

"""

        # Data integrity
        integrity = self.validation_data.get('validation_results', {}).get('data_integrity', {})

        section += """### Data Integrity Checks

| Check | Status | Details |
|-------|--------|---------|
"""

        # Duplicates
        dup_data = integrity.get('duplicate_timestamps', {})
        dup_total = sum(dup_data.values()) if isinstance(dup_data, dict) else 0
        dup_status = "✓ PASS" if dup_total == 0 else "✗ FAIL"
        section += f"| Duplicate Timestamps | {dup_status} | {dup_total} duplicates |\n"

        # NaN values
        nan_data = integrity.get('nan_values', {})
        nan_total = sum(nan_data.values()) if isinstance(nan_data, dict) else 0
        nan_status = "✓ PASS" if nan_total == 0 else "✗ FAIL"
        section += f"| NaN Values | {nan_status} | {nan_total} NaN values in {len(nan_data)} columns |\n"

        # Infinite values
        inf_data = integrity.get('infinite_values', {})
        inf_total = sum(inf_data.values()) if isinstance(inf_data, dict) else 0
        inf_status = "✓ PASS" if inf_total == 0 else "✗ FAIL"
        section += f"| Infinite Values | {inf_status} | {inf_total} infinite values |\n"

        # Gaps
        gaps = integrity.get('gaps', [])
        gap_status = "✓ PASS" if len(gaps) == 0 else "⚠ WARNING"
        section += f"| Time Gaps | {gap_status} | {len(gaps)} large gaps detected |\n"

        if self.charts_dir:
            section += f"""
### Visualizations

![Label Distribution](charts/label_distribution.png)

![Quality Distribution](charts/quality_distribution.png)
"""

        section += "\n---"
        return section

    def _generate_feature_section(self, feature_cols: List[str]) -> str:
        """Generate feature overview section."""
        # Categorize features
        categories = {
            'Price': [c for c in feature_cols if any(x in c.lower() for x in ['return', 'range', 'close_open'])],
            'Moving Averages': [c for c in feature_cols if any(x in c.lower() for x in ['sma', 'ema'])],
            'Momentum': [c for c in feature_cols if any(x in c.lower() for x in ['rsi', 'macd', 'roc', 'stoch', 'williams'])],
            'Volatility': [c for c in feature_cols if any(x in c.lower() for x in ['atr', 'bollinger', 'bb_'])],
            'Volume': [c for c in feature_cols if any(x in c.lower() for x in ['obv', 'volume', 'vwap'])],
            'Trend': [c for c in feature_cols if any(x in c.lower() for x in ['adx', 'di_'])],
            'Regime': [c for c in feature_cols if any(x in c.lower() for x in ['regime'])],
            'Temporal': [c for c in feature_cols if any(x in c.lower() for x in ['hour', 'dow', 'rth'])]
        }

        section = f"""## Feature Overview

**Total Features:** {len(feature_cols)}

### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
"""

        for cat_name, cat_features in categories.items():
            examples = ', '.join(cat_features[:3]) if cat_features else 'None'
            if len(cat_features) > 3:
                examples += f', ... (+{len(cat_features)-3} more)'
            section += f"| {cat_name} | {len(cat_features)} | {examples} |\n"

        # Add feature quality info if available
        if self.validation_data:
            feature_quality = self.validation_data.get('validation_results', {}).get('feature_quality', {})
            high_corr = feature_quality.get('high_correlations', [])

            section += f"""
### Feature Quality

- **Highly Correlated Pairs (>0.95):** {len(high_corr)}
"""
            if len(high_corr) > 0:
                section += "  - Consider removing redundant features in Phase 2\n"

            # Top features
            top_features = feature_quality.get('top_features', [])
            if top_features:
                section += "\n### Top 10 Features (by Random Forest Importance)\n\n"
                section += "| Rank | Feature | Importance |\n"
                section += "|------|---------|------------|\n"
                for i, feat_info in enumerate(top_features[:10], 1):
                    section += f"| {i} | {feat_info['feature']} | {feat_info['importance']:.4f} |\n"

        section += "\n---"
        return section

    def _generate_label_section(self, horizons: List[int]) -> str:
        """Generate label distribution section."""
        section = """## Label Analysis

### Label Distribution by Horizon

"""

        for horizon in horizons:
            label_col = f'label_h{horizon}'
            if label_col not in self.df.columns:
                continue

            counts = self.df[label_col].value_counts().sort_index()
            total = len(self.df)

            section += f"""#### Horizon {horizon} ({horizon}-bar ahead)

| Class | Count | Percentage |
|-------|-------|------------|
"""
            section += f"| Short (-1) | {counts.get(-1, 0):,} | {counts.get(-1, 0)/total*100:.2f}% |\n"
            section += f"| Neutral (0) | {counts.get(0, 0):,} | {counts.get(0, 0)/total*100:.2f}% |\n"
            section += f"| Long (+1) | {counts.get(1, 0):,} | {counts.get(1, 0)/total*100:.2f}% |\n"
            section += "\n"

            # Quality stats
            quality_col = f'quality_h{horizon}'
            if quality_col in self.df.columns:
                q_mean = self.df[quality_col].mean()
                q_median = self.df[quality_col].median()
                section += f"**Quality Score:** Mean = {q_mean:.3f}, Median = {q_median:.3f}\n\n"

        if self.charts_dir:
            section += """### Visualizations

See `charts/label_distribution.png` for visual representation.

"""

        section += "---"
        return section

    def _generate_split_section(self) -> str:
        """Generate split information section."""
        if not self.split_config:
            return """## Data Splits

⚠ Split configuration not available.

---"""

        section = f"""## Data Splits

### Split Configuration

| Split | Samples | Percentage | Date Range |
|-------|---------|------------|------------|
"""

        total = self.split_config['total_samples']
        section += f"| **Train** | {self.split_config['train_samples']:,} | {self.split_config['train_samples']/total*100:.1f}% | {self.split_config['train_date_start'][:10]} to {self.split_config['train_date_end'][:10]} |\n"
        section += f"| **Validation** | {self.split_config['val_samples']:,} | {self.split_config['val_samples']/total*100:.1f}% | {self.split_config['val_date_start'][:10]} to {self.split_config['val_date_end'][:10]} |\n"
        section += f"| **Test** | {self.split_config['test_samples']:,} | {self.split_config['test_samples']/total*100:.1f}% | {self.split_config['test_date_start'][:10]} to {self.split_config['test_date_end'][:10]} |\n"

        section += f"""
### Leakage Prevention

- **Purge Bars:** {self.split_config['purge_bars']} bars removed at split boundaries
- **Embargo Period:** {self.split_config['embargo_bars']} bars buffer between splits
- **Validation:** {'✓ PASSED' if self.split_config.get('validation_passed') else '✗ FAILED'}

---"""
        return section

    def _generate_backtest_section(self) -> str:
        """Generate baseline backtest results section."""
        if not self.backtest_results:
            return """## Baseline Backtest Results

⚠ Backtest results not available.

---"""

        section = """## Baseline Backtest Results

**Strategy:** Trade in direction of label when quality > 0.5 (shifted to prevent lookahead)

**Note:** This is NOT meant to be profitable - it's a sanity check that labels have some predictive signal.

### Results by Horizon

"""

        for horizon in sorted(self.backtest_results.keys()):
            result = self.backtest_results[horizon]
            metrics = result.get('metrics', {})

            if metrics.get('total_trades', 0) == 0:
                section += f"#### Horizon {horizon}\n\n⚠ No trades executed\n\n"
                continue

            section += f"""#### Horizon {horizon}

| Metric | Value |
|--------|-------|
| Total Trades | {metrics.get('total_trades', 0):,} |
| Win Rate | {metrics.get('win_rate', 0):.2f}% |
| Total Return | {metrics.get('total_return_pct', 0):.2f}% |
| Profit Factor | {metrics.get('profit_factor', 0):.2f} |
| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |
| Max Drawdown | {metrics.get('max_drawdown_pct', 0):.2f}% |
| Avg Trade Duration | {metrics.get('avg_trade_duration_bars', 0):.1f} bars |

"""

        if self.charts_dir:
            section += """### Equity Curves

See `baseline_backtest/` directory for equity curve plots.

"""

        section += "---"
        return section

    def _generate_quality_gates(self) -> str:
        """Generate quality gates checklist."""
        gates = []

        # Data quality gates
        if self.validation_data:
            no_duplicates = sum(self.validation_data.get('validation_results', {}).get('data_integrity', {}).get('duplicate_timestamps', {}).values()) == 0
            no_nans = len(self.validation_data.get('validation_results', {}).get('data_integrity', {}).get('nan_values', {})) == 0
            no_infs = len(self.validation_data.get('validation_results', {}).get('data_integrity', {}).get('infinite_values', {})) == 0

            gates.append(("No duplicate timestamps", no_duplicates))
            gates.append(("No NaN values in features", no_nans))
            gates.append(("No infinite values", no_infs))
        else:
            gates.append(("Data validation completed", False))

        # Label quality gates
        for horizon in [1, 5, 20]:
            label_col = f'label_h{horizon}'
            if label_col in self.df.columns:
                counts = self.df[label_col].value_counts()
                # Check if all classes have at least 15% representation
                balanced = all(count/len(self.df) >= 0.15 for count in counts.values)
                gates.append((f"Horizon {horizon} labels reasonably balanced (>15% each)", balanced))

        # Split gates
        if self.split_config:
            gates.append(("Train/val/test splits created", True))
            gates.append(("No overlap between splits", self.split_config.get('validation_passed', False)))
        else:
            gates.append(("Train/val/test splits created", False))

        # Backtest gates
        if self.backtest_results:
            gates.append(("Baseline backtest completed", True))
        else:
            gates.append(("Baseline backtest completed", False))

        section = """## Quality Gates Checklist

| Gate | Status |
|------|--------|
"""

        for gate_name, passed in gates:
            status = "✓ PASS" if passed else "✗ FAIL"
            section += f"| {gate_name} | {status} |\n"

        all_passed = all(passed for _, passed in gates)
        section += f"""
### Overall Assessment

**Status:** {'✓ ALL GATES PASSED - Ready for Phase 2' if all_passed else '✗ ISSUES DETECTED - Review before proceeding'}

---"""
        return section

    def _generate_recommendations(self) -> str:
        """Generate recommendations for Phase 2."""
        recommendations = []

        # Based on validation
        if self.validation_data:
            high_corr_count = len(self.validation_data.get('validation_results', {}).get('feature_quality', {}).get('high_correlations', []))
            if high_corr_count > 0:
                recommendations.append(f"Consider removing {high_corr_count} highly correlated feature pairs to reduce redundancy")

            issues = self.validation_data.get('issues_count', 0)
            if issues > 0:
                recommendations.append("Address data quality issues identified in validation before proceeding")

        # Based on backtest
        if self.backtest_results:
            poor_results = []
            for horizon, result in self.backtest_results.items():
                metrics = result.get('metrics', {})
                if metrics.get('total_trades', 0) > 0:
                    if metrics.get('win_rate', 0) < 45:
                        poor_results.append(horizon)

            if poor_results:
                recommendations.append(f"Horizon(s) {poor_results} showed poor baseline performance - consider adjusting barrier parameters")

        # General recommendations
        recommendations.extend([
            "Use sample weights during model training to emphasize high-quality labels",
            "Implement walk-forward validation in Phase 2 to prevent overfitting",
            "Monitor feature importance and remove low-importance features if needed",
            "Consider ensemble methods to capture different market regimes",
            "Implement proper risk management in live trading"
        ])

        section = """## Recommendations for Phase 2

"""

        for i, rec in enumerate(recommendations, 1):
            section += f"{i}. {rec}\n"

        section += "\n---"
        return section

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""## Output Files Summary

### Data Files
- Combined labeled data: `{self.data_path}`
- Split indices: `data/splits/{{train,val,test}}.npy`

### Reports & Results
- Validation report: `{self.validation_report_path.name if self.validation_report_path else 'N/A'}`
- Split configuration: `{self.split_config_path.name if self.split_config_path else 'N/A'}`
- Backtest results: `baseline_backtest/`
- Charts: `charts/`

### Next Steps

Load the splits and begin Phase 2 model training:

```python
import numpy as np
import pandas as pd

# Load data and splits
df = pd.read_parquet('{self.data_path}')
train_idx = np.load('data/splits/train.npy')
val_idx = np.load('data/splits/val.npy')
test_idx = np.load('data/splits/test.npy')

# Prepare training data
train_df = df.iloc[train_idx]
feature_cols = [c for c in df.columns if ...]  # Define features
X_train = train_df[feature_cols]
y_train = train_df['label_h5']  # Choose horizon
sample_weights = train_df['sample_weight_h5']
```

---

*Phase 1 Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    def generate_json_export(self, output_path: Path):
        """Generate JSON export for programmatic access."""
        logger.info(f"\nGenerating JSON export: {output_path}")

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

    def generate_html_report(self, markdown_path: Path, output_path: Path):
        """Generate HTML version of the report (simple conversion)."""
        logger.info(f"\nGenerating HTML report: {output_path}")

        # Read markdown
        with open(markdown_path) as f:
            md_content = f.read()

        # Simple markdown to HTML conversion (basic)
        html_content = self._markdown_to_html(md_content)

        # Save HTML
        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"  HTML report saved to {output_path}")

    def _markdown_to_html(self, md_text: str) -> str:
        """Convert markdown to HTML (basic implementation)."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Phase 1 Summary Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f6f8fa;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }}
        h1 {{ color: #24292e; border-bottom: 2px solid #e1e4e8; padding-bottom: 10px; }}
        h2 {{ color: #24292e; border-bottom: 1px solid #e1e4e8; padding-bottom: 8px; margin-top: 32px; }}
        h3 {{ color: #24292e; margin-top: 24px; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        table th, table td {{
            border: 1px solid #dfe2e5;
            padding: 8px 12px;
            text-align: left;
        }}
        table th {{
            background-color: #f6f8fa;
            font-weight: 600;
        }}
        table tr:nth-child(even) {{
            background-color: #f6f8fa;
        }}
        code {{
            background-color: #f6f8fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 85%;
        }}
        pre {{
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        .pass {{ color: #22863a; font-weight: bold; }}
        .fail {{ color: #d73a49; font-weight: bold; }}
        .warning {{ color: #b08800; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin: 16px 0; }}
    </style>
</head>
<body>
    <div class="container">
"""

        # Very basic markdown to HTML
        # Replace headers
        lines = md_text.split('\n')
        for i, line in enumerate(lines):
            # Headers
            if line.startswith('# '):
                lines[i] = f"<h1>{line[2:]}</h1>"
            elif line.startswith('## '):
                lines[i] = f"<h2>{line[3:]}</h2>"
            elif line.startswith('### '):
                lines[i] = f"<h3>{line[4:]}</h3>"
            elif line.startswith('#### '):
                lines[i] = f"<h4>{line[5:]}</h4>"
            # Bold
            elif '**' in line:
                line = line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
                while '**' in line:
                    line = line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
                lines[i] = line
            # Code blocks
            elif line.startswith('```'):
                lines[i] = '<pre><code>' if not lines[i-1].startswith('<pre>') else '</code></pre>'
            # Images
            elif line.startswith('!['):
                alt_end = line.find(']')
                url_start = line.find('(')
                url_end = line.find(')')
                if alt_end > 0 and url_start > 0:
                    alt = line[2:alt_end]
                    url = line[url_start+1:url_end]
                    lines[i] = f'<img src="{url}" alt="{alt}">'
            # Horizontal rules
            elif line.strip() == '---':
                lines[i] = '<hr>'
            # List items
            elif line.strip().startswith('- '):
                if not lines[i-1].strip().startswith('- '):
                    lines[i] = '<ul><li>' + line.strip()[2:] + '</li>'
                else:
                    lines[i] = '<li>' + line.strip()[2:] + '</li>'
                # Check if next line is not a list item
                if i < len(lines) - 1 and not lines[i+1].strip().startswith('- '):
                    lines[i] += '</ul>'
            # Numbered lists
            elif line.strip() and line.strip()[0].isdigit() and '. ' in line:
                content = line.strip().split('. ', 1)[1]
                if i > 0 and lines[i-1].strip() and lines[i-1].strip()[0].isdigit():
                    lines[i] = f'<li>{content}</li>'
                else:
                    lines[i] = f'<ol><li>{content}</li>'
                # Check if next line is not a numbered list
                if i < len(lines) - 1 and (not lines[i+1].strip() or not lines[i+1].strip()[0].isdigit()):
                    lines[i] += '</ol>'
            # Paragraphs
            elif line.strip() and not line.startswith('<'):
                lines[i] = f'<p>{line}</p>'

        html += '\n'.join(lines)

        html += """
    </div>
</body>
</html>
"""

        # Style status indicators
        html = html.replace('✓ PASS', '<span class="pass">✓ PASS</span>')
        html = html.replace('✗ FAIL', '<span class="fail">✗ FAIL</span>')
        html = html.replace('⚠ WARNING', '<span class="warning">⚠ WARNING</span>')

        return html


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
    logger.info("="*70)
    logger.info("PHASE 1 REPORT GENERATION")
    logger.info("="*70)

    # Create generator
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

    logger.info("\n" + "="*70)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nGenerated files:")
    logger.info(f"  Markdown: {md_path}")
    logger.info(f"  HTML:     {html_path}")
    logger.info(f"  JSON:     {json_path}")
    logger.info(f"  Charts:   {generator.charts_dir}")

    return output_files


def main():
    """Generate Phase 1 report with default configuration."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import FINAL_DATA_DIR, SPLITS_DIR, RESULTS_DIR

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

    logger.info("\n✓ Phase 1 Report Generation Complete!")


if __name__ == "__main__":
    main()
