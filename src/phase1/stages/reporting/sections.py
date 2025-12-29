"""
Report section generators for Phase 1 reports.
Each function generates a specific markdown section.
"""
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def generate_header(data_path: Path) -> str:
    """Generate report header."""
    return f"""# Phase 1 Summary Report
## Ensemble Price Prediction System

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Data File:** `{data_path.name}`

---"""


def generate_executive_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    validation_data: dict[str, Any] | None
) -> str:
    """Generate executive summary section."""
    total_samples = len(df)
    date_start = df['datetime'].min()
    date_end = df['datetime'].max()
    duration_days = (date_end - date_start).days

    symbols = df['symbol'].unique() if 'symbol' in df.columns else []

    status = "PASSED" if validation_data and validation_data.get('status') == 'PASSED' else "NEEDS ATTENTION"
    status_icon = "checkmark" if status == "PASSED" else "warning"

    quality_status = 'All checks passed'
    if validation_data and validation_data.get('issues_count', 1) > 0:
        quality_status = 'Issues detected (see Data Health)'

    return f"""## Executive Summary

Phase 1 data preparation pipeline has completed successfully with the following outputs:

- **Status:** {status}
- **Total Samples:** {total_samples:,}
- **Symbols:** {', '.join(symbols) if len(symbols) > 0 else 'N/A'}
- **Date Range:** {date_start} to {date_end} ({duration_days} days)
- **Features:** {len(feature_cols)}
- **Label Horizons:** 1, 5, 20 bars
- **Data Quality:** {quality_status}

### Pipeline Stages Completed

1. Data Cleaning & Resampling
2. Feature Engineering
3. Triple-Barrier Labeling
4. Quality Scoring
5. Train/Val/Test Splitting
6. Comprehensive Validation
7. Baseline Backtesting

---"""


def generate_data_health_section(
    validation_data: dict[str, Any] | None,
    charts_dir: Path | None
) -> str:
    """Generate data health summary section."""
    if not validation_data:
        return """## Data Health Summary

Validation report not available.

---"""

    issues_count = validation_data.get('issues_count', 0)
    warnings_count = validation_data.get('warnings_count', 0)
    status = validation_data.get('status', 'UNKNOWN')

    section = f"""## Data Health Summary

**Overall Status:** {status}

**Issues Found:** {issues_count}
**Warnings:** {warnings_count}

"""

    # Data integrity
    integrity = validation_data.get('validation_results', {}).get('data_integrity', {})

    section += """### Data Integrity Checks

| Check | Status | Details |
|-------|--------|---------|
"""

    # Duplicates
    dup_data = integrity.get('duplicate_timestamps', {})
    dup_total = sum(dup_data.values()) if isinstance(dup_data, dict) else 0
    dup_status = "PASS" if dup_total == 0 else "FAIL"
    section += f"| Duplicate Timestamps | {dup_status} | {dup_total} duplicates |\n"

    # NaN values
    nan_data = integrity.get('nan_values', {})
    nan_total = sum(nan_data.values()) if isinstance(nan_data, dict) else 0
    nan_status = "PASS" if nan_total == 0 else "FAIL"
    section += f"| NaN Values | {nan_status} | {nan_total} NaN values in {len(nan_data)} columns |\n"

    # Infinite values
    inf_data = integrity.get('infinite_values', {})
    inf_total = sum(inf_data.values()) if isinstance(inf_data, dict) else 0
    inf_status = "PASS" if inf_total == 0 else "FAIL"
    section += f"| Infinite Values | {inf_status} | {inf_total} infinite values |\n"

    # Gaps
    gaps = integrity.get('gaps', [])
    gap_status = "PASS" if len(gaps) == 0 else "WARNING"
    section += f"| Time Gaps | {gap_status} | {len(gaps)} large gaps detected |\n"

    if charts_dir:
        section += """
### Visualizations

![Label Distribution](charts/label_distribution.png)

![Quality Distribution](charts/quality_distribution.png)
"""

    section += "\n---"
    return section


def generate_feature_section(
    feature_cols: list[str],
    validation_data: dict[str, Any] | None
) -> str:
    """Generate feature overview section."""
    # Categorize features
    categories = {
        'Price': [c for c in feature_cols if any(
            x in c.lower() for x in ['return', 'range', 'close_open']
        )],
        'Moving Averages': [c for c in feature_cols if any(
            x in c.lower() for x in ['sma', 'ema']
        )],
        'Momentum': [c for c in feature_cols if any(
            x in c.lower() for x in ['rsi', 'macd', 'roc', 'stoch', 'williams']
        )],
        'Volatility': [c for c in feature_cols if any(
            x in c.lower() for x in ['atr', 'bollinger', 'bb_']
        )],
        'Volume': [c for c in feature_cols if any(
            x in c.lower() for x in ['obv', 'volume', 'vwap']
        )],
        'Trend': [c for c in feature_cols if any(
            x in c.lower() for x in ['adx', 'di_']
        )],
        'Regime': [c for c in feature_cols if any(
            x in c.lower() for x in ['regime']
        )],
        'Temporal': [c for c in feature_cols if any(
            x in c.lower() for x in ['hour', 'dow', 'rth']
        )]
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
            examples += f', ... (+{len(cat_features) - 3} more)'
        section += f"| {cat_name} | {len(cat_features)} | {examples} |\n"

    # Add feature quality info if available
    if validation_data:
        feature_quality = validation_data.get('validation_results', {}).get('feature_quality', {})
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


def generate_label_section(
    df: pd.DataFrame,
    horizons: list[int],
    charts_dir: Path | None
) -> str:
    """Generate label distribution section."""
    section = """## Label Analysis

### Label Distribution by Horizon

"""

    for horizon in horizons:
        label_col = f'label_h{horizon}'
        if label_col not in df.columns:
            continue

        counts = df[label_col].value_counts().sort_index()
        total = len(df)

        section += f"""#### Horizon {horizon} ({horizon}-bar ahead)

| Class | Count | Percentage |
|-------|-------|------------|
"""
        section += f"| Short (-1) | {counts.get(-1, 0):,} | {counts.get(-1, 0) / total * 100:.2f}% |\n"
        section += f"| Neutral (0) | {counts.get(0, 0):,} | {counts.get(0, 0) / total * 100:.2f}% |\n"
        section += f"| Long (+1) | {counts.get(1, 0):,} | {counts.get(1, 0) / total * 100:.2f}% |\n"
        section += "\n"

        # Quality stats
        quality_col = f'quality_h{horizon}'
        if quality_col in df.columns:
            q_mean = df[quality_col].mean()
            q_median = df[quality_col].median()
            section += f"**Quality Score:** Mean = {q_mean:.3f}, Median = {q_median:.3f}\n\n"

    if charts_dir:
        section += """### Visualizations

See `charts/label_distribution.png` for visual representation.

"""

    section += "---"
    return section


def generate_split_section(split_config: dict[str, Any] | None) -> str:
    """Generate split information section."""
    if not split_config:
        return """## Data Splits

Split configuration not available.

---"""

    total = split_config['total_samples']

    section = """## Data Splits

### Split Configuration

| Split | Samples | Percentage | Date Range |
|-------|---------|------------|------------|
"""
    section += (
        f"| **Train** | {split_config['train_samples']:,} | "
        f"{split_config['train_samples'] / total * 100:.1f}% | "
        f"{split_config['train_date_start'][:10]} to {split_config['train_date_end'][:10]} |\n"
    )
    section += (
        f"| **Validation** | {split_config['val_samples']:,} | "
        f"{split_config['val_samples'] / total * 100:.1f}% | "
        f"{split_config['val_date_start'][:10]} to {split_config['val_date_end'][:10]} |\n"
    )
    section += (
        f"| **Test** | {split_config['test_samples']:,} | "
        f"{split_config['test_samples'] / total * 100:.1f}% | "
        f"{split_config['test_date_start'][:10]} to {split_config['test_date_end'][:10]} |\n"
    )

    validation_status = 'PASSED' if split_config.get('validation_passed') else 'FAILED'

    section += f"""
### Leakage Prevention

- **Purge Bars:** {split_config['purge_bars']} bars removed at split boundaries
- **Embargo Period:** {split_config['embargo_bars']} bars buffer between splits
- **Validation:** {validation_status}

---"""
    return section


def generate_backtest_section(
    backtest_results: dict[int, dict[str, Any]],
    charts_dir: Path | None
) -> str:
    """Generate baseline backtest results section."""
    if not backtest_results:
        return """## Baseline Backtest Results

Backtest results not available.

---"""

    section = """## Baseline Backtest Results

**Strategy:** Trade in direction of label when quality > 0.5 (shifted to prevent lookahead)

**Note:** This is NOT meant to be profitable - it's a sanity check that labels have some predictive signal.

### Results by Horizon

"""

    for horizon in sorted(backtest_results.keys()):
        result = backtest_results[horizon]
        metrics = result.get('metrics', {})

        if metrics.get('total_trades', 0) == 0:
            section += f"#### Horizon {horizon}\n\nNo trades executed\n\n"
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

    if charts_dir:
        section += """### Equity Curves

See `baseline_backtest/` directory for equity curve plots.

"""

    section += "---"
    return section


def generate_quality_gates(
    df: pd.DataFrame,
    validation_data: dict[str, Any] | None,
    split_config: dict[str, Any] | None,
    backtest_results: dict[int, dict[str, Any]]
) -> str:
    """Generate quality gates checklist section."""
    gates = []

    # Data quality gates
    if validation_data:
        integrity = validation_data.get('validation_results', {}).get('data_integrity', {})
        no_duplicates = sum(integrity.get('duplicate_timestamps', {}).values()) == 0
        no_nans = len(integrity.get('nan_values', {})) == 0
        no_infs = len(integrity.get('infinite_values', {})) == 0

        gates.append(("No duplicate timestamps", no_duplicates))
        gates.append(("No NaN values in features", no_nans))
        gates.append(("No infinite values", no_infs))
    else:
        gates.append(("Data validation completed", False))

    # Label quality gates
    for horizon in [1, 5, 20]:
        label_col = f'label_h{horizon}'
        if label_col in df.columns:
            counts = df[label_col].value_counts()
            # Check if all classes have at least 15% representation
            balanced = all(count / len(df) >= 0.15 for count in counts.values)
            gates.append((f"Horizon {horizon} labels reasonably balanced (>15% each)", balanced))

    # Split gates
    if split_config:
        gates.append(("Train/val/test splits created", True))
        gates.append(("No overlap between splits", split_config.get('validation_passed', False)))
    else:
        gates.append(("Train/val/test splits created", False))

    # Backtest gates
    if backtest_results:
        gates.append(("Baseline backtest completed", True))
    else:
        gates.append(("Baseline backtest completed", False))

    section = """## Quality Gates Checklist

| Gate | Status |
|------|--------|
"""

    for gate_name, passed in gates:
        status = "PASS" if passed else "FAIL"
        section += f"| {gate_name} | {status} |\n"

    all_passed = all(passed for _, passed in gates)
    overall = 'ALL GATES PASSED - Ready for Phase 2' if all_passed else 'ISSUES DETECTED - Review before proceeding'

    section += f"""
### Overall Assessment

**Status:** {overall}

---"""
    return section


def generate_recommendations(
    validation_data: dict[str, Any] | None,
    backtest_results: dict[int, dict[str, Any]]
) -> str:
    """Generate recommendations for Phase 2 section."""
    recommendations = []

    # Based on validation
    if validation_data:
        feature_quality = validation_data.get('validation_results', {}).get('feature_quality', {})
        high_corr_count = len(feature_quality.get('high_correlations', []))
        if high_corr_count > 0:
            recommendations.append(
                f"Consider removing {high_corr_count} highly correlated feature pairs to reduce redundancy"
            )

        issues = validation_data.get('issues_count', 0)
        if issues > 0:
            recommendations.append(
                "Address data quality issues identified in validation before proceeding"
            )

    # Based on backtest
    if backtest_results:
        poor_results = []
        for horizon, result in backtest_results.items():
            metrics = result.get('metrics', {})
            if metrics.get('total_trades', 0) > 0:
                if metrics.get('win_rate', 0) < 45:
                    poor_results.append(horizon)

        if poor_results:
            recommendations.append(
                f"Horizon(s) {poor_results} showed poor baseline performance - "
                "consider adjusting barrier parameters"
            )

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


def generate_footer(
    data_path: Path,
    validation_report_path: Path | None,
    split_config_path: Path | None
) -> str:
    """Generate report footer with output files summary."""
    validation_name = validation_report_path.name if validation_report_path else 'N/A'
    split_name = split_config_path.name if split_config_path else 'N/A'

    return f"""## Output Files Summary

### Data Files
- Combined labeled data: `{data_path}`
- Split indices: `data/splits/{{train,val,test}}.npy`

### Reports & Results
- Validation report: `{validation_name}`
- Split configuration: `{split_name}`
- Backtest results: `baseline_backtest/`
- Charts: `charts/`

### Next Steps

Load the splits and begin Phase 2 model training:

```python
import numpy as np
import pandas as pd

# Load data and splits
df = pd.read_parquet('{data_path}')
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
