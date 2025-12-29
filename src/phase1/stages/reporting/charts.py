"""
Chart generation for Phase 1 reports.
Generates matplotlib visualizations for label distributions, quality scores, etc.
"""
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.common.horizon_config import LOOKBACK_HORIZONS

matplotlib.use('Agg')

logger = logging.getLogger(__name__)


def generate_all_charts(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Generate all charts for the report.

    Args:
        df: DataFrame with labeled data
        output_dir: Base output directory

    Returns:
        Path to charts directory
    """
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating charts in {charts_dir}")

    generate_label_distribution_chart(df, charts_dir)
    generate_quality_distribution_chart(df, charts_dir)
    generate_bars_to_hit_chart(df, charts_dir)

    if 'symbol' in df.columns:
        generate_symbol_distribution_chart(df, charts_dir)

    chart_count = len(list(charts_dir.glob('*.png')))
    logger.info(f"  Generated {chart_count} chart(s)")

    return charts_dir


def generate_label_distribution_chart(df: pd.DataFrame, charts_dir: Path) -> None:
    """Generate label distribution bar chart for all horizons."""
    horizons = LOOKBACK_HORIZONS
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, horizon in enumerate(horizons):
        label_col = f'label_h{horizon}'
        if label_col not in df.columns:
            continue

        counts = df[label_col].value_counts().sort_index()
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
    plt.savefig(charts_dir / 'label_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_quality_distribution_chart(df: pd.DataFrame, charts_dir: Path) -> None:
    """Generate quality score distribution histogram for all horizons."""
    horizons = LOOKBACK_HORIZONS
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, horizon in enumerate(horizons):
        quality_col = f'quality_h{horizon}'
        if quality_col not in df.columns:
            continue

        data = df[quality_col].dropna()
        axes[idx].hist(data, bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[idx].axvline(
            data.mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {data.mean():.3f}'
        )
        axes[idx].axvline(
            data.median(), color='green', linestyle='--',
            linewidth=2, label=f'Median: {data.median():.3f}'
        )
        axes[idx].set_title(f'Horizon {horizon}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Quality Score', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(axis='y', alpha=0.3)

    plt.suptitle('Quality Score Distribution by Horizon', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(charts_dir / 'quality_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_bars_to_hit_chart(df: pd.DataFrame, charts_dir: Path) -> None:
    """Generate bars to hit distribution for non-neutral labels."""
    horizons = LOOKBACK_HORIZONS
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, horizon in enumerate(horizons):
        bars_col = f'bars_to_hit_h{horizon}'
        label_col = f'label_h{horizon}'
        if bars_col not in df.columns or label_col not in df.columns:
            continue

        # Only for non-neutral labels
        hit_data = df[df[label_col] != 0][bars_col].dropna()

        if len(hit_data) > 0:
            axes[idx].hist(hit_data, bins=30, color='purple', alpha=0.7, edgecolor='black')
            axes[idx].axvline(
                hit_data.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {hit_data.mean():.1f}'
            )
            axes[idx].axvline(
                hit_data.median(), color='green', linestyle='--',
                linewidth=2, label=f'Median: {hit_data.median():.1f}'
            )
            axes[idx].set_title(f'Horizon {horizon}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Bars to Hit', fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(axis='y', alpha=0.3)

    plt.suptitle(
        'Bars to Hit Distribution (Non-Neutral Labels)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(charts_dir / 'bars_to_hit.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_symbol_distribution_chart(df: pd.DataFrame, charts_dir: Path) -> None:
    """Generate symbol-wise label distribution chart."""
    horizons = LOOKBACK_HORIZONS
    symbols = df['symbol'].unique()

    fig, axes = plt.subplots(len(symbols), 3, figsize=(15, 4 * len(symbols)))
    if len(symbols) == 1:
        axes = axes.reshape(1, -1)

    for sym_idx, symbol in enumerate(symbols):
        symbol_df = df[df['symbol'] == symbol]

        for hor_idx, horizon in enumerate(horizons):
            label_col = f'label_h{horizon}'
            if label_col not in symbol_df.columns:
                continue

            counts = symbol_df[label_col].value_counts().sort_index()
            labels = ['Short', 'Neutral', 'Long']
            values = [counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)]
            colors = ['red', 'gray', 'green']

            axes[sym_idx, hor_idx].bar(labels, values, color=colors, alpha=0.7)
            axes[sym_idx, hor_idx].set_title(
                f'{symbol} - Horizon {horizon}', fontsize=11, fontweight='bold'
            )
            axes[sym_idx, hor_idx].set_ylabel('Count', fontsize=9)
            axes[sym_idx, hor_idx].grid(axis='y', alpha=0.3)

            # Add percentages
            total = sum(values)
            for i, v in enumerate(values):
                pct = v / total * 100 if total > 0 else 0
                axes[sym_idx, hor_idx].text(
                    i, v, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8
                )

    plt.suptitle(
        'Label Distribution by Symbol and Horizon',
        fontsize=14, fontweight='bold', y=1.00
    )
    plt.tight_layout()
    plt.savefig(charts_dir / 'symbol_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
