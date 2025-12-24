#!/usr/bin/env python3
"""
Visualizations for Triple-Barrier Parameter Analysis
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
FEATURES_PATH = Path("/Users/sneh/research/data/features")
RESULTS_PATH = Path("/Users/sneh/research/results")

def load_analysis_results():
    """Load the analysis results."""
    with open(RESULTS_PATH / "barrier_analysis.json", 'r') as f:
        return json.load(f)

def create_hit_rate_vs_multiplier_plot(results):
    """Create plot showing hit rate vs ATR multiplier for each horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    horizons = ['h1', 'h5', 'h20']
    horizon_labels = ['H=1 (5 min)', 'H=5 (25 min)', 'H=20 (100 min)']
    colors = {'MES': '#1f77b4', 'MGC': '#ff7f0e'}

    for ax, h_key, h_label in zip(axes, horizons, horizon_labels):
        for symbol in ['MES', 'MGC']:
            hit_rates = results['detailed_results'][symbol][h_key]['hit_rates']

            multipliers = sorted([float(k) for k in hit_rates.keys()])
            rates = [hit_rates[str(m)]['either'] for m in multipliers]

            ax.plot(multipliers, rates, 'o-', color=colors[symbol],
                   label=symbol, markersize=4, linewidth=2)

        # Target zone
        ax.axhspan(65, 70, alpha=0.2, color='green', label='Target zone (65-70%)')
        ax.axhline(y=67.5, color='green', linestyle='--', alpha=0.5)

        ax.set_xlabel('ATR Multiplier (k)', fontsize=11)
        ax.set_ylabel('Hit Rate (%)', fontsize=11)
        ax.set_title(h_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.1)
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'hit_rate_vs_multiplier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_PATH / 'hit_rate_vs_multiplier.png'}")

def create_neutral_rate_plot(results):
    """Create plot showing neutral (no-hit) rate vs multiplier."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    horizons = ['h1', 'h5', 'h20']
    horizon_labels = ['H=1 (5 min)', 'H=5 (25 min)', 'H=20 (100 min)']
    colors = {'MES': '#1f77b4', 'MGC': '#ff7f0e'}

    for ax, h_key, h_label in zip(axes, horizons, horizon_labels):
        for symbol in ['MES', 'MGC']:
            hit_rates = results['detailed_results'][symbol][h_key]['hit_rates']

            multipliers = sorted([float(k) for k in hit_rates.keys()])
            neutral_rates = [hit_rates[str(m)]['neutral'] for m in multipliers]

            ax.plot(multipliers, neutral_rates, 'o-', color=colors[symbol],
                   label=symbol, markersize=4, linewidth=2)

        # Target zone (30-35% neutral)
        ax.axhspan(30, 35, alpha=0.2, color='green', label='Target zone (30-35%)')
        ax.axhline(y=32.5, color='green', linestyle='--', alpha=0.5)

        ax.set_xlabel('ATR Multiplier (k)', fontsize=11)
        ax.set_ylabel('Neutral Rate (%)', fontsize=11)
        ax.set_title(h_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.1)
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'neutral_rate_vs_multiplier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_PATH / 'neutral_rate_vs_multiplier.png'}")

def create_price_move_distribution(results):
    """Create distribution plot of price moves vs ATR."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    horizons = ['h1', 'h5', 'h20']
    horizon_labels = ['H=1 (5 min)', 'H=5 (25 min)', 'H=20 (100 min)']

    for col, (h_key, h_label) in enumerate(zip(horizons, horizon_labels)):
        for row, symbol in enumerate(['MES', 'MGC']):
            ax = axes[row, col]

            data = results['detailed_results'][symbol][h_key]
            mean_atr = data['mean_atr']
            mean_move = data['mean_abs_move']
            ratio = data['move_atr_ratio']

            # Create bar chart of key statistics
            metrics = ['Mean ATR', 'Mean Move', 'Ratio']
            values = [mean_atr, mean_move, ratio]

            bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)

            ax.set_title(f'{symbol} - {h_label}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Points / Ratio', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'price_move_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_PATH / 'price_move_distribution.png'}")

def create_optimal_k_summary(results):
    """Create summary visualization of optimal k values."""
    fig, ax = plt.subplots(figsize=(12, 6))

    recommendations = results['final_recommendations']

    horizons = ['h1', 'h5', 'h20']
    horizon_labels = ['H=1\n(5 min)', 'H=5\n(25 min)', 'H=20\n(100 min)']
    x = np.arange(len(horizons))
    width = 0.35

    k_values = [recommendations[h]['recommended_k_up'] for h in horizons]
    neutral_pcts = [recommendations[h]['expected_neutral_pct'] for h in horizons]

    bars1 = ax.bar(x - width/2, k_values, width, label='Optimal k', color='#1f77b4')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, neutral_pcts, width, label='Expected Neutral %', color='#ff7f0e')

    # Add value labels
    for bar, val in zip(bars1, k_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'k={val}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    for bar, val in zip(bars2, neutral_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Horizon', fontsize=12)
    ax.set_ylabel('Optimal ATR Multiplier (k)', fontsize=12, color='#1f77b4')
    ax2.set_ylabel('Expected Neutral Rate (%)', fontsize=12, color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_labels, fontsize=11)
    ax.set_ylim(0, 2.5)
    ax2.set_ylim(0, 50)

    ax.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#1f77b4', label='Optimal k'),
        mpatches.Patch(facecolor='#ff7f0e', label='Expected Neutral %')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_title('Recommended ATR Multipliers by Horizon\n(Target: 65-70% hit rate, 30-35% neutral)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'optimal_k_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_PATH / 'optimal_k_summary.png'}")

def create_comparison_table_image(results):
    """Create a visual table showing detailed hit rates."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Create table data
    header = ['Multiplier', 'MES H=1', 'MGC H=1', 'MES H=5', 'MGC H=5', 'MES H=20', 'MGC H=20']

    multipliers = ['0.25', '0.5', '0.75', '0.9', '1.0', '1.5', '2.0']

    data = []
    for m in multipliers:
        row = [m]
        for symbol in ['MES', 'MGC']:
            for h in ['h1', 'h5', 'h20']:
                try:
                    hit = results['detailed_results'][symbol][h]['hit_rates'][m]['either']
                    neutral = results['detailed_results'][symbol][h]['hit_rates'][m]['neutral']
                    row.append(f"{hit:.1f}% / {neutral:.1f}%")
                except KeyError:
                    row.append("N/A")
        data.append(row)

    # Reorganize for better readability
    data_reorg = []
    for m in multipliers:
        row = [m]
        for h in ['h1', 'h5', 'h20']:
            for symbol in ['MES', 'MGC']:
                try:
                    hit = results['detailed_results'][symbol][h]['hit_rates'][m]['either']
                    row.append(f"{hit:.1f}%")
                except KeyError:
                    row.append("N/A")
        data_reorg.append(row)

    header_reorg = ['k', 'MES H=1', 'MGC H=1', 'MES H=5', 'MGC H=5', 'MES H=20', 'MGC H=20']

    table = ax.table(cellText=data_reorg, colLabels=header_reorg, loc='center',
                    cellLoc='center', colLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color cells based on target range
    for i in range(len(data_reorg)):
        for j in range(1, len(data_reorg[i])):
            cell = table[(i+1, j)]
            try:
                val = float(data_reorg[i][j].replace('%', ''))
                if 65 <= val <= 70:
                    cell.set_facecolor('#90EE90')  # Light green - in target
                elif 60 <= val <= 75:
                    cell.set_facecolor('#FFFACD')  # Light yellow - close
                elif val > 90:
                    cell.set_facecolor('#FFB6C1')  # Light red - too high
            except:
                pass

    # Header styling
    for j in range(len(header_reorg)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Hit Rates by ATR Multiplier and Horizon\n(Green = Target 65-70%, Yellow = Close, Red = Too High)',
                fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'hit_rate_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_PATH / 'hit_rate_table.png'}")

def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    results = load_analysis_results()

    create_hit_rate_vs_multiplier_plot(results)
    create_neutral_rate_plot(results)
    create_price_move_distribution(results)
    create_optimal_k_summary(results)
    create_comparison_table_image(results)

    print("\nAll visualizations saved to:", RESULTS_PATH)

if __name__ == "__main__":
    main()
