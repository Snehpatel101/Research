#!/usr/bin/env python3
"""
Add regime detection and regime-aware evaluation to ML_Pipeline.ipynb.

This script modifies the notebook to add:
1. regime_labels field to NotebookConfig
2. Cell 3.4: Detect Market Regimes (HMM)
3. Updated Cell 4.2: Training Summary with per-regime performance
4. New Cell 4.3: Regime Performance Heatmap
"""

import json
from pathlib import Path


def create_regime_detection_cell():
    """Create Cell 3.4: Detect Market Regimes (HMM)."""
    source = '''#@title 3.4 Detect Market Regimes (HMM) { display-mode: "form" }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.phase1.stages.regime import HMMRegimeDetector

config = require_config()

if not config.data_ready:
    print("[ERROR] Data not ready. Run Section 3 first.")
else:
    print("=" * 70)
    print(" REGIME DETECTION (HMM)")
    print("=" * 70)

    # Configuration
    n_regimes = 4  # Per 2025 research (State Street)
    regime_type = "volatility"  # volatility, trend, or composite
    use_hmm = True

    print(f"\\n  Method: {'HMM' if use_hmm else 'Rule-based'}")
    print(f"  Regime Type: {regime_type}")
    print(f"  Number of Regimes: {n_regimes}")

    # Load processed data (combine train+val+test to see full history)
    train_file = config.splits_dir / 'train_scaled.parquet'
    val_file = config.splits_dir / 'val_scaled.parquet'
    test_file = config.splits_dir / 'test_scaled.parquet'

    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    test_df = pd.read_parquet(test_file)

    # Combine all data
    all_data = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    print(f"\\n  Total samples: {len(all_data):,}")

    # Extract OHLCV columns (need at least 'close')
    # The scaled data may have standardized features, need to check if raw OHLC exists
    if 'close' not in all_data.columns:
        # Try to find close-related columns
        close_cols = [c for c in all_data.columns if 'close' in c.lower()]
        if close_cols:
            print(f"\\n  [INFO] Using column: {close_cols[0]} as 'close'")
            ohlcv_data = all_data.copy()
            ohlcv_data['close'] = all_data[close_cols[0]]
        else:
            print("\\n  [ERROR] No 'close' column found. Cannot detect regimes.")
            print("  Available columns:", list(all_data.columns)[:10], "...")
            config.regime_labels = None
    else:
        ohlcv_data = all_data.copy()

    if 'close' in (ohlcv_data.columns if 'ohlcv_data' in locals() else []):
        # Detect regimes using HMM
        detector = HMMRegimeDetector(
            n_states=n_regimes,
            lookback=252,  # ~1 trading year at 5-min
            input_type="returns",
            expanding=True,  # Use expanding window
            retrain_interval=50,  # Retrain every 50 bars for efficiency
        )

        print("\\n  Running HMM regime detection...")
        regimes, probs = detector.detect_with_probabilities(ohlcv_data)

        # Store in config
        config.regime_labels = regimes.values
        config.regime_probabilities = probs.values

        # Add regime column to data
        all_data['regime'] = regimes.values

        # Regime statistics
        print("\\n  Regime Distribution:")
        regime_counts = regimes.value_counts(dropna=False)
        for regime, count in regime_counts.items():
            pct = count / len(regimes) * 100
            print(f"    {regime}: {count:,} bars ({pct:.1f}%)")

        # Calculate regime statistics
        returns = ohlcv_data['close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualized

        print("\\n  Regime Characteristics:")
        for regime in regimes.dropna().unique():
            mask = regimes == regime
            avg_return = returns[mask].mean() * 252 * 100  # Annualized %
            avg_vol = volatility[mask].mean() * 100  # Annualized %
            print(f"    {regime}:")
            print(f"      Avg Return: {avg_return:+.2f}% (annualized)")
            print(f"      Avg Volatility: {avg_vol:.2f}% (annualized)")

        # Visualize regime transitions
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Plot 1: Price + Regime background
        ax1 = axes[0]
        close_prices = ohlcv_data['close'].values

        # Color-code regimes
        regime_colors = {
            'low_vol': 'lightgreen',
            'normal': 'lightblue',
            'high_vol': 'orange',
            'crisis': 'red',
        }

        # Plot price
        ax1.plot(close_prices, color='black', linewidth=0.8, label='Close Price')

        # Add regime background shading
        for i in range(len(regimes) - 1):
            if pd.notna(regimes.iloc[i]):
                regime = regimes.iloc[i]
                color = regime_colors.get(regime, 'gray')
                ax1.axvspan(i, i+1, alpha=0.3, color=color)

        ax1.set_ylabel('Price')
        ax1.set_title(f'Market Regimes Over Time (n={n_regimes})')
        ax1.grid(True, alpha=0.3)

        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=regime, alpha=0.5)
                          for regime, color in regime_colors.items()
                          if regime in regimes.values]
        ax1.legend(handles=legend_elements, loc='upper left')

        # Plot 2: Regime ID over time
        ax2 = axes[1]
        regime_ids = regimes.map({
            'low_vol': 0, 'normal': 1, 'high_vol': 2, 'crisis': 3
        }).values
        ax2.plot(regime_ids, color='steelblue', linewidth=1)
        ax2.set_ylabel('Regime State')
        ax2.set_xlabel('Bar Index')
        ax2.set_yticks(range(n_regimes))
        ax2.set_yticklabels(['Low Vol', 'Normal', 'High Vol', 'Crisis'][:n_regimes])
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Regime State Transitions')

        plt.tight_layout()
        plt.show()

        # Transition statistics
        transitions = (regimes != regimes.shift(1)).sum() - 1
        avg_duration = len(regimes) / (transitions + 1)
        print(f"\\n  Regime Transitions: {transitions}")
        print(f"  Avg Regime Duration: {avg_duration:.1f} bars")

        # Store regime info in config for later use
        config.n_regimes = n_regimes
        config.regime_names = ['low_vol', 'normal', 'high_vol', 'crisis'][:n_regimes]

        # Clean up
        del train_df, val_df, test_df, all_data, ohlcv_data
        clear_memory(verbose=False)

        print(f"\\n  Regime labels stored in config.regime_labels")
        print("=" * 70)
    else:
        print("\\n[Skipped] Regime detection failed.")
        print("=" * 70)
'''

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split('\n')
    }


def create_regime_heatmap_cell():
    """Create Cell 4.3: Regime Performance Heatmap."""
    source = '''#@title 4.3 Regime Performance Heatmap { display-mode: "form" }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

config = require_config()

if not config.training_results:
    print("[WARNING] No training results found. Run Cell 4.1 first.")
elif config.regime_labels is None:
    print("[WARNING] No regime labels found. Run Cell 3.4 first.")
else:
    print("=" * 70)
    print(" REGIME PERFORMANCE HEATMAP")
    print("=" * 70)

    # Load validation data to get indices
    val_file = config.splits_dir / 'val_scaled.parquet'
    val_df = pd.read_parquet(val_file)
    target_col = f'label_h{config.training_horizon}'
    y_val = val_df[target_col].values

    # Get validation indices (assumes train/val/test split in order)
    train_file = config.splits_dir / 'train_scaled.parquet'
    train_df = pd.read_parquet(train_file)
    val_start_idx = len(train_df)
    val_end_idx = val_start_idx + len(val_df)

    # Extract regime labels for validation set
    val_regimes = config.regime_labels[val_start_idx:val_end_idx]

    # Get unique regimes (excluding NaN)
    unique_regimes = [r for r in np.unique(val_regimes) if pd.notna(r)]
    n_regimes = len(unique_regimes)

    if n_regimes == 0:
        print("\\n[ERROR] No valid regimes found in validation set.")
    else:
        print(f"\\n  Regimes in validation set: {unique_regimes}")
        print(f"  Validation samples: {len(val_df):,}")

        # Build performance matrix
        successful_models = [m for m, d in config.training_results.items()
                           if 'error' not in d and 'val_predictions' in d]

        if not successful_models:
            print("\\n[ERROR] No models with validation predictions found.")
        else:
            # Initialize matrix: models × regimes
            perf_matrix = np.zeros((len(successful_models), n_regimes))
            sample_counts = np.zeros((len(successful_models), n_regimes))

            for i, model_name in enumerate(successful_models):
                val_preds = config.training_results[model_name]['val_predictions']

                for j, regime in enumerate(unique_regimes):
                    regime_mask = (val_regimes == regime)
                    n_samples = regime_mask.sum()

                    if n_samples >= 10:  # Minimum samples for meaningful metric
                        # Calculate F1 score for this regime
                        regime_y_true = y_val[regime_mask]
                        regime_y_pred = val_preds[regime_mask]
                        f1 = f1_score(regime_y_true, regime_y_pred, average='macro', zero_division=0)
                        perf_matrix[i, j] = f1
                        sample_counts[i, j] = n_samples
                    else:
                        perf_matrix[i, j] = np.nan
                        sample_counts[i, j] = n_samples

            # Create DataFrame for better visualization
            perf_df = pd.DataFrame(
                perf_matrix,
                index=successful_models,
                columns=[f"{r}\\n(n={int(sample_counts[0, j])})"
                        for j, r in enumerate(unique_regimes)]
            )

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(12, max(6, len(successful_models) * 0.5)))

            sns.heatmap(
                perf_df,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                center=0.5,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Macro F1 Score'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )

            ax.set_title('Model Performance by Market Regime (Validation Set)',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Market Regime', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)

            plt.tight_layout()
            plt.show()

            # Analysis insights
            print("\\n  Key Insights:")

            # 1. Most regime-agnostic model (consistent across regimes)
            regime_std = perf_df.std(axis=1)
            most_consistent = regime_std.idxmin()
            print(f"    Most Regime-Agnostic: {most_consistent} (std={regime_std[most_consistent]:.4f})")

            # 2. Best specialist per regime
            print("\\n    Best Model per Regime:")
            for regime_col in perf_df.columns:
                best_model = perf_df[regime_col].idxmax()
                best_score = perf_df[regime_col].max()
                if pd.notna(best_score):
                    print(f"      {regime_col.split(chr(10))[0]}: {best_model} (F1={best_score:.4f})")

            # 3. Overall best model
            overall_avg = perf_df.mean(axis=1)
            best_overall = overall_avg.idxmax()
            print(f"\\n    Best Overall (avg across regimes): {best_overall} (F1={overall_avg[best_overall]:.4f})")

            # Store regime performance in config
            for model_name in successful_models:
                if model_name in config.training_results:
                    config.training_results[model_name]['regime_performance'] = \
                        perf_df.loc[model_name].to_dict()

            print("\\n  Regime performance stored in config.training_results[model]['regime_performance']")

            # Clean up
            del train_df, val_df
            clear_memory(verbose=False)

    print("=" * 70)
'''

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split('\n')
    }


def update_notebook_config(cell_source):
    """Add regime_labels field to NotebookConfig dataclass."""
    lines = cell_source.split('\n')

    # Find the line with "# State tracking" and add regime fields before it
    for i, line in enumerate(lines):
        if '# State tracking' in line:
            # Insert new fields before this line
            insert_lines = [
                '    # Regime detection',
                '    regime_labels: Optional[np.ndarray] = None',
                '    regime_probabilities: Optional[np.ndarray] = None',
                '    n_regimes: int = 4',
                '    regime_names: List[str] = field(default_factory=lambda: ["low_vol", "normal", "high_vol", "crisis"])',
                '    ',
            ]
            lines = lines[:i] + insert_lines + lines[i:]
            break

    # Also need to import numpy
    for i, line in enumerate(lines):
        if 'from typing import' in line:
            # Check if numpy is already imported
            has_numpy = any('import numpy' in l for l in lines[:i+10])
            if not has_numpy:
                lines.insert(i+1, 'import numpy as np')
            break

    return '\n'.join(lines)


def update_training_summary_cell(cell_source):
    """Update Cell 4.2 to add per-regime performance breakdown."""
    lines = cell_source.split('\n')

    # Find the end of the plotting section (right before plt.show())
    insert_idx = None
    for i, line in enumerate(lines):
        if 'plt.show()' in line:
            insert_idx = i + 1
            break

    if insert_idx is None:
        # Fallback: insert at end
        insert_idx = len(lines)

    # Add regime performance section
    regime_section = '''
    # ========================================
    # PER-REGIME PERFORMANCE BREAKDOWN
    # ========================================

    if hasattr(config, 'regime_labels') and config.regime_labels is not None:
        print("\\n" + "=" * 70)
        print(" PERFORMANCE BY MARKET REGIME")
        print("=" * 70)

        # Load validation data
        from sklearn.metrics import accuracy_score, f1_score

        val_file = config.splits_dir / 'val_scaled.parquet'
        val_df = pd.read_parquet(val_file)
        target_col = f'label_h{config.training_horizon}'
        y_val = val_df[target_col].values

        # Get validation indices
        train_file = config.splits_dir / 'train_scaled.parquet'
        train_df = pd.read_parquet(train_file)
        val_start_idx = len(train_df)
        val_end_idx = val_start_idx + len(val_df)

        # Extract regime labels for validation set
        val_regimes = config.regime_labels[val_start_idx:val_end_idx]

        # Get unique regimes
        unique_regimes = [r for r in np.unique(val_regimes) if pd.notna(r)]

        for model_name in successful['Model'].values:
            if model_name in config.training_results:
                result_data = config.training_results[model_name]
                if 'val_predictions' in result_data:
                    val_preds = result_data['val_predictions']

                    print(f"\\n{model_name}:")
                    for regime in unique_regimes:
                        regime_mask = (val_regimes == regime)
                        n_samples = regime_mask.sum()

                        if n_samples >= 10:  # Minimum for meaningful metric
                            regime_acc = accuracy_score(y_val[regime_mask], val_preds[regime_mask])
                            regime_f1 = f1_score(y_val[regime_mask], val_preds[regime_mask],
                                               average='macro', zero_division=0)
                            print(f"  {regime:12s}: Acc={regime_acc:.4f}, F1={regime_f1:.4f} ({n_samples:,} samples)")
                        else:
                            print(f"  {regime:12s}: Insufficient samples ({n_samples})")

        # Clean up
        del train_df, val_df
        clear_memory(verbose=False)
'''

    lines = lines[:insert_idx] + regime_section.split('\n') + lines[insert_idx:]

    return '\n'.join(lines)


def update_training_cell_to_store_predictions(cell_source):
    """Update Cell 4.1 to store val_predictions in training_results."""
    lines = cell_source.split('\n')

    # Find where training_results are stored
    # Look for: config.training_results[model_name] = {
    for i, line in enumerate(lines):
        if "config.training_results[model_name] = {" in line or \
           "config.training_results[model_name] =" in line:
            # Find the closing brace
            brace_count = 0
            for j in range(i, min(i+20, len(lines))):
                if '{' in lines[j]:
                    brace_count += 1
                if '}' in lines[j]:
                    brace_count -= 1
                    if brace_count == 0:
                        # Insert val_predictions before the closing brace
                        # Check if 'model_path' line exists
                        for k in range(i, j):
                            if "'model_path':" in lines[k]:
                                # Insert after model_path line
                                lines.insert(k+1, "                    'val_predictions': result.val_predictions,")
                                break
                        break
            break

    return '\n'.join(lines)


def main():
    """Main script to update the notebook."""
    notebook_path = Path('/home/jake/Desktop/Research/notebooks/ML_Pipeline.ipynb')

    print("Loading notebook...")
    with open(notebook_path) as f:
        notebook = json.load(f)

    print(f"Original notebook has {len(notebook['cells'])} cells")

    # Step 1: Update Cell 4 (NotebookConfig)
    print("\n1. Updating NotebookConfig (Cell 4)...")
    cell_4_source = ''.join(notebook['cells'][4]['source'])
    updated_config_source = update_notebook_config(cell_4_source)
    notebook['cells'][4]['source'] = updated_config_source.split('\n')
    print("   ✓ Added regime_labels, regime_probabilities, n_regimes, regime_names fields")

    # Step 2: Insert Cell 3.4 (Regime Detection) after Cell 10
    print("\n2. Inserting Cell 3.4 (Regime Detection) after Cell 10...")
    regime_cell = create_regime_detection_cell()
    notebook['cells'].insert(11, regime_cell)  # Insert after cell 10
    print("   ✓ Added Cell 3.4: Detect Market Regimes (HMM)")

    # Step 3: Update Cell 13 -> now Cell 14 (Train Models) to store val_predictions
    print("\n3. Updating Cell 4.1 (Train Models) to store val_predictions...")
    cell_14_source = ''.join(notebook['cells'][14]['source'])  # Now at index 14
    updated_train_source = update_training_cell_to_store_predictions(cell_14_source)
    notebook['cells'][14]['source'] = updated_train_source.split('\n')
    print("   ✓ Updated to store val_predictions in training_results")

    # Step 4: Update Cell 14 -> now Cell 15 (Training Summary)
    print("\n4. Updating Cell 4.2 (Training Summary)...")
    cell_15_source = ''.join(notebook['cells'][15]['source'])  # Now at index 15
    updated_summary_source = update_training_summary_cell(cell_15_source)
    notebook['cells'][15]['source'] = updated_summary_source.split('\n')
    print("   ✓ Added per-regime performance breakdown")

    # Step 5: Insert Cell 4.3 (Regime Heatmap) after Cell 15 (now 4.2)
    print("\n5. Inserting Cell 4.3 (Regime Performance Heatmap)...")
    heatmap_cell = create_regime_heatmap_cell()
    notebook['cells'].insert(16, heatmap_cell)  # Insert after cell 15
    print("   ✓ Added Cell 4.3: Regime Performance Heatmap")

    print(f"\nUpdated notebook has {len(notebook['cells'])} cells (was {len(notebook['cells']) - 2})")

    # Save updated notebook
    print("\nSaving updated notebook...")
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"✓ Saved: {notebook_path}")
    print("\n" + "=" * 70)
    print("SUCCESS: Regime detection added to notebook!")
    print("=" * 70)
    print("\nChanges made:")
    print("  1. NotebookConfig: Added regime detection fields")
    print("  2. Cell 3.4: Detect Market Regimes (HMM)")
    print("  3. Cell 4.1: Store val_predictions for regime analysis")
    print("  4. Cell 4.2: Show per-regime performance")
    print("  5. Cell 4.3: Regime Performance Heatmap")
    print("\nBackup saved to: notebooks/ML_Pipeline_backup.ipynb")


if __name__ == '__main__':
    main()
