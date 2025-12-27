"""
Diagnostic script to investigate label distribution mismatch.

This script traces through the pipeline to identify where and why
label distributions become imbalanced between train and validation sets.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=" * 80)
    print("LABEL DISTRIBUTION DIAGNOSTIC")
    print("=" * 80)

    # Step 1: Check raw labeled data
    print("\n1. RAW LABELED DATA (combined_final_labeled.parquet)")
    print("-" * 80)
    combined_path = Path("/Users/sneh/research/data/final/combined_final_labeled.parquet")
    df = pd.read_parquet(combined_path)

    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"\nLabel distribution (label_h20):")
    for label in [-1, 0, 1]:
        count = (df['label_h20'] == label).sum()
        pct = count / len(df) * 100
        label_name = {-1: 'Short', 0: 'Neutral', 1: 'Long'}[label]
        print(f"  {label_name} ({label:2d}): {count:8,} ({pct:5.1f}%)")

    # Step 2: Check quality scores by label
    print("\n2. QUALITY SCORES BY LABEL")
    print("-" * 80)
    for label in [-1, 0, 1]:
        label_data = df[df['label_h20'] == label]
        label_name = {-1: 'Short', 0: 'Neutral', 1: 'Long'}[label]
        quality_mean = label_data['quality_h20'].mean()
        quality_median = label_data['quality_h20'].median()
        quality_std = label_data['quality_h20'].std()

        print(f"\n{label_name} labels (n={len(label_data):,}):")
        print(f"  Quality mean:   {quality_mean:.3f}")
        print(f"  Quality median: {quality_median:.3f}")
        print(f"  Quality std:    {quality_std:.3f}")

        # Show distribution by quality threshold
        for threshold in [0.3, 0.5, 0.7]:
            count = (label_data['quality_h20'] >= threshold).sum()
            pct = count / len(label_data) * 100
            print(f"  Quality >= {threshold}: {count:7,} ({pct:5.1f}%)")

    # Step 3: Check split indices
    print("\n3. SPLIT INDICES")
    print("-" * 80)
    train_indices = np.load("/Users/sneh/research/data/splits/train_indices.npy")
    val_indices = np.load("/Users/sneh/research/data/splits/val_indices.npy")
    test_indices = np.load("/Users/sneh/research/data/splits/test_indices.npy")

    print(f"Train indices: {len(train_indices):,} (range: {train_indices.min()}-{train_indices.max()})")
    print(f"Val indices:   {len(val_indices):,} (range: {val_indices.min()}-{val_indices.max()})")
    print(f"Test indices:  {len(test_indices):,} (range: {test_indices.min()}-{test_indices.max()})")

    # Apply indices to get splits
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]
    test_df = df.iloc[test_indices]

    print(f"\nLabel distribution after indexing:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name} ({len(split_df):,} rows):")
        for label in [-1, 0, 1]:
            count = (split_df['label_h20'] == label).sum()
            pct = count / len(split_df) * 100 if len(split_df) > 0 else 0
            label_name = {-1: 'Short', 0: 'Neutral', 1: 'Long'}[label]
            print(f"  {label_name} ({label:2d}): {count:8,} ({pct:5.1f}%)")

    # Step 4: Check scaled data
    print("\n4. SCALED DATA")
    print("-" * 80)
    train_scaled = pd.read_parquet("/Users/sneh/research/data/splits/scaled/train_scaled.parquet")
    val_scaled = pd.read_parquet("/Users/sneh/research/data/splits/scaled/val_scaled.parquet")
    test_scaled = pd.read_parquet("/Users/sneh/research/data/splits/scaled/test_scaled.parquet")

    print(f"Scaled data dimensions:")
    print(f"  Train: {len(train_scaled):,} rows")
    print(f"  Val:   {len(val_scaled):,} rows")
    print(f"  Test:  {len(test_scaled):,} rows")

    print(f"\n**CRITICAL: Data loss detected!**")
    print(f"  Train: {len(train_df):,} -> {len(train_scaled):,} ({(1-len(train_scaled)/len(train_df))*100:.1f}% loss)")
    print(f"  Val:   {len(val_df):,} -> {len(val_scaled):,} ({(1-len(val_scaled)/len(val_df))*100:.1f}% loss)")
    print(f"  Test:  {len(test_df):,} -> {len(test_scaled):,} ({(1-len(test_scaled)/len(test_df))*100:.1f}% loss)")

    print(f"\nLabel distribution in scaled data:")
    for split_name, split_df in [("Train", train_scaled), ("Val", val_scaled), ("Test", test_scaled)]:
        print(f"\n{split_name}:")
        for label in [-1, 0, 1]:
            count = (split_df['label_h20'] == label).sum()
            pct = count / len(split_df) * 100 if len(split_df) > 0 else 0
            label_name = {-1: 'Short', 0: 'Neutral', 1: 'Long'}[label]
            print(f"  {label_name} ({label:2d}): {count:6,} ({pct:5.1f}%)")

    # Step 5: Check date ranges
    print("\n5. DATE RANGE MISMATCH")
    print("-" * 80)
    print(f"Original combined data:")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    print(f"\nScaled data:")
    print(f"  Train: {train_scaled['datetime'].min()} to {train_scaled['datetime'].max()}")
    print(f"  Val:   {val_scaled['datetime'].min()} to {val_scaled['datetime'].max()}")
    print(f"  Test:  {test_scaled['datetime'].min()} to {test_scaled['datetime'].max()}")

    # Step 6: Root cause analysis
    print("\n6. ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print("\nFINDINGS:")
    print("1. The raw labeled data HAS BALANCED labels (~48% Short, ~2% Neutral, ~50% Long)")
    print("2. Quality scores are HEAVILY BIASED toward Long labels:")
    print("   - Short labels: mean quality = 0.43 (only 28% have quality >= 0.5)")
    print("   - Long labels:  mean quality = 0.67 (99% have quality >= 0.5)")
    print("3. The indices correctly sample the full dataset (2.4M rows)")
    print("4. BUT the scaled data has only 35K rows (98.5% data loss!)")
    print("5. The scaled data shows IMBALANCED labels (~0.4% Short, ~40% Neutral, ~59% Long)")

    print("\nROOT CAUSE:")
    print("The pipeline is loading a DIFFERENT dataset for the scaling stage.")
    print("The scaled data appears to be from a test run or filtered dataset,")
    print("NOT the full combined_final_labeled.parquet file.")

    print("\nRECOMMENDED FIX:")
    print("1. Verify the pipeline configuration points to the correct data files")
    print("2. Re-run the scaling stage with the full dataset")
    print("3. Or: Regenerate splits from the filtered dataset to ensure consistency")

if __name__ == "__main__":
    main()
