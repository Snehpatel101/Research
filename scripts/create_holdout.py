#!/usr/bin/env python3
"""
Create Holdout Datasets for Final Model Evaluation

This script splits raw OHLCV data into:
1. Training data (2020) - used for model development
2. Holdout data (2021) - ONLY for final evaluation, never seen during training

The holdout data preserves the temporal split to ensure no lookahead bias.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def create_holdout_split(
    input_path: Path,
    holdout_dir: Path,
    train_output_path: Path,
    holdout_cutoff: str = "2021-01-01",
) -> dict:
    """
    Split raw data into training and holdout sets.

    Args:
        input_path: Path to raw parquet file
        holdout_dir: Directory for holdout data
        train_output_path: Path for training data (without holdout)
        holdout_cutoff: Date string for holdout split (YYYY-MM-DD)

    Returns:
        Dictionary with split statistics
    """
    # Read raw data
    df = pd.read_parquet(input_path)
    symbol = input_path.stem.replace("_1m", "")

    print(f"\n{'='*60}")
    print(f"Processing: {symbol}")
    print(f"{'='*60}")

    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Original data stats
    total_rows = len(df)
    date_min = df["datetime"].min()
    date_max = df["datetime"].max()

    print(f"\nOriginal Data:")
    print(f"  - Total rows: {total_rows:,}")
    print(f"  - Date range: {date_min} to {date_max}")

    # Split by cutoff date
    cutoff = pd.Timestamp(holdout_cutoff)

    train_mask = df["datetime"] < cutoff
    holdout_mask = df["datetime"] >= cutoff

    train_df = df[train_mask].copy()
    holdout_df = df[holdout_mask].copy()

    # Training data stats
    train_rows = len(train_df)
    train_pct = (train_rows / total_rows) * 100
    train_date_min = train_df["datetime"].min() if len(train_df) > 0 else None
    train_date_max = train_df["datetime"].max() if len(train_df) > 0 else None

    print(f"\nTraining Data (before {holdout_cutoff}):")
    print(f"  - Rows: {train_rows:,} ({train_pct:.1f}%)")
    if train_date_min:
        print(f"  - Date range: {train_date_min} to {train_date_max}")

    # Holdout data stats
    holdout_rows = len(holdout_df)
    holdout_pct = (holdout_rows / total_rows) * 100
    holdout_date_min = holdout_df["datetime"].min() if len(holdout_df) > 0 else None
    holdout_date_max = holdout_df["datetime"].max() if len(holdout_df) > 0 else None

    print(f"\nHoldout Data ({holdout_cutoff} onwards):")
    print(f"  - Rows: {holdout_rows:,} ({holdout_pct:.1f}%)")
    if holdout_date_min:
        print(f"  - Date range: {holdout_date_min} to {holdout_date_max}")

    # Validate no overlap
    if len(train_df) > 0 and len(holdout_df) > 0:
        gap = (holdout_date_min - train_date_max).total_seconds() / 60
        print(f"\nGap between train and holdout: {gap:.0f} minutes")

        if train_date_max >= holdout_date_min:
            raise ValueError("Data overlap detected between train and holdout!")

    # Create directories
    holdout_dir.mkdir(parents=True, exist_ok=True)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save files
    holdout_path = holdout_dir / f"{symbol}_1m_holdout.parquet"

    if len(holdout_df) > 0:
        holdout_df.to_parquet(holdout_path, index=False)
        print(f"\nSaved holdout: {holdout_path}")
    else:
        print(f"\nWARNING: No holdout data for {symbol}")

    if len(train_df) > 0:
        train_df.to_parquet(train_output_path, index=False)
        print(f"Saved training: {train_output_path}")
    else:
        print(f"\nWARNING: No training data for {symbol}")

    return {
        "symbol": symbol,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
        "train_date_range": (train_date_min, train_date_max),
        "holdout_date_range": (holdout_date_min, holdout_date_max),
        "train_pct": train_pct,
        "holdout_pct": holdout_pct,
    }


def main():
    """Create holdout datasets for all available symbols."""

    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    holdout_dir = base_dir / "data" / "holdout"

    print("\n" + "=" * 60)
    print("HOLDOUT DATASET CREATION")
    print("=" * 60)
    print(f"\nBase directory: {base_dir}")
    print(f"Raw data directory: {raw_dir}")
    print(f"Holdout directory: {holdout_dir}")

    # Find all raw parquet files
    parquet_files = list(raw_dir.glob("*_1m.parquet"))

    # Exclude already-split files
    parquet_files = [
        f for f in parquet_files if "_train" not in f.stem and "_holdout" not in f.stem
    ]

    if not parquet_files:
        print("\nNo raw parquet files found!")
        return

    print(f"\nFound {len(parquet_files)} raw data file(s):")
    for f in parquet_files:
        print(f"  - {f.name}")

    results = []
    for input_path in parquet_files:
        symbol = input_path.stem.replace("_1m", "")
        train_output = raw_dir / f"{symbol}_1m_train.parquet"

        result = create_holdout_split(
            input_path=input_path,
            holdout_dir=holdout_dir,
            train_output_path=train_output,
            holdout_cutoff="2021-01-01",
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"\n{'Symbol':<10} {'Total':>12} {'Train':>12} {'Holdout':>12} {'Train %':>10} {'Holdout %':>10}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['symbol']:<10} {r['total_rows']:>12,} {r['train_rows']:>12,} {r['holdout_rows']:>12,} {r['train_pct']:>9.1f}% {r['holdout_pct']:>9.1f}%"
        )

    print("\n" + "=" * 60)
    print("IMPORTANT: Holdout data should NEVER be used during training!")
    print("It is reserved for final model evaluation only.")
    print("=" * 60 + "\n")

    # Verify files were created
    print("Verification - Files created:")
    for r in results:
        symbol = r["symbol"]
        holdout_path = holdout_dir / f"{symbol}_1m_holdout.parquet"
        train_path = raw_dir / f"{symbol}_1m_train.parquet"

        if holdout_path.exists():
            size = holdout_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {holdout_path.name} ({size:.2f} MB)")
        else:
            print(f"  [MISSING] {holdout_path.name}")

        if train_path.exists():
            size = train_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {train_path.name} ({size:.2f} MB)")
        else:
            print(f"  [MISSING] {train_path.name}")


if __name__ == "__main__":
    main()
