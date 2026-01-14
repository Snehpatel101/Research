#!/usr/bin/env python3
"""
FINAL CRITICAL VERIFICATION of the complete data pipeline.
Performs exhaustive checks on temporal integrity, label distribution, scaling, and data quality.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# Paths
DATA_DIR = Path("/Users/sneh/research/data/splits/final_correct")
SCALED_DIR = DATA_DIR / "scaled"

# Expected horizons
EXPECTED_HORIZONS = [5, 10, 15, 20]

# ANSI colors for output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def color_status(passed: bool) -> str:
    """Return colored PASS/FAIL."""
    if passed:
        return f"{GREEN}{BOLD}PASS{RESET}"
    return f"{RED}{BOLD}FAIL{RESET}"


def load_configs() -> Tuple[dict, dict]:
    """Load split and scaling configs."""
    with open(DATA_DIR / "split_config.json") as f:
        split_config = json.load(f)
    with open(SCALED_DIR / "scaling_config.json") as f:
        scaling_config = json.load(f)
    return split_config, scaling_config


def load_splits() -> Dict[str, pd.DataFrame]:
    """Load all splits."""
    print(f"\n{BLUE}{BOLD}Loading data splits...{RESET}")
    splits = {}
    for split_name in ["train", "val", "test"]:
        path = SCALED_DIR / f"{split_name}_scaled.parquet"
        df = pd.read_parquet(path)
        splits[split_name] = df
        print(f"  {split_name}: {len(df):,} rows, {len(df.columns)} columns")
    return splits


def verify_temporal_integrity(splits: Dict[str, pd.DataFrame], split_config: dict) -> bool:
    """
    CRITICAL: Verify no timestamp overlap and proper gaps.
    """
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}1. TEMPORAL INTEGRITY (CRITICAL){RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    all_passed = True

    # Get timestamps (column is 'datetime', not 'timestamp')
    train_ts = pd.to_datetime(splits["train"]["datetime"])
    val_ts = pd.to_datetime(splits["val"]["datetime"])
    test_ts = pd.to_datetime(splits["test"]["datetime"])

    # Check 1: Zero overlap
    print(f"\n{YELLOW}Check 1.1: Zero timestamp overlap{RESET}")
    train_end = train_ts.max()
    val_start = val_ts.min()
    val_end = val_ts.max()
    test_start = test_ts.min()

    print(f"  Train end:  {train_end}")
    print(f"  Val start:  {val_start}")
    print(f"  Val end:    {val_end}")
    print(f"  Test start: {test_start}")

    train_val_overlap = train_end >= val_start
    val_test_overlap = val_end >= test_start

    check_1a = not train_val_overlap
    check_1b = not val_test_overlap

    print(f"  Train-Val overlap: {color_status(check_1a)}")
    print(f"  Val-Test overlap:  {color_status(check_1b)}")

    all_passed &= check_1a and check_1b

    # Check 2: Purge gap (60 bars)
    print(f"\n{YELLOW}Check 1.2: Purge gap (60 bars minimum){RESET}")
    expected_purge = split_config.get("purge_bars", 60)

    # Calculate actual gaps in bars (assuming 5-min bars)
    train_val_gap_minutes = (val_start - train_end).total_seconds() / 60
    train_val_gap_bars = int(train_val_gap_minutes / 5)

    val_test_gap_minutes = (test_start - val_end).total_seconds() / 60
    val_test_gap_bars = int(val_test_gap_minutes / 5)

    print(f"  Expected purge: {expected_purge} bars")
    print(f"  Train-Val gap:  {train_val_gap_bars} bars ({train_val_gap_minutes:.0f} minutes)")
    print(f"  Val-Test gap:   {val_test_gap_bars} bars ({val_test_gap_minutes:.0f} minutes)")

    check_2a = train_val_gap_bars >= expected_purge
    check_2b = val_test_gap_bars >= expected_purge

    print(f"  Train-Val purge: {color_status(check_2a)}")
    print(f"  Val-Test purge:  {color_status(check_2b)}")

    all_passed &= check_2a and check_2b

    # Check 3: Embargo gap (1440 bars)
    print(f"\n{YELLOW}Check 1.3: Embargo gap (1440 bars minimum){RESET}")
    expected_embargo = split_config.get("embargo_bars", 1440)

    check_3a = train_val_gap_bars >= expected_embargo
    check_3b = val_test_gap_bars >= expected_embargo

    print(f"  Expected embargo: {expected_embargo} bars")
    print(f"  Train-Val embargo: {color_status(check_3a)}")
    print(f"  Val-Test embargo:  {color_status(check_3b)}")

    all_passed &= check_3a and check_3b

    # Check 4: Chronological order
    print(f"\n{YELLOW}Check 1.4: Chronological ordering{RESET}")

    train_sorted = train_ts.is_monotonic_increasing
    val_sorted = val_ts.is_monotonic_increasing
    test_sorted = test_ts.is_monotonic_increasing

    print(f"  Train sorted: {color_status(train_sorted)}")
    print(f"  Val sorted:   {color_status(val_sorted)}")
    print(f"  Test sorted:  {color_status(test_sorted)}")

    all_passed &= train_sorted and val_sorted and test_sorted

    print(f"\n{BOLD}Overall Temporal Integrity: {color_status(all_passed)}{RESET}")
    return all_passed


def verify_label_distribution(splits: Dict[str, pd.DataFrame]) -> bool:
    """
    CRITICAL: Verify all horizons exist and neutral percentage is reasonable.
    """
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}2. LABEL DISTRIBUTION (CRITICAL){RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    all_passed = True

    for split_name, df in splits.items():
        print(f"\n{YELLOW}{split_name.upper()} Split:{RESET}")

        # Find label columns (lowercase label_h)
        label_cols = [c for c in df.columns if c.startswith("label_h")]

        # Check 1: All horizons present
        check_1 = len(label_cols) == len(EXPECTED_HORIZONS)
        print(f"  Horizons present: {len(label_cols)}/4 {color_status(check_1)}")
        all_passed &= check_1

        # Check each horizon
        for horizon in EXPECTED_HORIZONS:
            col = f"label_h{horizon}"
            if col not in df.columns:
                print(f"    H{horizon}: {color_status(False)} - MISSING")
                all_passed = False
                continue

            labels = df[col].dropna()

            # Get distribution
            counts = labels.value_counts().sort_index()
            total = len(labels)

            # Check for -99 (invalid markers)
            invalid_count = (labels == -99).sum()

            # Calculate percentages (excluding -99)
            valid_labels = labels[labels != -99]
            if len(valid_labels) == 0:
                print(f"    H{horizon}: {color_status(False)} - NO VALID LABELS")
                all_passed = False
                continue

            dist = valid_labels.value_counts(normalize=True).sort_index()

            neutral_pct = dist.get(0, 0) * 100
            long_pct = dist.get(1, 0) * 100
            short_pct = dist.get(-1, 0) * 100

            # Check 2: Neutral percentage 10-40%
            check_2 = 10 <= neutral_pct <= 40

            # Check 3: No extreme class imbalance (at least 15% each)
            check_3 = (long_pct >= 15) and (short_pct >= 15)

            status_icon = "✓" if (check_2 and check_3) else "✗"

            print(
                f"    H{horizon}: {status_icon} Long={long_pct:.1f}%, Neutral={neutral_pct:.1f}%, Short={short_pct:.1f}%",
                end="",
            )
            if invalid_count > 0:
                print(f" (invalid={invalid_count})", end="")

            if not check_2:
                print(f" {RED}[Neutral OOR]{RESET}", end="")
                all_passed = False
            if not check_3:
                print(f" {RED}[Imbalanced]{RESET}", end="")
                all_passed = False

            print()

    print(f"\n{BOLD}Overall Label Distribution: {color_status(all_passed)}{RESET}")
    return all_passed


def verify_symbol_balance(splits: Dict[str, pd.DataFrame]) -> bool:
    """
    Verify both symbols present and reasonably balanced.
    """
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}3. SYMBOL BALANCE{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    all_passed = True

    for split_name, df in splits.items():
        print(f"\n{YELLOW}{split_name.upper()} Split:{RESET}")

        symbol_counts = df["symbol"].value_counts()
        total = len(df)

        # Check 1: Both symbols present
        check_1 = len(symbol_counts) == 2
        print(f"  Symbols present: {len(symbol_counts)}/2 {color_status(check_1)}")
        all_passed &= check_1

        # Show distribution
        for symbol, count in symbol_counts.items():
            pct = count / total * 100
            print(f"    {symbol}: {count:,} ({pct:.1f}%)")

        # Check 2: Reasonable balance (30-70%)
        if len(symbol_counts) == 2:
            min_pct = symbol_counts.min() / total * 100
            check_2 = min_pct >= 30
            print(f"  Balance check (≥30%): {color_status(check_2)}")
            all_passed &= check_2

    print(f"\n{BOLD}Overall Symbol Balance: {color_status(all_passed)}{RESET}")
    return all_passed


def verify_feature_scaling(splits: Dict[str, pd.DataFrame], scaling_config: dict) -> bool:
    """
    Verify scaler was fit on train only and features are properly scaled.
    """
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}4. FEATURE SCALING INTEGRITY{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    all_passed = True

    # Check 1: Scaler fit on train only
    print(f"\n{YELLOW}Check 4.1: Scaler configuration{RESET}")
    fit_on = scaling_config.get("fit_on", "unknown")
    check_1 = fit_on == "train"
    print(f"  Fit on: {fit_on} {color_status(check_1)}")
    all_passed &= check_1

    # Get feature columns (exclude metadata and label-related columns)
    exclude_patterns = [
        "datetime",
        "symbol",
        "label_",
        "bars_to_hit_",
        "mae_",
        "mfe_",
        "touch_type_",
        "quality_",
        "sample_weight_",
        "pain_to_gain_",
        "time_weighted_dd_",
        "fwd_return_",
    ]
    feature_cols = [
        c
        for c in splits["train"].columns
        if not any(c.startswith(p) or c == p for p in exclude_patterns)
    ]

    print(f"  Feature count: {len(feature_cols)}")

    # Check 2: Sample features for proper scaling
    print(f"\n{YELLOW}Check 4.2: Feature scaling verification (random sample){RESET}")

    np.random.seed(42)
    sample_features = np.random.choice(feature_cols, size=min(5, len(feature_cols)), replace=False)

    for split_name, df in splits.items():
        print(f"\n  {split_name.upper()}:")

        for feat in sample_features:
            values = df[feat].dropna()

            if len(values) == 0:
                print(f"    {feat}: {color_status(False)} - ALL NAN")
                all_passed = False
                continue

            mean = values.mean()
            std = values.std()
            q01 = values.quantile(0.01)
            q99 = values.quantile(0.99)

            # Check for proper scaling (robust scaler should center around 0)
            # and no extreme outliers
            check_centered = abs(mean) < 2.0  # Median should be near 0
            check_outliers = (q01 > -10) and (q99 < 10)  # No extreme outliers

            status = check_centered and check_outliers

            print(
                f"    {feat[:30]:<30}: mean={mean:>7.3f}, std={std:>6.3f}, "
                f"[{q01:>7.2f}, {q99:>7.2f}] {color_status(status)}"
            )

            if not status:
                all_passed = False

    # Check 3: No all-NaN or all-zero features
    print(f"\n{YELLOW}Check 4.3: Feature completeness{RESET}")

    for split_name, df in splits.items():
        all_nan_features = []
        all_zero_features = []

        for feat in feature_cols:
            values = df[feat]

            if values.isna().all():
                all_nan_features.append(feat)
            elif (values.dropna() == 0).all():
                all_zero_features.append(feat)

        check_3a = len(all_nan_features) == 0
        check_3b = len(all_zero_features) == 0

        print(
            f"  {split_name.upper()}: all-NaN={len(all_nan_features)} {color_status(check_3a)}, "
            f"all-zero={len(all_zero_features)} {color_status(check_3b)}"
        )

        all_passed &= check_3a and check_3b

        if all_nan_features:
            print(f"    All-NaN features: {all_nan_features[:5]}")
        if all_zero_features:
            print(f"    All-zero features: {all_zero_features[:5]}")

    print(f"\n{BOLD}Overall Feature Scaling: {color_status(all_passed)}{RESET}")
    return all_passed


def verify_data_completeness(splits: Dict[str, pd.DataFrame], split_config: dict) -> bool:
    """
    Verify row counts and check for unexpected patterns.
    """
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}5. DATA COMPLETENESS{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    all_passed = True

    # Check 1: Row counts match expected ratios
    print(f"\n{YELLOW}Check 5.1: Split ratios{RESET}")

    total_rows = sum(len(df) for df in splits.values())
    expected_ratios = split_config.get("split_ratios", {})

    print(f"  Total rows: {total_rows:,}")

    for split_name, df in splits.items():
        actual_pct = len(df) / total_rows * 100
        expected_pct = expected_ratios.get(split_name, 0)

        # Allow ±5% deviation
        check = abs(actual_pct - expected_pct) <= 5

        print(
            f"  {split_name}: {len(df):,} rows ({actual_pct:.1f}%, "
            f"expected ~{expected_pct:.0f}%) {color_status(check)}"
        )

        all_passed &= check

    # Check 2: No duplicate rows
    print(f"\n{YELLOW}Check 5.2: Duplicate rows{RESET}")

    for split_name, df in splits.items():
        # Check duplicates on datetime + symbol
        dup_count = df.duplicated(subset=["datetime", "symbol"]).sum()
        check = dup_count == 0

        print(f"  {split_name}: {dup_count} duplicates {color_status(check)}")
        all_passed &= check

    # Check 3: NaN patterns
    print(f"\n{YELLOW}Check 5.3: NaN patterns{RESET}")

    for split_name, df in splits.items():
        # Check label NaNs
        label_cols = [c for c in df.columns if c.startswith("label_h")]
        label_nan_pct = df[label_cols].isna().mean().mean() * 100

        # Check feature NaNs
        exclude_patterns = [
            "datetime",
            "symbol",
            "label_",
            "bars_to_hit_",
            "mae_",
            "mfe_",
            "touch_type_",
            "quality_",
            "sample_weight_",
            "pain_to_gain_",
            "time_weighted_dd_",
            "fwd_return_",
        ]
        feature_cols = [
            c for c in df.columns if not any(c.startswith(p) or c == p for p in exclude_patterns)
        ]
        feature_nan_pct = df[feature_cols].isna().mean().mean() * 100

        # Labels should have very few NaNs (< 5%)
        check_labels = label_nan_pct < 5
        # Features should have reasonable NaN rate (< 10%)
        check_features = feature_nan_pct < 10

        print(
            f"  {split_name}: labels={label_nan_pct:.2f}% NaN {color_status(check_labels)}, "
            f"features={feature_nan_pct:.2f}% NaN {color_status(check_features)}"
        )

        all_passed &= check_labels and check_features

    print(f"\n{BOLD}Overall Data Completeness: {color_status(all_passed)}{RESET}")
    return all_passed


def verify_invalid_labels(splits: Dict[str, pd.DataFrame]) -> bool:
    """
    Check for label=-99 (invalid markers) and verify proper handling.
    """
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}6. INVALID LABEL HANDLING{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    all_passed = True

    for split_name, df in splits.items():
        print(f"\n{YELLOW}{split_name.upper()} Split:{RESET}")

        label_cols = [c for c in df.columns if c.startswith("label_h")]

        total_invalid = 0

        for col in label_cols:
            invalid_count = (df[col] == -99).sum()
            total_invalid += invalid_count

            if invalid_count > 0:
                pct = invalid_count / len(df) * 100
                # Invalid labels at end of series should be < 1%
                check = pct < 1.0

                horizon = col.split("_h")[1]
                print(f"  H{horizon}: {invalid_count} invalid ({pct:.2f}%) {color_status(check)}")

                all_passed &= check

        if total_invalid == 0:
            print(f"  No invalid labels found {color_status(True)}")

    print(f"\n{BOLD}Overall Invalid Label Handling: {color_status(all_passed)}{RESET}")
    return all_passed


def main():
    """Run all verification checks."""
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}FINAL CRITICAL VERIFICATION - Complete Data Pipeline{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    # Load data
    split_config, scaling_config = load_configs()
    splits = load_splits()

    # Run all checks
    results = {}

    results["temporal_integrity"] = verify_temporal_integrity(splits, split_config)
    results["label_distribution"] = verify_label_distribution(splits)
    results["symbol_balance"] = verify_symbol_balance(splits)
    results["feature_scaling"] = verify_feature_scaling(splits, scaling_config)
    results["data_completeness"] = verify_data_completeness(splits, split_config)
    results["invalid_labels"] = verify_invalid_labels(splits)

    # Final summary
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}VERIFICATION SUMMARY{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    for check_name, passed in results.items():
        check_display = check_name.replace("_", " ").title()
        print(f"  {check_display:<30}: {color_status(passed)}")

    overall_passed = all(results.values())

    print(f"\n{BOLD}{'='*80}{RESET}")
    if overall_passed:
        print(f"{GREEN}{BOLD}OVERALL STATUS: PASS - Pipeline is production-ready{RESET}")
    else:
        print(f"{RED}{BOLD}OVERALL STATUS: FAIL - Critical issues found{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    exit(main())
