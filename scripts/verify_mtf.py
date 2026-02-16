"""
MTF (Multi-Timeframe) Feature Verification Script
===================================================
Loads MES 1-minute data, runs feature engineering with MTF enabled,
and verifies that MTF columns are generated properly.
"""
import sys
import os

os.chdir("/Users/sneh/research")
sys.path.insert(0, "/Users/sneh/research")

import logging

logging.basicConfig(level=logging.WARNING)

import pandas as pd

from src.config.experiment import (
    ExperimentConfig,
    DataSection,
    TrainingSection,
    EvaluationSection,
    BundlingSection,
)
from src.config.training import OptunaConfig
from src.factory import MLFactory


def main():
    print("=" * 70)
    print("MTF FEATURE VERIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load raw data and show basic info
    # -------------------------------------------------------------------------
    data_path = "data/raw/MES_1m_1week.parquet"
    raw_df = pd.read_parquet(data_path)
    print(f"\n[1] Raw data loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(f"    Columns: {list(raw_df.columns)}")

    # -------------------------------------------------------------------------
    # Step 2: Run factory data pipeline (uses default MTF: 15min, 60min)
    # -------------------------------------------------------------------------
    print("\n[2] Running MLFactory data pipeline...")

    config = ExperimentConfig(
        name="mtf_verify",
        output_dir="experiments/mtf_verify",
        verbose=0,
        data=DataSection(
            symbol="MES",
            data_path=data_path,
        ),
        training=TrainingSection(
            models=["xgboost"],
            horizons=[5],
            n_splits=2,
            purge_bars=10,
            embargo_bars=60,
            optuna=OptunaConfig(n_trials=0),
        ),
        evaluation=EvaluationSection(run_backtest=False),
        bundling=BundlingSection(create_bundle=False),
    )

    factory = MLFactory(config, verbose=0)
    df, additional_dfs = factory._run_data_pipeline()

    print(f"    Output shape: {df.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Detect MTF columns using known suffixes
    # -------------------------------------------------------------------------
    from src.core.common.timeframes import get_timeframe_suffix

    # The FeatureEngineer defaults to timeframes=['15min', '60min']
    default_mtf_tfs = ["15min", "60min"]
    expected_suffixes = {tf: get_timeframe_suffix(tf) for tf in default_mtf_tfs}
    print(f"\n[3] Expected MTF suffixes: {expected_suffixes}")

    all_cols = list(df.columns)
    mtf_cols = []
    for col in all_cols:
        for tf, suffix in expected_suffixes.items():
            if col.endswith(suffix):
                mtf_cols.append((col, tf, suffix))
                break

    # Also do a broader search for any timeframe-like pattern
    broad_search_terms = ["15m", "30m", "1h", "60m", "4h", "1d"]
    broad_mtf_cols = [
        c for c in all_cols if any(term in c.lower() for term in broad_search_terms)
    ]

    # -------------------------------------------------------------------------
    # Step 4: Report findings
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("MTF VERIFICATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Total columns in output:  {len(all_cols)}")
    print(f"MTF columns (suffix):     {len(mtf_cols)}")
    print(f"MTF columns (broad):      {len(broad_mtf_cols)}")

    if mtf_cols:
        # Group by timeframe
        by_tf = {}
        for col, tf, suffix in mtf_cols:
            by_tf.setdefault(tf, []).append(col)

        print(f"\nBreakdown by timeframe:")
        for tf in sorted(by_tf.keys()):
            cols = by_tf[tf]
            print(f"  {tf} ({expected_suffixes[tf]}): {len(cols)} features")
            # Show first 5 sample column names
            for c in cols[:5]:
                print(f"    - {c}")
            if len(cols) > 5:
                print(f"    ... and {len(cols) - 5} more")
    else:
        print("\n  WARNING: No MTF columns found via suffix matching!")

    if broad_mtf_cols and len(broad_mtf_cols) != len(mtf_cols):
        extra = set(broad_mtf_cols) - set(c for c, _, _ in mtf_cols)
        if extra:
            print(f"\n  Additional MTF-like columns (broad search): {len(extra)}")
            for c in sorted(extra)[:5]:
                print(f"    - {c}")

    # -------------------------------------------------------------------------
    # Step 5: Check additional_dfs for multi-stream models
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("ADDITIONAL DFS (multi-stream)")
    print(f"{'=' * 70}")
    if additional_dfs:
        for k, v in additional_dfs.items():
            print(f"  {k}: shape={v.shape}")
    else:
        print("  None (multi-stream not generated — expected for default config)")

    # -------------------------------------------------------------------------
    # Step 6: Final verdict
    # -------------------------------------------------------------------------
    n_mtf = len(mtf_cols)
    status = "PASS" if n_mtf > 0 else "FAIL"
    print(f"\n{'=' * 70}")
    print(f"MTF STATUS: {status}  ({n_mtf} MTF features detected)")
    print(f"{'=' * 70}")

    return 0 if n_mtf > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
