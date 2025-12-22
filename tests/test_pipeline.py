#!/usr/bin/env python3
"""
Quick test script to validate the pipeline stages work correctly.

This script tests each stage with the existing data using modern Python 3.12+ patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stages.stage1_ingest import DataIngestor
from stages.stage2_clean import DataCleaner
from stages.features.engineer import FeatureEngineer

# Test configuration matching pipeline defaults
TEST_SYMBOLS: Final[list[str]] = ['MES', 'MGC']
TEST_HORIZONS: Final[list[int]] = [5, 20]  # H1 excluded (transaction costs > profit)

def test_stage1() -> bool:
    """
    Test Stage 1: Data Ingestion.

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "="*60)
    print("Testing Stage 1: Data Ingestion")
    print("="*60)

    # Use absolute paths for test directories
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'data' / 'raw_test'

    ingestor = DataIngestor(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        source_timezone='UTC'
    )

    # Test with first available symbol
    test_file = raw_data_dir / 'MES_1m.parquet'
    if not test_file.exists():
        # Try alternate file pattern
        test_file = raw_data_dir / 'MES.parquet'

    if test_file.exists():
        df, metadata = ingestor.ingest_file(test_file, validate=True)
        print(f"\n✓ Stage 1 test passed")
        print(f"  Loaded {len(df):,} rows")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        return True
    else:
        print(f"✗ Test file not found: {test_file}")
        return False


def test_stage2() -> bool:
    """
    Test Stage 2: Data Cleaning.

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "="*60)
    print("Testing Stage 2: Data Cleaning")
    print("="*60)

    # Use absolute paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'data' / 'clean_test'

    cleaner = DataCleaner(
        input_dir=input_dir,
        output_dir=output_dir,
        timeframe='1min',
        gap_fill_method='forward',
        max_gap_fill_minutes=5,
        outlier_method='atr',
        atr_threshold=5.0
    )

    # Test with first available symbol
    test_file = input_dir / 'MES_1m.parquet'
    if not test_file.exists():
        test_file = input_dir / 'MES.parquet'

    if test_file.exists():
        df, report = cleaner.clean_file(test_file)
        print(f"\n✓ Stage 2 test passed")
        print(f"  Initial rows: {report['initial_rows']:,}")
        print(f"  Final rows: {report['final_rows']:,}")
        print(f"  Retention: {report['retention_pct']:.2f}%")
        print(f"  Duplicates removed: {report['duplicates']['n_duplicates']}")
        print(f"  Outliers removed: {report['outliers']['total_outliers']}")
        return True
    else:
        print(f"✗ Test file not found: {test_file}")
        return False


def test_stage3() -> bool:
    """
    Test Stage 3: Feature Engineering.

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "="*60)
    print("Testing Stage 3: Feature Engineering")
    print("="*60)

    # Use absolute paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / 'data' / 'raw'
    clean_dir = project_root / 'data' / 'clean_test'
    features_dir = project_root / 'data' / 'features_test'

    # First clean a file
    cleaner = DataCleaner(
        input_dir=input_dir,
        output_dir=clean_dir,
        timeframe='1min',
        gap_fill_method='forward',
        max_gap_fill_minutes=5
    )

    test_file = input_dir / 'MES_1m.parquet'
    if not test_file.exists():
        test_file = input_dir / 'MES.parquet'

    if test_file.exists():
        df_clean, _ = cleaner.clean_file(test_file)

        # Now test feature engineering
        engineer = FeatureEngineer(
            input_dir=clean_dir,
            output_dir=features_dir,
            timeframe='5min'  # Using 5min as per pipeline config
        )

        # Save cleaned file temporarily
        temp_file = clean_dir / 'MES.parquet'
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(temp_file, index=False)

        # Engineer features
        df_features, report = engineer.engineer_features(df_clean, 'MES')

        print(f"\n✓ Stage 3 test passed")
        print(f"  Initial columns: {report['initial_columns']}")
        print(f"  Final columns: {report['final_columns']}")
        print(f"  Features added: {report['features_added']}")
        print(f"  Final rows: {report['final_rows']:,}")

        # Show some feature names
        base_cols = {'datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol'}
        feature_cols = [c for c in df_features.columns if c not in base_cols]
        print(f"\n  Sample features ({len(feature_cols)} total):")
        for feat in feature_cols[:20]:
            print(f"    - {feat}")
        if len(feature_cols) > 20:
            print(f"    ... and {len(feature_cols) - 20} more")

        return True
    else:
        print(f"✗ Test file not found: {test_file}")
        return False


def main() -> None:
    """Run all pipeline stage tests."""
    print("\n" + "="*80)
    print("PIPELINE STAGE TESTS")
    print("="*80)
    print(f"Test configuration: symbols={TEST_SYMBOLS}, horizons={TEST_HORIZONS}")

    results: dict[str, bool] = {
        'Stage 1 (Ingestion)': test_stage1(),
        'Stage 2 (Cleaning)': test_stage2(),
        'Stage 3 (Features)': test_stage3()
    }

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = True
    for stage, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{stage}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
