#!/usr/bin/env python3
"""
Quick test script to validate the pipeline stages work correctly.

This script tests each stage with the existing data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stages import DataIngestor, DataCleaner, FeatureEngineer

def test_stage1():
    """Test Stage 1: Data Ingestion"""
    print("\n" + "="*60)
    print("Testing Stage 1: Data Ingestion")
    print("="*60)

    ingestor = DataIngestor(
        raw_data_dir='data/raw',
        output_dir='data/raw_test',
        source_timezone='UTC'
    )

    # Test with one file
    test_file = Path('data/raw/MES_1m.parquet')
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


def test_stage2():
    """Test Stage 2: Data Cleaning"""
    print("\n" + "="*60)
    print("Testing Stage 2: Data Cleaning")
    print("="*60)

    cleaner = DataCleaner(
        input_dir='data/raw',
        output_dir='data/clean_test',
        timeframe='1min',
        gap_fill_method='forward',
        max_gap_fill_minutes=5,
        outlier_method='atr',
        atr_threshold=5.0
    )

    # Test with one file
    test_file = Path('data/raw/MES_1m.parquet')
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


def test_stage3():
    """Test Stage 3: Feature Engineering"""
    print("\n" + "="*60)
    print("Testing Stage 3: Feature Engineering")
    print("="*60)

    # First clean a file
    cleaner = DataCleaner(
        input_dir='data/raw',
        output_dir='data/clean_test',
        timeframe='1min',
        gap_fill_method='forward',
        max_gap_fill_minutes=5
    )

    test_file = Path('data/raw/MES_1m.parquet')
    if test_file.exists():
        df_clean, _ = cleaner.clean_file(test_file)

        # Now test feature engineering
        engineer = FeatureEngineer(
            input_dir='data/clean_test',
            output_dir='data/features_test',
            timeframe='1min'
        )

        # Save cleaned file temporarily
        temp_file = Path('data/clean_test/MES.parquet')
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
        feature_cols = [c for c in df_features.columns if c not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
        print(f"\n  Sample features ({len(feature_cols)} total):")
        for feat in feature_cols[:20]:
            print(f"    - {feat}")
        if len(feature_cols) > 20:
            print(f"    ... and {len(feature_cols) - 20} more")

        return True
    else:
        print(f"✗ Test file not found: {test_file}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("PIPELINE STAGE TESTS")
    print("="*80)

    results = {
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
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
