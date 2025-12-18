#!/usr/bin/env python3
"""
Quick verification that all modules can be imported and instantiated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("="*60)
    print("MODULE VERIFICATION")
    print("="*60)

    # Test imports
    print("\n1. Testing imports...")
    try:
        from stages import DataIngestor, DataCleaner, FeatureEngineer
        print("   ✓ All modules imported successfully")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False

    # Test instantiation
    print("\n2. Testing instantiation...")
    try:
        ingestor = DataIngestor(
            raw_data_dir='data/raw',
            output_dir='data/raw',
            source_timezone='UTC'
        )
        print("   ✓ DataIngestor instantiated")

        cleaner = DataCleaner(
            input_dir='data/raw',
            output_dir='data/clean',
            timeframe='1min'
        )
        print("   ✓ DataCleaner instantiated")

        engineer = FeatureEngineer(
            input_dir='data/clean',
            output_dir='data/features',
            timeframe='1min'
        )
        print("   ✓ FeatureEngineer instantiated")

    except Exception as e:
        print(f"   ✗ Instantiation failed: {e}")
        return False

    # Test file existence
    print("\n3. Checking module files...")
    stage_files = [
        'src/stages/stage1_ingest.py',
        'src/stages/stage2_clean.py',
        'src/stages/stage3_features.py',
        'src/stages/__init__.py'
    ]

    for file_path in stage_files:
        path = Path(file_path)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"   ✓ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"   ✗ {file_path} not found")

    # Check data files
    print("\n4. Checking data files...")
    data_files = list(Path('data/raw').glob('*.parquet'))
    if data_files:
        print(f"   ✓ Found {len(data_files)} data files in data/raw/")
        for f in data_files[:5]:
            print(f"     - {f.name}")
    else:
        print("   ⚠ No data files found in data/raw/")

    print("\n" + "="*60)
    print("✓ VERIFICATION COMPLETE - All modules are ready to use!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run individual stages:")
    print("     python src/stages/stage1_ingest.py --raw-dir data/raw")
    print("     python src/stages/stage2_clean.py --input-dir data/raw --output-dir data/clean")
    print("     python src/stages/stage3_features.py --input-dir data/clean --output-dir data/features")
    print("\n  2. Or run complete pipeline:")
    print("     python src/run_pipeline.py --raw-dir data/raw --output-dir data")
    print()

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
