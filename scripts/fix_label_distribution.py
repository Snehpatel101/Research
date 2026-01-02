"""
Fix label distribution mismatch by regenerating splits and scaled data.

This script:
1. Backs up current state
2. Deletes stale indices and scaled data
3. Regenerates splits from the full combined dataset
4. Regenerates scaled data
5. Validates the fix

Usage:
    python scripts/fix_label_distribution.py --backup-only  # Just create backup
    python scripts/fix_label_distribution.py --fix          # Run the full fix
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def backup_current_state(backup_dir: Path) -> None:
    """Create backup of current splits and scaled data."""
    print(f"\n{'='*80}")
    print("CREATING BACKUP")
    print(f"{'='*80}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    # Backup splits directory
    splits_dir = Path("/Users/sneh/research/data/splits")
    splits_backup = backup_dir / "splits"

    if splits_dir.exists():
        shutil.copytree(splits_dir, splits_backup, dirs_exist_ok=True)
        print(f"✓ Backed up splits to: {splits_backup}")
    else:
        print(f"⚠ Splits directory not found: {splits_dir}")

    # Create backup manifest
    manifest = {
        "backup_date": datetime.now().isoformat(),
        "original_splits_dir": str(splits_dir),
        "backup_location": str(splits_backup),
        "files_backed_up": [
            str(p.relative_to(backup_dir)) for p in backup_dir.rglob("*") if p.is_file()
        ],
    }

    manifest_path = backup_dir / "backup_manifest.json"
    import json

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created backup manifest: {manifest_path}")
    print(f"\nBackup complete. Total files backed up: {len(manifest['files_backed_up'])}")


def delete_stale_data() -> None:
    """Delete stale indices and scaled data."""
    print(f"\n{'='*80}")
    print("DELETING STALE DATA")
    print(f"{'='*80}")

    splits_dir = Path("/Users/sneh/research/data/splits")

    # Delete index files
    for index_file in ["train_indices.npy", "val_indices.npy", "test_indices.npy"]:
        path = splits_dir / index_file
        if path.exists():
            path.unlink()
            print(f"✓ Deleted: {index_file}")
        else:
            print(f"⚠ Not found: {index_file}")

    # Delete scaled directory
    scaled_dir = splits_dir / "scaled"
    if scaled_dir.exists():
        shutil.rmtree(scaled_dir)
        print(f"✓ Deleted: scaled/")
    else:
        print(f"⚠ Not found: scaled/")

    # Delete datasets directory (will be regenerated)
    datasets_dir = splits_dir / "datasets"
    if datasets_dir.exists():
        shutil.rmtree(datasets_dir)
        print(f"✓ Deleted: datasets/")
    else:
        print(f"⚠ Not found: datasets/")

    print("\nStale data deletion complete.")


def validate_fix() -> bool:
    """Validate that the fix worked correctly."""
    print(f"\n{'='*80}")
    print("VALIDATING FIX")
    print(f"{'='*80}")

    try:
        # Load combined data
        combined_path = Path("/Users/sneh/research/data/final/combined_final_labeled.parquet")
        df = pd.read_parquet(combined_path)
        print(f"✓ Loaded combined data: {len(df):,} rows")

        # Load new indices
        splits_dir = Path("/Users/sneh/research/data/splits")
        train_indices = np.load(splits_dir / "train_indices.npy")
        val_indices = np.load(splits_dir / "val_indices.npy")
        test_indices = np.load(splits_dir / "test_indices.npy")

        print(f"✓ Loaded indices:")
        print(f"  Train: {len(train_indices):,}")
        print(f"  Val:   {len(val_indices):,}")
        print(f"  Test:  {len(test_indices):,}")

        # Load scaled data
        train_scaled = pd.read_parquet(splits_dir / "scaled" / "train_scaled.parquet")
        val_scaled = pd.read_parquet(splits_dir / "scaled" / "val_scaled.parquet")
        test_scaled = pd.read_parquet(splits_dir / "scaled" / "test_scaled.parquet")

        print(f"✓ Loaded scaled data:")
        print(f"  Train: {len(train_scaled):,}")
        print(f"  Val:   {len(val_scaled):,}")
        print(f"  Test:  {len(test_scaled):,}")

        # Validate sizes match
        if len(train_scaled) != len(train_indices):
            print(
                f"✗ VALIDATION FAILED: Train size mismatch "
                f"({len(train_scaled)} vs {len(train_indices)})"
            )
            return False

        if len(val_scaled) != len(val_indices):
            print(
                f"✗ VALIDATION FAILED: Val size mismatch "
                f"({len(val_scaled)} vs {len(val_indices)})"
            )
            return False

        if len(test_scaled) != len(test_indices):
            print(
                f"✗ VALIDATION FAILED: Test size mismatch "
                f"({len(test_scaled)} vs {len(test_indices)})"
            )
            return False

        # Check label distributions
        print(f"\n✓ Label distributions:")
        for split_name, split_df in [
            ("Train", train_scaled),
            ("Val", val_scaled),
            ("Test", test_scaled),
        ]:
            label_counts = split_df["label_h20"].value_counts().sort_index()
            print(f"\n  {split_name}:")
            for label in [-1, 0, 1]:
                count = label_counts.get(label, 0)
                pct = count / len(split_df) * 100 if len(split_df) > 0 else 0
                label_name = {-1: "Short", 0: "Neutral", 1: "Long"}[label]
                print(f"    {label_name:7s} ({label:2d}): {count:8,} ({pct:5.1f}%)")

                # Validate distribution is balanced (within 5% of expected 48-50%)
                if label in [-1, 1]:  # Check Short and Long
                    if pct < 40 or pct > 55:
                        print(
                            f"    ⚠ WARNING: {label_name} percentage {pct:.1f}% "
                            f"outside expected range (40-55%)"
                        )

        print(f"\n{'='*80}")
        print("✓ VALIDATION PASSED")
        print(f"{'='*80}")
        return True

    except Exception as e:
        print(f"\n✗ VALIDATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Fix label distribution mismatch in pipeline data")
    parser.add_argument(
        "--backup-only",
        action="store_true",
        help="Only create backup, don't fix",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Run the full fix (backup + regenerate)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate current state",
    )

    args = parser.parse_args()

    if not any([args.backup_only, args.fix, args.validate_only]):
        parser.print_help()
        print("\n⚠ No action specified. Use --backup-only, --fix, or --validate-only")
        return 1

    # Create backup directory
    backup_dir = Path(
        f"/Users/sneh/research/data/backups/" f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pre_fix"
    )

    if args.validate_only:
        success = validate_fix()
        return 0 if success else 1

    if args.backup_only or args.fix:
        backup_current_state(backup_dir)

    if args.fix:
        delete_stale_data()

        print(f"\n{'='*80}")
        print("NEXT STEPS")
        print(f"{'='*80}")
        print("\nTo complete the fix, run these commands:")
        print("\n1. Regenerate splits:")
        print("   cd /Users/sneh/research")
        print("   python -m src.phase1.stages.splits.run")
        print("\n2. Regenerate scaled data:")
        print("   python -m src.phase1.stages.scaling.run")
        print("\n3. Regenerate datasets:")
        print("   python -m src.phase1.stages.datasets.run")
        print("\n4. Validate the fix:")
        print("   python scripts/fix_label_distribution.py --validate-only")
        print("\nOr use the pipeline CLI:")
        print("   ./pipeline run --start-from create_splits")

    return 0


if __name__ == "__main__":
    sys.exit(main())
