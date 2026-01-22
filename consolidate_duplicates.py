#!/usr/bin/env python3
"""
Script to consolidate duplicate directories by updating imports and deleting duplicates.
"""
import os
import re
import shutil
from pathlib import Path

# Define the mapping from old import paths to new canonical paths
IMPORT_MAPPINGS = [
    # Data domain mappings
    (r'\bfrom src\.pipeline\.', 'from src.data.pipeline.'),
    (r'\bfrom src\.pipeline\b', 'from src.data.pipeline'),
    (r'\bimport src\.pipeline\.', 'import src.data.pipeline.'),
    (r'\bimport src\.pipeline\b', 'import src.data.pipeline'),

    (r'\bfrom src\.features\.', 'from src.data.features.'),
    (r'\bfrom src\.features\b', 'from src.data.features'),
    (r'\bimport src\.features\.', 'import src.data.features.'),
    (r'\bimport src\.features\b', 'import src.data.features'),

    (r'\bfrom src\.labeling\.', 'from src.data.labeling.'),
    (r'\bfrom src\.labeling\b', 'from src.data.labeling'),
    (r'\bimport src\.labeling\.', 'import src.data.labeling.'),
    (r'\bimport src\.labeling\b', 'import src.data.labeling'),

    (r'\bfrom src\.adapters\.', 'from src.data.adapters.'),
    (r'\bfrom src\.adapters\b', 'from src.data.adapters'),
    (r'\bimport src\.adapters\.', 'import src.data.adapters.'),
    (r'\bimport src\.adapters\b', 'import src.data.adapters'),

    (r'\bfrom src\.feature_store\.', 'from src.data.store.'),
    (r'\bfrom src\.feature_store\b', 'from src.data.store'),
    (r'\bimport src\.feature_store\.', 'import src.data.store.'),
    (r'\bimport src\.feature_store\b', 'import src.data.store'),

    # Core domain mappings
    (r'\bfrom src\.utils\.', 'from src.core.utils.'),
    (r'\bfrom src\.utils\b', 'from src.core.utils'),
    (r'\bimport src\.utils\.', 'import src.core.utils.'),
    (r'\bimport src\.utils\b', 'import src.core.utils'),

    (r'\bfrom src\.common\.', 'from src.core.common.'),
    (r'\bfrom src\.common\b', 'from src.core.common'),
    (r'\bimport src\.common\.', 'import src.core.common.'),
    (r'\bimport src\.common\b', 'import src.core.common'),

    (r'\bfrom src\.contracts\.', 'from src.core.contracts.'),
    (r'\bfrom src\.contracts\b', 'from src.core.contracts'),
    (r'\bimport src\.contracts\.', 'import src.core.contracts.'),
    (r'\bimport src\.contracts\b', 'import src.core.contracts'),

    # Validation domain mappings
    (r'\bfrom src\.cross_validation\.', 'from src.validation.cv.'),
    (r'\bfrom src\.cross_validation\b', 'from src.validation.cv'),
    (r'\bimport src\.cross_validation\.', 'import src.validation.cv.'),
    (r'\bimport src\.cross_validation\b', 'import src.validation.cv'),

    (r'\bfrom src\.evaluation\.', 'from src.validation.evaluation.'),
    (r'\bfrom src\.evaluation\b', 'from src.validation.evaluation'),
    (r'\bimport src\.evaluation\.', 'import src.validation.evaluation.'),
    (r'\bimport src\.evaluation\b', 'import src.validation.evaluation'),

    (r'\bfrom src\.monitoring\.', 'from src.validation.monitoring.'),
    (r'\bfrom src\.monitoring\b', 'from src.validation.monitoring'),
    (r'\bimport src\.monitoring\.', 'import src.validation.monitoring.'),
    (r'\bimport src\.monitoring\b', 'import src.validation.monitoring'),

    # Inference domain mappings
    (r'\bfrom src\.backtesting\.', 'from src.inference.backtesting.'),
    (r'\bfrom src\.backtesting\b', 'from src.inference.backtesting'),
    (r'\bimport src\.backtesting\.', 'import src.inference.backtesting.'),
    (r'\bimport src\.backtesting\b', 'import src.inference.backtesting'),
]

# Directories to delete after import updates
DUPLICATE_DIRS = [
    'src/pipeline',
    'src/features',
    'src/labeling',
    'src/adapters',
    'src/feature_store',
    'src/utils',
    'src/common',
    'src/contracts',
    'src/cross_validation',
    'src/evaluation',
    'src/monitoring',
    'src/backtesting',
]

def update_imports_in_file(filepath: Path, dry_run: bool = False) -> tuple[bool, list[str]]:
    """Update imports in a single file. Returns (was_modified, list_of_changes)."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError) as e:
        print(f"  Skipping {filepath}: {e}")
        return False, []

    original_content = content
    changes = []

    for pattern, replacement in IMPORT_MAPPINGS:
        # Find all matches before replacing
        matches = re.findall(pattern, content)
        if matches:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                changes.append(f"{pattern} -> {replacement} ({len(matches)} occurrences)")
                content = new_content

    if content != original_content:
        if not dry_run:
            filepath.write_text(content, encoding='utf-8')
        return True, changes

    return False, []


def find_python_files(root_dir: Path) -> list[Path]:
    """Find all Python files in the directory tree."""
    python_files = []
    for path in root_dir.rglob('*.py'):
        # Skip __pycache__ directories
        if '__pycache__' in str(path):
            continue
        # Skip the consolidation script itself
        if path.name == 'consolidate_duplicates.py':
            continue
        python_files.append(path)
    return python_files


def update_all_imports(root_dir: Path, dry_run: bool = False) -> int:
    """Update imports in all Python files. Returns count of modified files."""
    python_files = find_python_files(root_dir)
    modified_count = 0

    print(f"\nScanning {len(python_files)} Python files...")

    for filepath in sorted(python_files):
        was_modified, changes = update_imports_in_file(filepath, dry_run)
        if was_modified:
            modified_count += 1
            rel_path = filepath.relative_to(root_dir)
            print(f"\n  Modified: {rel_path}")
            for change in changes:
                print(f"    - {change}")

    return modified_count


def delete_duplicate_directories(root_dir: Path, dry_run: bool = False) -> int:
    """Delete duplicate directories. Returns count of deleted directories."""
    deleted_count = 0

    print("\n" + "="*60)
    print("Deleting duplicate directories...")
    print("="*60)

    for dir_path in DUPLICATE_DIRS:
        full_path = root_dir / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"\n  Deleting: {dir_path}")
            if not dry_run:
                shutil.rmtree(full_path)
            deleted_count += 1
        else:
            print(f"\n  Not found (already deleted?): {dir_path}")

    return deleted_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Consolidate duplicate directories')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--root', default='.', help='Root directory (default: current directory)')
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()

    print("="*60)
    print("DIRECTORY CONSOLIDATION SCRIPT")
    print("="*60)
    print(f"Root directory: {root_dir}")
    print(f"Dry run: {args.dry_run}")

    # Step 1: Update imports
    print("\n" + "="*60)
    print("Step 1: Updating imports...")
    print("="*60)
    modified_count = update_all_imports(root_dir, dry_run=args.dry_run)
    print(f"\n  Total files modified: {modified_count}")

    # Step 2: Delete duplicate directories
    print("\n" + "="*60)
    print("Step 2: Deleting duplicate directories...")
    print("="*60)
    deleted_count = delete_duplicate_directories(root_dir, dry_run=args.dry_run)
    print(f"\n  Total directories deleted: {deleted_count}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Files modified: {modified_count}")
    print(f"  Directories deleted: {deleted_count}")
    if args.dry_run:
        print("\n  NOTE: This was a DRY RUN. No actual changes were made.")
        print("  Run without --dry-run to apply changes.")


if __name__ == '__main__':
    main()
