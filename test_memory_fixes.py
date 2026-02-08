#!/usr/bin/env python3
"""
test_memory_fixes.py - Verify Phase 37 memory fixes work correctly.

Run with: python test_memory_fixes.py

This script verifies:
1. gc module is imported at module level in unified_orchestrator.py
2. gc.collect() is called between models after del prepared
3. OOF predictions are cleared after ensemble building

Author: Diagnostic Agent 7/9
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Add src to path for imports
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR.parent))


def test_gc_import_at_module_level() -> bool:
    """Verify gc is imported at module level, not inside functions."""
    print("\n[TEST 1] Verify gc import at module level...")

    orchestrator_path = SRC_DIR / "models" / "training" / "unified_orchestrator.py"

    if not orchestrator_path.exists():
        print(f"  FAIL: File not found: {orchestrator_path}")
        return False

    content = orchestrator_path.read_text()
    tree = ast.parse(content)

    # Check for module-level import of gc
    gc_imported = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "gc":
                    gc_imported = True
                    print(f"  Found: 'import gc' at line {node.lineno}")
                    break

    if gc_imported:
        print("  PASS: gc is imported at module level")
        return True
    else:
        print("  FAIL: gc is not imported at module level")
        return False


def test_gc_collect_after_del_prepared() -> bool:
    """Verify gc.collect() is called after del prepared in the training loop."""
    print("\n[TEST 2] Verify gc.collect() after del prepared...")

    orchestrator_path = SRC_DIR / "models" / "training" / "unified_orchestrator.py"

    if not orchestrator_path.exists():
        print(f"  FAIL: File not found: {orchestrator_path}")
        return False

    content = orchestrator_path.read_text()
    lines = content.split("\n")

    # Find "del prepared" and check if gc.collect() follows
    found_del_prepared = False
    gc_collect_follows = False
    del_line_num = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "del prepared":
            found_del_prepared = True
            del_line_num = i + 1
            print(f"  Found: 'del prepared' at line {del_line_num}")

            # Check next few non-empty lines for gc.collect()
            for j in range(i + 1, min(i + 5, len(lines))):
                next_stripped = lines[j].strip()
                if next_stripped == "gc.collect()":
                    gc_collect_follows = True
                    print(f"  Found: 'gc.collect()' at line {j + 1}")
                    break
                elif next_stripped and not next_stripped.startswith("#"):
                    # Non-comment, non-empty line that isn't gc.collect()
                    break
            break

    if not found_del_prepared:
        print("  FAIL: 'del prepared' not found")
        return False

    if gc_collect_follows:
        print("  PASS: gc.collect() is called after del prepared")
        return True
    else:
        print("  FAIL: gc.collect() not found after del prepared")
        return False


def test_oof_predictions_cleared_after_ensemble() -> bool:
    """Verify OOF predictions are cleared after ensemble building."""
    print("\n[TEST 3] Verify OOF predictions cleared after ensemble...")

    orchestrator_path = SRC_DIR / "models" / "training" / "unified_orchestrator.py"

    if not orchestrator_path.exists():
        print(f"  FAIL: File not found: {orchestrator_path}")
        return False

    content = orchestrator_path.read_text()

    # Check for the pattern: _build_ensemble call followed by _oof_predictions.clear()
    if "_build_ensemble" not in content:
        print("  FAIL: _build_ensemble method not found")
        return False

    if "self._oof_predictions.clear()" not in content:
        print("  FAIL: self._oof_predictions.clear() not found")
        return False

    # Verify the clear happens after build_ensemble in the train method
    lines = content.split("\n")
    build_ensemble_line = None
    clear_oof_line = None

    for i, line in enumerate(lines):
        if "_build_ensemble(df)" in line or "_build_ensemble(" in line:
            if "aligned_oof" in line and "stacking_dataset" in line:
                build_ensemble_line = i + 1
                print(f"  Found: _build_ensemble call at line {build_ensemble_line}")

        if "self._oof_predictions.clear()" in line:
            clear_oof_line = i + 1
            print(f"  Found: _oof_predictions.clear() at line {clear_oof_line}")

    if build_ensemble_line and clear_oof_line:
        if clear_oof_line > build_ensemble_line:
            print("  PASS: OOF predictions cleared after ensemble building")
            return True
        else:
            print("  FAIL: clear() called before _build_ensemble()")
            return False
    else:
        print("  FAIL: Could not locate both build_ensemble and clear() calls")
        return False


def test_imports_work() -> bool:
    """Verify all relevant imports work without errors."""
    print("\n[TEST 4] Verify imports work...")

    try:
        # Import gc (should always work)
        import gc
        print("  OK: import gc")

        # Import the orchestrator module
        from src.models.training import UnifiedTrainingOrchestrator
        print("  OK: from src.models.training import UnifiedTrainingOrchestrator")

        # Verify the class has the expected attributes
        from src.core import PipelineConfig
        print("  OK: from src.core import PipelineConfig")

        print("  PASS: All imports successful")
        return True

    except ImportError as e:
        print(f"  FAIL: Import error: {e}")
        return False
    except Exception as e:
        print(f"  FAIL: Unexpected error: {e}")
        return False


def test_orchestrator_has_oof_storage() -> bool:
    """Verify orchestrator initializes _oof_predictions dict."""
    print("\n[TEST 5] Verify orchestrator has _oof_predictions storage...")

    orchestrator_path = SRC_DIR / "models" / "training" / "unified_orchestrator.py"

    if not orchestrator_path.exists():
        print(f"  FAIL: File not found: {orchestrator_path}")
        return False

    content = orchestrator_path.read_text()

    # Check for _oof_predictions initialization
    if "self._oof_predictions: dict" in content:
        print("  Found: self._oof_predictions initialization with type hint")
        print("  PASS: _oof_predictions storage exists")
        return True
    elif "self._oof_predictions = {}" in content:
        print("  Found: self._oof_predictions = {}")
        print("  PASS: _oof_predictions storage exists")
        return True
    else:
        print("  FAIL: _oof_predictions not found in orchestrator")
        return False


def main() -> int:
    """Run all verification tests."""
    print("=" * 60)
    print("Phase 37 Memory Fixes Verification")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("gc import at module level", test_gc_import_at_module_level()))
    results.append(("gc.collect() after del prepared", test_gc_collect_after_del_prepared()))
    results.append(("OOF predictions cleared after ensemble", test_oof_predictions_cleared_after_ensemble()))
    results.append(("Imports work", test_imports_work()))
    results.append(("Orchestrator has _oof_predictions", test_orchestrator_has_oof_storage()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll memory fixes verified successfully!")
        return 0
    else:
        print("\nSome tests failed. Please check the fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
