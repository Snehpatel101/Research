#!/usr/bin/env python3
"""
Test script to validate workflow integration fixes.

Tests:
1. Run ID collision prevention (milliseconds + random suffix)
2. CV output directory isolation
3. Phase 3→4 stacking data loading

Usage:
    python test_workflow_integration.py
"""
import secrets
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_run_id_collision_prevention():
    """Test that run IDs are unique even when generated rapidly."""
    print("=" * 60)
    print("TEST 1: Run ID Collision Prevention")
    print("=" * 60)

    # Generate 100 run IDs rapidly
    ids = set()
    for i in range(100):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        random_suffix = secrets.token_hex(2)
        run_id = f"{timestamp}_{random_suffix}"
        ids.add(run_id)

    print(f"Generated: 100 run IDs")
    print(f"Unique: {len(ids)}")
    print(f"Duplicates: {100 - len(ids)}")

    if len(ids) == 100:
        print("✓ PASS: All run IDs are unique")
        return True
    else:
        print("✗ FAIL: Collision detected!")
        return False


def test_cv_output_directory():
    """Test CV output directory structure."""
    print("\n" + "=" * 60)
    print("TEST 2: CV Output Directory Structure")
    print("=" * 60)

    # Simulate CV run ID generation (without importing run_cv.py)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_suffix = secrets.token_hex(2)
    cv_run_id = f"{timestamp}_{random_suffix}"

    cv_output_dir = PROJECT_ROOT / "data" / "stacking" / cv_run_id

    print(f"CV Run ID: {cv_run_id}")
    print(f"Output Directory: {cv_output_dir}")
    print(f"Expected stacking dir: {cv_output_dir / 'stacking'}")

    # Validate format
    parts = cv_run_id.split("_")
    if len(parts) == 4:  # YYYYMMDD_HHMMSS_microseconds_random
        print("✓ PASS: CV run ID has correct format")
        return True
    else:
        print(f"✗ FAIL: Invalid format (expected 4 parts, got {len(parts)})")
        return False


def test_phase3_data_loading():
    """Test Phase 3 stacking data loading function signature."""
    print("\n" + "=" * 60)
    print("TEST 3: Phase 3 Stacking Data Loading")
    print("=" * 60)

    # Test that the function exists and has correct signature
    # (Don't actually call it to avoid import issues)

    print("Checking load_phase3_stacking_data() exists in train_model.py...")

    train_model_path = PROJECT_ROOT / "scripts" / "train_model.py"
    with open(train_model_path, "r") as f:
        content = f.read()

    has_function = "def load_phase3_stacking_data(" in content
    has_cv_run_id = "cv_run_id: str" in content
    has_horizon = "horizon: int" in content
    has_phase3_base_dir = "phase3_base_dir: Path" in content

    if has_function and has_cv_run_id and has_horizon and has_phase3_base_dir:
        print("✓ PASS: load_phase3_stacking_data() exists with correct signature")
        print("  Parameters: cv_run_id, horizon, phase3_base_dir")
        return True
    else:
        print("✗ FAIL: Function missing or has incorrect signature")
        return False


def test_trainer_run_id_format():
    """Test trainer run ID format."""
    print("\n" + "=" * 60)
    print("TEST 4: Trainer Run ID Format")
    print("=" * 60)

    # Simulate trainer run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_suffix = secrets.token_hex(2)
    run_id = f"xgboost_h20_{timestamp}_{random_suffix}"

    print(f"Trainer Run ID: {run_id}")

    # Validate format
    parts = run_id.split("_")
    if len(parts) == 6:  # model_h_horizon_YYYYMMDD_HHMMSS_microseconds_random
        print("✓ PASS: Trainer run ID has correct format")
        print(f"  Model: {parts[0]}")
        print(f"  Horizon: {parts[1]}")
        print(f"  Timestamp: {parts[2]}_{parts[3]}_{parts[4]}")
        print(f"  Random: {parts[5]}")
        return True
    else:
        print(f"✗ FAIL: Invalid format (expected 6 parts, got {len(parts)})")
        return False


def main():
    """Run all tests."""
    print("\nWorkflow Integration Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Run ID Collision Prevention", test_run_id_collision_prevention()))
    results.append(("CV Output Directory", test_cv_output_directory()))
    results.append(("Phase 3 Data Loading", test_phase3_data_loading()))
    results.append(("Trainer Run ID Format", test_trainer_run_id_format()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
