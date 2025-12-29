"""
Standalone test runner for quality score tests.
This verifies the tests work without requiring full pytest infrastructure.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/user/Research')

try:
    import numpy as np
    from src.phase1.stages.final_labels.core import compute_quality_scores

    print("=" * 80)
    print("STANDALONE TEST RUNNER FOR QUALITY SCORE CALCULATION")
    print("=" * 80)
    print()

    # Test 1: LONG trade with positive MFE
    print("Test 1: LONG trade with favorable movement")
    print("-" * 60)
    bars_to_hit = np.array([10], dtype=np.int32)
    mae = np.array([-0.5], dtype=np.float32)  # Max downside
    mfe = np.array([1.0], dtype=np.float32)   # Max upside (favorable for LONG)
    labels = np.array([1], dtype=np.int8)     # LONG

    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, 20, symbol='MES'
    )

    print(f"  Input: MAE={mae[0]:.2f}, MFE={mfe[0]:.2f}, Label=LONG")
    print(f"  Expected: favorable=1.0, adverse=0.5, pain_to_gain=0.5")
    print(f"  Actual: pain_to_gain={pain_to_gain[0]:.3f}")
    assert abs(pain_to_gain[0] - 0.5) < 0.01, f"Expected 0.5, got {pain_to_gain[0]}"
    print("  ✓ PASS")
    print()

    # Test 2: SHORT trade with negative MAE
    print("Test 2: SHORT trade with favorable movement")
    print("-" * 60)
    mae = np.array([-1.0], dtype=np.float32)  # Max downside (favorable for SHORT)
    mfe = np.array([0.5], dtype=np.float32)   # Max upside (adverse for SHORT)
    labels = np.array([-1], dtype=np.int8)    # SHORT

    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, 20, symbol='MES'
    )

    print(f"  Input: MAE={mae[0]:.2f}, MFE={mfe[0]:.2f}, Label=SHORT")
    print(f"  Expected: favorable=1.0, adverse=0.5, pain_to_gain=0.5")
    print(f"  Actual: pain_to_gain={pain_to_gain[0]:.3f}")
    assert abs(pain_to_gain[0] - 0.5) < 0.01, f"Expected 0.5, got {pain_to_gain[0]}"
    print("  ✓ PASS")
    print()

    # Test 3: NEUTRAL trade
    print("Test 3: NEUTRAL trade")
    print("-" * 60)
    mae = np.array([-1.0], dtype=np.float32)
    mfe = np.array([1.0], dtype=np.float32)
    labels = np.array([0], dtype=np.int8)     # NEUTRAL

    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, 20, symbol='MES'
    )

    print(f"  Input: MAE={mae[0]:.2f}, MFE={mfe[0]:.2f}, Label=NEUTRAL")
    print(f"  Expected: pain_to_gain=1.0 (default for neutral)")
    print(f"  Actual: pain_to_gain={pain_to_gain[0]:.3f}")
    assert abs(pain_to_gain[0] - 1.0) < 0.01, f"Expected 1.0, got {pain_to_gain[0]}"
    print("  ✓ PASS")
    print()

    # Test 4: LONG with zero adverse movement (ideal trade)
    print("Test 4: LONG trade with zero adverse movement")
    print("-" * 60)
    mae = np.array([0.0], dtype=np.float32)   # No downside
    mfe = np.array([2.0], dtype=np.float32)   # Price went up
    labels = np.array([1], dtype=np.int8)

    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, 20, symbol='MES'
    )

    print(f"  Input: MAE={mae[0]:.2f}, MFE={mfe[0]:.2f}, Label=LONG")
    print(f"  Expected: pain_to_gain=0.0 (ideal trade)")
    print(f"  Actual: pain_to_gain={pain_to_gain[0]:.6f}")
    assert pain_to_gain[0] < 0.01, f"Expected near 0, got {pain_to_gain[0]}"
    print("  ✓ PASS")
    print()

    # Test 5: SHORT with zero adverse movement (ideal trade)
    print("Test 5: SHORT trade with zero adverse movement")
    print("-" * 60)
    mae = np.array([-2.0], dtype=np.float32)  # Price went down
    mfe = np.array([0.0], dtype=np.float32)   # No upside
    labels = np.array([-1], dtype=np.int8)

    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, 20, symbol='MES'
    )

    print(f"  Input: MAE={mae[0]:.2f}, MFE={mfe[0]:.2f}, Label=SHORT")
    print(f"  Expected: pain_to_gain=0.0 (ideal trade)")
    print(f"  Actual: pain_to_gain={pain_to_gain[0]:.6f}")
    assert pain_to_gain[0] < 0.01, f"Expected near 0, got {pain_to_gain[0]}"
    print("  ✓ PASS")
    print()

    # Test 6: Mixed batch
    print("Test 6: Mixed batch with LONG, SHORT, and NEUTRAL")
    print("-" * 60)
    bars_to_hit = np.array([10, 12, 15], dtype=np.int32)
    mae = np.array([-0.5, -1.0, -0.3], dtype=np.float32)
    mfe = np.array([1.0, 0.5, 1.5], dtype=np.float32)
    labels = np.array([1, -1, 0], dtype=np.int8)  # LONG, SHORT, NEUTRAL

    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, 20, symbol='MES'
    )

    print(f"  Sample 0 (LONG): MAE={mae[0]:.2f}, MFE={mfe[0]:.2f}")
    print(f"    Expected: pain_to_gain=0.5")
    print(f"    Actual: pain_to_gain={pain_to_gain[0]:.3f}")
    assert abs(pain_to_gain[0] - 0.5) < 0.05, f"Expected 0.5, got {pain_to_gain[0]}"

    print(f"  Sample 1 (SHORT): MAE={mae[1]:.2f}, MFE={mfe[1]:.2f}")
    print(f"    Expected: pain_to_gain=0.5")
    print(f"    Actual: pain_to_gain={pain_to_gain[1]:.3f}")
    assert abs(pain_to_gain[1] - 0.5) < 0.05, f"Expected 0.5, got {pain_to_gain[1]}"

    print(f"  Sample 2 (NEUTRAL): MAE={mae[2]:.2f}, MFE={mfe[2]:.2f}")
    print(f"    Expected: pain_to_gain=1.0")
    print(f"    Actual: pain_to_gain={pain_to_gain[2]:.3f}")
    assert abs(pain_to_gain[2] - 1.0) < 0.05, f"Expected 1.0, got {pain_to_gain[2]}"
    print("  ✓ PASS")
    print()

    # Test 7: Edge case - all zeros
    print("Test 7: Edge case with zero values")
    print("-" * 60)
    bars_to_hit = np.array([0, 0, 0], dtype=np.int32)
    mae = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    mfe = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    labels = np.array([1, -1, 0], dtype=np.int8)

    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, 20, symbol='MES'
    )

    print(f"  Verified no crashes with zero values")
    assert len(quality_scores) == 3
    assert np.all(np.isfinite(quality_scores))
    assert np.all(np.isfinite(pain_to_gain))
    print("  ✓ PASS")
    print()

    # Test 8: Direction-aware correctness
    print("Test 8: Direction-aware correctness verification")
    print("-" * 60)
    bars_to_hit = np.array([10, 10], dtype=np.int32)
    mae = np.array([-2.0, -2.0], dtype=np.float32)
    mfe = np.array([0.5, 0.5], dtype=np.float32)
    labels = np.array([1, -1], dtype=np.int8)  # LONG, SHORT

    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, 20, symbol='MES'
    )

    print(f"  LONG (label=1): favorable should be MFE (0.5), adverse should be |MAE| (2.0)")
    print(f"    pain_to_gain = 2.0 / 0.5 = 4.0")
    print(f"    Actual: {pain_to_gain[0]:.3f}")
    assert abs(pain_to_gain[0] - 4.0) < 0.1, f"Expected 4.0, got {pain_to_gain[0]}"

    print(f"  SHORT (label=-1): favorable should be |MAE| (2.0), adverse should be MFE (0.5)")
    print(f"    pain_to_gain = 0.5 / 2.0 = 0.25")
    print(f"    Actual: {pain_to_gain[1]:.3f}")
    assert abs(pain_to_gain[1] - 0.25) < 0.05, f"Expected 0.25, got {pain_to_gain[1]}"
    print("  ✓ PASS")
    print()

    print("=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - 8 test scenarios executed")
    print("  - All direction-aware logic verified (LONG/SHORT/NEUTRAL)")
    print("  - Edge cases handled correctly")
    print("  - Quality score calculation working as expected")
    print()
    print("To run the full pytest suite (once environment is set up):")
    print("  poetry run pytest tests/phase_1_tests/stages/final_labels/test_quality_scores.py -v")
    print()

except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print()
    print("Please run: poetry install")
    print("Then run this test with: poetry run python tests/phase_1_tests/stages/final_labels/run_tests_standalone.py")
    sys.exit(1)
except AssertionError as e:
    print(f"TEST FAILED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
