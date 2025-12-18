#!/usr/bin/env python3
"""
Quick test script for labeling stages
Tests imports and basic functionality with small dataset
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from stages import stage4_labeling
        print("  ✓ stage4_labeling imported")
    except ImportError as e:
        print(f"  ✗ stage4_labeling failed: {e}")
        return False
    
    try:
        from stages import stage5_ga_optimize
        print("  ✓ stage5_ga_optimize imported")
    except ImportError as e:
        print(f"  ✗ stage5_ga_optimize failed: {e}")
        return False
    
    try:
        from stages import stage6_final_labels
        print("  ✓ stage6_final_labels imported")
    except ImportError as e:
        print(f"  ✗ stage6_final_labels failed: {e}")
        return False
    
    print("  ✓ All imports successful\n")
    return True


def test_numba_function():
    """Test the numba triple barrier function."""
    print("Testing numba triple barrier function...")
    
    from stages.stage4_labeling import triple_barrier_numba
    
    # Create small test dataset
    n = 100
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    atr = np.ones(n) * 1.0
    
    # Run labeling
    labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
        close, high, low, atr, k_up=2.0, k_down=1.0, max_bars=10
    )
    
    # Basic checks
    assert len(labels) == n, "Labels length mismatch"
    assert len(bars_to_hit) == n, "Bars_to_hit length mismatch"
    assert len(mae) == n, "MAE length mismatch"
    assert len(mfe) == n, "MFE length mismatch"
    assert set(labels).issubset({-1, 0, 1}), "Invalid label values"
    
    print(f"  ✓ Generated {n} labels")
    print(f"  ✓ Label distribution: {np.bincount(labels + 1)}")
    print(f"  ✓ Numba function works correctly\n")
    return True


def test_quality_scoring():
    """Test quality scoring function."""
    print("Testing quality scoring...")
    
    from stages.stage6_final_labels import compute_quality_scores
    
    # Create test data
    n = 100
    np.random.seed(42)
    bars_to_hit = np.random.randint(1, 20, n)
    mae = np.random.randn(n) * 0.01
    mfe = np.abs(np.random.randn(n) * 0.02)
    horizon = 5
    
    # Compute scores
    quality_scores = compute_quality_scores(bars_to_hit, mae, mfe, horizon)
    
    # Basic checks
    assert len(quality_scores) == n, "Quality scores length mismatch"
    assert np.all(quality_scores >= 0), "Quality scores should be non-negative"
    assert np.all(quality_scores <= 1.5), "Quality scores should be reasonable"
    
    print(f"  ✓ Generated {n} quality scores")
    print(f"  ✓ Score range: [{quality_scores.min():.3f}, {quality_scores.max():.3f}]")
    print(f"  ✓ Mean score: {quality_scores.mean():.3f}")
    print(f"  ✓ Quality scoring works correctly\n")
    return True


def test_sample_weights():
    """Test sample weight assignment."""
    print("Testing sample weight assignment...")
    
    from stages.stage6_final_labels import assign_sample_weights
    
    # Create test quality scores
    n = 1000
    np.random.seed(42)
    quality_scores = np.random.rand(n)
    
    # Assign weights
    sample_weights = assign_sample_weights(quality_scores)
    
    # Check tiers
    unique_weights = np.unique(sample_weights)
    expected_weights = {0.5, 1.0, 1.5}
    
    assert set(unique_weights) == expected_weights, f"Unexpected weights: {unique_weights}"
    
    # Check distribution (should be ~20%, ~60%, ~20%)
    tier1 = (sample_weights == 1.5).sum()
    tier2 = (sample_weights == 1.0).sum()
    tier3 = (sample_weights == 0.5).sum()
    
    print(f"  ✓ Tier 1 (1.5x): {tier1} ({tier1/n*100:.1f}%)")
    print(f"  ✓ Tier 2 (1.0x): {tier2} ({tier2/n*100:.1f}%)")
    print(f"  ✓ Tier 3 (0.5x): {tier3} ({tier3/n*100:.1f}%)")
    print(f"  ✓ Sample weight assignment works correctly\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("LABELING STAGES - UNIT TESTS")
    print("=" * 60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Numba Triple Barrier", test_numba_function),
        ("Quality Scoring", test_quality_scoring),
        ("Sample Weights", test_sample_weights),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  ✗ {name} failed with error: {e}\n")
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
