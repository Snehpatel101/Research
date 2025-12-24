#!/usr/bin/env python3
"""
Quick test script for labeling stages using modern Python 3.12+ patterns.
Tests imports and basic functionality with small dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

# Test configuration matching pipeline defaults
TEST_SYMBOLS: Final[list[str]] = ['MES', 'MGC']
TEST_HORIZONS: Final[list[int]] = [5, 20]  # H1 excluded (transaction costs > profit)

def test_imports() -> bool:
    """
    Test that all labeling stage modules can be imported.

    Returns:
        True if all imports successful, False otherwise
    """
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


def test_numba_function() -> bool:
    """
    Test the numba triple barrier function.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing numba triple barrier function...")

    from stages.stage4_labeling import triple_barrier_numba

    # Create small test dataset
    n: int = 100
    np.random.seed(42)
    close: np.ndarray = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high: np.ndarray = close + np.abs(np.random.randn(n) * 0.2)
    low: np.ndarray = close - np.abs(np.random.randn(n) * 0.2)
    open_prices: np.ndarray = close - np.random.randn(n) * 0.1  # Open prices
    atr: np.ndarray = np.ones(n) * 1.0

    # Run labeling with realistic parameters (matching config.py)
    labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
        close, high, low, open_prices, atr, k_up=2.0, k_down=1.0, max_bars=15
    )

    # Basic checks with proper type assertions
    assert len(labels) == n, "Labels length mismatch"
    assert len(bars_to_hit) == n, "Bars_to_hit length mismatch"
    assert len(mae) == n, "MAE length mismatch"
    assert len(mfe) == n, "MFE length mismatch"
    # Labels can be -99 (invalid), -1 (short loss), 0 (timeout), 1 (long win)
    assert set(labels).issubset({-99, -1, 0, 1}), "Invalid label values"

    print(f"  ✓ Generated {n} labels")
    # Show label distribution (handle -99 separately since bincount can't handle it)
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    print(f"  ✓ Label distribution: {label_dist}")
    print(f"  ✓ Numba function works correctly\n")
    return True


def test_quality_scoring() -> bool:
    """
    Test quality scoring function.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing quality scoring...")

    from stages.stage6_final_labels import compute_quality_scores

    # Create test data
    n: int = 100
    np.random.seed(42)
    bars_to_hit: np.ndarray = np.random.randint(1, 20, n)
    mae: np.ndarray = np.random.randn(n) * 0.01
    mfe: np.ndarray = np.abs(np.random.randn(n) * 0.02)
    labels: np.ndarray = np.random.choice([-1, 0, 1], n)  # Random labels
    horizon: int = 5  # Using active horizon from TEST_HORIZONS

    # Compute scores (returns 3 values: quality_scores, pain_to_gain, time_weighted_dd)
    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, horizon
    )

    # Basic checks with explicit type validation
    assert len(quality_scores) == n, "Quality scores length mismatch"
    assert np.all(quality_scores >= 0), "Quality scores should be non-negative"
    assert np.all(quality_scores <= 1.5), "Quality scores should be reasonable"

    print(f"  ✓ Generated {n} quality scores")
    print(f"  ✓ Score range: [{quality_scores.min():.3f}, {quality_scores.max():.3f}]")
    print(f"  ✓ Mean score: {quality_scores.mean():.3f}")
    print(f"  ✓ Quality scoring works correctly\n")
    return True


def test_sample_weights() -> bool:
    """
    Test sample weight assignment.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing sample weight assignment...")

    from stages.stage6_final_labels import assign_sample_weights

    # Create test quality scores
    n: int = 1000
    np.random.seed(42)
    quality_scores: np.ndarray = np.random.rand(n)

    # Assign weights
    sample_weights: np.ndarray = assign_sample_weights(quality_scores)

    # Check tiers with type-safe assertions
    unique_weights: np.ndarray = np.unique(sample_weights)
    expected_weights: set[float] = {0.5, 1.0, 1.5}

    assert set(unique_weights) == expected_weights, f"Unexpected weights: {unique_weights}"

    # Check distribution (should be ~20%, ~60%, ~20%)
    tier1: int = (sample_weights == 1.5).sum()
    tier2: int = (sample_weights == 1.0).sum()
    tier3: int = (sample_weights == 0.5).sum()

    print(f"  ✓ Tier 1 (1.5x): {tier1} ({tier1/n*100:.1f}%)")
    print(f"  ✓ Tier 2 (1.0x): {tier2} ({tier2/n*100:.1f}%)")
    print(f"  ✓ Tier 3 (0.5x): {tier3} ({tier3/n*100:.1f}%)")
    print(f"  ✓ Sample weight assignment works correctly\n")
    return True


def main() -> int:
    """
    Run all labeling stage tests.

    Returns:
        0 if all tests passed, 1 if any failed
    """
    print("=" * 60)
    print("LABELING STAGES - UNIT TESTS")
    print("=" * 60)
    print(f"Test configuration: symbols={TEST_SYMBOLS}, horizons={TEST_HORIZONS}")
    print()

    tests: list[tuple[str, callable]] = [
        ("Imports", test_imports),
        ("Numba Triple Barrier", test_numba_function),
        ("Quality Scoring", test_quality_scoring),
        ("Sample Weights", test_sample_weights),
    ]

    results: list[tuple[str, bool]] = []
    for name, test_func in tests:
        try:
            success: bool = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  ✗ {name} failed with error: {e}\n")
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, success in results:
        status: str = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    total: int = len(results)
    passed: int = sum(1 for _, s in results if s)

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
