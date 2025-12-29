"""
Benchmark script for PurgedKFold vectorized optimization.

Compares performance of vectorized label overlap checking vs. the original
loop-based implementation across different dataset sizes.

Usage:
    python scripts/benchmark_purged_kfold.py
    python scripts/benchmark_purged_kfold.py --sizes 1000,10000,100000
    python scripts/benchmark_purged_kfold.py --n-splits 5 --verify
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_data(
    n_samples: int,
    n_features: int = 10,
    horizon: int = 20,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate synthetic OHLCV-like data for benchmarking.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        horizon: Label horizon in bars

    Returns:
        Tuple of (X, y, label_end_times)
    """
    # Generate datetime index (5-min bars)
    start_time = pd.Timestamp("2020-01-01 09:30:00")
    index = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    # Generate random features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        index=index,
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Generate random labels
    y = pd.Series(
        np.random.randint(0, 3, size=n_samples),
        index=index,
        name="label",
    )

    # Generate label end times (bar_time + horizon * 5 minutes)
    label_end_times = pd.Series(
        [index[i] + pd.Timedelta(minutes=horizon * 5) for i in range(n_samples)],
        index=index,
        name="label_end_time",
    )

    return X, y, label_end_times


# =============================================================================
# ORIGINAL LOOP-BASED IMPLEMENTATION (FOR COMPARISON)
# =============================================================================

def purge_label_overlap_loop(
    X: pd.DataFrame,
    label_end_times: pd.Series,
    train_mask: np.ndarray,
    test_start: int,
    test_end: int,
) -> np.ndarray:
    """
    Original loop-based label overlap purging (for benchmarking).

    Args:
        X: Features DataFrame with DatetimeIndex
        label_end_times: When each label's outcome is known
        train_mask: Boolean mask of training samples
        test_start: Test set start index
        test_end: Test set end index

    Returns:
        Updated train_mask with overlapping samples removed
    """
    n_samples = len(X)
    test_start_time = X.index[test_start]
    test_end_time = X.index[test_end - 1]

    # Original loop-based implementation
    for i in range(n_samples):
        if train_mask[i]:
            label_end = label_end_times.iloc[i]
            if label_end >= test_start_time and X.index[i] <= test_end_time:
                train_mask[i] = False

    return train_mask


def purge_label_overlap_vectorized(
    X: pd.DataFrame,
    label_end_times: pd.Series,
    train_mask: np.ndarray,
    test_start: int,
    test_end: int,
) -> np.ndarray:
    """
    Vectorized label overlap purging (current implementation).

    Args:
        X: Features DataFrame with DatetimeIndex
        label_end_times: When each label's outcome is known
        train_mask: Boolean mask of training samples
        test_start: Test set start index
        test_end: Test set end index

    Returns:
        Updated train_mask with overlapping samples removed
    """
    test_start_time = X.index[test_start]
    test_end_time = X.index[test_end - 1]

    # Vectorized implementation
    overlapping = (label_end_times >= test_start_time) & (X.index <= test_end_time)
    train_mask[overlapping] = False

    return train_mask


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    n_samples: int
    n_splits: int
    loop_time_ms: float
    vectorized_time_ms: float
    speedup: float
    identical_results: bool


def benchmark_single_size(
    n_samples: int,
    n_splits: int,
    verify: bool = True,
) -> BenchmarkResult:
    """
    Benchmark PurgedKFold for a single dataset size.

    Args:
        n_samples: Number of samples to generate
        n_splits: Number of CV splits
        verify: Whether to verify identical results

    Returns:
        BenchmarkResult with timing and verification info
    """
    logger.info(f"Benchmarking with {n_samples:,} samples, {n_splits} splits...")

    # Generate synthetic data
    X, y, label_end_times = generate_synthetic_data(n_samples)

    # Create CV config
    config = PurgedKFoldConfig(
        n_splits=n_splits,
        purge_bars=60,
        embargo_bars=1440,
    )

    # Get test fold boundaries (we'll benchmark first fold)
    fold_size = n_samples // n_splits
    test_start = 0
    test_end = fold_size

    # Prepare train mask (simulate the state before label overlap check)
    train_mask_base = np.ones(n_samples, dtype=bool)
    train_mask_base[test_start:test_end] = False  # Remove test period
    purge_start = max(0, test_start - config.purge_bars)
    train_mask_base[purge_start:test_start] = False  # Remove purge
    embargo_end = min(n_samples, test_end + config.embargo_bars)
    train_mask_base[test_end:embargo_end] = False  # Remove embargo

    # Benchmark loop-based implementation
    train_mask_loop = train_mask_base.copy()
    start_time = time.perf_counter()
    train_mask_loop = purge_label_overlap_loop(
        X, label_end_times, train_mask_loop, test_start, test_end
    )
    loop_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

    # Benchmark vectorized implementation
    train_mask_vec = train_mask_base.copy()
    start_time = time.perf_counter()
    train_mask_vec = purge_label_overlap_vectorized(
        X, label_end_times, train_mask_vec, test_start, test_end
    )
    vec_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

    # Verify identical results
    identical = True
    if verify:
        identical = np.array_equal(train_mask_loop, train_mask_vec)
        if not identical:
            diff_count = np.sum(train_mask_loop != train_mask_vec)
            logger.error(
                f"Results differ! {diff_count} samples have different mask values"
            )

    # Calculate speedup
    speedup = loop_time / vec_time if vec_time > 0 else float('inf')

    logger.info(
        f"  Loop: {loop_time:.2f}ms | "
        f"Vectorized: {vec_time:.2f}ms | "
        f"Speedup: {speedup:.1f}x | "
        f"Identical: {identical}"
    )

    return BenchmarkResult(
        n_samples=n_samples,
        n_splits=n_splits,
        loop_time_ms=loop_time,
        vectorized_time_ms=vec_time,
        speedup=speedup,
        identical_results=identical,
    )


def run_benchmarks(
    sizes: List[int],
    n_splits: int = 5,
    verify: bool = True,
) -> List[BenchmarkResult]:
    """
    Run benchmarks across multiple dataset sizes.

    Args:
        sizes: List of dataset sizes to benchmark
        n_splits: Number of CV splits
        verify: Whether to verify identical results

    Returns:
        List of BenchmarkResult objects
    """
    results = []

    logger.info("=" * 80)
    logger.info("PurgedKFold Vectorization Benchmark")
    logger.info("=" * 80)

    for size in sizes:
        result = benchmark_single_size(size, n_splits, verify)
        results.append(result)

    return results


def print_summary(results: List[BenchmarkResult]) -> None:
    """
    Print summary table of benchmark results.

    Args:
        results: List of BenchmarkResult objects
    """
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)

    # Print header
    header = (
        f"{'Samples':<12} | "
        f"{'Loop (ms)':<12} | "
        f"{'Vectorized (ms)':<16} | "
        f"{'Speedup':<10} | "
        f"{'Identical':<10}"
    )
    logger.info(header)
    logger.info("-" * 80)

    # Print results
    for result in results:
        row = (
            f"{result.n_samples:<12,} | "
            f"{result.loop_time_ms:<12.2f} | "
            f"{result.vectorized_time_ms:<16.2f} | "
            f"{result.speedup:<10.1f}x | "
            f"{'✓' if result.identical_results else '✗':<10}"
        )
        logger.info(row)

    # Print summary statistics
    avg_speedup = np.mean([r.speedup for r in results])
    all_identical = all(r.identical_results for r in results)

    logger.info("-" * 80)
    logger.info(f"Average Speedup: {avg_speedup:.1f}x")
    logger.info(f"All Results Identical: {'✓ YES' if all_identical else '✗ NO'}")
    logger.info("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark PurgedKFold vectorized optimization"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="1000,10000,100000",
        help="Comma-separated list of dataset sizes (default: 1000,10000,100000)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits (default: 5)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify vectorized results match loop-based (default: True)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
        help="Skip verification of identical results",
    )

    args = parser.parse_args()

    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    # Run benchmarks
    results = run_benchmarks(sizes, args.n_splits, args.verify)

    # Print summary
    print_summary(results)

    # Exit with error if results don't match
    if args.verify and not all(r.identical_results for r in results):
        logger.error("VERIFICATION FAILED: Results differ between implementations!")
        exit(1)


if __name__ == "__main__":
    main()
