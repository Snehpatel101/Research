#!/usr/bin/env python3
"""
Benchmark script for boosting-only voting ensemble latency.

Measures inference latency for XGBoost + LightGBM + CatBoost ensemble
in both sequential and parallel modes.

Usage:
    python scripts/benchmark_ensemble.py
    python scripts/benchmark_ensemble.py --n-samples 1000 --n-features 150
    python scripts/benchmark_ensemble.py --warmup 5 --iterations 100

Expected output:
    Parallel mode should show ~40-60% latency reduction compared to sequential.
    Target: < 10ms for single-sample inference, < 50ms for 1000 samples.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry import ModelRegistry


def create_mock_data(
    n_samples: int = 100,
    n_features: int = 150,
    n_classes: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create mock training and inference data."""
    rng = np.random.default_rng(seed)

    X_train = rng.standard_normal((n_samples * 10, n_features)).astype(np.float32)
    # Labels in trading format: -1 (sell), 0 (hold), 1 (buy)
    y_train = rng.integers(-1, 2, n_samples * 10).astype(np.int32)

    X_val = rng.standard_normal((n_samples * 2, n_features)).astype(np.float32)
    y_val = rng.integers(-1, 2, n_samples * 2).astype(np.int32)

    return X_train, y_train, X_val, y_val


def train_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_names: List[str],
) -> Tuple[List, List[str]]:
    """Train boosting models for benchmarking."""
    print("\n[1/3] Training base models...")
    models = []
    trained_names = []

    # Fast config for benchmarking (fewer trees)
    fast_configs = {
        "xgboost": {"n_estimators": 50, "max_depth": 4, "use_gpu": False},
        "lightgbm": {"n_estimators": 50, "max_depth": 4, "use_gpu": False},
        "catboost": {"iterations": 50, "depth": 4, "use_gpu": False, "verbose": False},
    }

    for name in model_names:
        print(f"  Training {name}...", end=" ", flush=True)
        start = time.perf_counter()

        try:
            config = fast_configs.get(name, {})
            model = ModelRegistry.create(name, config=config)
            model.fit(X_train, y_train, X_val, y_val)
            models.append(model)
            trained_names.append(name)
            elapsed = time.perf_counter() - start
            print(f"done ({elapsed:.1f}s)")
        except ImportError as e:
            print(f"skipped (not installed)")

    if len(models) < 2:
        print("\n  [ERROR] Need at least 2 models for ensemble benchmark")
        print("  Install missing packages: pip install xgboost lightgbm catboost")
        sys.exit(1)

    return models, trained_names


def benchmark_single_model(
    model,
    X: np.ndarray,
    warmup: int = 3,
    iterations: int = 50,
) -> Dict[str, float]:
    """Benchmark a single model's inference latency."""
    # Warmup
    for _ in range(warmup):
        model.predict(X)

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.predict(X)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
    }


def benchmark_ensemble(
    models: List,
    X: np.ndarray,
    parallel: bool,
    warmup: int = 3,
    iterations: int = 50,
) -> Dict[str, float]:
    """Benchmark ensemble inference latency."""
    from src.models.ensemble.voting import VotingEnsemble

    config = {"voting": "soft", "parallel": parallel}
    ensemble = VotingEnsemble(config=config)
    ensemble.set_base_models(models)

    # Warmup
    for _ in range(warmup):
        ensemble.predict(X)

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = ensemble.predict(X)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "metadata_inference_ms": output.metadata.get("inference_ms", 0),
    }


def print_results(
    name: str,
    results: Dict[str, float],
    indent: int = 0,
) -> None:
    """Print benchmark results."""
    prefix = "  " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  Mean:   {results['mean_ms']:6.2f} ms")
    print(f"{prefix}  Std:    {results['std_ms']:6.2f} ms")
    print(f"{prefix}  P50:    {results['p50_ms']:6.2f} ms")
    print(f"{prefix}  P95:    {results['p95_ms']:6.2f} ms")
    print(f"{prefix}  P99:    {results['p99_ms']:6.2f} ms")
    print(f"{prefix}  Range:  {results['min_ms']:.2f} - {results['max_ms']:.2f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark boosting ensemble inference latency"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of samples for inference (default: 100)"
    )
    parser.add_argument(
        "--n-features", type=int, default=150,
        help="Number of features (default: 150)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of benchmark iterations (default: 50)"
    )
    parser.add_argument(
        "--single-sample", action="store_true",
        help="Also benchmark single-sample latency"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BOOSTING ENSEMBLE LATENCY BENCHMARK")
    print("=" * 60)
    print(f"Samples: {args.n_samples}, Features: {args.n_features}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")

    # Create data
    X_train, y_train, X_val, y_val = create_mock_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
    )
    X_test = np.random.randn(args.n_samples, args.n_features).astype(np.float32)

    # Train models
    model_names = ["xgboost", "lightgbm", "catboost"]
    models, trained_names = train_base_models(X_train, y_train, X_val, y_val, model_names)

    # Benchmark individual models
    print("\n[2/3] Benchmarking individual models...")
    individual_results = {}
    for i, (name, model) in enumerate(zip(trained_names, models)):
        results = benchmark_single_model(
            model, X_test,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        individual_results[name] = results
        print_results(name, results, indent=1)

    # Benchmark ensemble
    print("\n[3/3] Benchmarking ensemble...")

    # Sequential mode
    print("\n  Sequential mode (parallel=False):")
    seq_results = benchmark_ensemble(
        models, X_test,
        parallel=False,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print_results("Sequential", seq_results, indent=2)

    # Parallel mode
    print("\n  Parallel mode (parallel=True):")
    par_results = benchmark_ensemble(
        models, X_test,
        parallel=True,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print_results("Parallel", par_results, indent=2)

    # Single sample benchmark
    if args.single_sample:
        print("\n[Extra] Single-sample latency:")
        X_single = X_test[:1]

        seq_single = benchmark_ensemble(
            models, X_single,
            parallel=False,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        par_single = benchmark_ensemble(
            models, X_single,
            parallel=True,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(f"  Sequential: {seq_single['mean_ms']:.2f} ms (p95: {seq_single['p95_ms']:.2f} ms)")
        print(f"  Parallel:   {par_single['mean_ms']:.2f} ms (p95: {par_single['p95_ms']:.2f} ms)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    sum_individual = sum(r["mean_ms"] for r in individual_results.values())
    speedup = seq_results["mean_ms"] / par_results["mean_ms"]
    improvement = (1 - par_results["mean_ms"] / seq_results["mean_ms"]) * 100

    print(f"Individual models sum:  {sum_individual:.2f} ms")
    print(f"Ensemble sequential:    {seq_results['mean_ms']:.2f} ms")
    print(f"Ensemble parallel:      {par_results['mean_ms']:.2f} ms")
    print(f"Speedup:                {speedup:.2f}x")
    print(f"Latency reduction:      {improvement:.1f}%")

    # Pass/fail thresholds
    print("\n" + "-" * 60)
    if par_results["p95_ms"] < 100:
        print(f"[PASS] P95 latency ({par_results['p95_ms']:.2f}ms) < 100ms threshold")
    else:
        print(f"[WARN] P95 latency ({par_results['p95_ms']:.2f}ms) >= 100ms threshold")

    if speedup > 1.3:
        print(f"[PASS] Parallel speedup ({speedup:.2f}x) > 1.3x threshold")
    else:
        print(f"[WARN] Parallel speedup ({speedup:.2f}x) <= 1.3x threshold")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
