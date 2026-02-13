#!/usr/bin/env python3
"""
Test all 12 production models with 1 month of data.

Usage:
    python scripts/test_all_models.py                    # Run all models
    python scripts/test_all_models.py --group boosting   # Run specific group
    python scripts/test_all_models.py --group neural_rnn
    python scripts/test_all_models.py --group neural_cnn
    python scripts/test_all_models.py --group transformer
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Suppress verbose loggers
for name in [
    "src.data", "src.core", "src.models", "src.config",
    "src.validation", "src.optimization", "urllib3", "numba",
]:
    logging.getLogger(name).setLevel(logging.ERROR)


# Model groups matching CLAUDE.md's 12 production models
MODEL_GROUPS = {
    "boosting": ["xgboost", "lightgbm", "catboost"],
    "neural_rnn": ["lstm", "gru"],
    "neural_cnn": ["tcn", "inceptiontime", "resnet1d"],
    "transformer": ["patchtst", "itransformer", "tft", "nbeats"],
}

ALL_MODELS = []
for group in MODEL_GROUPS.values():
    ALL_MODELS.extend(group)


def run_data_pipeline():
    """Run data pipeline on MES 1-month data to produce train/val/test splits."""
    from src.data.pipeline.data_config import DataConfig
    from src.data.pipeline.runner import PipelineRunner

    print("=" * 70)
    print("STEP 1: Running Data Pipeline (MES, 1 month, horizon=20)")
    print("=" * 70)

    config = DataConfig(
        symbols=["MES"],
        label_horizons=[20],
        target_timeframe="5min",
        project_root=PROJECT_ROOT,
        description="Test all models - 1 month data",
    )

    runner = PipelineRunner(config)
    success = runner.run()

    if not success:
        print("ERROR: Data pipeline failed!")
        sys.exit(1)

    # Find the scaled splits directory
    scaled_dir = config.run_splits_dir / "scaled"
    if not scaled_dir.exists():
        # Try to find it in the run directory
        run_dir = PROJECT_ROOT / "data" / "runs" / config.run_id
        scaled_dir = run_dir / "artifacts" / "splits" / "scaled"

    if not scaled_dir.exists():
        # Search for it
        import glob
        candidates = glob.glob(
            str(PROJECT_ROOT / "data" / "runs" / config.run_id / "**" / "scaled"),
            recursive=True,
        )
        if candidates:
            scaled_dir = Path(candidates[0])
        else:
            print(f"ERROR: Cannot find scaled splits directory for run {config.run_id}")
            print(f"Run dir: {PROJECT_ROOT / 'data' / 'runs' / config.run_id}")
            sys.exit(1)

    print(f"\nData pipeline complete. Scaled splits: {scaled_dir}")
    return scaled_dir, config.run_id


def test_model(model_name, scaled_dir, horizon=20, max_epochs=3):
    """Test a single model. Returns (success, duration, error_msg)."""
    start = time.time()
    try:
        import src.models  # noqa: F401 - trigger registration
        from src.core.container import TimeSeriesDataContainer
        from src.models.config import TrainerConfig
        from src.models.training import Trainer

        # Load data
        container = TimeSeriesDataContainer.from_parquet_dir(
            path=scaled_dir,
            horizon=horizon,
        )

        # Create minimal config for quick test
        config = TrainerConfig(
            model_name=model_name,
            horizon=horizon,
            model_config={
                "max_epochs": max_epochs,
                "early_stopping_patience": 2,
                "batch_size": 128,
                "device": "cpu",
                "mixed_precision": False,
            },
            output_dir=PROJECT_ROOT / "experiments" / "model_tests",
        )

        # Train
        trainer = Trainer(config)
        results = trainer.run(container, skip_save=True)

        duration = time.time() - start

        # Extract key metrics
        metrics = results.get("evaluation_metrics", results.get("metrics", {}))
        val_acc = metrics.get("val_accuracy", metrics.get("accuracy", "N/A"))
        val_f1 = metrics.get("val_f1", metrics.get("f1", "N/A"))

        return True, duration, None, {"accuracy": val_acc, "f1": val_f1}

    except Exception as e:
        duration = time.time() - start
        error_msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        return False, duration, error_msg, tb


def test_model_group(group_name, models, scaled_dir):
    """Test a group of models and print results."""
    print(f"\n{'=' * 70}")
    print(f"Testing {group_name.upper()} models: {models}")
    print(f"{'=' * 70}\n")

    results = {}
    for model_name in models:
        print(f"  [{model_name}] Training...", end=" ", flush=True)
        success, duration, error, extra = test_model(model_name, scaled_dir)

        if success:
            metrics = extra
            print(f"OK ({duration:.1f}s) - acc={metrics.get('accuracy', '?')}, f1={metrics.get('f1', '?')}")
            results[model_name] = {
                "status": "PASS",
                "duration": round(duration, 1),
                "metrics": metrics,
            }
        else:
            print(f"FAIL ({duration:.1f}s)")
            print(f"    Error: {error}")
            results[model_name] = {
                "status": "FAIL",
                "duration": round(duration, 1),
                "error": error,
                "traceback": extra,
            }

    return results


def print_summary(all_results):
    """Print final summary table."""
    print(f"\n{'=' * 70}")
    print("SUMMARY: All 12 Models Test Results")
    print(f"{'=' * 70}")
    print(f"{'Model':<15} {'Status':<8} {'Time':<10} {'Accuracy':<12} {'F1':<12}")
    print("-" * 57)

    passed = 0
    failed = 0
    for model_name in ALL_MODELS:
        r = all_results.get(model_name, {"status": "SKIP", "duration": 0})
        status = r["status"]
        duration = f"{r['duration']:.1f}s" if r.get("duration") else "N/A"

        if status == "PASS":
            passed += 1
            metrics = r.get("metrics", {})
            acc = metrics.get("accuracy", "N/A")
            f1 = metrics.get("f1", "N/A")
            if isinstance(acc, float):
                acc = f"{acc:.4f}"
            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            print(f"{model_name:<15} PASS     {duration:<10} {acc:<12} {f1:<12}")
        else:
            failed += 1
            err_short = r.get("error", "unknown")[:40]
            print(f"{model_name:<15} FAIL     {duration:<10} {err_short}")

    print("-" * 57)
    print(f"Total: {passed} passed, {failed} failed out of {passed + failed}")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test ML Factory models with 1 month data")
    parser.add_argument(
        "--group", choices=list(MODEL_GROUPS.keys()) + ["all"],
        default="all", help="Which model group to test",
    )
    parser.add_argument(
        "--scaled-dir", type=str, default=None,
        help="Path to pre-existing scaled splits (skip data pipeline)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=3,
        help="Max epochs per model (default: 3 for quick test)",
    )
    args = parser.parse_args()

    # Step 1: Get or create data
    if args.scaled_dir:
        scaled_dir = Path(args.scaled_dir)
        if not scaled_dir.exists():
            print(f"ERROR: Scaled dir not found: {scaled_dir}")
            sys.exit(1)
        print(f"Using existing scaled data: {scaled_dir}")
    else:
        scaled_dir, run_id = run_data_pipeline()

    # Step 2: Test models
    if args.group == "all":
        groups_to_test = MODEL_GROUPS
    else:
        groups_to_test = {args.group: MODEL_GROUPS[args.group]}

    all_results = {}
    for group_name, models in groups_to_test.items():
        group_results = test_model_group(group_name, models, scaled_dir)
        all_results.update(group_results)

    # Step 3: Summary
    all_pass = print_summary(all_results)

    # Save results to JSON
    results_path = PROJECT_ROOT / "experiments" / "model_test_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        # Filter out tracebacks for JSON
        clean_results = {}
        for k, v in all_results.items():
            clean_results[k] = {kk: vv for kk, vv in v.items() if kk != "traceback"}
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
