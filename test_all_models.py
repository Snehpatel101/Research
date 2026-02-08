#!/usr/bin/env python
"""
All-Models Pipeline Test - Tests all 12 models on ~1 month of data.
Logs results and errors to test_logs/ directory.

Hardware: RTX 4070 Ti (12GB VRAM), 32GB RAM
Data: ~30K rows (1 month of 1-min MES data)
Models: xgboost, lightgbm, catboost, lstm, gru, tcn, inceptiontime,
        resnet1d, patchtst, itransformer, tft, nbeats
"""
import gc
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/jake/Desktop/Research")

import psutil
import torch

# =============================================================================
# CONSTANTS
# =============================================================================
ALL_MODELS = [
    # Boosting (2D tabular)
    "xgboost",
    "lightgbm",
    "catboost",
    # RNN (3D sequence)
    "lstm",
    "gru",
    # CNN (3D sequence)
    "tcn",
    "inceptiontime",
    "resnet1d",
    # Transformer (4D multi-timeframe)
    "patchtst",
    "itransformer",
    "tft",
    # MLP (3D sequence)
    "nbeats",
]

DATA_PATH = "data/raw/mes-1m_data_2020.parquet"
LOG_DIR = Path("test_logs")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# HELPERS
# =============================================================================
def get_memory_stats():
    ram = psutil.virtual_memory()
    gpu_allocated = 0
    gpu_reserved = 0
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
    return {
        "ram_used_gb": round(ram.used / 1e9, 2),
        "ram_total_gb": round(ram.total / 1e9, 2),
        "ram_pct": round(ram.percent, 1),
        "gpu_allocated_gb": round(gpu_allocated, 3),
        "gpu_reserved_gb": round(gpu_reserved, 3),
    }


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log(msg, file=None):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if file:
        file.write(line + "\n")
        file.flush()


# =============================================================================
# SINGLE MODEL TEST
# =============================================================================
def test_single_model(model_name, data_path, log_file):
    """Run pipeline for a single model. Returns dict with results."""
    from src.config.data import FeatureConfig, LabelingConfig, MTFConfig
    from src.config.experiment import (
        DataSection,
        EvaluationSection,
        ExperimentConfig,
        TrainingSection,
    )
    from src.config.training import OptunaConfig
    from src.factory import MLFactory

    result_info = {
        "model": model_name,
        "status": "UNKNOWN",
        "duration_s": 0,
        "error": None,
        "metrics": None,
        "memory_before": get_memory_stats(),
    }

    log(f"--- Testing {model_name} ---", log_file)
    log(f"  Memory: {result_info['memory_before']}", log_file)

    config = ExperimentConfig(
        name=f"test_{model_name}",
        random_seed=42,
        data=DataSection(
            symbol="MES",
            data_path=data_path,
            features=FeatureConfig(
                families=["price", "momentum", "volatility", "volume"]
            ),
            labeling=LabelingConfig(method="triple_barrier"),
            mtf=MTFConfig(
                enabled=True,
                mode="indicators",
                timeframes=["15min", "30min"],
                primary_timeframe="5min",
            ),
        ),
        training=TrainingSection(
            models=[model_name],
            horizons=[15],
            optuna=OptunaConfig(n_trials=5, metric="f1_weighted"),
            batch_size=64,
            max_epochs=30,
            early_stopping_patience=10,
            build_ensemble=False,
        ),
        evaluation=EvaluationSection(run_backtest=False),
    )

    start = time.time()
    try:
        factory = MLFactory(config, enable_checkpoints=False)
        result = factory.run()
        elapsed = time.time() - start
        result_info["duration_s"] = round(elapsed, 1)

        if result.success:
            result_info["status"] = "PASS"
            result_info["metrics"] = {}
            if result.metrics:
                for mname, mvals in result.metrics.items():
                    result_info["metrics"][mname] = {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in mvals.items()
                        if k.startswith("val_")
                    }
            log(f"  PASS in {elapsed:.1f}s", log_file)
            if result_info["metrics"]:
                for mname, mvals in result_info["metrics"].items():
                    log(f"    {mname}: {mvals}", log_file)
        else:
            result_info["status"] = "FAIL"
            result_info["error"] = str(result.error_message)
            log(f"  FAIL in {elapsed:.1f}s: {result.error_message}", log_file)

    except Exception as e:
        elapsed = time.time() - start
        result_info["duration_s"] = round(elapsed, 1)
        result_info["status"] = "ERROR"
        result_info["error"] = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        log(f"  ERROR in {elapsed:.1f}s: {e}", log_file)
        log(f"  Traceback:\n{tb}", log_file)

    result_info["memory_after"] = get_memory_stats()
    log(f"  Memory after: {result_info['memory_after']}", log_file)

    # Cleanup between models
    clear_gpu()

    return result_info


# =============================================================================
# MAIN
# =============================================================================
def main():
    LOG_DIR.mkdir(exist_ok=True)

    summary_path = LOG_DIR / f"summary_{TIMESTAMP}.txt"
    detail_path = LOG_DIR / f"detail_{TIMESTAMP}.log"
    json_path = LOG_DIR / f"results_{TIMESTAMP}.json"

    with open(detail_path, "w") as detail_log:
        log("=" * 70, detail_log)
        log("ML FACTORY - ALL MODELS PIPELINE TEST", detail_log)
        log("=" * 70, detail_log)

        # System info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            log(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)", detail_log)
        else:
            log("WARNING: No GPU detected!", detail_log)

        mem = get_memory_stats()
        log(f"RAM: {mem['ram_used_gb']}/{mem['ram_total_gb']} GB", detail_log)
        log(f"Models to test: {ALL_MODELS}", detail_log)
        log(f"Data: {DATA_PATH} (1 month subset)", detail_log)
        log("", detail_log)

        # Subset data to ~1 month
        import pandas as pd

        df = pd.read_parquet(DATA_PATH)
        total_rows = len(df)
        df.columns = [c.lower().strip() for c in df.columns]
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
        # Take last ~1 month (22 trading days * 1440 min/day)
        one_month = df.tail(1440 * 22).copy()
        log(
            f"Data: {total_rows:,} total rows -> {len(one_month):,} rows "
            f"({one_month.index.min()} to {one_month.index.max()})",
            detail_log,
        )

        # Save subset for the pipeline to load
        subset_path = Path("data/raw/_test_subset_1mo.parquet")
        one_month.to_parquet(subset_path)
        log(f"Saved 1-month subset to {subset_path}", detail_log)
        log("", detail_log)

        # Run all models
        all_results = []
        total_start = time.time()
        test_data_path = str(subset_path)

        for i, model_name in enumerate(ALL_MODELS, 1):
            log(f"\n[{i}/{len(ALL_MODELS)}] {model_name.upper()}", detail_log)
            result = test_single_model(model_name, test_data_path, detail_log)
            all_results.append(result)

            # Report failure immediately
            if result["status"] != "PASS":
                print(f"\n  *** FAILURE: {model_name} - {result['status']}: "
                      f"{result['error']}\n")

        total_elapsed = time.time() - total_start

        # =====================================================================
        # SUMMARY
        # =====================================================================
        log("\n" + "=" * 70, detail_log)
        log("SUMMARY", detail_log)
        log("=" * 70, detail_log)

        passed = [r for r in all_results if r["status"] == "PASS"]
        failed = [r for r in all_results if r["status"] == "FAIL"]
        errored = [r for r in all_results if r["status"] == "ERROR"]

        log(
            f"Total: {len(all_results)} | "
            f"PASS: {len(passed)} | "
            f"FAIL: {len(failed)} | "
            f"ERROR: {len(errored)} | "
            f"Time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)",
            detail_log,
        )

        log("", detail_log)
        for r in all_results:
            status_marker = {
                "PASS": "OK ",
                "FAIL": "FAIL",
                "ERROR": "ERR",
                "UNKNOWN": "???",
            }[r["status"]]
            err_msg = f" - {r['error'][:80]}" if r["error"] else ""
            log(
                f"  [{status_marker}] {r['model']:15s} ({r['duration_s']:6.1f}s){err_msg}",
                detail_log,
            )

        if failed or errored:
            log("\n--- FAILURES ---", detail_log)
            for r in failed + errored:
                log(f"  {r['model']}: {r['error']}", detail_log)

        # Save JSON results
        with open(json_path, "w") as f:
            json.dump(
                {
                    "timestamp": TIMESTAMP,
                    "total_duration_s": round(total_elapsed, 1),
                    "pass_count": len(passed),
                    "fail_count": len(failed),
                    "error_count": len(errored),
                    "results": all_results,
                },
                f,
                indent=2,
                default=str,
            )

        log(f"\nLogs saved to:", detail_log)
        log(f"  Detail: {detail_path}", detail_log)
        log(f"  JSON:   {json_path}", detail_log)

    # Write concise summary
    with open(summary_path, "w") as f:
        f.write(f"ML Factory All-Models Test - {TIMESTAMP}\n")
        f.write(f"{'='*50}\n")
        f.write(
            f"PASS: {len(passed)}/{len(all_results)} | "
            f"FAIL: {len(failed)} | ERROR: {len(errored)} | "
            f"Time: {total_elapsed:.0f}s\n\n"
        )
        for r in all_results:
            marker = "PASS" if r["status"] == "PASS" else r["status"]
            f.write(f"[{marker:5s}] {r['model']:15s} {r['duration_s']:7.1f}s\n")
        if failed or errored:
            f.write(f"\nFailed models:\n")
            for r in failed + errored:
                f.write(f"  {r['model']}: {r['error']}\n")

    print(f"\nSummary: {summary_path}")

    # Cleanup temp file
    if subset_path.exists():
        subset_path.unlink()

    return 1 if (failed or errored) else 0


if __name__ == "__main__":
    exit(main())
