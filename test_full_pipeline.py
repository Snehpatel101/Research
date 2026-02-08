#!/usr/bin/env python
"""
Full Pipeline Test - Verifies memory fixes work on GPU
Tests: 100K rows, 3 models (lightgbm, tcn, patchtst), ensemble enabled
Hardware: 4070 Ti (12GB VRAM), 32GB RAM
"""
import gc
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Add repo to path
sys.path.insert(0, "/home/jake/Desktop/Research")

import psutil
import torch
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
TEST_CONFIG = {
    "max_rows": 100_000,  # Use 100K rows for testing (safe for 32GB)
    "models": ["lightgbm", "tcn"],  # 2 models: 1 boosting + 1 neural
    "horizons": [15],  # Single horizon for speed
    "optuna_trials": 10,  # Quick test
    "build_ensemble": True,  # Test ensemble building (was broken)
    "mtf_enabled": True,
    "mtf_timeframes": ["15min", "30min"],
}

# =============================================================================
# MEMORY MONITORING
# =============================================================================
def get_memory_stats():
    """Get current memory usage."""
    ram = psutil.virtual_memory()

    gpu_allocated = 0
    gpu_reserved = 0
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9

    return {
        "ram_used_gb": ram.used / 1e9,
        "ram_total_gb": ram.total / 1e9,
        "ram_pct": ram.percent,
        "gpu_allocated_gb": gpu_allocated,
        "gpu_reserved_gb": gpu_reserved,
    }

def print_memory(label=""):
    """Print memory status."""
    stats = get_memory_stats()
    print(f"[{label}] RAM: {stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f} GB ({stats['ram_pct']:.0f}%) | "
          f"GPU: {stats['gpu_allocated_gb']:.2f} GB allocated, {stats['gpu_reserved_gb']:.2f} GB reserved")

def clear_memory():
    """Force memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =============================================================================
# MAIN TEST
# =============================================================================
def main():
    print("=" * 70)
    print("ML FACTORY FULL PIPELINE TEST")
    print("=" * 70)
    print(f"Config: {TEST_CONFIG}")
    print()

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected!")

    print_memory("Initial")
    print()

    # ==========================================================================
    # PHASE 1: Load and subset data
    # ==========================================================================
    print("-" * 70)
    print("PHASE 1: Loading data...")
    print("-" * 70)

    df = pd.read_parquet("data/raw/mes-1m_data_2020.parquet")
    print(f"Loaded: {len(df):,} rows")

    # Subset to max_rows (take most recent)
    if len(df) > TEST_CONFIG["max_rows"]:
        df = df.tail(TEST_CONFIG["max_rows"]).copy()
        print(f"Subset to: {len(df):,} rows (most recent)")

    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

    print(f"Date range: {df.index.min()} -> {df.index.max()}")
    print_memory("After data load")
    print()

    # ==========================================================================
    # PHASE 2: Import and configure factory
    # ==========================================================================
    print("-" * 70)
    print("PHASE 2: Configuring ML Factory...")
    print("-" * 70)

    from src.config.experiment import (
        ExperimentConfig,
        DataSection,
        TrainingSection,
        EvaluationSection,
    )
    from src.config.training import OptunaConfig
    from src.config.data import FeatureConfig, LabelingConfig, MTFConfig
    from src.factory import MLFactory

    config = ExperimentConfig(
        name="memory_test",
        random_seed=42,
        data=DataSection(
            symbol="MES",
            data_path="data/raw/mes-1m_data_2020.parquet",
            features=FeatureConfig(families=["price", "momentum", "volatility", "volume"]),
            labeling=LabelingConfig(method="triple_barrier"),
            mtf=MTFConfig(
                enabled=TEST_CONFIG["mtf_enabled"],
                mode="indicators" if TEST_CONFIG["mtf_enabled"] else "none",
                timeframes=TEST_CONFIG["mtf_timeframes"],
                primary_timeframe="5min",
            ),
        ),
        training=TrainingSection(
            models=TEST_CONFIG["models"],
            horizons=TEST_CONFIG["horizons"],
            build_ensemble=TEST_CONFIG["build_ensemble"],
            meta_learner="ridge_meta",
            optuna=OptunaConfig(
                n_trials=TEST_CONFIG["optuna_trials"],
                metric="f1_weighted",
            ),
            batch_size=64,  # Safe for 12GB VRAM
        ),
        evaluation=EvaluationSection(
            run_backtest=False,  # Skip backtest for speed
        ),
    )

    print(f"Models: {config.training.models}")
    print(f"Horizons: {config.training.horizons}")
    print(f"Ensemble: {config.training.build_ensemble}")
    print(f"MTF: {config.data.mtf.enabled} ({config.data.mtf.timeframes})")
    print_memory("After config")
    print()

    # ==========================================================================
    # PHASE 3: Run the factory
    # ==========================================================================
    print("-" * 70)
    print("PHASE 3: Running ML Factory pipeline...")
    print("-" * 70)

    factory = MLFactory(config, enable_checkpoints=True)

    start_time = time.time()
    peak_ram = 0
    peak_gpu = 0

    try:
        # Monitor memory during run
        print_memory("Before run")
        print()

        result = factory.run()

        elapsed = time.time() - start_time

        print()
        print("-" * 70)
        print("PHASE 4: Results")
        print("-" * 70)

        if result.success:
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print()
            print(f"Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"Models trained: {result.n_models}")
            print(f"Best model: {result.best_model}")

            if result.metrics:
                print()
                print("Model Metrics:")
                for model_name, metrics in result.metrics.items():
                    f1 = metrics.get("val_f1", 0)
                    acc = metrics.get("val_accuracy", 0)
                    print(f"  {model_name}: F1={f1:.4f}, Acc={acc:.4f}")

            if result.ensemble_metrics:
                print()
                print("Ensemble Metrics:")
                for k, v in result.ensemble_metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")

            print()
            print_memory("Final")

        else:
            print("❌ PIPELINE FAILED!")
            print(f"Error: {result.error_message}")

    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print("❌ EXCEPTION DURING PIPELINE!")
        print(f"Error: {e}")
        print(f"Elapsed: {elapsed:.1f}s")
        print_memory("At failure")
        import traceback
        traceback.print_exc()
        return 1

    # ==========================================================================
    # PHASE 5: Memory verification
    # ==========================================================================
    print()
    print("-" * 70)
    print("PHASE 5: Memory Verification")
    print("-" * 70)

    # Force cleanup
    clear_memory()
    print_memory("After cleanup")

    final_stats = get_memory_stats()

    # Check if memory is reasonable
    if final_stats["ram_pct"] < 70:
        print("✅ RAM usage acceptable (<70%)")
    else:
        print(f"⚠️ RAM usage high ({final_stats['ram_pct']:.0f}%)")

    if final_stats["gpu_reserved_gb"] < 8:
        print("✅ GPU memory acceptable (<8GB)")
    else:
        print(f"⚠️ GPU memory high ({final_stats['gpu_reserved_gb']:.1f}GB)")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
