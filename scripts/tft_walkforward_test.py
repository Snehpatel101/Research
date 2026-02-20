"""
TFT Walk-Forward Smoke Test.

Runs TFT in walk-forward mode with reduced memory settings to fit within 16GB RAM.
Previous attempt OOM'd with default batch_size=256. Using batch_size=32 + max_epochs=5.
"""
from __future__ import annotations

import gc
import os
import sys
import time
import traceback

# Disable torch.compile (no MSVC on system)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config.experiment import (
    BundlingSection,
    DataSection,
    EvaluationSection,
    ExperimentConfig,
    TrainingSection,
)
from src.config.cv import WalkForwardConfig
from src.config.training import OptunaConfig
from src.factory import MLFactory


def main() -> None:
    print("=" * 70)
    print("TFT Walk-Forward Smoke Test")
    print("=" * 70)

    # Force garbage collection before starting
    gc.collect()

    config = ExperimentConfig(
        name="tft_walk_forward_smoke",
        description="TFT walk-forward validation - memory-reduced settings",
        output_dir="experiments/tft_wf_smoke",
        verbose=2,
        data=DataSection(
            symbol="MES",
            data_path="data/raw/MES_1m_1week.parquet",
        ),
        training=TrainingSection(
            models=["tft"],
            horizons=[5],
            training_mode="walk_forward",
            n_splits=3,
            purge_bars=5,
            embargo_bars=2,
            batch_size=32,           # Reduced from 256 to save RAM
            max_epochs=5,            # Reduced from 100 — smoke test only
            early_stopping_patience=3,
            walk_forward=WalkForwardConfig(
                n_windows=3,
                window_type="expanding",
                min_train_pct=0.4,
                test_pct=0.1,
                gap_bars=5,
                embargo_bars=2,
            ),
            optuna=OptunaConfig(n_trials=1),
            build_ensemble=False,    # Single model, no ensemble needed
        ),
        evaluation=EvaluationSection(
            run_backtest=True,
            generate_report=True,
        ),
        bundling=BundlingSection(
            create_bundle=True,
            deploy_artifact=True,
        ),
    )

    start = time.time()
    try:
        factory = MLFactory(config)
        result = factory.run()

        elapsed = time.time() - start
        print("\n" + "=" * 70)
        print("TFT Walk-Forward COMPLETE")
        print("=" * 70)
        print(f"Duration: {elapsed:.0f}s")
        print(f"Success: {result.success}")
        print(f"N models: {result.n_models}")

        if result.metrics:
            for model_name, metrics in result.metrics.items():
                if isinstance(metrics, dict):
                    f1 = metrics.get("macro_f1", metrics.get("f1", "N/A"))
                    acc = metrics.get("accuracy", "N/A")
                    print(f"  {model_name}: F1={f1}, Accuracy={acc}")
                else:
                    print(f"  {model_name}: {metrics}")

        if result.ensemble_metrics:
            print(f"Ensemble: {result.ensemble_metrics}")

        print(f"\nResult summary:\n{result.summary() if hasattr(result, 'summary') else 'N/A'}")

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n{'=' * 70}")
        print(f"TFT Walk-Forward FAILED after {elapsed:.0f}s")
        print(f"{'=' * 70}")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
