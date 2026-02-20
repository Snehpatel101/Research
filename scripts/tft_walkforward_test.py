"""
TFT Walk-Forward Smoke Test (Ultra-Fast Mode).

Reduced features (price+momentum only) + short sequences + 2 epochs.
Target: complete in ~15-20 min on CPU.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import time
import traceback

os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from src.config.experiment import (
    BundlingSection,
    DataSection,
    EvaluationSection,
    ExperimentConfig,
    TrainingSection,
)
from src.config.cv import WalkForwardConfig
from src.config.data import FeatureConfig, SequenceConfig
from src.config.training import OptunaConfig
from src.factory import MLFactory


def main() -> None:
    print("=" * 70, flush=True)
    print("TFT Walk-Forward Smoke Test (ULTRA-FAST)", flush=True)
    print("  Features: price+momentum only (~40 features)", flush=True)
    print("  seq_len=20, batch_size=64, max_epochs=2, n_windows=2", flush=True)
    print("=" * 70, flush=True)

    gc.collect()

    config = ExperimentConfig(
        name="tft_walk_forward_ultrafast",
        description="TFT walk-forward - minimal features for fast E2E validation",
        output_dir="experiments/tft_wf_ultrafast",
        verbose=2,
        data=DataSection(
            symbol="MES",
            data_path="data/raw/MES_1m_1week.parquet",
            features=FeatureConfig(
                mode="minimal",
                families=["price", "momentum"],
                sma_periods=[10, 20],
                ema_periods=[9, 21],
                atr_periods=[14],
                selection_n_features=30,
            ),
            sequence=SequenceConfig(
                seq_len=20,  # Reduced from 60
            ),
        ),
        training=TrainingSection(
            models=["tft"],
            horizons=[5],
            training_mode="walk_forward",
            n_splits=2,
            purge_bars=5,
            embargo_bars=2,
            batch_size=64,
            max_epochs=2,
            early_stopping_patience=2,
            walk_forward=WalkForwardConfig(
                n_windows=2,
                window_type="expanding",
                min_train_pct=0.5,
                test_pct=0.15,
                gap_bars=5,
                embargo_bars=2,
            ),
            optuna=OptunaConfig(n_trials=0),
            build_ensemble=False,
        ),
        evaluation=EvaluationSection(
            run_backtest=False,
            generate_report=False,
        ),
        bundling=BundlingSection(
            create_bundle=False,
            deploy_artifact=False,
        ),
    )

    start = time.time()
    try:
        factory = MLFactory(config)
        result = factory.run()

        elapsed = time.time() - start
        print(f"\n{'=' * 70}", flush=True)
        print("TFT Walk-Forward COMPLETE", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"Duration: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
        print(f"Success: {result.success}", flush=True)
        print(f"N models: {result.n_models}", flush=True)

        if result.metrics:
            for model_name, metrics in result.metrics.items():
                if isinstance(metrics, dict):
                    f1 = metrics.get("macro_f1", metrics.get("f1", "N/A"))
                    acc = metrics.get("accuracy", "N/A")
                    print(f"  {model_name}: F1={f1}, Accuracy={acc}", flush=True)
                else:
                    print(f"  {model_name}: {metrics}", flush=True)

        if hasattr(result, 'summary'):
            print(f"\nResult summary:\n{result.summary()}", flush=True)

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n{'=' * 70}", flush=True)
        print(f"TFT Walk-Forward FAILED after {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"Error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
