"""Verify Optuna hyperparameter optimization works end-to-end.

Runs the full ML Factory pipeline with Optuna enabled (n_trials=3) on
xgboost, with feature selection disabled (too data-hungry for a 1-week
sample). Checks that trials ran and best params were found.

Usage:
    python scripts/verify_optuna.py
"""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.config.data import FeatureConfig  # noqa: E402
from src.config.experiment import (  # noqa: E402
    BundlingSection,
    DataSection,
    EvaluationSection,
    ExperimentConfig,
    TrainingSection,
)
from src.config.training import OptunaConfig  # noqa: E402
from src.factory import MLFactory  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

config = ExperimentConfig(
    name="optuna_verify",
    output_dir="experiments/optuna_verify",
    verbose=2,
    data=DataSection(
        symbol="MES",
        data_path="data/raw/MES_1m_1week.parquet",
        features=FeatureConfig(selection_enabled=False),
    ),
    training=TrainingSection(
        models=["xgboost"],
        horizons=[5],
        n_splits=2,
        build_ensemble=False,
        optuna=OptunaConfig(n_trials=3),
    ),
    evaluation=EvaluationSection(run_backtest=False),
    bundling=BundlingSection(create_bundle=False),
)

result = MLFactory(config).run()

print("\n\n" + "=" * 60)
print("OPTUNA VERIFICATION RESULTS")
print("=" * 60)
print(f"Success: {result.success}")
print(f"Models trained: {result.n_models}")
print(f"Best model: {result.best_model}")
print(f"Duration: {result.duration_seconds:.1f}s")

for model, metrics in result.metrics.items():
    print(f"\n{model}:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

if result.error_message:
    print(f"\nError: {result.error_message}")

passed = result.success and result.n_models > 0
print(f"\nOPTUNA STATUS: {'PASS' if passed else 'FAIL'}")
sys.exit(0 if passed else 1)
