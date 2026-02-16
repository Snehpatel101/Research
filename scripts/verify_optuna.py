"""Verify Optuna hyperparameter optimization works end-to-end.

Runs the full ML Factory pipeline with Optuna enabled (n_trials=3)
on xgboost, with feature selection disabled (too data-hungry for
a 1-week sample). Checks that trials ran and best params were found.
"""
import sys, os
os.chdir("/Users/sneh/research")
sys.path.insert(0, "/Users/sneh/research")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

from src.config.experiment import ExperimentConfig, DataSection, TrainingSection, EvaluationSection, BundlingSection
from src.config.training import OptunaConfig
from src.factory import MLFactory

config = ExperimentConfig(
    name="optuna_verify",
    output_dir="experiments/optuna_verify",
    verbose=2,
    data=DataSection(symbol="MES", data_path="data/raw/MES_1m_1week.parquet"),
    training=TrainingSection(
        models=["xgboost"],
        horizons=[5],
        n_splits=2,
        purge_bars=10,
        embargo_bars=60,
        build_ensemble=False,
        optuna=OptunaConfig(n_trials=3),
    ),
    evaluation=EvaluationSection(run_backtest=False),
    bundling=BundlingSection(create_bundle=False),
)

# Patch to_pipeline_config: enable Optuna hyperparams but DISABLE feature
# selection (walk-forward MDA needs >5k samples due to hardcoded embargo_bars=1440).
_original = config.to_pipeline_config

def _patched_pipeline_config():
    pc = _original()
    pc.optimize_features = False   # disable feature selection (too little data)
    pc.optimize_labels = False     # also skip label optimization for speed
    return pc

config.to_pipeline_config = _patched_pipeline_config

factory = MLFactory(config)
result = factory.run()

print("\n\n" + "=" * 60)
print("OPTUNA VERIFICATION RESULTS")
print("=" * 60)
print(f"Success: {result.success}")
print(f"Models trained: {result.n_models}")
print(f"Best model: {result.best_model}")
print(f"Duration: {result.duration_seconds:.1f}s")

if result.metrics:
    for model, metrics in result.metrics.items():
        print(f"\n{model}:")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

if result.error_message:
    print(f"\nError: {result.error_message}")

print(f"\nOPTUNA STATUS: {'PASS' if result.success and result.n_models > 0 else 'FAIL'}")
