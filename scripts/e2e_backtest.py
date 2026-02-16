"""E2E test with backtest enabled — boosting only for speed."""
import sys, os
os.chdir("/Users/sneh/research")
sys.path.insert(0, "/Users/sneh/research")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

from src.config.experiment import ExperimentConfig, DataSection, TrainingSection, EvaluationSection, BundlingSection
from src.config.training import OptunaConfig
from src.factory import MLFactory

config = ExperimentConfig(
    name="mes_backtest_smoke",
    output_dir="experiments/smoke_backtest",
    verbose=2,
    data=DataSection(symbol="MES", data_path="data/raw/MES_1m_1week.parquet"),
    training=TrainingSection(
        models=["xgboost", "lightgbm", "catboost"],
        horizons=[5],
        n_splits=2, purge_bars=10, embargo_bars=60,
        build_ensemble=True,
        optuna=OptunaConfig(n_trials=0),
    ),
    evaluation=EvaluationSection(run_backtest=True, generate_report=True),
    bundling=BundlingSection(create_bundle=True, deploy_artifact=True),
)

factory = MLFactory(config)
result = factory.run()
print("\n\n" + result.summary())

# Show backtest metrics specifically
print("\n" + "="*60)
print("BACKTEST METRICS")
print("="*60)
if result.backtest_metrics:
    for k, v in sorted(result.backtest_metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
else:
    print("  No backtest metrics returned")
