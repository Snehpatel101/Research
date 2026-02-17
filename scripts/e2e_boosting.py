import sys, os
os.chdir("/Users/sneh/research")
sys.path.insert(0, "/Users/sneh/research")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

from src.config.experiment import ExperimentConfig, DataSection, TrainingSection, EvaluationSection, BundlingSection
from src.config.training import OptunaConfig
from src.factory import MLFactory

config = ExperimentConfig(
    name="mes_boosting_smoke",
    output_dir="experiments/smoke_boosting",
    verbose=2,
    data=DataSection(symbol="MES", data_path="data/raw/MES_1m_1week.parquet"),
    training=TrainingSection(
        models=["xgboost", "lightgbm", "catboost"],
        horizons=[5, 10],
        n_splits=2, purge_bars=10, embargo_bars=60,
        build_ensemble=True,
        optuna=OptunaConfig(n_trials=0),
    ),
    evaluation=EvaluationSection(run_backtest=False),
    bundling=BundlingSection(create_bundle=True, deploy_artifact=True),
)

factory = MLFactory(config)
result = factory.run()
print("\n\n" + result.summary())
