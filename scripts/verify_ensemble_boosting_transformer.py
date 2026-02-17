"""Verify ensemble: boosting + transformer (2D + 4D data ranks)."""
import sys, os
os.chdir("/Users/sneh/research")
sys.path.insert(0, "/Users/sneh/research")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

from src.config.experiment import ExperimentConfig, DataSection, TrainingSection, EvaluationSection, BundlingSection
from src.config.training import OptunaConfig
from src.factory import MLFactory

config = ExperimentConfig(
    name="ensemble_boost_transformer",
    output_dir="experiments/ensemble_boost_transformer",
    verbose=2,
    data=DataSection(symbol="MES", data_path="data/raw/MES_1m_1week.parquet"),
    training=TrainingSection(
        models=["xgboost", "patchtst"],
        horizons=[5],
        n_splits=2,
        purge_bars=10,
        embargo_bars=60,
        build_ensemble=True,
        max_epochs=10,
        early_stopping_patience=5,
        batch_size=256,
        optuna=OptunaConfig(n_trials=0),
    ),
    evaluation=EvaluationSection(run_backtest=True),
    bundling=BundlingSection(create_bundle=True, deploy_artifact=True),
)

factory = MLFactory(config)
result = factory.run()

print("\n\nENSEMBLE VERIFICATION: BOOSTING + TRANSFORMER")
print("=" * 50)
print(f"Success: {result.success}")
print(f"Models trained: {result.n_models}")
print(f"Best model: {result.best_model}")
if result.metrics:
    for model, metrics in result.metrics.items():
        f1 = metrics.get("macro_f1", metrics.get("val_f1", 0))
        acc = metrics.get("val_accuracy", 0)
        print(f"  {model}: F1={f1:.4f}, Acc={acc:.4f}")

if result.backtest_metrics:
    print(f"\nBacktest:")
    print(f"  Trades: {result.backtest_metrics.get('total_trades', 0)}")
    print(f"  Sharpe: {result.backtest_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  PnL: ${result.backtest_metrics.get('total_pnl', 0):.2f}")

print(f"\nSTATUS: {'PASS' if result.success else 'FAIL'}")
if not result.success:
    print(f"Error: {result.error_message}")
