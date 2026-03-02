#!/usr/bin/env python3
"""
Dry-run test: xgboost + lightgbm, 3 epochs, standard mode.
Tests the full pipeline end-to-end with minimal compute.
"""
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factory import MLFactory
from src.config.experiment import (
    ExperimentConfig, DataSection, TrainingSection, EvaluationSection, BundlingSection
)
from src.config.training import OptunaConfig
from src.config.data import FeatureConfig, LabelingConfig, MTFConfig
from src.config.cv import WalkForwardConfig

print("=== DRY RUN: xgboost + lightgbm, 3 epochs, standard mode ===")
t0 = time.time()

config = ExperimentConfig(
    name="dry_run_test",
    random_seed=42,
    verbose=1,
    data=DataSection(
        symbol="MGC",
        data_path="data/raw/MGC_1m_5year.parquet",
        features=FeatureConfig(
            mode="full",
            selection_enabled=False,
        ),
        labeling=LabelingConfig(method="triple_barrier", binary_mode=False),
        mtf=MTFConfig(
            enabled=True,
            mode="indicators",
            timeframes=["15min", "30min", "1h"],
            primary_timeframe="5min",
        ),
    ),
    training=TrainingSection(
        models=["xgboost", "lightgbm"],
        horizons=[10, 20],
        training_mode="standard",
        n_splits=2,
        purge_bars=20,
        embargo_bars=60,
        device="auto",
        batch_size=512,
        max_epochs=3,
        early_stopping_patience=3,
        build_ensemble=True,
        meta_learner="ridge_meta",
        walk_forward=WalkForwardConfig(
            n_windows=2, window_type="expanding",
            min_train_pct=0.4, test_pct=0.1,
            embargo_bars=60, gap_bars=20,
        ),
        optuna=OptunaConfig(n_trials=0, n_startup_trials=5, timeout=0),
    ),
    evaluation=EvaluationSection(
        run_backtest=False,
        generate_report=True,
        commission_per_contract=2.50,
        slippage_ticks=1.0,
        initial_equity=100000.0,
    ),
    bundling=BundlingSection(
        create_bundle=False,
        deploy_artifact=False,
    ),
)

print(f"Config: models={config.training.models}, horizons={config.training.horizons}, purge={config.training.purge_bars}")

factory = MLFactory(config, enable_checkpoints=False)
print("Pipeline starting...")
sys.stdout.flush()

try:
    result = factory.run()
    elapsed = time.time() - t0
    print()
    if result.success:
        print(f"=== DRY RUN SUCCESS ({elapsed:.0f}s) ===")
        print(result.summary())
    else:
        print(f"=== DRY RUN FAILED ({elapsed:.0f}s) ===")
        print(f"Error: {result.error_message}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"\n=== DRY RUN EXCEPTION ({elapsed:.0f}s) ===")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
