#!/usr/bin/env python3
"""
Integration test: Mixed-model ensemble end-to-end.

Trains a heterogeneous ensemble (xgboost + lightgbm + tcn) on a small
data slice, verifying that all components work together seamlessly:

1. Feature engineering
2. Labeling
3. Data preparation (2D tabular + 3D sequence, per-model routing)
4. Training (parallel boosting + sequential neural)
5. OOF generation (tabular + sequence generators)
6. OOF alignment (intersection of heterogeneous sample sets)
7. Meta-learner training (ridge_meta on stacking features)
8. Artifact bundling

Usage:
    python scripts/integration_test.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("integration_test")

# ── Configuration ─────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "raw" / "MES_1m_1week.parquet"
SYMBOL = "MES"

# Models: 2 boosting (parallel-eligible) + 1 sequence (sequential)
MODELS = ["xgboost", "lightgbm", "tcn"]
HORIZONS = [5]  # Single horizon for speed
TRAINING_MODE = "standard"
BUILD_ENSEMBLE = True
META_LEARNER = "ridge_meta"

# CV settings (keep small for speed)
N_SPLITS = 3
PURGE_BARS = 10
EMBARGO_BARS = 5

# Feature engineering families (minimal set for speed)
FEATURE_FAMILIES = ["price", "volume", "volatility"]

# Sequence settings
SEQUENCE_LENGTH = 30  # Keep short for small dataset

# Training settings
MAX_EPOCHS = 3  # Minimal for speed
BATCH_SIZE = 64


def main() -> int:
    """Run the integration test and return exit code (0=pass, 1=fail)."""
    start = time.time()
    logger.info("=" * 70)
    logger.info("  INTEGRATION TEST: Mixed-Model Ensemble (xgboost + lightgbm + tcn)")
    logger.info("=" * 70)

    # ── Step 0: Import everything ─────────────────────────────
    logger.info("\n[Step 0] Importing pipeline modules...")
    try:
        import pandas as pd

        from src.config.experiment import (
            BundlingSection,
            DataSection,
            EvaluationSection,
            ExperimentConfig,
            TrainingSection,
        )
        from src.config.data import FeatureConfig, LabelingConfig, SequenceConfig
        from src.config.training import CalibrationConfig, OptunaConfig
        from src.factory import MLFactory

        logger.info("  ✓ All imports successful")
    except ImportError as e:
        logger.error(f"  ✗ Import failed: {e}")
        return 1

    # ── Step 1: Build ExperimentConfig ────────────────────────
    logger.info("\n[Step 1] Building ExperimentConfig...")
    try:
        with tempfile.TemporaryDirectory(prefix="integration_test_") as tmpdir:
            config = ExperimentConfig(
                name="integration_test_mixed_ensemble",
                output_dir=Path(tmpdir),
                data=DataSection(
                    symbol=SYMBOL,
                    data_path=str(DATA_PATH),
                    features=FeatureConfig(families=FEATURE_FAMILIES),
                    labeling=LabelingConfig(method="triple_barrier"),
                    sequence=SequenceConfig(seq_len=SEQUENCE_LENGTH),
                ),
                training=TrainingSection(
                    models=MODELS,
                    horizons=HORIZONS,
                    training_mode=TRAINING_MODE,
                    n_splits=N_SPLITS,
                    purge_bars=PURGE_BARS,
                    embargo_bars=EMBARGO_BARS,
                    build_ensemble=BUILD_ENSEMBLE,
                    meta_learner=META_LEARNER,
                    max_epochs=MAX_EPOCHS,
                    batch_size=BATCH_SIZE,
                    optuna=OptunaConfig(n_trials=0),  # Disable Optuna for speed
                    calibration=CalibrationConfig(enabled=False),
                ),
                evaluation=EvaluationSection(
                    run_backtest=False,
                    compute_shap=False,
                ),
                bundling=BundlingSection(
                    create_bundle=True,
                    deploy_artifact=False,  # Skip deploy for test
                ),
            )
            logger.info(f"  ✓ ExperimentConfig created: {len(config.models)} models, "
                        f"horizons={config.horizons}")

            # ── Step 2: Create MLFactory and run ──────────────
            logger.info("\n[Step 2] Creating MLFactory and running full pipeline...")
            factory = MLFactory(config)
            result = factory.run()

            # ── Step 3: Validate results ──────────────────────
            logger.info("\n[Step 3] Validating results...")
            errors = []

            # Check we got a result at all
            if result is None:
                errors.append("MLFactory.run() returned None")
            else:
                training_result = result.training_result
                if training_result is None:
                    errors.append("training_result is None")
                else:
                    # Check model results
                    n_model_results = training_result.n_models
                    expected_models = len(MODELS) * len(HORIZONS)
                    if n_model_results < expected_models:
                        errors.append(
                            f"Expected {expected_models} model results, "
                            f"got {n_model_results}"
                        )
                    else:
                        logger.info(f"  ✓ {n_model_results} model results")

                    # Check each model has metrics
                    for key, model_result in training_result.model_results.items():
                        if not model_result.metrics:
                            errors.append(f"Model {key} has empty metrics")
                        else:
                            val_f1 = model_result.metrics.get("val_f1", -1)
                            logger.info(
                                f"  ✓ {key}: val_f1={val_f1:.4f}, "
                                f"rank={model_result.data_rank}D, "
                                f"n_features={model_result.n_features}, "
                                f"time={model_result.training_time_seconds:.1f}s"
                            )

                    # Check ensemble result
                    if BUILD_ENSEMBLE:
                        if training_result.ensemble_result is None:
                            errors.append("Ensemble result is None despite build_ensemble=True")
                        else:
                            ens = training_result.ensemble_result
                            ens_f1 = ens.metrics.get("val_f1", -1)
                            logger.info(
                                f"  ✓ Ensemble ({ens.model_name}): "
                                f"val_f1={ens_f1:.4f}, "
                                f"time={ens.training_time_seconds:.1f}s"
                            )

                        # Check stacking dataset
                        if training_result.stacking_dataset is None:
                            errors.append("Stacking dataset is None")
                        else:
                            sd = training_result.stacking_dataset
                            logger.info(
                                f"  ✓ Stacking dataset: {sd.n_samples} samples, "
                                f"models={sd.model_names}"
                            )

                        # Check aligned OOF
                        if training_result.aligned_oof is None:
                            errors.append("Aligned OOF is None")
                        else:
                            aoof = training_result.aligned_oof
                            logger.info(
                                f"  ✓ Aligned OOF: {aoof.n_common} samples, "
                                f"{aoof.n_models} models"
                            )
                            logger.info(f"    Coverage: {aoof.coverage}")

                    # Check output artifacts exist
                    output_path = Path(tmpdir)
                    artifacts = list(output_path.rglob("*"))
                    n_artifacts = len([a for a in artifacts if a.is_file()])
                    logger.info(f"  ✓ {n_artifacts} artifact files saved")

            # ── Step 4: Report ────────────────────────────────
            elapsed = time.time() - start
            logger.info("\n" + "=" * 70)
            if errors:
                logger.error(f"  INTEGRATION TEST FAILED ({elapsed:.1f}s)")
                for e in errors:
                    logger.error(f"    ✗ {e}")
                return 1
            else:
                logger.info(f"  INTEGRATION TEST PASSED ✓ ({elapsed:.1f}s)")
                logger.info("=" * 70)
                return 0

    except Exception as e:
        elapsed = time.time() - start
        logger.exception(f"  INTEGRATION TEST CRASHED ({elapsed:.1f}s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
