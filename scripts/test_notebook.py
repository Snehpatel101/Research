#!/usr/bin/env python
"""
Test script that simulates running notebook cells 1-7 locally.
Uses boosting-only models (xgboost, lightgbm, catboost) for speed.
"""
import os
import sys
import time
import traceback

# Ensure repo root
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

IN_COLAB = False
results = {}


def run_cell(name, func):
    """Run a cell function, capture pass/fail."""
    print(f"\n{'='*60}")
    print(f"CELL {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        ret = func()
        elapsed = time.time() - t0
        results[name] = "PASS"
        print(f"\n  >> PASS ({elapsed:.1f}s)")
        return ret
    except Exception as e:
        elapsed = time.time() - t0
        results[name] = f"FAIL: {e}"
        print(f"\n  >> FAIL ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        return None


# ── CELL 1: Setup / Imports ──────────────────────────────────
def cell_1_setup():
    from src.factory import MLFactory
    from src.config.experiment import ExperimentConfig
    import torch

    print(f"ML Factory loaded successfully!")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {'cuda' if torch.cuda.is_available() else 'cpu (no GPU)'}")
    print(f"Repo: {REPO_DIR}")
    return True


# ── CELL 2: Configuration ────────────────────────────────────
# Boosting only, single horizon, minimal settings
SYMBOL = "MES"
DATA_PATH = os.path.join(REPO_DIR, "data", "raw", "MES_1m_1week.parquet")
TARGET_TIMEFRAME = "5min"
MODELS = ["xgboost", "lightgbm", "catboost"]
HORIZONS = [5]
N_SPLITS = 2
PURGE_BARS = 10
EMBARGO_BARS = 60
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 256
DEVICE = "auto"
TRAINING_MODE = "standard"
MTF_ENABLED = True
MTF_TIMEFRAMES = ["15min", "30min", "1h"]
FEATURE_SELECTION_ENABLED = False
FEATURE_SELECTION_METHOD = "mda"
BUILD_ENSEMBLE = True
META_LEARNER = "ridge_meta"
OPTUNA_ENABLED = False
OPTUNA_TRIALS = 0
RUN_BACKTEST = False
GENERATE_REPORT = True
CREATE_BUNDLE = True
DEPLOY_ARTIFACT = True
EXPERIMENT_NAME = "test_notebook_boosting"
RANDOM_SEED = 42


def cell_2_config():
    print(f"Models: {len(MODELS)} selected -> {MODELS}")
    print(f"Epochs: {MAX_EPOCHS}, Horizons: {HORIZONS}, Device: {DEVICE}")
    print(f"Training mode: {TRAINING_MODE}")
    print(f"Ensemble: {BUILD_ENSEMBLE}, Optuna trials: disabled")
    print(f"Bundling: create={CREATE_BUNDLE}, deploy={DEPLOY_ARTIFACT}")
    assert len(MODELS) == 3, f"Expected 3 models, got {len(MODELS)}"
    assert HORIZONS == [5], f"Expected [5], got {HORIZONS}"
    return True


# ── CELL 3: Validate Configuration ──────────────────────────
def cell_3_validate():
    import torch

    VALID_MODELS = {
        "xgboost", "lightgbm", "catboost",
        "lstm", "gru", "tcn", "nbeats",
        "inceptiontime", "resnet1d",
        "patchtst", "itransformer", "tft",
    }

    errors = []
    for m in MODELS:
        if m not in VALID_MODELS:
            errors.append(f"Unknown model: '{m}'")
    if not MODELS:
        errors.append("No models selected!")
    if not os.path.exists(DATA_PATH):
        errors.append(f"Data file not found: {DATA_PATH}")
    if TRAINING_MODE not in ("standard", "walk_forward", "regime_aware", "meta_labeling"):
        errors.append(f"Unknown training mode: '{TRAINING_MODE}'")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        raise ValueError("Fix errors above")

    print(f"Config OK: {len(MODELS)} models, {N_SPLITS} CV folds, "
          f"purge={PURGE_BARS}, embargo={EMBARGO_BARS}, device={DEVICE}, mode={TRAINING_MODE}")
    return True


# ── CELL 4: Load Data ────────────────────────────────────────
raw_data = None


def cell_4_data():
    global raw_data
    import pandas as pd

    raw_data = pd.read_parquet(DATA_PATH)
    raw_data.columns = [c.lower().strip() for c in raw_data.columns]

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
    missing = [c for c in REQUIRED_COLUMNS if c not in raw_data.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    # Ensure datetime index
    if "datetime" in raw_data.columns:
        raw_data["datetime"] = pd.to_datetime(raw_data["datetime"])
        raw_data = raw_data.set_index("datetime").sort_index()
    elif "date" in raw_data.columns:
        raw_data["date"] = pd.to_datetime(raw_data["date"])
        raw_data = raw_data.set_index("date").sort_index()
        raw_data.index.name = "datetime"
    elif not isinstance(raw_data.index, pd.DatetimeIndex):
        try:
            raw_data.index = pd.to_datetime(raw_data.index)
            raw_data.index.name = "datetime"
            raw_data = raw_data.sort_index()
        except Exception:
            raise ValueError("Could not parse datetime index")

    print(f"  Symbol:      {SYMBOL}")
    print(f"  Rows:        {len(raw_data):,}")
    print(f"  Shape:       {raw_data.shape}")
    print(f"  Columns:     {list(raw_data.columns)}")
    print(f"  Date range:  {raw_data.index.min()} -> {raw_data.index.max()}")

    missing_counts = raw_data[REQUIRED_COLUMNS].isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        print(f"WARNING: {total_missing} missing values in OHLCV columns")
    else:
        print("No missing values in OHLCV columns.")

    assert len(raw_data) > 0, "Empty DataFrame"
    return True


# ── CELL 5: Run ML Factory ───────────────────────────────────
result = None


def cell_5_run():
    global result
    from src.config.experiment import (
        ExperimentConfig,
        DataSection,
        TrainingSection,
        EvaluationSection,
        BundlingSection,
    )
    from src.config.training import OptunaConfig
    from src.config.data import FeatureConfig, LabelingConfig, MTFConfig
    from src.factory import MLFactory

    config = ExperimentConfig(
        name=EXPERIMENT_NAME,
        random_seed=RANDOM_SEED,
        verbose=1,
        data=DataSection(
            symbol=SYMBOL,
            data_path=DATA_PATH,
            features=FeatureConfig(
                mode="full",
                selection_enabled=FEATURE_SELECTION_ENABLED,
                selection_method=FEATURE_SELECTION_METHOD,
            ),
            labeling=LabelingConfig(method="triple_barrier"),
            mtf=MTFConfig(
                enabled=MTF_ENABLED,
                mode="indicators" if MTF_ENABLED else "none",
                timeframes=MTF_TIMEFRAMES if MTF_ENABLED else [],
                primary_timeframe=TARGET_TIMEFRAME,
            ),
        ),
        training=TrainingSection(
            models=MODELS,
            horizons=HORIZONS,
            training_mode=TRAINING_MODE,
            n_splits=N_SPLITS,
            purge_bars=PURGE_BARS,
            embargo_bars=EMBARGO_BARS,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            build_ensemble=BUILD_ENSEMBLE,
            meta_learner=META_LEARNER,
            optuna=OptunaConfig(n_trials=OPTUNA_TRIALS),
        ),
        evaluation=EvaluationSection(
            run_backtest=RUN_BACKTEST,
            generate_report=GENERATE_REPORT,
        ),
        bundling=BundlingSection(
            create_bundle=CREATE_BUNDLE,
            deploy_artifact=DEPLOY_ARTIFACT,
        ),
    )

    print(f"Experiment: {config.name}")
    print(f"Models: {config.training.models}")
    print(f"Training mode: {config.training.training_mode}")
    print(f"CV: {config.training.n_splits} folds, purge={config.training.purge_bars}, embargo={config.training.embargo_bars}")
    print(f"Optuna: disabled ({config.training.optuna.n_trials} trials)")
    print(f"Bundling: create={config.bundling.create_bundle}, deploy={config.bundling.deploy_artifact}")

    factory = MLFactory(config, enable_checkpoints=True)
    result = factory.run()

    if result.success:
        print(f"\nPipeline SUCCESS")
        print(result.summary())
    else:
        raise RuntimeError(f"Pipeline failed: {result.error_message}")

    return True


# ── CELL 6: Results Verification ─────────────────────────────
def cell_6_results():
    assert result is not None, "No result object"
    assert result.success, f"result.success is False: {result.error_message}"
    assert result.n_models > 0, f"n_models={result.n_models}, expected > 0"

    print(f"  Run ID:          {result.run_id}")
    print(f"  Models trained:  {result.n_models}")
    print(f"  Best model:      {result.best_model}")
    print(f"  Duration:        {result.duration_seconds:.1f}s ({result.duration_seconds/60:.1f} min)")

    if result.metrics:
        import pandas as pd
        metrics_df = pd.DataFrame(result.metrics).T
        metrics_df.index.name = "model"
        print(f"\n  Metrics:\n{metrics_df.round(4).to_string()}")

    return True


# ── CELL 7: Deploy Artifact Verification ─────────────────────
def cell_7_deploy():
    from pathlib import Path

    assert result is not None and result.success, "No successful result"
    assert result.deploy_path, "No deploy_path set"
    assert Path(result.deploy_path).exists(), f"Deploy path does not exist: {result.deploy_path}"

    from src.inference.deploy import (
        load_deploy_artifact,
        validate_deploy_artifact,
        DeployManifest,
        DEPLOY_MANIFEST_FILE,
    )

    deploy_dir = Path(result.deploy_path)

    # Validate
    validation = validate_deploy_artifact(deploy_dir)
    print(f"  Path:     {deploy_dir}")
    print(f"  Valid:    {validation['valid']}")
    print(f"  Horizons: {validation.get('n_horizons', 0)}")
    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"  ISSUE: {issue}")

    # Load manifest and check model names
    manifest = DeployManifest.load(deploy_dir / DEPLOY_MANIFEST_FILE)

    unknown_names = []
    missing_primary = []
    for h, h_manifest in sorted(manifest.horizons.items()):
        print(f"\n  Horizon {h}:")
        print(f"    Primary model: {h_manifest.primary_model}")
        if not h_manifest.primary_model:
            missing_primary.append(h)
        for entry in h_manifest.entries:
            tag = " [ensemble]" if entry.is_ensemble else ""
            f1 = entry.metrics.get("macro_f1", 0.0)
            print(f"    - {entry.model_name}{tag} (F1={f1:.4f}) -> {entry.bundle_path}")
            if entry.model_name == "unknown":
                unknown_names.append(f"horizon={h}, entry={entry.bundle_path}")

    assert not unknown_names, f"Found 'unknown' model names: {unknown_names}"
    assert not missing_primary, f"Missing primary_model for horizons: {missing_primary}"
    print(f"\n  All model names valid, primary_model set for all horizons.")
    return True


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Test Notebook Script — Boosting Only")
    print(f"Repo: {REPO_DIR}")
    print(f"Data: {DATA_PATH}")
    t_start = time.time()

    run_cell("1-Setup", cell_1_setup)
    run_cell("2-Config", cell_2_config)
    run_cell("3-Validate", cell_3_validate)
    run_cell("4-Data", cell_4_data)
    run_cell("5-RunFactory", cell_5_run)
    run_cell("6-Results", cell_6_results)
    run_cell("7-Deploy", cell_7_deploy)

    total = time.time() - t_start

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY ({total:.1f}s total)")
    print(f"{'='*60}")
    all_pass = True
    for cell, status in results.items():
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  {icon}  Cell {cell}")
        if status != "PASS":
            print(f"       {status}")
            all_pass = False

    if all_pass:
        print(f"\nALL CELLS PASSED")
        sys.exit(0)
    else:
        failed = [k for k, v in results.items() if v != "PASS"]
        print(f"\nFAILED CELLS: {failed}")
        sys.exit(1)
