"""
One MLFactory run for the determinism test (``test_determinism_e2e.py``).

Runs in its own interpreter so each run gets a fresh process (own
PYTHONHASHSEED, own import state), then writes what the test compares:

- ``arrays.npz``: every model's OOF probabilities and class predictions, the
  stacking meta-learner's holdout predictions, and the deployed bundles'
  ``predict_from_raw`` probabilities (primary = ensemble, and each model)
- ``facts.json``: metrics, the seed each model received, manifest and
  tracking facts

With a third argument ``features`` it runs only the data step and the live
feature selection (MDA ranking + per-model feature sets) and writes
``features.json``.

Usage: python -m tests.e2e.determinism_driver <data.parquet> <output_root> [features]
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["xgboost", "logistic"]
HORIZON = 5
SEED = 7


def main(data_path: Path, output_root: Path) -> None:
    logging.basicConfig(level=logging.ERROR)
    warnings.simplefilter("ignore")

    from src.config.experiment import ExperimentConfig, TrackingSection
    from src.factory import MLFactory
    from src.inference import load_deploy_artifact
    from src.inference.deploy import validate_deploy_artifact
    from src.models.tracking import LocalTracker

    cfg = ExperimentConfig(name="determinism", run_id="determinism", output_dir=output_root)
    cfg.random_seed = SEED
    cfg.verbose = 0
    cfg.data.symbol = "MES"
    cfg.data.data_path = data_path
    cfg.data.mtf.enabled = False
    cfg.training.models = list(MODELS)
    cfg.training.horizons = [HORIZON]
    cfg.training.n_splits = 2
    cfg.training.purge_bars = 15
    cfg.training.embargo_bars = 30
    cfg.training.optuna.n_trials = 0
    cfg.training.build_ensemble = True
    cfg.evaluation.run_backtest = True
    cfg.bundling.create_bundle = True
    cfg.bundling.deploy_artifact = True
    cfg.tracking = TrackingSection(backend="local")

    result = MLFactory(cfg, verbose=0, enable_checkpoints=False).run()
    training = result.training_result
    assert training is not None and result.deploy_path is not None

    arrays: dict[str, np.ndarray] = {}
    seeds: dict[str, int] = {}
    for key, model_result in training.model_results.items():
        oof = model_result.oof_prediction
        arrays[f"oof_proba__{key}"] = oof.get_probabilities()
        arrays[f"oof_pred__{key}"] = oof.get_class_predictions()
        seeds[key] = int(model_result.trainer.model.config["random_state"])

    ensemble = training.ensemble_result
    holdout = ensemble.metadata["holdout_predictions"]
    for column in holdout.columns:
        arrays[f"holdout__{column}"] = holdout[column].to_numpy()
    seeds["meta_learner"] = int(ensemble.trainer.config["random_state"])

    raw = pd.read_parquet(data_path)
    recent = raw.iloc[len(raw) // 2 :]
    arrays["deploy__primary"] = (
        load_deploy_artifact(result.deploy_path, horizon=HORIZON)
        .predict_from_raw(recent)
        .class_probabilities
    )
    for model in MODELS:
        bundle = load_deploy_artifact(result.deploy_path, horizon=HORIZON, model_name=model)
        arrays[f"deploy__{model}"] = bundle.predict_from_raw(recent).class_probabilities
    np.savez(output_root / "arrays.npz", **arrays)

    manifest = json.loads(Path(result.manifest_path).read_text())
    tracking_root = Path(manifest["tracking"]["uri"])
    runs = LocalTracker.list_runs(tracking_root)
    parent_id = manifest["tracking"]["run_id"]
    facts = {
        "metrics": result.metrics,
        "ensemble_metrics": {k: v for k, v in result.ensemble_metrics.items() if "time" not in k},
        "backtest_metrics": result.backtest_metrics,
        "seeds": seeds,
        "manifest": {
            "status": manifest["status"],
            "config_hash": manifest["provenance"]["config_hash"],
            "data_sha256": manifest["provenance"]["data_source"]["sha256"],
            "data": manifest["data"],
            "results_metrics": sorted(manifest["results"]["metrics"]),
        },
        "deploy": {
            "valid": validate_deploy_artifact(result.deploy_path)["valid"],
            "run_manifest": json.loads((Path(result.deploy_path) / "manifest.json").read_text())[
                "run_manifest"
            ],
            "provenance_sha256": manifest["provenance_sha256"],
        },
        "tracking": {
            "parent_status": next(r["status"] for r in runs if r["run_id"] == parent_id),
            "children": sorted(
                r["run_name"].split("_h")[0] for r in runs if r.get("parent_run_id") == parent_id
            ),
            "children_status": sorted(
                {r["status"] for r in runs if r.get("parent_run_id") == parent_id}
            ),
        },
    }
    (output_root / "facts.json").write_text(json.dumps(facts, indent=2, default=str))


# Models whose contract caps (logistic 100, svm 80) cut into the ~200 features
FEATURE_SELECTION_MODELS = ["logistic", "svm", "random_forest"]


def feature_selection(data_path: Path, output_root: Path) -> None:
    """Live feature selection only: the ranking and every model's feature set."""
    logging.basicConfig(level=logging.ERROR)
    warnings.simplefilter("ignore")

    from src.config.experiment import ExperimentConfig
    from src.factory import MLFactory
    from src.models.training import UnifiedTrainingOrchestrator

    cfg = ExperimentConfig(name="fs_determinism", run_id="fs", output_dir=output_root)
    cfg.random_seed = SEED
    cfg.verbose = 0
    cfg.data.data_path = data_path
    cfg.data.mtf.enabled = False
    cfg.training.models = list(FEATURE_SELECTION_MODELS)
    cfg.training.horizons = [HORIZON]
    cfg.training.n_splits = 2
    cfg.training.purge_bars = 15
    cfg.training.embargo_bars = 30

    factory = MLFactory(cfg, verbose=0, enable_checkpoints=False)
    factory._seed_everything()
    df, _ = factory.prepare_data()
    orchestrator = UnifiedTrainingOrchestrator(factory._pipeline_config(df=df))
    orchestrator._run_feature_selection_on_train_data(df)
    features = [c for c in df.columns if c in set(orchestrator._all_feature_names)]
    train = df.iloc[: int(len(df) * cfg.data.splits.train_ratio)]
    ranking = orchestrator._compute_mda_ranking(train, features)
    assert ranking is not None
    result = {
        "per_model_features": orchestrator._per_model_features,
        "ranking": [[name, repr(float(value))] for name, value in ranking.items()],
    }
    (output_root / "features.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "features":
        feature_selection(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
