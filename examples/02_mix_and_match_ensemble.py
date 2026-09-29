"""
Mix and match: a gradient-boosted tree, an LSTM and a multi-timeframe PatchTST,
stacked with a soft-voting meta-learner, deployed and served from raw bars.

    python examples/02_mix_and_match_ensemble.py

The three models see different inputs: XGBoost a 2D feature table, the LSTM
3D windows of the feature table, PatchTST 4D windows of OHLCV at several
timeframes. Each produces out-of-fold (OOF) predictions under purged k-fold,
the OOF rows are aligned on the bar each prediction belongs to, and the
meta-learner combines them. The deploy manifest makes the ensemble the primary
artifact, so ``load_deploy_artifact`` returns it and ``predict_from_raw`` runs
all three base bundles plus the meta-learner on raw bars.

Swap ``models`` / ``meta_learner`` for any other combination (see
docs/mix-and-match.md); nothing else changes.
"""

from __future__ import annotations

import time

import pandas as pd
from _synthetic import OUTPUT_ROOT, make_synthetic_ohlcv, quiet_logging, use_this_checkout

use_this_checkout()
quiet_logging()

from src.config.experiment import ExperimentConfig  # noqa: E402
from src.factory import MLFactory  # noqa: E402
from src.inference import load_deploy_artifact  # noqa: E402


def main() -> None:
    t0 = time.time()
    data_path = make_synthetic_ohlcv(OUTPUT_ROOT / "data" / "synthetic_5min.parquet")

    cfg = ExperimentConfig(name="mix_and_match", output_dir=OUTPUT_ROOT / "mix_and_match")
    cfg.data.symbol = "MES"
    cfg.data.data_path = data_path
    cfg.data.mtf.enabled = False  # MTF *features*; PatchTST still gets its 4D streams
    cfg.training.models = ["xgboost", "lstm", "patchtst"]  # 2D + 3D + 4D
    cfg.training.meta_learner = "voting_meta"  # or ridge_meta, xgboost_meta, ...
    cfg.training.build_ensemble = True
    cfg.training.horizons = [5]
    cfg.training.n_splits = 3
    cfg.training.optuna.n_trials = 0
    # Tiny neural budget so the demo finishes in minutes on a CPU
    cfg.training.max_epochs = 1
    cfg.training.early_stopping_patience = 1
    cfg.training.batch_size = 128

    result = MLFactory(cfg, verbose=0).run()

    print(f"run directory : {result.output_dir}")
    for key, m in result.metrics.items():
        print(f"  {key:<14} macro F1 {m.get('macro_f1', float('nan')):.3f}")
    ens = result.ensemble_metrics
    print(
        f"  {'ensemble':<14} macro F1 {ens.get('macro_f1', float('nan')):.3f} "
        "(scored on a purged holdout of the OOF rows)"
    )

    artifact = load_deploy_artifact(result.deploy_path, horizon=5)  # primary = ensemble
    raw = pd.read_parquet(data_path).iloc[-1000:]
    pred = artifact.predict_from_raw(raw)
    stamps = pred.metadata["timestamps"]
    print(
        f"\n{type(artifact).__name__}: {len(pred.class_predictions)} predictions from "
        f"{len(raw)} raw bars ({stamps[0]} .. {stamps[-1]})"
    )
    print("  (the first bars go to feature and sequence-window warmup)")
    print(f"  signal counts: {pd.Series(pred.class_predictions).value_counts().to_dict()}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
