"""
Two training modes on the same bars: walk-forward evaluation and meta-labeling.

    python examples/03_walk_forward_and_meta_labeling.py

Walk-forward (``training_mode="walk_forward"``)
    The model is re-fit on successive expanding windows and always predicts the
    window after its training cutoff (with a purge gap), the way it would have
    run live. Each window selects features and fits scalers on its own training
    rows only. The deployed model is trained like in standard mode.

Meta-labeling (``training_mode="meta_labeling"``)
    ``training.models[0]`` is the primary model that picks the side. A second
    model learns P(the primary's bet pays off) from the primary's out-of-fold
    predictions, and a bet is taken only when that probability clears
    ``training.meta_labeling.threshold``. The deployed ``MetaLabelingBundle``
    returns directions, P(win), a trade mask and sized positions.
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


def base_config(name: str, data_path) -> ExperimentConfig:
    cfg = ExperimentConfig(name=name, output_dir=OUTPUT_ROOT / name)
    cfg.data.symbol = "MES"
    cfg.data.data_path = data_path
    cfg.data.mtf.enabled = False
    cfg.training.horizons = [5]
    cfg.training.n_splits = 3
    cfg.training.optuna.n_trials = 0
    cfg.evaluation.run_backtest = True
    return cfg


def walk_forward(data_path) -> None:
    cfg = base_config("walk_forward", data_path)
    cfg.training.training_mode = "walk_forward"
    cfg.training.models = ["lightgbm"]
    cfg.training.walk_forward.n_windows = 3
    cfg.training.walk_forward.window_type = "expanding"  # or "rolling"

    result = MLFactory(cfg, verbose=0).run()
    bt = result.backtest_metrics
    print("walk-forward")
    for key, m in result.metrics.items():  # mean over the out-of-sample windows
        print(f"  {key}: mean window F1 {m['val_f1']:.3f}, accuracy {m['val_accuracy']:.3f}")
    print(
        f"  backtest of the walk-forward signals: {bt['total_trades']} trades, "
        f"Sharpe {bt['sharpe_ratio']:.2f}"
    )
    print(f"  run directory: {result.output_dir}")


def meta_labeling(data_path) -> None:
    cfg = base_config("meta_labeling", data_path)
    cfg.training.training_mode = "meta_labeling"
    cfg.training.models = ["xgboost"]  # models[0] is the primary (side) model
    cfg.training.meta_labeling.meta_model = "logistic"
    cfg.training.meta_labeling.threshold = 0.5

    result = MLFactory(cfg, verbose=0).run()
    m = next(iter(result.metrics.values()))
    print("\nmeta-labeling")
    print(
        f"  primary precision {m.get('primary_precision', float('nan')):.3f} -> "
        f"filtered {m.get('meta_precision', float('nan')):.3f}, "
        f"{m.get('trades_taken', 0):.0f} of {m.get('primary_bets', 0):.0f} primary bets taken"
    )

    bundle = load_deploy_artifact(result.deploy_path, horizon=5)  # MetaLabelingBundle
    raw = pd.read_parquet(data_path).iloc[-1000:]
    meta = bundle.predict_meta(raw)
    table = pd.DataFrame(
        {
            "direction": meta.directions,
            "p_win": meta.meta_probabilities.round(3),
            "trade": meta.trade_mask,
            "position": meta.positions.round(3),
        },
        index=meta.timestamps,
    )
    print(f"  {meta.n_trades} of {len(table)} bars traded; last 5:")
    print(table.tail(5).to_string())
    print(f"  run directory: {result.output_dir}")


def main() -> None:
    t0 = time.time()
    data_path = make_synthetic_ohlcv(OUTPUT_ROOT / "data" / "synthetic_5min.parquet")
    walk_forward(data_path)
    meta_labeling(data_path)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
