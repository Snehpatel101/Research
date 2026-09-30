"""
Quickstart: raw OHLCV bars in, a deployed model out, predictions from raw bars.

    python examples/01_quickstart.py

What happens:
1. 4,000 synthetic 5-minute bars are written to parquet.
2. ``MLFactory`` computes features, triple-barrier labels (horizon 5), trains
   XGBoost with purged cross-validation, backtests its out-of-sample signals
   (fills at the next bar's open, costs included), and writes a model bundle
   plus a deploy manifest.
3. ``load_deploy_artifact`` reloads the deployed model and ``predict_from_raw``
   scores raw bars: same cleaning, features, scaling and calibration as training.
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

    # The run lands in <output_dir>/<run_id>/
    cfg = ExperimentConfig(name="quickstart", output_dir=OUTPUT_ROOT / "quickstart")
    cfg.data.symbol = "MES"  # barrier table, tick size, costs, session hours
    cfg.data.data_path = data_path
    cfg.data.mtf.enabled = False  # keep the demo fast (MTF features are on by default)
    cfg.training.models = ["xgboost"]
    cfg.training.horizons = [5]
    cfg.training.n_splits = 3
    cfg.training.optuna.n_trials = 0  # no hyperparameter search for a first run
    cfg.evaluation.run_backtest = True
    # purge_bars / embargo_bars stay None: derived from the label span (max_bars of
    # the MES h5 barrier) and one trading day of 5-minute bars.

    result = MLFactory(cfg, verbose=0).run()

    print(f"run directory : {result.output_dir}")
    val = result.metrics["xgboost_h5"]  # keys are <model>_h<horizon>
    print(
        f"validation    : accuracy {val['accuracy']:.3f}, macro F1 {val['macro_f1']:.3f}, "
        f"log loss {val['logloss_unweighted']:.3f}"
    )
    bt = result.backtest_metrics
    print(
        f"backtest      : {bt['total_trades']} trades, win rate {bt['win_rate_pct']:.1f}%, "
        f"Sharpe {bt['sharpe_ratio']:.2f}, net P&L ${bt['total_pnl']:,.2f} (after costs)"
    )

    # Serve: raw bars in, predictions out. Pass enough history for warmup
    # (about 320 bars here; about 1,900 with MTF features enabled).
    artifact = load_deploy_artifact(result.deploy_path, horizon=5)
    raw = pd.read_parquet(data_path).iloc[-1000:]
    pred = artifact.predict_from_raw(raw)
    latest = pd.DataFrame(
        pred.class_probabilities[-5:],
        columns=["p_short", "p_neutral", "p_long"],
        index=pred.metadata["timestamps"][-5:],
    )
    latest["signal"] = pred.class_predictions[-5:]
    print(f"\nlast 5 of {len(pred.class_predictions)} predictions from raw bars:")
    print(latest.round(3).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
