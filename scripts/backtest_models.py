#!/usr/bin/env python3
"""
Train fast models + backtest to get financial metrics (Sharpe, DD, profit, etc.).

Uses validation set predictions (properly returned by the Trainer) to simulate
trading with realistic transaction costs via the backtester.

Runs 6 fast models: N-BEATS, LightGBM, XGBoost, CatBoost, iTransformer, PatchTST
Expected time: ~10 minutes on CPU.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)
for name in [
    "src.data", "src.core", "src.models", "src.config",
    "src.validation", "src.optimization", "urllib3", "numba",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

FAST_MODELS = ["nbeats", "lightgbm", "xgboost", "catboost", "itransformer", "patchtst"]
SCALED_DIR = PROJECT_ROOT / "runs" / "20260212_120129_588531_22ff" / "data" / "splits" / "scaled"
HORIZON = 20


def train_model(model_name: str):
    """Train model and return results with val predictions."""
    import src.models  # noqa: F401
    from src.core.container import TimeSeriesDataContainer
    from src.models.config import TrainerConfig
    from src.models.training import Trainer

    container = TimeSeriesDataContainer.from_parquet_dir(
        path=SCALED_DIR, horizon=HORIZON,
    )

    config = TrainerConfig(
        model_name=model_name,
        horizon=HORIZON,
        model_config={
            "max_epochs": 3,
            "early_stopping_patience": 2,
            "batch_size": 128,
            "device": "cpu",
            "mixed_precision": False,
        },
        output_dir=PROJECT_ROOT / "experiments" / "backtest_runs",
        evaluate_test_set=True,  # Also evaluate on test set
    )

    trainer = Trainer(config)
    results = trainer.run(container, skip_save=True)
    return results, container


def run_backtest(predictions_arr, prices, timestamps, labels):
    """Run backtest on predictions and return financial metrics."""
    from src.inference.backtesting.backtest import BacktestConfig, Backtester

    pred_df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "prediction": predictions_arr.astype(int),
        "label": labels.astype(int),
    })

    price_df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "open": prices,
        "high": prices * 1.001,
        "low": prices * 0.999,
        "close": prices,
    })

    config = BacktestConfig.for_mes(
        initial_equity=100000.0,
        commission_per_contract=2.50,
        slippage_ticks=1.0,
        fixed_contracts=1,
        max_holding_period=HORIZON,
    )

    backtester = Backtester(predictions=pred_df, prices=price_df, config=config)
    result = backtester.run()
    m = result.metrics
    s = result.summary()

    return {
        "total_return_pct": round(s["total_return_pct"], 4),
        "total_pnl": round(s["total_pnl"], 2),
        "final_equity": round(s["final_equity"], 2),
        "total_trades": m.total_trades,
        "win_rate": round(m.win_rate, 4),
        "profit_factor": round(m.profit_factor, 4),
        "sharpe_ratio": round(m.sharpe_ratio, 4),
        "sortino_ratio": round(m.sortino_ratio, 4),
        "calmar_ratio": round(m.calmar_ratio, 4),
        "max_drawdown_pct": round(m.max_drawdown * 100, 4),
        "max_drawdown_duration": m.max_drawdown_duration,
        "expectancy": round(m.expectancy, 4),
        "payoff_ratio": round(m.payoff_ratio, 4),
        "var_95": round(m.var_95, 6),
        "cvar_95": round(m.cvar_95, 6),
    }


def main():
    print("=" * 70)
    print("BACKTEST: Train + Backtest 6 Fast Models")
    print(f"Data: {SCALED_DIR}")
    print("=" * 70)

    # Load val data for prices and timestamps (for backtesting)
    val_df = pd.read_parquet(SCALED_DIR / "val_scaled.parquet")
    val_prices = val_df["close"].values
    val_timestamps = val_df["datetime"].values
    val_labels = val_df["label_h20"].values
    print(f"\nVal set: {len(val_df)} samples")
    print(f"Label distribution: {dict(zip(*np.unique(val_labels[val_labels != -99], return_counts=True)))}")

    all_results = {}
    total_start = time.time()

    for model_name in FAST_MODELS:
        print(f"\n{'─' * 60}")
        print(f"  [{model_name.upper()}] Training...", end=" ", flush=True)
        start = time.time()

        try:
            results, container = train_model(model_name)
            train_time = time.time() - start
            print(f"OK ({train_time:.1f}s)")

            eval_metrics = results.get("evaluation_metrics", {})
            test_metrics = results.get("test_metrics", {})
            val_acc = eval_metrics.get("accuracy", 0)
            val_f1 = eval_metrics.get("macro_f1", 0)
            test_acc = test_metrics.get("accuracy", 0) if test_metrics else 0
            test_f1 = test_metrics.get("macro_f1", 0) if test_metrics else 0
            print(f"    Val: acc={val_acc:.4f}, F1={val_f1:.4f}")
            print(f"    Test: acc={test_acc:.4f}, F1={test_f1:.4f}")

            # Get val predictions (properly returned by trainer)
            val_preds = results.get("val_predictions")
            val_true = results.get("val_true")

            if val_preds is None:
                print(f"    No val predictions — skipping backtest")
                all_results[model_name] = {
                    "train_time": round(train_time, 1),
                    "classification": {
                        "val_accuracy": round(val_acc, 4),
                        "val_f1": round(val_f1, 4),
                        "test_accuracy": round(test_acc, 4),
                        "test_f1": round(test_f1, 4),
                    },
                    "backtest": None,
                }
                continue

            preds = np.asarray(val_preds)

            # Align predictions with val price data
            # Sequence models may have fewer predictions (due to windowing)
            n_preds = len(preds)
            offset = len(val_prices) - n_preds
            prices_aligned = val_prices[offset:offset + n_preds]
            ts_aligned = val_timestamps[offset:offset + n_preds]
            labels_aligned = val_labels[offset:offset + n_preds]

            # Filter out invalid labels for distribution display
            valid_mask = labels_aligned != -99
            pred_dist = dict(zip(*np.unique(preds.astype(int), return_counts=True)))
            print(f"    Predictions: {pred_dist} ({n_preds} samples)")

            # Run backtest
            print(f"    Backtesting...", end=" ", flush=True)
            bt_start = time.time()
            bt_metrics = run_backtest(preds, prices_aligned, ts_aligned, labels_aligned)
            bt_time = time.time() - bt_start
            print(f"OK ({bt_time:.1f}s)")

            print(f"    Sharpe={bt_metrics['sharpe_ratio']:.2f}, "
                  f"Sortino={bt_metrics['sortino_ratio']:.2f}, "
                  f"MaxDD={bt_metrics['max_drawdown_pct']:.2f}%, "
                  f"WinRate={bt_metrics['win_rate']*100:.1f}%, "
                  f"PnL=${bt_metrics['total_pnl']:.2f}, "
                  f"Trades={bt_metrics['total_trades']}, "
                  f"PF={bt_metrics['profit_factor']:.2f}")

            all_results[model_name] = {
                "train_time": round(train_time, 1),
                "classification": {
                    "val_accuracy": round(val_acc, 4),
                    "val_f1": round(val_f1, 4),
                    "val_precision": round(eval_metrics.get("precision", 0), 4),
                    "val_recall": round(eval_metrics.get("recall", 0), 4),
                    "test_accuracy": round(test_acc, 4),
                    "test_f1": round(test_f1, 4),
                },
                "backtest": bt_metrics,
            }

        except Exception as e:
            train_time = time.time() - start
            print(f"FAIL ({train_time:.1f}s) - {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {
                "train_time": round(train_time, 1),
                "error": str(e),
            }

    total_time = time.time() - total_start

    # Print summary
    print(f"\n{'=' * 95}")
    print(f"BACKTEST SUMMARY — Validation Set (Total: {total_time:.1f}s)")
    print(f"{'=' * 95}")
    print(f"{'Model':<15} {'Time':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD%':>8} "
          f"{'WinRate':>8} {'PnL':>10} {'Trades':>7} {'PF':>7}")
    print("─" * 95)

    for model_name in FAST_MODELS:
        r = all_results.get(model_name, {})
        bt = r.get("backtest")
        t = r.get("train_time", 0)
        if bt:
            print(f"{model_name:<15} {t:>6.1f}s {bt['sharpe_ratio']:>8.2f} "
                  f"{bt['sortino_ratio']:>8.2f} {bt['max_drawdown_pct']:>7.2f}% "
                  f"{bt['win_rate']*100:>7.1f}% ${bt['total_pnl']:>9.2f} "
                  f"{bt['total_trades']:>7} {bt['profit_factor']:>7.2f}")
        else:
            err = r.get("error", "no predictions")[:40]
            print(f"{model_name:<15} {t:>6.1f}s {'—':>8} {'—':>8} {'—':>8} "
                  f"{'—':>8} {'—':>10} {'—':>7} {err}")

    # Save results
    output_path = PROJECT_ROOT / "experiments" / "backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
