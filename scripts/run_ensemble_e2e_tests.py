"""
E2E Cross-Family Ensemble Tests — All combinations on 1 week MES data.

Tests all cross-family ensemble combinations:
  1. 2D+2D   (XGBoost + LightGBM)
  2. 2D+2D+2D (XGBoost + LightGBM + CatBoost)
  3. 2D+3D   (XGBoost + LSTM)
  4. 2D+4D   (XGBoost + PatchTST)
  5. 3D+3D   (LSTM + TCN)
  6. 3D+4D   (GRU + iTransformer)
  7. 4D+4D   (PatchTST + iTransformer)
  8. 2D+3D+4D (XGBoost + LSTM + PatchTST)
"""

import json
import os
import sys
import time
import traceback

os.environ["TORCHDYNAMO_DISABLE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
# Suppress noisy loggers
for name in ["optuna", "lightgbm", "urllib3"]:
    logging.getLogger(name).setLevel(logging.WARNING)

from src.config.experiment import (
    BundlingSection,
    DataSection,
    EvaluationSection,
    ExperimentConfig,
    TrainingSection,
)
from src.config.training import OptunaConfig
from src.factory import MLFactory

# ── Test cases ──────────────────────────────────────────────────────────────
TEST_CASES = [
    (["xgboost", "lightgbm"], "2D+2D"),
    (["xgboost", "lightgbm", "catboost"], "2D+2D+2D"),
    (["xgboost", "lstm"], "2D+3D"),
    (["xgboost", "patchtst"], "2D+4D"),
    (["lstm", "tcn"], "3D+3D"),
    (["gru", "itransformer"], "3D+4D"),
    (["patchtst", "itransformer"], "4D+4D"),
    (["xgboost", "lstm", "patchtst"], "2D+3D+4D"),
]

DATA_PATH = "data/raw/MES_1m_1month.parquet"

# ── Results storage ─────────────────────────────────────────────────────────
results = []


def run_test(models, label, idx):
    """Run a single ensemble test case."""
    print(f"\n{'=' * 70}")
    print(f"  TEST {idx}/{len(TEST_CASES)}: {label} — {models}")
    print(f"{'=' * 70}")

    start = time.time()
    try:
        config = ExperimentConfig(
            name=f"e2e_ensemble_{label.replace('+', '_').lower()}",
            output_dir=f"experiments/e2e_ensemble_{label.replace('+', '_').lower()}",
            verbose=1,
            data=DataSection(
                symbol="MES",
                data_path=DATA_PATH,
            ),
            training=TrainingSection(
                models=models,
                horizons=[5],
                n_splits=3,
                purge_bars=5,
                embargo_bars=2,
                build_ensemble=True,
                meta_learner="ridge_meta",
                max_epochs=1,
                early_stopping_patience=1,
                batch_size=256,
                optuna=OptunaConfig(n_trials=0),
            ),
            evaluation=EvaluationSection(run_backtest=True, generate_report=True),
            bundling=BundlingSection(create_bundle=True, deploy_artifact=True),
        )

        factory = MLFactory(config)
        result = factory.run()
        duration = time.time() - start

        # Extract metrics
        model_metrics = {}
        if result.metrics:
            for model, metrics in result.metrics.items():
                f1 = metrics.get("macro_f1", metrics.get("val_f1", 0))
                acc = metrics.get("val_accuracy", 0)
                model_metrics[model] = {"f1": f1, "accuracy": acc}

        backtest = {}
        if result.backtest_metrics:
            backtest = {
                "trades": result.backtest_metrics.get("total_trades", 0),
                "sharpe": result.backtest_metrics.get("sharpe_ratio", 0),
                "pnl": result.backtest_metrics.get("total_pnl", 0),
                "win_rate": result.backtest_metrics.get("win_rate", 0),
                "max_drawdown": result.backtest_metrics.get("max_drawdown_pct", 0),
                "final_equity": result.backtest_metrics.get("final_equity", 100000),
            }

        ensemble_f1 = 0
        if result.metrics:
            for k, v in result.metrics.items():
                if "ensemble" in k.lower() or "stacking" in k.lower():
                    ensemble_f1 = v.get("macro_f1", v.get("val_f1", 0))

        entry = {
            "label": label,
            "models": models,
            "status": "PASS" if result.success else "FAIL",
            "duration": round(duration, 1),
            "model_metrics": model_metrics,
            "ensemble_f1": ensemble_f1,
            "backtest": backtest,
            "error": None,
        }

        print(f"\n  STATUS: {'PASS' if result.success else 'FAIL'}")
        print(f"  Duration: {duration:.1f}s")
        for model, m in model_metrics.items():
            print(f"  {model}: F1={m['f1']:.4f}, Acc={m['accuracy']:.4f}")
        if backtest:
            print(f"  Backtest: {backtest.get('trades', 0)} trades, Sharpe={backtest.get('sharpe', 0):.4f}")

        return entry

    except Exception as e:
        duration = time.time() - start
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n  STATUS: FAIL")
        print(f"  Error: {error_msg}")
        traceback.print_exc()
        return {
            "label": label,
            "models": models,
            "status": "FAIL",
            "duration": round(duration, 1),
            "model_metrics": {},
            "ensemble_f1": 0,
            "backtest": {},
            "error": error_msg,
        }


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  ML FACTORY — CROSS-FAMILY ENSEMBLE E2E TESTS")
    print(f"  Data: {DATA_PATH} (MES Jan 2020, ~30K rows, 1 epoch, standard mode)")
    print(f"  Test cases: {len(TEST_CASES)}")
    print("=" * 70)

    total_start = time.time()

    for idx, (models, label) in enumerate(TEST_CASES, 1):
        entry = run_test(models, label, idx)
        results.append(entry)

    total_duration = time.time() - total_start

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Total time: {total_duration:.1f}s")
    print(f"  Passed: {sum(1 for r in results if r['status'] == 'PASS')}/{len(results)}")
    print(f"\n  {'#':<4} {'Combo':<12} {'Status':<8} {'Duration':<10} {'Models'}")
    print(f"  {'-'*4} {'-'*12} {'-'*8} {'-'*10} {'-'*40}")
    for i, r in enumerate(results, 1):
        print(
            f"  {i:<4} {r['label']:<12} {r['status']:<8} {r['duration']:<10.1f} {', '.join(r['models'])}"
        )

    # Save results to JSON
    output_path = "experiments/ensemble_e2e_results.json"
    os.makedirs("experiments", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "date": "2026-02-20",
                "data": DATA_PATH,
                "date_range": "2020-01-06 to 2020-01-12",
                "total_duration": round(total_duration, 1),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to {output_path}")
