#!/usr/bin/env python3
"""
ML Factory Compatibility Test — Exhaustive Model Combination Verification.

Tests EVERY registered base model in ensemble combinations to prove they
work together seamlessly. Covers:

  • All data ranks:  2D (tabular) × 3D (sequence) × 4D (multi-stream)
  • All model families: boosting, classical, neural, transformer
  • Both meta-learners: ridge_meta, xgboost_meta
  • Heterogeneous ensembles: 2D+3D, 2D+4D, 3D+4D, 2D+3D+4D

Test Matrix (each row is a separate MLFactory run):
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  #  │ Models                           │ Ranks  │ Meta-learner      ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║  1  │ xgboost + lightgbm               │ 2D+2D  │ ridge_meta        ║
  ║  2  │ catboost + random_forest          │ 2D+2D  │ xgboost_meta      ║
  ║  3  │ logistic + svm                    │ 2D+2D  │ ridge_meta        ║
  ║  4  │ xgboost + tcn                     │ 2D+3D  │ ridge_meta        ║
  ║  5  │ lightgbm + lstm                   │ 2D+3D  │ xgboost_meta      ║
  ║  6  │ xgboost + gru                     │ 2D+3D  │ ridge_meta        ║
  ║  7  │ lightgbm + transformer            │ 2D+3D  │ ridge_meta        ║
  ║  8  │ xgboost + tft                     │ 2D+3D  │ ridge_meta        ║
  ║  9  │ lightgbm + nbeats                 │ 2D+3D  │ ridge_meta        ║
  ║ 10  │ xgboost + inceptiontime           │ 2D+3D  │ ridge_meta        ║
  ║ 11  │ lightgbm + resnet1d               │ 2D+3D  │ ridge_meta        ║
  ║ 12  │ tcn + lstm                        │ 3D+3D  │ ridge_meta        ║
  ║ 13  │ gru + inceptiontime               │ 3D+3D  │ xgboost_meta      ║
  ║ 14  │ xgboost + lightgbm + tcn          │ 2D+2D  │ ridge_meta        ║
  ║     │                                   │  +3D   │                   ║
  ║ 15  │ xgboost + patchtst                │ 2D+4D  │ ridge_meta        ║
  ║ 16  │ lightgbm + itransformer           │ 2D+4D  │ ridge_meta        ║
  ║ 17  │ xgboost + tcn + patchtst          │ 2D+3D  │ ridge_meta        ║
  ║     │                                   │  +4D   │                   ║
  ╚═══════════════════════════════════════════════════════════════════════╝

Usage:
    python scripts/compatibility_test.py              # Run all tests
    python scripts/compatibility_test.py --quick      # Run 2D-only tests (fast)
    python scripts/compatibility_test.py --skip-4d    # Skip 4D model tests
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=logging.WARNING,  # Keep quiet — we only want our own output
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# But allow our own logger to be verbose
logger = logging.getLogger("compatibility_test")
logger.setLevel(logging.INFO)

# Quiet down noisy libraries during testing
for name in [
    "src", "torch", "xgboost", "lightgbm", "catboost",
    "sklearn", "numba", "matplotlib", "PIL",
]:
    logging.getLogger(name).setLevel(logging.ERROR)


# ── Constants ─────────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "raw" / "MES_1m_1week.parquet"
SYMBOL = "MES"


def _cleanup_gpu() -> None:
    """Free GPU memory between test runs to prevent OOM."""
    from src.models.device import release_gpu_memory

    release_gpu_memory()


@dataclass
class TestCase:
    """A single compatibility test case."""

    name: str
    models: list[str]
    meta_learner: str
    rank_description: str
    needs_4d: bool = False


# ── Test Matrix ──────────────────────────────────────────────
TEST_CASES: list[TestCase] = [
    # ── 2D + 2D (tabular pairs) ──────────────────────────────
    TestCase("2D+2D boosting pair", ["xgboost", "lightgbm"], "ridge_meta", "2D+2D"),
    TestCase("2D+2D catboost+rf", ["catboost", "random_forest"], "xgboost_meta", "2D+2D"),
    TestCase("2D+2D classical pair", ["logistic", "svm"], "ridge_meta", "2D+2D"),
    # ── 2D + 3D (heterogeneous — every 3D model) ─────────────
    TestCase("2D+3D xgb+tcn", ["xgboost", "tcn"], "ridge_meta", "2D+3D"),
    TestCase("2D+3D lgb+lstm", ["lightgbm", "lstm"], "xgboost_meta", "2D+3D"),
    TestCase("2D+3D xgb+gru", ["xgboost", "gru"], "ridge_meta", "2D+3D"),
    TestCase("2D+3D lgb+transformer", ["lightgbm", "transformer"], "ridge_meta", "2D+3D"),
    TestCase("2D+3D xgb+tft", ["xgboost", "tft"], "ridge_meta", "2D+3D"),
    TestCase("2D+3D lgb+nbeats", ["lightgbm", "nbeats"], "ridge_meta", "2D+3D"),
    TestCase("2D+3D xgb+inceptiontime", ["xgboost", "inceptiontime"], "ridge_meta", "2D+3D"),
    TestCase("2D+3D lgb+resnet1d", ["lightgbm", "resnet1d"], "ridge_meta", "2D+3D"),
    # ── 3D + 3D (neural pairs) ───────────────────────────────
    TestCase("3D+3D tcn+lstm", ["tcn", "lstm"], "ridge_meta", "3D+3D"),
    TestCase("3D+3D gru+inceptiontime", ["gru", "inceptiontime"], "xgboost_meta", "3D+3D"),
    # ── 2D + 2D + 3D (three-model heterogeneous) ─────────────
    TestCase("2D+2D+3D triple", ["xgboost", "lightgbm", "tcn"], "ridge_meta", "2D+2D+3D"),
    # ── 4D combinations (multi-stream) ────────────────────────
    TestCase("2D+4D xgb+patchtst", ["xgboost", "patchtst"], "ridge_meta", "2D+4D", needs_4d=True),
    TestCase("2D+4D lgb+itransformer", ["lightgbm", "itransformer"], "ridge_meta", "2D+4D", needs_4d=True),
    TestCase("2D+3D+4D full hetero", ["xgboost", "tcn", "patchtst"], "ridge_meta", "2D+3D+4D", needs_4d=True),
]


@dataclass
class TestResult:
    """Result of a single test case."""

    test_case: TestCase
    passed: bool
    duration: float
    n_models_trained: int = 0
    ensemble_f1: float | None = None
    stacking_samples: int | None = None
    error: str | None = None


def run_single_test(tc: TestCase) -> TestResult:
    """Run a single compatibility test case through the full MLFactory pipeline."""

    from src.config.data import FeatureConfig, LabelingConfig, SequenceConfig
    from src.config.experiment import (
        BundlingSection,
        DataSection,
        EvaluationSection,
        ExperimentConfig,
        TrainingSection,
    )
    from src.config.training import CalibrationConfig, OptunaConfig
    from src.factory import MLFactory

    start = time.time()

    try:
        with tempfile.TemporaryDirectory(prefix=f"compat_{tc.name.replace(' ', '_')}_") as tmpdir:
            config = ExperimentConfig(
                name=f"compat_test_{tc.name.replace(' ', '_')}",
                output_dir=Path(tmpdir),
                data=DataSection(
                    symbol=SYMBOL,
                    data_path=str(DATA_PATH),
                    features=FeatureConfig(families=["price", "volume", "volatility"]),
                    labeling=LabelingConfig(method="triple_barrier"),
                    sequence=SequenceConfig(seq_len=30),
                ),
                training=TrainingSection(
                    models=tc.models,
                    horizons=[5],
                    training_mode="standard",
                    n_splits=3,
                    purge_bars=10,
                    embargo_bars=5,
                    build_ensemble=True,
                    meta_learner=tc.meta_learner,
                    max_epochs=3,
                    batch_size=64,
                    optuna=OptunaConfig(n_trials=0),
                    calibration=CalibrationConfig(enabled=False),
                ),
                evaluation=EvaluationSection(
                    run_backtest=False,
                    compute_shap=False,
                ),
                bundling=BundlingSection(
                    create_bundle=True,
                    deploy_artifact=False,
                ),
            )

            factory = MLFactory(config)
            result = factory.run()

            elapsed = time.time() - start

            if result is None:
                return TestResult(tc, False, elapsed, error="MLFactory.run() returned None")

            tr = result.training_result
            if tr is None:
                return TestResult(tc, False, elapsed, error="No training_result")

            n_trained = tr.n_models
            if n_trained != len(tc.models):
                return TestResult(
                    tc, False, elapsed, n_trained,
                    error=f"Expected {len(tc.models)} models, got {n_trained}",
                )

            # Check ensemble result
            ensemble_f1 = None
            stacking_n = None
            has_ensemble = False
            for name, mr in tr.model_results.items():
                if "ensemble" in name:
                    has_ensemble = True
                    ensemble_f1 = mr.val_f1
            if hasattr(tr, "ensemble_result") and tr.ensemble_result is not None:
                has_ensemble = True
                if hasattr(tr.ensemble_result, "stacking_dataset"):
                    sd = tr.ensemble_result.stacking_dataset
                    if sd is not None:
                        stacking_n = sd.n_samples

            if not has_ensemble:
                return TestResult(
                    tc, False, elapsed, n_trained,
                    error="No ensemble result — meta-learner training failed",
                )

            return TestResult(tc, True, elapsed, n_trained, ensemble_f1, stacking_n)

    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        # Extract last 3 lines of traceback for compact error
        short_tb = "\n".join(tb.strip().split("\n")[-3:])
        return TestResult(tc, False, elapsed, error=f"{type(e).__name__}: {e}\n{short_tb}")


def main() -> int:
    """Run all compatibility tests and report results."""
    parser = argparse.ArgumentParser(description="ML Factory Compatibility Test")
    parser.add_argument("--quick", action="store_true", help="Run 2D-only tests (fast)")
    parser.add_argument("--skip-4d", action="store_true", help="Skip 4D model tests")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode for all models (avoids GPU OOM)")
    args = parser.parse_args()

    # Force CPU mode if requested — prevents GPU OOM in sequential tests
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("  CPU mode: forced (CUDA_VISIBLE_DEVICES='')")

    # Filter test cases
    if args.quick:
        cases = [tc for tc in TEST_CASES if tc.rank_description == "2D+2D"]
    elif args.skip_4d:
        cases = [tc for tc in TEST_CASES if not tc.needs_4d]
    else:
        cases = TEST_CASES

    n_total = len(cases)
    logger.info("=" * 72)
    logger.info("  ML FACTORY COMPATIBILITY TEST")
    logger.info("=" * 72)
    logger.info(f"  Test cases: {n_total}")
    logger.info(f"  Data: {DATA_PATH.name} ({SYMBOL})")
    logger.info(f"  Mode: {'quick (2D only)' if args.quick else 'skip-4D' if args.skip_4d else 'full'}")
    logger.info("=" * 72)

    results: list[TestResult] = []
    total_start = time.time()

    for i, tc in enumerate(cases, 1):
        # Clean up GPU memory between tests to prevent OOM
        _cleanup_gpu()

        logger.info(f"\n[{i}/{n_total}] {tc.name} "
                     f"({' + '.join(tc.models)}, meta={tc.meta_learner})")
        logger.info(f"  Ranks: {tc.rank_description}")

        result = run_single_test(tc)
        results.append(result)

        if result.passed:
            extras = []
            if result.ensemble_f1 is not None:
                extras.append(f"ensemble_f1={result.ensemble_f1:.4f}")
            if result.stacking_samples is not None:
                extras.append(f"stacking={result.stacking_samples}")
            extras_str = ", ".join(extras)
            logger.info(
                f"  ✅ PASS — {result.n_models_trained} models, "
                f"{result.duration:.1f}s{', ' + extras_str if extras_str else ''}"
            )
        else:
            logger.error(f"  ❌ FAIL — {result.duration:.1f}s")
            logger.error(f"     {result.error}")

    # ── Summary ───────────────────────────────────────────────
    total_time = time.time() - total_start
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    logger.info("\n" + "=" * 72)
    logger.info("  COMPATIBILITY TEST SUMMARY")
    logger.info("=" * 72)
    logger.info(f"  Total: {n_total} | Passed: {len(passed)} | Failed: {len(failed)}")
    logger.info(f"  Time: {total_time:.1f}s")

    if failed:
        logger.info("\n  FAILURES:")
        for r in failed:
            logger.info(f"    ❌ {r.test_case.name}: {r.error}")

    # ── Per-rank summary ──────────────────────────────────────
    rank_groups: dict[str, list[TestResult]] = {}
    for r in results:
        rd = r.test_case.rank_description
        rank_groups.setdefault(rd, []).append(r)

    logger.info("\n  Per-rank results:")
    for rank, group in sorted(rank_groups.items()):
        p = sum(1 for r in group if r.passed)
        logger.info(f"    {rank}: {p}/{len(group)} passed")

    # ── Model coverage ────────────────────────────────────────
    tested_models = set()
    for r in results:
        if r.passed:
            tested_models.update(r.test_case.models)

    all_base_models = {
        "xgboost", "lightgbm", "catboost", "random_forest", "logistic", "svm",
        "lstm", "gru", "tcn", "transformer", "tft", "nbeats", "inceptiontime",
        "resnet1d", "patchtst", "itransformer",
    }
    if args.quick:
        expected = {"xgboost", "lightgbm", "catboost", "random_forest", "logistic", "svm"}
    elif args.skip_4d:
        expected = all_base_models - {"patchtst", "itransformer"}
    else:
        expected = all_base_models

    untested = expected - tested_models
    logger.info(f"\n  Model coverage: {len(tested_models)}/{len(expected)}")
    if untested:
        logger.warning(f"  ⚠ Untested models: {sorted(untested)}")

    tested_meta = set()
    for r in results:
        if r.passed:
            tested_meta.add(r.test_case.meta_learner)
    logger.info(f"  Meta-learner coverage: {sorted(tested_meta)}")

    logger.info("\n" + "=" * 72)
    if not failed:
        logger.info("  ✅ ALL COMPATIBILITY TESTS PASSED")
    else:
        logger.info(f"  ❌ {len(failed)} TESTS FAILED")
    logger.info("=" * 72)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
