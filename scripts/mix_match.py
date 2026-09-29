"""
Mix-and-match matrix runner for ML Factory.

Runs MLFactory end-to-end (features -> labels -> train -> OOF -> ensemble ->
backtest -> bundle -> deploy -> reload + predict) for many combinations of
base models, meta-learners and training modes on tiny synthetic data, each in
its own subprocess, and prints a pass/fail matrix.

Usage:
    python scripts/mix_match.py solo                 # every base model alone
    python scripts/mix_match.py pairs                # every pair of base models
    python scripts/mix_match.py meta                 # every meta-learner on a cross-family set
    python scripts/mix_match.py modes                # every training mode on a cross-family set
    python scripts/mix_match.py modes-solo           # every model alone in each non-standard mode
    python scripts/mix_match.py binary               # binary labels x meta-learners x modes
    python scripts/mix_match.py all-in               # all base models in one ensemble
    python scripts/mix_match.py report               # render docs/MIX_AND_MATCH.md from results
    python scripts/mix_match.py custom xgboost,lstm --meta stacking --mode standard

Options:
    --jobs N        parallel subprocesses (default 3)
    --timeout S     per-run timeout in seconds (default 900)
    --out DIR       results directory (default experiments/mix_match)
    --data PATH     real OHLCV file (custom runs); --bar-timeframe 5min resamples it
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_MODELS = [
    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest",
    "logistic",
    "svm",
    "lstm",
    "gru",
    "tcn",
    "transformer",
    "inceptiontime",
    "resnet1d",
    "nbeats",
    "patchtst",
    "itransformer",
    "tft",
]

META_LEARNERS = [
    "ridge_meta",
    "xgboost_meta",
    "mlp_meta",
    "calibrated_meta",
    "voting_meta",
]

TRAINING_MODES = ["standard", "walk_forward", "regime_aware", "meta_labeling"]

# One model from each input-rank family: 2D tabular, 3D sequence, 4D multi-stream
CROSS_FAMILY = ["xgboost", "lstm", "patchtst"]

N_ROWS = 4000


def make_synthetic_ohlcv(path: Path, n_rows: int = N_ROWS, seed: int = 7) -> None:
    """Synthetic 5-min OHLCV random walk written to parquet."""
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-02 09:30", periods=n_rows, freq="5min")
    close = 5000.0 + np.cumsum(rng.normal(0, 2.0, n_rows))
    open_ = np.roll(close, 1) + rng.normal(0, 0.5, n_rows)
    open_[0] = close[0]
    eps = np.abs(rng.normal(0, 0.5, n_rows)) + 0.25
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + eps,
            "low": np.minimum(open_, close) - eps,
            "close": close,
            "volume": rng.randint(100, 5000, n_rows).astype(float),
        },
        index=idx,
    )
    df.index.name = "datetime"
    df.to_parquet(path)


def run_one(spec: dict, data_path: Path, out_dir: Path) -> dict:
    """Run one factory configuration in-process and return a result record."""
    import logging
    import warnings

    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    if os.environ.get("MIX_MATCH_TRACE"):
        # Attach the in-flight exception to any warning logged inside an except
        # block, so swallowed errors show their origin in the run log.
        class _AttachExcInfo(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                if record.levelno >= logging.WARNING and not record.exc_info:
                    exc = sys.exc_info()
                    if exc[0] is not None:
                        record.exc_info = exc
                return True

        for handler in logging.getLogger().handlers:
            handler.addFilter(_AttachExcInfo())

    from src.config.experiment import ExperimentConfig
    from src.factory import MLFactory

    models = spec["models"]
    cfg = ExperimentConfig()
    cfg.name = spec["name"]
    cfg.random_seed = 42
    cfg.verbose = 0
    cfg.output_dir = out_dir / spec["name"]
    if cfg.output_dir.name != cfg.run_id:
        cfg.output_dir = cfg.output_dir / cfg.run_id

    cfg.data.symbol = "MES"
    data_path = Path(spec.get("data") or data_path)
    cfg.data.data_path = data_path
    cfg.data.bar_timeframe = spec.get("bar_timeframe")
    cfg.data.labeling.binary_mode = bool(spec.get("binary"))
    cfg.data.mtf.enabled = spec.get("mtf", False)

    cfg.training.models = models
    cfg.training.horizons = [5]
    cfg.training.training_mode = spec.get("mode", "standard")
    cfg.training.n_splits = 3
    cfg.training.purge_bars = 15
    cfg.training.embargo_bars = 30
    cfg.training.max_epochs = 1
    cfg.training.batch_size = 128
    cfg.training.early_stopping_patience = 1
    cfg.training.build_ensemble = len(models) > 1
    cfg.training.meta_learner = spec.get("meta", "ridge_meta")
    cfg.training.optuna.n_trials = 0
    wf = cfg.training.walk_forward
    if hasattr(wf, "n_windows"):
        wf.n_windows = 3

    cfg.evaluation.run_backtest = True
    cfg.bundling.create_bundle = True
    cfg.bundling.deploy_artifact = True

    record: dict = {"name": spec["name"], "spec": spec, "ok": False}
    t0 = time.time()
    try:
        factory = MLFactory(cfg, verbose=0)
        result = factory.run()
        record["success"] = bool(result.success)
        record["error"] = result.error_message
        record["n_models"] = result.n_models
        record["metrics"] = {
            model: {k: float(v) for k, v in m.items() if isinstance(v, int | float | np.floating)}
            for model, m in (result.metrics or {}).items()
            if isinstance(m, dict)
        }
        record["ensemble_metrics"] = {
            k: float(v)
            for k, v in (result.ensemble_metrics or {}).items()
            if isinstance(v, int | float | np.floating)
        }
        record["backtest_keys"] = sorted((result.backtest_metrics or {}).keys())
        record["deploy_path"] = str(result.deploy_path) if result.deploy_path else None
        record["bundle_path"] = str(result.bundle_path) if result.bundle_path else None

        # Meta-labeling trains one primary+meta system regardless of `models`
        is_meta_labeling = spec.get("mode") == "meta_labeling"
        expected_models = 1 if is_meta_labeling else len(models)
        problems = []
        if not result.success:
            problems.append(f"result.success=False: {record['error']}")
        if result.n_models != expected_models:
            problems.append(f"n_models={result.n_models} != {expected_models}")
        if len(record["metrics"]) < expected_models:
            problems.append(f"metrics for {sorted(record['metrics'])} only")
        if expected_models > 1 and not record["ensemble_metrics"]:
            problems.append("no ensemble metrics")
        problems.extend(_check_ensemble_alignment(result))
        problems.extend(_check_prediction_parity(result, data_path))
        problems.extend(_check_meta_labeling_parity(result, data_path))
        problems.extend(_check_universal_pipeline(result, data_path))
        if not record["backtest_keys"] and not spec.get("binary"):
            problems.append("empty backtest metrics")
        if not result.deploy_path:
            problems.append("no deploy artifact")
        else:
            problems.extend(
                _check_deploy_predict(
                    Path(result.deploy_path), data_path, 2 if spec.get("binary") else 3
                )
            )
        record["problems"] = problems
        record["ok"] = not problems
    except BaseException as exc:  # noqa: BLE001 - harness must report every failure
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    record["seconds"] = round(time.time() - t0, 1)
    return record


def _check_ensemble_alignment(result) -> list[str]:
    """Stacking rows must pair every model's prediction with the label of the SAME bar."""
    import numpy as np
    import pandas as pd

    tr = result.training_result
    aligned = getattr(tr, "aligned_oof", None)
    stacking = getattr(tr, "stacking_dataset", None)
    if aligned is None or stacking is None or result.output_dir is None:
        return []
    cache = Path(result.output_dir) / "cache" / "data_pipeline.parquet"
    if not cache.exists():
        return []
    labels = pd.read_parquet(cache)["label_h5"].to_numpy()
    y_true = stacking.data["y_true"].to_numpy()
    expected = labels[aligned.common_indices[: len(y_true)]]
    if not np.array_equal(y_true, expected):
        return [f"stacking labels misaligned ({(y_true != expected).mean():.0%} differ)"]
    return []


def _check_prediction_parity(result, data_path: Path) -> list[str]:
    """Deployed bundle == trained model: same probabilities on the validation bars.

    Re-prepares the validation split from the cached training frame, scores it
    with the in-memory trained model, and compares with the bundle's
    predict_from_raw on raw OHLCV at the same bar timestamps.
    """
    import numpy as np
    import pandas as pd

    from src.core.constants import OHLCV_COLUMNS
    from src.data.adapters.preparation import UnifiedDataPreparation
    from src.inference.bundle import ModelBundle

    tr = result.training_result
    out = Path(result.output_dir)
    cache = out / "cache" / "data_pipeline.parquet"
    if tr is None or not cache.exists():
        return []
    df = pd.read_parquet(cache)
    raw = pd.read_parquet(data_path)
    problems: list[str] = []
    for mr in tr.model_results.values():
        trainer = mr.trainer
        bundle_dir = out / "bundles" / f"{mr.model_name}_h{mr.horizon}"
        if (
            trainer is None
            or not hasattr(trainer, "model")
            or not (bundle_dir / "features.json").exists()
        ):
            continue
        bundle = ModelBundle.load(bundle_dir)
        if bundle.metadata.requires_4d:
            continue  # needs the factory's multi-stream frames; covered by deploy predict
        features = set(bundle.feature_columns)
        keep = [
            c
            for c in df.columns
            if c in features or c in OHLCV_COLUMNS or c.startswith(("label", "sample_weight"))
        ]
        prep = (
            UnifiedDataPreparation(tr.config)
            .prepare(df=df[keep], model_name=mr.model_name)
            .filter_invalid_labels()
        )
        expected = trainer.model.predict(prep.X_val).class_probabilities
        served = bundle.predict_from_raw(raw, calibrate=False)
        got = pd.DataFrame(
            served.class_probabilities, index=pd.DatetimeIndex(served.metadata["timestamps"])
        ).reindex(df.index[prep.val_indices])
        covered = got.notna().all(axis=1).to_numpy()
        close = np.isclose(got.to_numpy()[covered], expected[covered], atol=1e-3).all(axis=1)
        if covered.mean() < 0.95 or close.mean() < 0.99:
            problems.append(
                f"prediction parity {mr.model_name}: covered {covered.mean():.0%}, "
                f"match {close.mean():.0%}"
            )
    return problems


def _check_meta_labeling_parity(result, data_path: Path) -> list[str]:
    """Deployed MetaLabelingBundle == trained system on the validation bars.

    Rebuilds the meta features exactly as training does (primary model input +
    uncalibrated primary probabilities, ``build_meta_features``) from the
    in-memory primary and meta models, and compares P(bet pays off) and the
    trade decisions with the bundle's predict_from_raw at the same timestamps.
    """
    import numpy as np
    import pandas as pd

    from src.core.constants import OHLCV_COLUMNS
    from src.data.adapters.preparation import UnifiedDataPreparation
    from src.inference.meta_labeling_bundle import (
        MetaLabelingBundle,
        build_meta_features,
        primary_sides,
    )

    tr = result.training_result
    out = Path(result.output_dir)
    cache = out / "cache" / "data_pipeline.parquet"
    if tr is None or not cache.exists():
        return []
    df = pd.read_parquet(cache)
    raw = pd.read_parquet(data_path)
    problems: list[str] = []
    for mr in tr.model_results.values():
        art = mr.mode_artifacts
        if art.get("kind") != "meta_labeling":
            continue
        bundle = MetaLabelingBundle.load(out / "bundles" / f"{mr.model_name}_h{mr.horizon}")
        trainer = mr.trainer
        features = set(trainer.feature_columns or [])
        keep = [
            c
            for c in df.columns
            if c in features or c in OHLCV_COLUMNS or c.startswith(("label", "sample_weight"))
        ]
        prep = (
            UnifiedDataPreparation(tr.config)
            .prepare(df=df[keep], model_name=art["primary_model"])
            .filter_invalid_labels()
        )
        model_input = prep.X_val
        if prep.data_rank == 2 and features and list(trainer.feature_columns) != prep.feature_names:
            model_input = model_input[
                :, [prep.feature_names.index(c) for c in trainer.feature_columns]
            ]
        primary = trainer.model.predict(model_input)
        expected_p = art["meta_model"].predict_proba(
            build_meta_features(model_input, primary.class_probabilities)
        )[:, 1]
        expected_trade = primary_sides(primary.class_predictions) & (expected_p >= art["threshold"])

        served = bundle.predict_from_raw(raw, calibrate=False)
        index = pd.DatetimeIndex(served.metadata["timestamps"])
        rows = df.index[prep.val_indices]
        got_p = pd.Series(served.metadata["meta_probability"], index=index).reindex(rows)
        got_trade = pd.Series(served.metadata["trade_mask"], index=index).reindex(rows)
        covered = got_p.notna().to_numpy()
        close = np.isclose(got_p.to_numpy()[covered], expected_p[covered], atol=1e-3)
        same_trades = got_trade.to_numpy()[covered].astype(bool) == expected_trade[covered]
        if covered.mean() < 0.95 or close.mean() < 0.99 or same_trades.mean() < 0.99:
            problems.append(
                f"meta-labeling parity {mr.model_name}: covered {covered.mean():.0%}, "
                f"P(win) match {close.mean():.0%}, trade match {same_trades.mean():.0%}"
            )
    return problems


def _check_universal_pipeline(result, data_path: Path) -> list[str]:
    """UniversalInferencePipeline.from_experiment serves every model and the ensemble."""
    import pandas as pd

    from src.inference import UniversalInferencePipeline

    tr = result.training_result
    if tr is None:
        return []
    try:
        pipeline = UniversalInferencePipeline.from_experiment(tr.config)
    except Exception as exc:  # noqa: BLE001
        # Experiments with only regime / meta-labeling bundles have nothing to load here
        if "No valid bundles" in str(exc):
            return []
        return [f"universal pipeline load failed: {type(exc).__name__}: {exc}"]
    raw = pd.read_parquet(data_path)
    raw = raw.iloc[len(raw) // 2 :]
    served = pipeline.predict_all(raw)
    problems = []
    if len(served) != pipeline.n_models:
        problems.append(f"universal pipeline served {len(served)}/{pipeline.n_models} models")
    if pipeline.has_ensemble:
        try:
            pipeline.predict_ensemble(raw)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"universal pipeline ensemble failed: {type(exc).__name__}: {exc}")
    return problems


def _check_deploy_predict(deploy_path: Path, data_path: Path, n_classes: int = 3) -> list[str]:
    """Reload the deploy artifact, predict on raw OHLCV, and check feature parity."""
    import numpy as np
    import pandas as pd

    from src.inference import load_deploy_artifact
    from src.inference.bundle import ModelBundle

    problems: list[str] = []
    raw = pd.read_parquet(data_path)
    try:
        artifact = load_deploy_artifact(deploy_path, horizon=5)
        # Recent half of the history: enough warmup for MTF/wavelet features
        pred = artifact.predict_from_raw(raw.iloc[len(raw) // 2 :])
        n = len(pred.class_predictions)
        if n == 0:
            problems.append("deploy predict returned no rows")
        elif pred.class_probabilities.shape != (n, n_classes):
            problems.append(f"bad proba shape {pred.class_probabilities.shape} for {n} rows")
        elif not np.isfinite(pred.class_probabilities).all():
            problems.append("non-finite deploy probabilities")
    except Exception as exc:  # noqa: BLE001
        return [f"deploy predict failed: {type(exc).__name__}: {exc}"]

    # Train/serve parity: features recomputed from the FULL raw history by each
    # bundle's preprocessing graph must equal the training features.
    cache = deploy_path.parent / "cache" / "data_pipeline.parquet"
    if not cache.exists():
        return problems
    trained = pd.read_parquet(cache)
    for bundle_dir in sorted((deploy_path.parent / "bundles").glob("*_h5")):
        if not (bundle_dir / "features.json").exists():
            continue  # regime / meta-labeling / ensemble wrappers
        bundle = ModelBundle.load(bundle_dir)
        if bundle.metadata.requires_4d or bundle.preprocessing_graph is None:
            continue
        served = bundle.preprocess(raw)
        common = served.index.intersection(trained.index)
        cols = bundle.feature_columns
        a = served.loc[common, cols].to_numpy(np.float64)
        b = trained.loc[common, cols].to_numpy(np.float64)
        scale = np.maximum(np.abs(b), 1.0)
        bad = ~np.isclose(a / scale, b / scale, atol=1e-3, equal_nan=True)
        if len(common) < 0.9 * len(trained) or bad.any():
            worst = [cols[j] for j in np.where(bad.any(axis=0))[0][:5]]
            problems.append(
                f"parity {bundle_dir.name}: {len(common)}/{len(trained)} rows, "
                f"{bad.any(axis=0).sum()} mismatched cols e.g. {worst}"
            )
    return problems


def build_specs(kind: str, args: argparse.Namespace) -> list[dict]:
    specs: list[dict] = []
    if kind == "solo":
        for m in BASE_MODELS:
            specs.append({"name": f"solo_{m}", "models": [m]})
    elif kind == "pairs":
        for a, b in itertools.combinations(BASE_MODELS, 2):
            specs.append({"name": f"pair_{a}+{b}", "models": [a, b]})
    elif kind == "meta":
        for meta in META_LEARNERS:
            specs.append({"name": f"meta_{meta}", "models": CROSS_FAMILY, "meta": meta})
    elif kind == "modes":
        for mode in TRAINING_MODES:
            specs.append({"name": f"mode_{mode}", "models": CROSS_FAMILY, "mode": mode})
    elif kind == "modes-solo":
        for mode in TRAINING_MODES[1:]:
            for m in BASE_MODELS:
                specs.append({"name": f"mode_{mode}_{m}", "models": [m], "mode": mode})
    elif kind == "binary":
        for meta in META_LEARNERS:
            specs.append(
                {"name": f"binary_{meta}", "models": CROSS_FAMILY, "meta": meta, "binary": True}
            )
        for mode in TRAINING_MODES[1:]:
            specs.append(
                {"name": f"binary_{mode}", "models": CROSS_FAMILY, "mode": mode, "binary": True}
            )
    elif kind == "all-in":
        specs.append({"name": "all_in", "models": BASE_MODELS, "meta": args.meta})
    elif kind == "custom":
        models = args.models.split(",")
        specs.append(
            {
                "name": f"custom_{'+'.join(models)}_{args.meta}_{args.mode}",
                "data": args.data or None,
                "bar_timeframe": args.bar_timeframe or None,
                "binary": args.binary,
                "models": models,
                "meta": args.meta,
                "mode": args.mode,
            }
        )
    else:
        raise SystemExit(f"unknown kind: {kind}")
    return specs


def write_report(out_dir: Path, report_path: Path) -> None:
    """Render every summary_<kind>.json into a markdown verification report."""
    from datetime import date

    sections = [
        ("solo", "Every base model alone"),
        ("pairs", "Every pair of base models (stacking ensemble)"),
        ("meta", f"Every meta-learner on a cross-family ensemble ({' + '.join(CROSS_FAMILY)})"),
        ("modes", f"Every training mode on a cross-family ensemble ({' + '.join(CROSS_FAMILY)})"),
        ("modes-solo", "Every base model in every non-standard training mode"),
        ("binary", "Binary labels: every meta-learner and every non-standard mode"),
        ("all-in", "All base models in one ensemble"),
    ]
    lines = [
        "# Mix-and-Match Verification Matrix",
        "",
        f"Generated {date.today().isoformat()} by `python scripts/mix_match.py report` from the",
        "results of `python scripts/mix_match.py {solo,pairs,meta,modes,modes-solo,binary,all-in}`.",
        "",
        "Every run is a full `MLFactory.run()` on synthetic 5-minute OHLCV (4,000 bars,",
        "1 epoch, CPU): features -> labels -> per-model feature selection -> training ->",
        "OOF -> stacking ensemble -> backtest -> bundles -> deploy artifact -> reload ->",
        "`predict_from_raw`. A run passes only if **all** of these hold:",
        "",
        "- training succeeds and every model reports metrics (plus ensemble metrics for >1 model)",
        "- stacking rows pair every model's OOF prediction with the label of the *same* bar",
        "- the deployed bundle reproduces the trained model's validation probabilities",
        "  (re-prepared validation split vs `predict_from_raw` on raw bars)",
        "- features recomputed from raw OHLCV equal the training features",
        "- the backtest produces metrics and the deploy artifact reloads and predicts",
        "",
        "## Building blocks",
        "",
        f"- **Base models ({len(BASE_MODELS)}):** " + ", ".join(f"`{m}`" for m in BASE_MODELS),
        f"- **Meta-learners ({len(META_LEARNERS)}):** "
        + ", ".join(f"`{m}`" for m in META_LEARNERS),
        f"- **Training modes ({len(TRAINING_MODES)}):** "
        + ", ".join(f"`{m}`" for m in TRAINING_MODES),
        "",
    ]
    total = passed = 0
    for kind, title in sections:
        path = out_dir / f"summary_{kind}.json"
        if not path.exists():
            continue
        records = json.loads(path.read_text())
        n_pass = sum(bool(r.get("ok")) for r in records)
        total += len(records)
        passed += n_pass
        lines += [f"## {title} — {n_pass}/{len(records)} pass", ""]
        if kind == "pairs":
            ok = {tuple(r["name"].removeprefix("pair_").split("+")): r.get("ok") for r in records}
            lines.append("| | " + " | ".join(BASE_MODELS) + " |")
            lines.append("|---" * (len(BASE_MODELS) + 1) + "|")
            for a in BASE_MODELS:
                cells = []
                for b in BASE_MODELS:
                    if a == b:
                        cells.append("—")
                    else:
                        res = ok.get((a, b), ok.get((b, a)))
                        cells.append("✅" if res else ("❌" if res is not None else "·"))
                lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
        else:
            lines += ["| Run | Result | Seconds |", "|---|---|---|"]
            for r in records:
                detail = (
                    "PASS"
                    if r.get("ok")
                    else (r.get("exception") or "; ".join(r.get("problems", [])))[:120]
                )
                lines.append(f"| `{r['name']}` | {detail} | {r.get('seconds', 0)} |")
        lines.append("")
    lines.insert(8, f"**Overall: {passed}/{total} runs pass.**\n")
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path} ({passed}/{total} pass)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("kind")
    parser.add_argument("models", nargs="?", default="")
    parser.add_argument("--meta", default="ridge_meta")
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--data", default="", help="OHLCV file to use instead of synthetic data")
    parser.add_argument("--bar-timeframe", default="", help="resample input bars, e.g. 5min")
    parser.add_argument("--binary", action="store_true", help="binary labels (move vs no move)")
    parser.add_argument("--out", default=str(REPO_ROOT / "experiments" / "mix_match"))
    parser.add_argument("--only", default="", help="comma-separated spec names to run")
    parser.add_argument("--_child", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "synthetic_5min.parquet"

    if args._child:
        spec = json.loads(args._child)
        record = run_one(spec, data_path, out_dir / "runs")
        (out_dir / "results" / f"{spec['name']}.json").write_text(json.dumps(record, indent=2))
        return

    if args.kind == "report":
        write_report(out_dir, REPO_ROOT / "docs" / "MIX_AND_MATCH.md")
        return

    if not data_path.exists():
        make_synthetic_ohlcv(data_path)
    (out_dir / "results").mkdir(exist_ok=True)

    specs = build_specs(args.kind, args)
    if args.only:
        wanted = set(args.only.split(","))
        specs = [s for s in specs if s["name"] in wanted]

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(REPO_ROOT))

    def launch(spec: dict) -> dict:
        result_file = out_dir / "results" / f"{spec['name']}.json"
        result_file.unlink(missing_ok=True)
        log_file = out_dir / "results" / f"{spec['name']}.log"
        t0 = time.time()
        try:
            with open(log_file, "w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        args.kind,
                        "--out",
                        str(out_dir),
                        "--_child",
                        json.dumps(spec),
                    ],
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return {
                "name": spec["name"],
                "ok": False,
                "exception": "TIMEOUT",
                "seconds": args.timeout,
            }
        if result_file.exists():
            return json.loads(result_file.read_text())
        tail = log_file.read_text()[-2000:] if log_file.exists() else ""
        return {
            "name": spec["name"],
            "ok": False,
            "exception": "CRASH (no result)",
            "traceback": tail,
            "seconds": round(time.time() - t0, 1),
        }

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for rec in pool.map(launch, specs):
            records.append(rec)
            status = "PASS" if rec.get("ok") else "FAIL"
            detail = rec.get("exception") or "; ".join(rec.get("problems", []))
            print(
                f"[{status}] {rec['name']:<45} {rec.get('seconds', 0):>7}s  {detail[:200]}",
                flush=True,
            )

    n_pass = sum(r.get("ok", False) for r in records)
    print(f"\n{n_pass}/{len(records)} passed")
    summary = out_dir / f"summary_{args.kind}.json"
    summary.write_text(json.dumps(records, indent=2))
    sys.exit(0 if n_pass == len(records) else 1)


if __name__ == "__main__":
    main()
