"""
Determinism: the same config, seed and data produce bit-identical outputs.

Two identical CPU runs (xgboost + logistic, stacking ensemble, one horizon, no
Optuna, local tracking) execute in SEPARATE interpreters with DIFFERENT
``PYTHONHASHSEED`` values, so any dependence on set/dict ordering of strings,
unseeded global RNG state or import-order side effects shows up as a
difference. Compared bit for bit (``np.array_equal``, no tolerance):

- every model's out-of-fold probabilities and class predictions
- the stacking meta-learner's holdout predictions (what the backtest replays)
- the deployed bundles' ``predict_from_raw`` output (ensemble and each model)
- every reported metric (timing excluded)

Also checks that the non-default seed reached every model and the
meta-learner, and that both runs share their config hash and data fingerprint
in ``run_manifest.json``, which the deploy manifest references.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.e2e.determinism_driver import FEATURE_SELECTION_MODELS, MODELS, SEED
from tests.helpers import REPO_ROOT, make_intraday_ohlcv

pytestmark = pytest.mark.slow

RUN_TIMEOUT_SECONDS = 900


def _run_driver(data_path: Path, output_root: Path, hash_seed: int, *mode: str) -> None:
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed), "OMP_NUM_THREADS": "1"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.e2e.determinism_driver",
            str(data_path),
            str(output_root),
            *mode,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
        check=False,
    )
    assert proc.returncode == 0, f"run failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"


def _run(data_path: Path, output_root: Path, hash_seed: int) -> tuple[dict[str, Any], Any]:
    _run_driver(data_path, output_root, hash_seed)
    facts = json.loads((output_root / "facts.json").read_text())
    return facts, np.load(output_root / "arrays.npz")


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "mes_5min.parquet"
    make_intraday_ohlcv(2500, seed=7).to_parquet(path)
    return path


@pytest.fixture(scope="module")
def two_runs(data_path: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any]:
    # Separate output roots: the second run must not reuse the first's feature cache
    run_a = _run(data_path, tmp_path_factory.mktemp("run_a"), hash_seed=1)
    run_b = _run(data_path, tmp_path_factory.mktemp("run_b"), hash_seed=2)
    return run_a, run_b


def _assert_bit_identical(a: Any, b: Any, prefix: str) -> None:
    keys = sorted(k for k in a.files if k.startswith(prefix))
    assert keys, f"no arrays with prefix {prefix}"
    assert keys == sorted(k for k in b.files if k.startswith(prefix))
    for key in keys:
        assert a[key].shape == b[key].shape, key
        assert np.array_equal(
            a[key], b[key], equal_nan=a[key].dtype.kind == "f"
        ), f"{key} differs between identical runs"


def _json(value: Any) -> str:
    # Text comparison: NaN metrics compare equal as "NaN"
    return json.dumps(value, sort_keys=True, default=str)


class TestBitIdenticalOutputs:
    def test_oof_predictions(self, two_runs: tuple[Any, Any]) -> None:
        (_, arrays_a), (_, arrays_b) = two_runs
        _assert_bit_identical(arrays_a, arrays_b, "oof_proba__")
        _assert_bit_identical(arrays_a, arrays_b, "oof_pred__")
        assert {k.split("__")[1] for k in arrays_a.files if k.startswith("oof_proba__")} == {
            f"{m}_h5" for m in MODELS
        }

    def test_ensemble_holdout_predictions(self, two_runs: tuple[Any, Any]) -> None:
        (_, arrays_a), (_, arrays_b) = two_runs
        _assert_bit_identical(arrays_a, arrays_b, "holdout__")

    def test_deployed_bundle_predictions(self, two_runs: tuple[Any, Any]) -> None:
        (_, arrays_a), (_, arrays_b) = two_runs
        _assert_bit_identical(arrays_a, arrays_b, "deploy__")
        assert np.isfinite(arrays_a["deploy__primary"]).all()

    def test_metrics(self, two_runs: tuple[Any, Any]) -> None:
        (facts_a, _), (facts_b, _) = two_runs
        for section in ("metrics", "ensemble_metrics", "backtest_metrics"):
            assert facts_a[section], f"{section} is empty"
            assert _json(facts_a[section]) == _json(facts_b[section]), section


class TestSeedAndProvenance:
    def test_run_seed_reaches_every_model(self, two_runs: tuple[Any, Any]) -> None:
        (facts, _), _ = two_runs
        assert facts["seeds"] == {**{f"{m}_h5": SEED for m in MODELS}, "meta_learner": SEED}

    def test_manifests_share_identity(self, two_runs: tuple[Any, Any]) -> None:
        (facts_a, _), (facts_b, _) = two_runs
        man_a, man_b = facts_a["manifest"], facts_b["manifest"]
        assert man_a["status"] == man_b["status"] == "success"
        assert man_a["config_hash"] == man_b["config_hash"]
        assert man_a["data_sha256"] == man_b["data_sha256"] is not None
        assert man_a["data"] == man_b["data"]
        assert man_a["data"]["n_rows"] == 2500
        assert man_a["results_metrics"] == [f"{m}_h5" for m in sorted(MODELS)]

    def test_deploy_manifest_references_run_manifest(self, two_runs: tuple[Any, Any]) -> None:
        (facts, _), _ = two_runs
        deploy = facts["deploy"]
        assert deploy["valid"]
        reference = deploy["run_manifest"]
        assert reference["path"] == "../run_manifest.json"
        assert reference["provenance_sha256"] == deploy["provenance_sha256"]
        assert reference["config_hash"] == facts["manifest"]["config_hash"]
        assert reference["data_sha256"] == facts["manifest"]["data_sha256"]

    def test_tracking_parent_with_one_child_per_model(self, two_runs: tuple[Any, Any]) -> None:
        (facts, _), _ = two_runs
        tracking = facts["tracking"]
        assert tracking["parent_status"] == "FINISHED"
        assert tracking["children"] == sorted(MODELS)
        assert tracking["children_status"] == ["FINISHED"]


def test_live_feature_selection_identical_across_hash_seeds(
    data_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The MDA ranking (names, order and exact values) and every model's feature
    set are the same in interpreters with different PYTHONHASHSEEDs, including
    models whose contract cap cuts into near-tied features."""
    selections = []
    for hash_seed in (1, 3):
        out = tmp_path_factory.mktemp(f"features_{hash_seed}")
        _run_driver(data_path, out, hash_seed, "features")
        selections.append(json.loads((out / "features.json").read_text()))
    first, second = selections
    assert first["ranking"] == second["ranking"]
    assert first["per_model_features"] == second["per_model_features"]
    capped = [m for m in FEATURE_SELECTION_MODELS if m in ("logistic", "svm")]
    for model in capped:
        assert len(first["per_model_features"][model]) < len(first["ranking"]), model
