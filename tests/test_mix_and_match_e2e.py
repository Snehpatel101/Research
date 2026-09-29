"""
End-to-end mix-and-match tests: MLFactory.run -> ensemble -> backtest -> bundle ->
deploy -> reload -> predict from raw OHLCV, for representative combinations of
input ranks (2D tabular, 3D sequence, 4D multi-stream), meta-learners and
training modes.

Each run is checked by scripts/mix_match.py's validators:
- stacking rows pair every model's prediction with the label of the same bar
- the deployed bundle reproduces the trained model's validation probabilities
- features recomputed from raw OHLCV equal the training features
- the deploy artifact reloads and predicts finite probabilities
- UniversalInferencePipeline.from_experiment serves every model and the ensemble

The full matrix (every model, every pair, every meta-learner, every mode) is
``python scripts/mix_match.py {solo,pairs,meta,modes}``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "mix_match", REPO_ROOT / "scripts" / "mix_match.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mix_match") / "synthetic_5min.parquet"
    harness.make_synthetic_ohlcv(path)
    return path


CASES = [
    # (name, models, meta_learner, training_mode, binary_labels)
    ("tabular_pair_voting", ["xgboost", "logistic"], "voting_meta", "standard", False),
    (
        "tabular_triple_ridge",
        ["lightgbm", "catboost", "random_forest"],
        "ridge_meta",
        "standard",
        False,
    ),
    ("tabular_plus_sequence", ["xgboost", "gru"], "xgboost_meta", "standard", False),
    ("tabular_plus_multistream", ["svm", "patchtst"], "calibrated_meta", "standard", False),
    ("binary_labels", ["xgboost", "tcn"], "ridge_meta", "standard", True),
    ("walk_forward", ["xgboost", "lightgbm"], "mlp_meta", "walk_forward", False),
    ("regime_aware", ["xgboost", "lightgbm"], "ridge_meta", "regime_aware", False),
    ("meta_labeling", ["xgboost"], "ridge_meta", "meta_labeling", False),
]


@pytest.mark.parametrize(
    ("name", "models", "meta", "mode", "binary"), CASES, ids=[c[0] for c in CASES]
)
def test_combination_trains_deploys_and_serves(
    name: str,
    models: list[str],
    meta: str,
    mode: str,
    binary: bool,
    data_path: Path,
    tmp_path: Path,
) -> None:
    spec = {"name": name, "models": models, "meta": meta, "mode": mode, "binary": binary}
    record = harness.run_one(spec, data_path, tmp_path)

    assert "exception" not in record, record.get("traceback")
    assert record["ok"], record["problems"]
