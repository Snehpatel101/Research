"""
Phase 117: feature governance as opt-in diagnostics.

Guarantees pinned here:
- config surface (``data.features.governance``) round-trips, tolerates unknown keys,
  and reaches PipelineConfig;
- the report is produced only when enabled and never changes the selection;
- diagnostics only see train rows and label spans stay valid inside blocks;
- the registry persists across runs and advances lifecycles;
- a failing diagnostic never interrupts training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config.data import FeatureGovernanceConfig
from src.config.experiment import ExperimentConfig
from src.core import PipelineConfig
from src.core.label_spans import label_end_positions
from src.data.labeling import TripleBarrierConfig, TripleBarrierLabeler
from src.models.training.feature_governance import (
    FeatureGovernance,
    default_registry_path,
    experiment_context,
)
from src.models.training.feature_selection import FeatureSelectionMixin
from src.optimization.feature_selection.registry import FeatureRegistry

N_ROWS = 3000
TRAIN_ROWS = 2100
N_FEATURES = 45  # xgboost's contract needs >= 40 features


def _make_frame(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """OHLC random walk + real triple-barrier label/label-end + features (f0, f1 informative)."""
    rng = np.random.default_rng(seed)
    close = 5000.0 + np.cumsum(rng.normal(0, 2.0, n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.5, n)
    eps = np.abs(rng.normal(0, 0.5, n)) + 0.25
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + eps,
            "low": np.minimum(open_, close) - eps,
            "close": close,
            "volume": rng.integers(100, 5000, n).astype(float),
        }
    )
    labeler = TripleBarrierLabeler(
        TripleBarrierConfig(
            horizon=12,
            upper_mult=1.5,
            lower_mult=1.0,
            atr_period=14,
            atr_column=None,
            symbol="MES",
            cost_calibration_fraction=0.7,
        )
    )
    result = labeler.compute_labels(df, horizon=12)
    labels = np.asarray(result.labels)
    feats = {f"f{i}": rng.normal(size=n) for i in range(N_FEATURES)}
    feats["f0"] = labels * 0.8 + rng.normal(size=n)
    feats["f1"] = labels * 0.5 + rng.normal(size=n)
    out = pd.concat([df, pd.DataFrame(feats)], axis=1)
    out["label_h5"] = labels
    out["label_end_h5"] = label_end_positions(labels, result.metadata["bars_to_hit"])
    return out


def _feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def _pipeline_config(
    output_dir: Path, governance: dict[str, Any] | None = None, **kw: Any
) -> PipelineConfig:
    return PipelineConfig(
        symbol="MES",
        data_path="unused.parquet",
        output_dir=output_dir,
        models=["xgboost"],
        horizons=[5],
        purge_bars=15,
        embargo_bars=30,
        governance=governance or {},
        label_barriers={"5": {"k_up": 1.5, "k_down": 1.0, "max_bars": 12}},
        **kw,
    )


class _Host(FeatureSelectionMixin):
    """Just enough of the orchestrator for the selection pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._per_model_features: dict[str, list[str]] = {}
        self._all_feature_names: list[str] = []


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _make_frame()


@pytest.fixture(autouse=True)
def single_threaded_selection_forest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the selection's own MDA bit-reproducible so on/off runs can be compared exactly.

    The clustered MDA scores permutations with a multi-threaded RandomForest, whose
    tree-by-tree probability sums accumulate in thread-completion order. That float noise
    (~1e-16) is amplified by the within-cluster tie-break normalisation, so two calls on
    IDENTICAL data can rank near-tied features differently -- with governance off as well
    (a pre-existing property of the selection, independent of these diagnostics). One
    thread removes it; anything that still differs between an on and an off run is then
    caused by governance.
    """
    from sklearn.ensemble import RandomForestClassifier

    import src.optimization.feature_selection.walk_forward as walk_forward

    def single_threaded(*args: Any, **kwargs: Any) -> RandomForestClassifier:
        kwargs["n_jobs"] = 1
        return RandomForestClassifier(*args, **kwargs)

    monkeypatch.setattr(walk_forward, "RandomForestClassifier", single_threaded)


GOV_ON = {"report": True, "n_bootstrap": 3}


def _registry_files(tmp_path: Path) -> list[Path]:
    """Default-location registries created under ``<tmp>/runs``."""
    return sorted((tmp_path / "runs").glob("feature_registry_MES_*.json"))


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------
class TestGovernanceConfig:
    def test_default_is_off(self) -> None:
        gov = ExperimentConfig().data.features.governance
        assert gov.report is False
        assert gov.to_dict()["report"] is False

    def test_roundtrip_dict_and_yaml(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(run_id="fixed")
        gov = cfg.data.features.governance
        gov.report = True
        gov.n_bootstrap = 5
        gov.barrier_scales = [0.5, 1.5, 2.0]
        gov.registry_path = "reg.json"

        restored = ExperimentConfig.from_dict(cfg.to_dict())
        assert restored.data.features.governance == gov

        path = tmp_path / "cfg.yaml"
        cfg.save_yaml(path)
        assert ExperimentConfig.from_yaml(path).data.features.governance == gov

    def test_old_configs_without_governance_load_with_defaults(self) -> None:
        data = ExperimentConfig(run_id="fixed").to_dict()
        del data["data"]["features"]["governance"]
        restored = ExperimentConfig.from_dict(data)
        assert restored.data.features.governance == FeatureGovernanceConfig()

    def test_unknown_governance_keys_are_tolerated(self, caplog: pytest.LogCaptureFixture) -> None:
        data = ExperimentConfig(run_id="fixed").to_dict()
        data["data"]["features"]["governance"]["retired_option"] = 1
        with caplog.at_level(logging.WARNING):
            restored = ExperimentConfig.from_dict(data)
        assert restored.data.features.governance.report is False
        assert "retired_option" in caplog.text

    @pytest.mark.parametrize(
        "bad",
        [
            {"n_bootstrap": 0},
            {"stability_threshold": 0.0},
            {"window_fraction": 1.5},
            {"barrier_scales": [1.0, -0.5]},
            {"max_degraded_runs": 0},
        ],
    )
    def test_invalid_values_rejected(self, bad: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="governance"):
            FeatureGovernanceConfig(**bad)

    def test_reaches_pipeline_config_and_survives_its_json_roundtrip(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(run_id="fixed", output_dir=tmp_path)
        cfg.training.horizons = [5, 20]
        cfg.data.features.governance.report = True
        pc = cfg.to_pipeline_config(cv_gaps=(15, 30))
        assert pc.governance["report"] is True
        assert set(pc.label_barriers) == {"5", "20"}
        assert pc.label_barriers["5"] == {"k_up": 1.5, "k_down": 1.0, "max_bars": 12}

        loaded = PipelineConfig.load(pc.save(tmp_path / "pc.json"))
        assert loaded.governance == pc.governance
        assert loaded.label_barriers == pc.label_barriers

    def test_disabled_config_adds_nothing_heavy_to_pipeline_config(self, tmp_path: Path) -> None:
        pc = ExperimentConfig(run_id="fixed", output_dir=tmp_path).to_pipeline_config(
            cv_gaps=(15, 30)
        )
        assert pc.governance["report"] is False
        assert pc.label_barriers == {}


# ---------------------------------------------------------------------------
# Selection pipeline: report present when enabled, absent when disabled, selection identical
# ---------------------------------------------------------------------------
class TestGovernanceInSelectionPipeline:
    def _select(self, frame: pd.DataFrame, out_dir: Path, governance: dict[str, Any] | None):
        host = _Host(_pipeline_config(out_dir, governance))
        host._run_feature_selection_pipeline(frame.iloc[:TRAIN_ROWS], _feature_names(frame))
        return host

    def test_report_only_when_enabled_and_selection_is_identical(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        off = self._select(frame, tmp_path / "runs" / "run_off", None)
        on = self._select(frame, tmp_path / "runs" / "run_on", GOV_ON)

        assert on._per_model_features == off._per_model_features
        assert on._per_model_features["xgboost"], "selection produced no features"

        assert not (tmp_path / "runs" / "run_off" / "feature_governance").exists()
        report_path = tmp_path / "runs" / "run_on" / "feature_governance" / "h5.json"
        assert report_path.exists()

    def test_report_content(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        host = self._select(frame, tmp_path / "runs" / "run_1", GOV_ON)
        report = json.loads((tmp_path / "runs/run_1/feature_governance/h5.json").read_text())

        selected = host._per_model_features["xgboost"]
        assert report["ranking_label"] == "label_h5"
        assert report["ranking_method"] == "MDA"
        assert report["train_rows"] == TRAIN_ROWS
        assert report["n_candidates"] == N_FEATURES
        assert report["selection"]["models"]["xgboost"] == selected
        assert report["stability"]["status"] == "ok"
        assert report["label_perturbation"]["status"] == "ok"
        assert report["label_perturbation"]["variants"] == ["barriers_x0.75", "barriers_x1.25"]
        assert set(report["features"]) == set(_feature_names(frame))
        entry = report["features"]["f0"]
        assert entry["selected"] is (("f0") in selected)
        assert entry["selection_frequency"] is not None
        assert entry["label_robust"] is not None
        assert set(report["flags"]) == {
            "selected_unstable",
            "selected_unstable_by_model",
            "selected_label_fragile",
        }
        assert set(entry["model_frequency"]) == {"xgboost"}
        assert report["stability"]["n_blocks_used"] == 3
        assert report["registry"]["context"]["ranking_label"] == "label_h5"
        # The informative feature outranks pure noise in the live importance
        imps = {k: v["mda_importance"] for k, v in report["features"].items()}
        assert imps["f0"] > np.median(list(imps.values()))

    def test_registry_persists_across_runs_and_advances_lifecycle(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        first = self._select(frame, tmp_path / "runs" / "run_1", GOV_ON)
        (registry_path,) = _registry_files(tmp_path)
        selected = set(first._per_model_features["xgboost"])

        registry = FeatureRegistry(registry_path)
        assert len(registry) == N_FEATURES
        assert {r.feature_name for r in registry.get_by_state("selected")} == selected

        self._select(frame, tmp_path / "runs" / "run_2", GOV_ON)
        reopened = FeatureRegistry(registry_path)
        promoted = {r.feature_name for r in reopened.get_by_state("active")}
        assert promoted and promoted <= selected
        report = json.loads((tmp_path / "runs/run_2/feature_governance/h5.json").read_text())
        assert report["registry"]["n_new"] == 0
        assert {t["to"] for t in report["registry"]["transitions"]} == {"active"}
        assert all(
            r.last_run_id == "run_2" for r in reopened.all_features()
        ), "every feature is judged each run"

    def test_registry_can_be_switched_off_or_relocated(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        self._select(frame, tmp_path / "runs" / "r1", {**GOV_ON, "registry": False})
        assert _registry_files(tmp_path) == []
        custom = tmp_path / "elsewhere" / "reg.json"
        self._select(frame, tmp_path / "runs" / "r2", {**GOV_ON, "registry_path": str(custom)})
        assert custom.exists()

    def test_diagnostics_can_be_switched_off_individually(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        self._select(
            frame,
            tmp_path / "runs" / "r1",
            {**GOV_ON, "bootstrap_stability": False, "label_perturbation": False},
        )
        report = json.loads((tmp_path / "runs/r1/feature_governance/h5.json").read_text())
        assert report["stability"]["status"] == "disabled"
        assert report["label_perturbation"]["status"] == "disabled"
        assert report["features"]["f0"]["stable"] is None

    def test_failing_governance_never_interrupts_selection(
        self,
        frame: pd.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        baseline = self._select(frame, tmp_path / "runs" / "off", None)

        def boom(self: FeatureGovernance, *a: Any, **k: Any) -> None:
            raise RuntimeError("diagnostic exploded")

        monkeypatch.setattr(FeatureGovernance, "run", boom)
        with caplog.at_level(logging.WARNING):
            host = self._select(frame, tmp_path / "runs" / "on", GOV_ON)
        assert host._per_model_features == baseline._per_model_features
        assert "diagnostic exploded" in caplog.text


# ---------------------------------------------------------------------------
# FeatureGovernance internals: train-only rows, valid label spans, relabeling
# ---------------------------------------------------------------------------
class TestFeatureGovernanceInternals:
    def _run(self, frame: pd.DataFrame, tmp_path: Path, **overrides: Any):
        calls: list[dict[str, Any]] = []

        def fake_importance(df, features, labels, ends):
            calls.append({"rows": df.index.to_numpy(), "labels": labels.copy(), "ends": ends})
            rng = np.random.default_rng(len(calls))
            imp = pd.Series(rng.uniform(0, 0.05, len(features)), index=features)
            imp["f0"] = 1.0
            return imp.sort_values(ascending=False)

        cfg = _pipeline_config(
            tmp_path / "runs" / "run_x", {**GOV_ON, "n_bootstrap": 4, **overrides.pop("gov", {})}
        )
        gov = FeatureGovernance(cfg, fake_importance, _Host(cfg)._replay_selection)
        train = frame.iloc[:TRAIN_ROWS]
        names = _feature_names(frame)
        raw = pd.Series(np.linspace(1, 0, len(names)), index=names)
        report = gov.run(
            train,
            label_col="label_h5",
            candidates=names,
            raw_importance=overrides.pop("raw_importance", raw),
            selected_by_model={"xgboost": names[:10]},
        )
        return report, calls

    def test_only_train_rows_are_touched(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        _report, calls = self._run(frame, tmp_path)
        assert calls, "expected stability + perturbation rankings"
        assert max(int(c["rows"].max()) for c in calls) < TRAIN_ROWS

    def test_block_label_ends_are_in_block_coordinates(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        _report, calls = self._run(frame, tmp_path, gov={"bootstrap_stability": True})
        blocks = [c for c in calls if len(c["rows"]) < TRAIN_ROWS]
        assert len(blocks) == 4
        for c in blocks:
            ends = c["ends"]
            positions = np.arange(len(ends))
            known = ends >= 0
            assert len(c["rows"]) == int(TRAIN_ROWS * 0.5)
            assert np.all(ends[known] >= positions[known]), "a label cannot resolve before its bar"
            assert ends.max() < len(ends) + 12  # spans reach at most max_bars past the block

    def test_perturbed_labels_differ_and_are_full_train_length(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        report, calls = self._run(frame, tmp_path)
        relabeled = [c for c in calls if len(c["rows"]) == TRAIN_ROWS]
        assert len(relabeled) == 3  # x1.0 control + the two scaled variants
        control, *variants = relabeled
        for v in relabeled:
            assert len(v["labels"]) == TRAIN_ROWS
            assert np.all(v["ends"][v["ends"] >= 0] >= np.flatnonzero(v["ends"] >= 0))
        for v in variants:
            assert not np.array_equal(v["labels"], control["labels"])
        assert report["label_perturbation"]["status"] == "ok"
        assert report["label_perturbation"]["control_rank_correlation"] is not None

    def test_perturbation_skipped_without_mda_baseline_or_barriers(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        report, _ = self._run(frame, tmp_path, raw_importance=None)
        assert report["ranking_method"] == "variance"
        assert report["label_perturbation"]["status"].startswith("skipped")

        cfg = _pipeline_config(tmp_path / "runs" / "run_y", GOV_ON)
        cfg.label_barriers = {}
        names = _feature_names(frame)
        out = FeatureGovernance(
            cfg, lambda *a: pd.Series(1.0, index=names), _Host(cfg)._replay_selection
        ).run(
            frame.iloc[:TRAIN_ROWS],
            label_col="label_h5",
            candidates=names,
            raw_importance=pd.Series(1.0, index=names),
            selected_by_model={"xgboost": names},
        )
        assert out["label_perturbation"]["status"].startswith("skipped")

    def test_run_does_not_mutate_the_frame(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        train = frame.iloc[:TRAIN_ROWS].copy()
        before = train.copy()
        cfg = _pipeline_config(tmp_path / "runs" / "run_z", GOV_ON)
        names = _feature_names(frame)
        FeatureGovernance(
            cfg,
            lambda *a: pd.Series(np.arange(len(names)), index=names),
            _Host(cfg)._replay_selection,
        ).run(
            train,
            label_col="label_h5",
            candidates=names,
            raw_importance=pd.Series(np.arange(len(names), dtype=float), index=names),
            selected_by_model={"xgboost": names},
        )
        pd.testing.assert_frame_equal(train, before)

    def test_default_registry_path_is_per_context_and_shared_by_its_runs(
        self, tmp_path: Path
    ) -> None:
        def path_for(run: str, **kw: Any) -> Path:
            cfg = _pipeline_config(tmp_path / "runs" / run, **kw)
            return default_registry_path(cfg, experiment_context(cfg, "label_h5"))

        a, b = path_for("run_a"), path_for("run_b")
        assert (
            a == b and a.parent == tmp_path / "runs" and a.name.startswith("feature_registry_MES_")
        )
        assert path_for("run_c", bar_timeframe="15min") != a
        assert path_for("run_d", mtf_timeframes=["5min"]) != a
        cfg = _pipeline_config(tmp_path / "runs" / "run_e")
        assert default_registry_path(cfg, experiment_context(cfg, "label_h20")) != a

    def test_experiment_context_fields(self, tmp_path: Path) -> None:
        cfg = _pipeline_config(
            tmp_path / "runs" / "r", bar_timeframe="5min", mtf_timeframes=["15min", "5min"]
        )
        assert experiment_context(cfg, "label_h5") == {
            "symbol": "MES",
            "bar_timeframe": "5min",
            "ranking_label": "label_h5",
            "mtf_timeframes": ["15min", "5min"],
            "models": ["xgboost"],
        }

    def test_stability_replays_the_real_selection_not_the_raw_top_k(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Decorrelation keeps lower-ranked representatives; they must not look unstable."""
        train = frame.iloc[:TRAIN_ROWS].copy()
        rng = np.random.default_rng(3)
        train["f_dup"] = train["f2"] + rng.normal(0, 0.01, len(train))  # ~duplicate of f2
        names = _feature_names(train)
        order = ["f2", "f_dup"] + [n for n in names if n not in ("f2", "f_dup")]
        importance = pd.Series(np.arange(len(order), 0, -1, dtype=float), index=order)

        cfg = _pipeline_config(tmp_path / "runs" / "run_dup", {**GOV_ON, "n_bootstrap": 4})
        host = _Host(cfg)
        live = host._select_from_ranking(train, list(order), importance, "MDA")[2]["xgboost"]
        assert "f_dup" not in live and "f2" in live, "decorrelation keeps the higher-ranked twin"
        lowest = order[-1]
        assert lowest in live
        assert len(live) < order.index(lowest) + 1, "raw top-K over the selection size drops it"

        report = FeatureGovernance(
            cfg, lambda df, feats, y, ends: importance.copy(), host._replay_selection
        ).run(
            train,
            label_col="label_h5",
            candidates=list(order),
            raw_importance=importance,
            selected_by_model={"xgboost": live},
        )
        entry = report["features"][lowest]
        assert entry["selected"] and entry["stable"] is True
        assert entry["selection_frequency"] == 1.0
        assert entry["model_frequency"] == {"xgboost": 1.0}
        assert lowest not in report["flags"]["selected_unstable"]
        assert report["flags"]["selected_unstable_by_model"] == {"xgboost": []}
        # The dropped twin is never kept, and is not flagged (it was never selected)
        assert report["features"]["f_dup"]["selection_frequency"] == 0.0
        assert "f_dup" not in report["flags"]["selected_unstable"]

    def test_governance_receives_only_the_train_prefix(
        self, frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def spy(self: FeatureGovernance, df_train: pd.DataFrame, **kw: Any) -> dict[str, Any]:
            seen["rows"] = len(df_train)
            seen["last_index"] = int(df_train.index[-1])
            return {}

        monkeypatch.setattr(FeatureGovernance, "run", spy)
        cfg = _pipeline_config(tmp_path / "runs" / "run_spy", GOV_ON, train_ratio=0.7)
        host = _Host(cfg)
        host._run_feature_selection_on_train_data(frame)
        assert len(frame) == N_ROWS
        assert seen["rows"] == int(N_ROWS * 0.7) == TRAIN_ROWS
        assert seen["last_index"] == TRAIN_ROWS - 1

    def test_global_rng_state_is_untouched(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        import random

        def rng_state(governance: dict[str, Any] | None, run: str) -> tuple[Any, Any]:
            random.seed(11)
            np.random.seed(11)  # noqa: NPY002 (legacy global RNG is what is being guarded)
            self_ = _Host(_pipeline_config(tmp_path / "runs" / run, governance))
            self_._run_feature_selection_pipeline(frame.iloc[:TRAIN_ROWS], _feature_names(frame))
            return random.getstate(), np.random.get_state()  # noqa: NPY002

        py_off, np_off = rng_state(None, "off")
        py_on, np_on = rng_state(GOV_ON, "on")
        assert py_off == py_on
        assert np_off[2:] == np_on[2:] and np.array_equal(np_off[1], np_on[1])

    def test_rerun_of_the_same_run_id_is_a_registry_noop(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        train = frame.iloc[:TRAIN_ROWS]
        names = _feature_names(frame)
        reg = tmp_path / "reg.json"

        def run() -> dict[str, Any]:
            cfg = _pipeline_config(
                tmp_path / "runs" / "run_same", {**GOV_ON, "registry_path": str(reg)}
            )
            host = _Host(cfg)
            return FeatureGovernance(
                cfg,
                lambda df, f, y, e: pd.Series(np.arange(len(f), 0, -1.0), index=f),
                host._replay_selection,
            ).run(
                train,
                label_col="label_h5",
                candidates=names,
                raw_importance=pd.Series(np.arange(len(names), 0, -1.0), index=names),
                selected_by_model={"xgboost": names[:40]},
            )

        first, second = run()["registry"], run()["registry"]
        assert first["status"] == "ok" and first["n_new"] == N_FEATURES
        assert second["status"] == "skipped" and "already recorded" in second["skipped_reason"]
        assert {r.state for r in FeatureRegistry(reg).all_features()} <= {"selected", "candidate"}

    def test_registry_from_a_different_context_is_not_advanced(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        reg = tmp_path / "shared.json"
        gov = {**GOV_ON, "registry_path": str(reg)}
        self_first = _Host(_pipeline_config(tmp_path / "runs" / "r1", gov))
        self_first._run_feature_selection_pipeline(frame.iloc[:TRAIN_ROWS], _feature_names(frame))
        before = json.loads(reg.read_text())

        other = _Host(_pipeline_config(tmp_path / "runs" / "r2", gov, bar_timeframe="15min"))
        other._run_feature_selection_pipeline(frame.iloc[:TRAIN_ROWS], _feature_names(frame))
        report = json.loads((tmp_path / "runs/r2/feature_governance/h5.json").read_text())
        assert report["registry"]["status"] == "skipped"
        assert "context changed" in report["registry"]["skipped_reason"]
        assert json.loads(reg.read_text()) == before

    def test_checkpoint_hash_ignores_governance_settings_only(self) -> None:
        from src.core.checkpoint import compute_config_hash

        cfg = ExperimentConfig(run_id="fixed")
        base = compute_config_hash(cfg)
        cfg.data.features.governance.report = True
        cfg.data.features.governance.n_bootstrap = 20
        assert compute_config_hash(cfg) == base, "diagnostics must not invalidate checkpoints"
        cfg.data.features.selection_enabled = False
        assert compute_config_hash(cfg) != base
        cfg.data.features.selection_enabled = True
        cfg.training.n_splits += 1
        assert compute_config_hash(cfg) != base


# ---------------------------------------------------------------------------
# Full factory run (tiny synthetic bars, xgboost, one horizon)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestGovernanceFactoryE2E:
    def _run(self, data_path: Path, base: Path, governance: bool) -> dict[str, Any]:
        """Run once and snapshot what parity is judged on (result objects can be shared/mutated)."""
        import copy

        from src.factory import MLFactory
        from tests.e2e.test_factory_e2e import _make_config

        cfg = _make_config(data_path, base)
        cfg.evaluation.run_backtest = False
        gov = cfg.data.features.governance
        gov.report = governance
        gov.n_bootstrap = 3
        result = MLFactory(cfg, verbose=0, enable_checkpoints=False).run()
        assert result.success
        model = next(iter(result.training_result.model_results.values()))
        return {
            "cfg": cfg,
            "features": list(model.trainer.feature_columns),
            "metrics": copy.deepcopy(result.metrics),
            "oof_classes": np.array(model.oof_prediction.get_class_predictions()),
        }

    def test_report_present_only_when_enabled_and_features_identical(self, tmp_path: Path) -> None:
        from tests.e2e.test_factory_e2e import N_ROWS
        from tests.helpers import make_intraday_ohlcv

        data_path = tmp_path / "bars.parquet"
        make_intraday_ohlcv(N_ROWS, seed=7).to_parquet(data_path)

        off = self._run(data_path, tmp_path / "off", governance=False)
        on = self._run(data_path, tmp_path / "on", governance=True)

        assert not (Path(off["cfg"].output_dir) / "feature_governance").exists()
        report_path = Path(on["cfg"].output_dir) / "feature_governance" / "h5.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["selection"]["models"]["xgboost"]
        assert list(Path(on["cfg"].output_dir).parent.glob("feature_registry_MES_*.json"))

        assert on["features"] == off["features"]
        assert sorted(on["features"]) == sorted(report["selection"]["models"]["xgboost"])

        # Prediction parity, not just the feature lists: the report changes nothing downstream
        assert set(on["metrics"]) == set(off["metrics"])
        for model_key, metrics_off in off["metrics"].items():
            metrics_on = on["metrics"][model_key]
            assert set(metrics_on) == set(metrics_off), f"metric keys differ for {model_key}"
            for name, value_off in metrics_off.items():
                value_on = metrics_on[name]
                if isinstance(value_off, float) and np.isnan(value_off):
                    assert np.isnan(value_on), f"{model_key}.{name}: {value_on} != {value_off}"
                else:
                    assert value_on == value_off, f"{model_key}.{name}: {value_on} != {value_off}"
        np.testing.assert_array_equal(on["oof_classes"], off["oof_classes"])
