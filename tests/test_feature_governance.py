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


GOV_ON = {"report": True, "n_bootstrap": 3}


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
        assert set(report["flags"]) == {"selected_unstable", "selected_label_fragile"}
        # The informative feature outranks pure noise in the live importance
        imps = {k: v["mda_importance"] for k, v in report["features"].items()}
        assert imps["f0"] > np.median(list(imps.values()))

    def test_registry_persists_across_runs_and_advances_lifecycle(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        first = self._select(frame, tmp_path / "runs" / "run_1", GOV_ON)
        registry_path = tmp_path / "runs" / "feature_registry_MES.json"
        assert registry_path.exists()
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
        assert not (tmp_path / "runs" / "feature_registry_MES.json").exists()
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
        gov = FeatureGovernance(cfg, fake_importance)
        train = frame.iloc[:TRAIN_ROWS]
        names = _feature_names(frame)
        raw = pd.Series(np.linspace(1, 0, len(names)), index=names)
        report = gov.run(
            train,
            label_col="label_h5",
            candidates=names,
            raw_importance=overrides.pop("raw_importance", raw),
            selected_by_model={"xgboost": names[:10]},
            top_k=10,
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
        variants = [c for c in calls if len(c["rows"]) == TRAIN_ROWS]
        assert len(variants) == 2
        base = frame["label_h5"].to_numpy()[:TRAIN_ROWS]
        for v in variants:
            assert len(v["labels"]) == TRAIN_ROWS
            assert not np.array_equal(v["labels"], base)
            assert np.all(v["ends"][v["ends"] >= 0] >= np.flatnonzero(v["ends"] >= 0))
        assert report["label_perturbation"]["status"] == "ok"

    def test_perturbation_skipped_without_mda_baseline_or_barriers(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        report, _ = self._run(frame, tmp_path, raw_importance=None)
        assert report["ranking_method"] == "variance"
        assert report["label_perturbation"]["status"].startswith("skipped")

        cfg = _pipeline_config(tmp_path / "runs" / "run_y", GOV_ON)
        cfg.label_barriers = {}
        names = _feature_names(frame)
        out = FeatureGovernance(cfg, lambda *a: pd.Series(1.0, index=names)).run(
            frame.iloc[:TRAIN_ROWS],
            label_col="label_h5",
            candidates=names,
            raw_importance=pd.Series(1.0, index=names),
            selected_by_model={"xgboost": names},
            top_k=10,
        )
        assert out["label_perturbation"]["status"].startswith("skipped")

    def test_run_does_not_mutate_the_frame(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        train = frame.iloc[:TRAIN_ROWS].copy()
        before = train.copy()
        cfg = _pipeline_config(tmp_path / "runs" / "run_z", GOV_ON)
        names = _feature_names(frame)
        FeatureGovernance(cfg, lambda *a: pd.Series(np.arange(len(names)), index=names)).run(
            train,
            label_col="label_h5",
            candidates=names,
            raw_importance=pd.Series(np.arange(len(names), dtype=float), index=names),
            selected_by_model={"xgboost": names},
            top_k=10,
        )
        pd.testing.assert_frame_equal(train, before)

    def test_default_registry_path_is_shared_by_runs_of_a_symbol(self, tmp_path: Path) -> None:
        a = default_registry_path(_pipeline_config(tmp_path / "runs" / "run_a"))
        b = default_registry_path(_pipeline_config(tmp_path / "runs" / "run_b"))
        assert a == b == tmp_path / "runs" / "feature_registry_MES.json"


# ---------------------------------------------------------------------------
# Full factory run (tiny synthetic bars, xgboost, one horizon)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestGovernanceFactoryE2E:
    def _run(self, data_path: Path, base: Path, governance: bool):
        from src.factory import MLFactory
        from tests.test_factory_e2e import _make_config

        cfg = _make_config(data_path, base)
        cfg.evaluation.run_backtest = False
        gov = cfg.data.features.governance
        gov.report = governance
        gov.n_bootstrap = 3
        result = MLFactory(cfg, verbose=0, enable_checkpoints=False).run()
        assert result.success
        return cfg, result

    def test_report_present_only_when_enabled_and_features_identical(self, tmp_path: Path) -> None:
        from tests.test_factory_e2e import _make_synthetic_ohlcv

        data_path = tmp_path / "bars.parquet"
        _make_synthetic_ohlcv().to_parquet(data_path)

        cfg_off, res_off = self._run(data_path, tmp_path / "off", governance=False)
        cfg_on, res_on = self._run(data_path, tmp_path / "on", governance=True)

        assert not (Path(cfg_off.output_dir) / "feature_governance").exists()
        report_path = Path(cfg_on.output_dir) / "feature_governance" / "h5.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["selection"]["models"]["xgboost"]
        assert (Path(cfg_on.output_dir).parent / "feature_registry_MES.json").exists()

        def features_of(result: Any) -> list[str]:
            trainer = next(iter(result.training_result.model_results.values())).trainer
            return list(trainer.feature_columns)

        assert features_of(res_on) == features_of(res_off)
        assert sorted(features_of(res_on)) == sorted(report["selection"]["models"]["xgboost"])
