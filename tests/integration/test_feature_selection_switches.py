"""Feature-selection switches set on ExperimentConfig reach the orchestrator's
selection (``data.features.selection_enabled``, ``data.features.mtf_max_per_timeframe``).

Each test builds the PipelineConfig through ``ExperimentConfig.to_pipeline_config``
(the factory's only path) and runs the real train-only selection, MDA included.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig
from src.core.exceptions import PreTrainingValidationError
from src.core.validation import ValidationError
from src.models.training import UnifiedTrainingOrchestrator

N_ROWS = 800
N_PER_TIMEFRAME = 10
# Column suffixes the MTF generator writes for 15min / 60min
SUFFIXES = ("_15m", "_1h")


def _frame(n_base: int = 20) -> pd.DataFrame:
    """Independent features (base + two MTF timeframes); the label depends on two MTF ones."""
    rng = np.random.default_rng(3)
    names = [f"base{i}" for i in range(n_base)]
    names += [f"f{i}{s}" for s in SUFFIXES for i in range(N_PER_TIMEFRAME)]
    df = pd.DataFrame(rng.normal(size=(N_ROWS, len(names))), columns=names)
    signal = df["f7_15m"] + df["f3_1h"] + 0.5 * rng.normal(size=N_ROWS)
    df["label_h5"] = np.digitize(signal, [-0.6, 0.6]) - 1
    return df


def _orchestrator(tmp_path: Path, **features) -> UnifiedTrainingOrchestrator:
    cfg = ExperimentConfig(run_id="r", output_dir=tmp_path, random_seed=11)
    cfg.training.models = ["logistic"]
    cfg.training.horizons = [5]
    cfg.data.mtf.timeframes = ["15min", "60min"]
    for key, value in features.items():
        if key == "mtf_enabled":
            cfg.data.mtf.enabled = value
        else:
            setattr(cfg.data.features, key, value)
    assert cfg.validate() == []
    return UnifiedTrainingOrchestrator(cfg.to_pipeline_config(cv_gaps=(15, 30)))


def _per_timeframe(features: list[str]) -> dict[str, int]:
    return {s: sum(f.endswith(s) for f in features) for s in SUFFIXES}


class TestTimeframeBudget:
    def test_budget_caps_each_timeframe_and_keeps_the_informative_features(
        self, tmp_path: Path
    ) -> None:
        orch = _orchestrator(tmp_path, mtf_max_per_timeframe=3)
        orch._run_feature_selection_on_train_data(_frame())
        selected = orch._per_model_features["logistic"]
        assert _per_timeframe(selected) == {"_15m": 3, "_1h": 3}
        # Top of the (train-only, purged) MDA ranking survives the budget
        assert {"f7_15m", "f3_1h"} <= set(selected)
        # Base-timeframe features are never budgeted
        assert {f"base{i}" for i in range(20)} <= set(selected)

    @pytest.mark.parametrize(
        "features",
        [{}, {"mtf_max_per_timeframe": 3, "mtf_enabled": False}],
        ids=["default-off", "mtf-disabled"],
    )
    def test_no_budget_by_default_or_without_mtf(self, tmp_path: Path, features: dict) -> None:
        orch = _orchestrator(tmp_path, **features)
        orch._run_feature_selection_on_train_data(_frame())
        selected = orch._per_model_features["logistic"]
        assert _per_timeframe(selected) == {"_15m": N_PER_TIMEFRAME, "_1h": N_PER_TIMEFRAME}

    def test_governance_replay_applies_the_same_budget(self, tmp_path: Path) -> None:
        """The stability replay runs the live post-ranking code, budget included."""
        orch = _orchestrator(tmp_path, mtf_max_per_timeframe=2)
        df = _frame()
        candidates = [c for c in df.columns if c != "label_h5"]
        importance = pd.Series(np.linspace(1.0, 0.1, len(candidates)), index=candidates)
        replay = orch._replay_selection(df, candidates, importance)
        assert _per_timeframe(replay["logistic"]) == {"_15m": 2, "_1h": 2}


class TestSelectionEnabled:
    def test_off_skips_selection_and_trainer_reselection(self, tmp_path: Path) -> None:
        orch = _orchestrator(tmp_path, selection_enabled=False)
        orch._run_feature_selection_on_train_data(_frame())
        assert orch._per_model_features == {}
        assert orch._trainer_feature_selection("logistic") is False

    def test_off_with_more_features_than_the_contract_fails_loudly(self, tmp_path: Path) -> None:
        orch = _orchestrator(tmp_path, selection_enabled=False)
        df = _frame(n_base=120)  # 140 features > logistic max_features (100)
        orch._run_feature_selection_on_train_data(df)
        with pytest.raises(PreTrainingValidationError, match="selection_enabled=False"):
            orch._post_selection_contract_validation(df)

    def test_on_cuts_to_the_contract(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        orch = _orchestrator(tmp_path)
        df = _frame(n_base=120)
        # The ranking itself is covered above; a fixed one keeps this test fast
        monkeypatch.setattr(
            orch,
            "_compute_mda_ranking",
            lambda frame, names: pd.Series(np.linspace(1.0, 0.1, len(names)), index=names),
        )
        orch._run_feature_selection_on_train_data(df)
        assert len(orch._per_model_features["logistic"]) == 100
        orch._post_selection_contract_validation(df)


class TestConfigSurface:
    def test_yaml_round_trip_and_pipeline_config(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(run_id="r", output_dir=tmp_path)
        cfg.data.features.mtf_max_per_timeframe = 4
        cfg.save_yaml(tmp_path / "c.yaml")
        loaded = ExperimentConfig.from_yaml(tmp_path / "c.yaml")
        assert loaded.data.features.mtf_max_per_timeframe == 4
        assert loaded.to_pipeline_config(cv_gaps=(1, 1)).mtf_max_per_timeframe == 4

    def test_budget_is_result_affecting(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(run_id="r", output_dir=tmp_path)
        before = cfg.config_hash()
        cfg.data.features.mtf_max_per_timeframe = 4
        assert cfg.config_hash() != before

    @pytest.mark.parametrize("bad", [0, -1, True, 2.5, "3"])
    def test_invalid_budget_rejected(self, tmp_path: Path, bad: object) -> None:
        cfg = ExperimentConfig(run_id="r", output_dir=tmp_path)
        cfg.data.features.mtf_max_per_timeframe = bad  # type: ignore[assignment]
        assert any("mtf_max_per_timeframe" in issue for issue in cfg.validate())
        with pytest.raises(ValidationError, match="mtf_max_per_timeframe"):
            cfg.to_pipeline_config(cv_gaps=(1, 1))

    def test_regime_conditional_key_is_ignored_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The deleted regime blend switch never existed on ExperimentConfig;
        a YAML carrying it loads with a warning instead of silently doing nothing."""
        cfg = ExperimentConfig.from_dict(
            {"output_dir": str(tmp_path), "data": {"features": {"regime_conditional": True}}}
        )
        assert not hasattr(cfg.data.features, "regime_conditional")
        assert "regime_conditional" in caplog.text
