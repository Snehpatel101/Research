"""Regression tests for the Phase 116 code-review findings.

- OOF generation only survives out-of-memory; any other error propagates
  instead of silently dropping the model from stacking.
- AdapterFactory resolves the per-model contract sequence length when
  PipelineConfig.sequence_length is left at None (the default).
- ridge_meta config surfaces use its real parameters (C, class_weight).
- Ensemble bundles written before the ridge_meta change refuse to load.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.training.services import oof_generation
from src.models.training.services.oof_generation import OOFGenerationService, OOFRequest


def _request() -> OOFRequest:
    return OOFRequest(model_name="xgboost", horizon=5, prepared_data=SimpleNamespace())  # type: ignore[arg-type]


class TestOOFErrorsSurface:
    def test_non_oom_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(self: Any, request: OOFRequest) -> None:
            raise ValueError("label_spans has 10 samples but X has 12")

        monkeypatch.setattr(OOFGenerationService, "_generate_oof_inner", _fail)
        with pytest.raises(ValueError, match="label_spans"):
            OOFGenerationService().generate_oof(_request())

    def test_non_oom_runtime_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(self: Any, request: OOFRequest) -> None:
            raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")

        monkeypatch.setattr(OOFGenerationService, "_generate_oof_inner", _fail)
        with pytest.raises(RuntimeError, match="shapes"):
            OOFGenerationService().generate_oof(_request())

    def test_persistent_oom_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        def _oom(self: Any, request: OOFRequest) -> None:
            calls.append(1)
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        monkeypatch.setattr(OOFGenerationService, "_generate_oof_inner", _oom)
        assert OOFGenerationService().generate_oof(_request()) is None
        assert len(calls) == 3  # first try, GPU retry, CPU fallback

    def test_oom_then_success_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        calls: list[int] = []

        def _flaky(self: Any, request: OOFRequest) -> object:
            calls.append(1)
            if len(calls) == 1:
                raise MemoryError
            return sentinel

        monkeypatch.setattr(OOFGenerationService, "_generate_oof_inner", _flaky)
        assert OOFGenerationService().generate_oof(_request()) is sentinel

    def test_oom_classifier(self) -> None:
        assert oof_generation._is_out_of_memory(RuntimeError("CUDA out of memory"))
        assert oof_generation._is_out_of_memory(MemoryError())
        assert not oof_generation._is_out_of_memory(RuntimeError("cuDNN error"))
        assert not oof_generation._is_out_of_memory(ValueError("out of memory"))


class TestAdapterFactorySequenceLength:
    def test_prepare_data_lstm_uses_contract_length(self, tmp_path: Path) -> None:
        from src.core.config import PipelineConfig
        from src.core.contracts import get_model_contract
        from src.data.adapters import AdapterFactory

        cfg = PipelineConfig(
            symbol="MES", data_path=tmp_path / "x.parquet", output_dir=tmp_path / "out"
        )
        assert cfg.sequence_length is None
        n = 200
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "f1": rng.normal(size=n),
                "f2": rng.normal(size=n),
                "label_h20": rng.choice([-1, 0, 1], size=n),
                "sample_weight_h20": 1.0,
            },
            index=pd.date_range("2024-01-02", periods=n, freq="5min"),
        )
        factory = AdapterFactory(cfg)
        result = factory.prepare_data("lstm", df)
        seq_len = int(get_model_contract("lstm").sequence_length)
        assert result.X.ndim == 3
        assert result.X.shape[1] == seq_len
        assert factory.get_model_info("lstm")["sequence_length"] == seq_len

    def test_explicit_override_wins(self, tmp_path: Path) -> None:
        from src.core.config import PipelineConfig
        from src.data.adapters import AdapterFactory

        cfg = PipelineConfig(
            symbol="MES",
            data_path=tmp_path / "x.parquet",
            output_dir=tmp_path / "out",
            sequence_length=16,
        )
        assert AdapterFactory(cfg).get_model_info("gru")["sequence_length"] == 16


class TestRidgeMetaConfig:
    def test_param_space_matches_estimator(self) -> None:
        from src.validation.cv.param_spaces import PARAM_SPACES

        space = PARAM_SPACES["ridge_meta"]
        assert set(space) == {"C", "class_weight"}
        assert space["class_weight"]["choices"] == [None, "balanced"]

    def test_meta_learner_config_maps_to_c(self) -> None:
        from src.models.ensemble import MetaLearnerConfig, get_meta_learner

        params = MetaLearnerConfig(name="ridge_meta", C=0.5).to_dict()
        assert params == {"C": 0.5, "class_weight": None}
        meta = get_meta_learner("ridge_meta", **params)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(120, 6))
        y = rng.choice([-1, 0, 1], size=120)
        meta.fit(X[:90], y[:90], X[90:], y[90:])
        assert meta.predict_proba(X[90:]).shape == (30, 3)


class TestEnsembleBundleVersion:
    def test_old_bundle_refuses_to_load(self, tmp_path: Path) -> None:
        from src.inference.ensemble_bundle import (
            ENSEMBLE_BUNDLE_VERSION,
            ENSEMBLE_MANIFEST_FILE,
            EnsembleBundle,
        )

        assert ENSEMBLE_BUNDLE_VERSION == "2.0.0"
        (tmp_path / ENSEMBLE_MANIFEST_FILE).write_text(
            json.dumps({"version": "1.0.0", "files": []})
        )
        with pytest.raises(ValueError, match="1.0.0.*predates.*Retrain"):
            EnsembleBundle.load(tmp_path)
