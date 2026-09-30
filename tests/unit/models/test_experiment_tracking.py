"""Experiment tracking: backends ("none", "local", "mlflow"), parent/child runs, and
the Trainer's per-model run.

MLflow is an optional extra and is not installed in the dev environment, so the
mlflow backend is exercised against an in-memory stand-in for the MlflowClient API
(installed into ``sys.modules`` per test); the local backend is tested for real.
"""

from __future__ import annotations

import json
import sys
import types
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from src.models.tracking import (
    DisabledTracker,
    LocalTracker,
    MLflowTracker,
    TrackerConfig,
    flatten_params,
    get_tracker,
)
from src.models.tracking.mlflow_tracker import MLFLOW_INSTALL_HINT, PARENT_RUN_TAG
from tests.helpers import tiny_prepared_data

# =============================================================================
# In-memory MLflow stand-in
# =============================================================================


class FakeMlflowStore:
    """What a tracking server would hold, recorded per run."""

    def __init__(self) -> None:
        self.experiments: dict[str, str] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.clients: list[str | None] = []
        self.batch_sizes: list[int] = []


def install_fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMlflowStore:
    """Install ``mlflow``, ``mlflow.tracking`` and ``mlflow.entities`` stand-ins."""
    store = FakeMlflowStore()
    Param = namedtuple("Param", "key value")
    Metric = namedtuple("Metric", "key value timestamp step")
    RunTag = namedtuple("RunTag", "key value")

    class MlflowClient:
        def __init__(self, tracking_uri: str | None = None) -> None:
            store.clients.append(tracking_uri)

        def get_experiment_by_name(self, name: str) -> Any:
            if name not in store.experiments:
                return None
            return types.SimpleNamespace(experiment_id=store.experiments[name])

        def create_experiment(self, name: str) -> str:
            store.experiments[name] = str(len(store.experiments) + 1)
            return store.experiments[name]

        def create_run(self, experiment_id: str, tags: dict, run_name: str | None) -> Any:
            run_id = f"run{len(store.runs) + 1}"
            store.runs[run_id] = {
                "experiment_id": experiment_id,
                "run_name": run_name,
                "tags": dict(tags),
                "params": {},
                "metrics": {},
                "artifacts": [],
                "status": "RUNNING",
            }
            return types.SimpleNamespace(info=types.SimpleNamespace(run_id=run_id))

        def log_batch(self, run_id: str, metrics=(), params=(), tags=()) -> None:
            run = store.runs[run_id]
            store.batch_sizes.append(len(params) + len(metrics) + len(tags))
            for p in params:
                assert p.key not in run["params"], f"param {p.key} logged twice"
                run["params"][p.key] = p.value
            for m in metrics:
                run["metrics"][m.key] = m.value
            for t in tags:
                run["tags"][t.key] = t.value

        def log_artifact(self, run_id: str, local_path: str, artifact_path=None) -> None:
            store.runs[run_id]["artifacts"].append((Path(local_path).name, artifact_path))

        def log_artifacts(self, run_id: str, local_dir: str, artifact_path=None) -> None:
            raise AssertionError("directories are recorded by reference, never uploaded")

        def set_terminated(self, run_id: str, status: str) -> None:
            store.runs[run_id]["status"] = status

    mlflow = types.ModuleType("mlflow")
    tracking = types.ModuleType("mlflow.tracking")
    entities = types.ModuleType("mlflow.entities")
    tracking.MlflowClient = MlflowClient  # type: ignore[attr-defined]
    entities.Param, entities.Metric, entities.RunTag = Param, Metric, RunTag  # type: ignore[attr-defined]
    mlflow.tracking, mlflow.entities = tracking, entities  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities)
    return store


# =============================================================================
# Config and factory
# =============================================================================


class TestTrackerConfig:
    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown tracking backend"):
            TrackerConfig(backend="disabled")

    def test_local_root(self, tmp_path: Path) -> None:
        assert TrackerConfig(output_dir=tmp_path).local_root == tmp_path
        assert TrackerConfig(tracking_uri=str(tmp_path / "x")).local_root == tmp_path / "x"

    def test_get_tracker_backends(self, tmp_path: Path) -> None:
        assert isinstance(get_tracker(None), DisabledTracker)
        assert isinstance(get_tracker(TrackerConfig(backend="none")), DisabledTracker)
        local = get_tracker(TrackerConfig(backend="local", output_dir=tmp_path))
        assert isinstance(local, LocalTracker)

    def test_mlflow_missing_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A run that asked for MLflow never silently falls back to another backend."""
        monkeypatch.setitem(sys.modules, "mlflow", None)
        with pytest.raises(ImportError, match=MLFLOW_INSTALL_HINT.replace("[", r"\[")):
            get_tracker(TrackerConfig(backend="mlflow"))

    def test_flatten_params(self) -> None:
        flat = flatten_params({"a": 1, "b": {"c": [1, 2], "d": {"e": None}}, "f": {}})
        assert flat == {"a": 1, "b.c": [1, 2], "b.d.e": None, "f": {}}


# =============================================================================
# Local backend
# =============================================================================


class TestLocalTracker:
    def test_parent_and_child_runs(self, tmp_path: Path) -> None:
        parent = LocalTracker(
            TrackerConfig(backend="local", tracking_uri=str(tmp_path), experiment_name="exp")
        )
        parent_id = parent.start_run(run_name="factory_run", tags={"config_hash": "abc"})
        parent.log_params(flatten_params({"training": {"models": ["xgboost"]}}))

        child = LocalTracker(
            TrackerConfig(
                backend="local",
                tracking_uri=str(tmp_path),
                experiment_name="exp",
                parent_run_id=parent_id,
            )
        )
        child_id = child.start_run(run_name="xgboost_h5")
        child.log_metrics({"val_macro_f1": 0.5})
        child.end_run()
        parent.end_run(status="FAILED")

        runs = {r["run_id"]: r for r in LocalTracker.list_runs(tmp_path, "exp")}
        assert runs[child_id]["parent_run_id"] == parent_id
        assert runs[child_id]["status"] == "FINISHED"
        assert runs[parent_id]["parent_run_id"] is None
        assert runs[parent_id]["status"] == "FAILED"
        params = json.loads((tmp_path / "exp" / parent_id / "params.json").read_text())
        assert params == {"training.models": ["xgboost"]}
        tags = json.loads((tmp_path / "exp" / parent_id / "tags.json").read_text())
        assert tags == {"config_hash": "abc"}

    def test_run_ids_unique_within_a_second(self, tmp_path: Path) -> None:
        tracker = LocalTracker(TrackerConfig(backend="local", tracking_uri=str(tmp_path)))
        ids = set()
        for _ in range(5):
            ids.add(tracker.start_run(run_name="same_name"))
            tracker.end_run()
        assert len(ids) == 5

    def test_artifacts_recorded_by_reference(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "checkpoints"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"\0" * 1024)
        root = tmp_path / "tracking"
        tracker = LocalTracker(TrackerConfig(backend="local", tracking_uri=str(root)))
        run_id = tracker.start_run(run_name="r")
        tracker.log_artifact(model_dir, "model")
        tracker.set_tags({"artifact.deploy": "/x/deploy"})
        tracker.end_run()

        run_dir = root / "default" / run_id
        artifacts = json.loads((run_dir / "artifacts.json").read_text())
        assert artifacts == [{"path": str(model_dir.resolve()), "type": "model"}]
        # Nothing copied: the model file exists once
        assert not list(root.rglob("model.bin"))
        assert json.loads((run_dir / "tags.json").read_text()) == {"artifact.deploy": "/x/deploy"}


# =============================================================================
# MLflow backend (stand-in client)
# =============================================================================


class TestMLflowTracker:
    def test_parent_child_params_metrics_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = install_fake_mlflow(monkeypatch)
        cfg = TrackerConfig(
            backend="mlflow", tracking_uri="http://mlflow:5000", experiment_name="mes"
        )
        parent = get_tracker(cfg)
        assert isinstance(parent, MLflowTracker)
        parent_id = parent.start_run(run_name="run_1", tags={"config_hash": "abc"})
        params = {f"p{i}": i for i in range(250)}
        params["long"] = "x" * 10_000
        parent.log_params(params)

        child = get_tracker(
            TrackerConfig(backend="mlflow", experiment_name="mes", parent_run_id=parent_id)
        )
        child_id = child.start_run(run_name="xgboost_h5", tags={"model_name": "xgboost"})
        child.log_metrics({"val_macro_f1": 0.5, "skipped": None, "bad": "n/a"}, step=3)  # type: ignore[dict-item]
        child.end_run(status="INTERRUPTED")
        parent.end_run()

        assert store.clients == ["http://mlflow:5000", None]
        assert list(store.experiments) == ["mes"]  # created once, then reused
        parent_run, child_run = store.runs[parent_id], store.runs[child_id]
        assert parent_run["tags"] == {"config_hash": "abc"}
        assert parent_run["run_name"] == "run_1"
        assert child_run["tags"][PARENT_RUN_TAG] == parent_id
        assert child_run["tags"]["model_name"] == "xgboost"
        assert len(parent_run["params"]) == 251
        assert len(parent_run["params"]["long"]) == 6000
        assert max(store.batch_sizes) <= 100
        assert child_run["metrics"] == {"val_macro_f1": 0.5}
        assert child_run["status"] == "KILLED"
        assert parent_run["status"] == "FINISHED"

    def test_artifacts_and_tags(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        store = install_fake_mlflow(monkeypatch)
        (tmp_path / "manifest.json").write_text("{}")
        tracker = get_tracker(TrackerConfig(backend="mlflow"))
        run_id = tracker.start_run(run_name="r")
        tracker.log_artifact(tmp_path / "manifest.json", "config")
        tracker.log_artifact(tmp_path, "dir")
        tracker.log_artifact(tmp_path / "missing.json", "config")
        tracker.set_tags({"artifact.deploy": "/deploy"})
        tracker.end_run(status="FAILED")
        run = store.runs[run_id]
        # Files are uploaded; directories (checkpoints can be GBs) only referenced
        assert run["artifacts"] == [("manifest.json", "config")]
        assert run["tags"]["artifact_path.dir"] == str(tmp_path.resolve())
        assert run["tags"]["artifact.deploy"] == "/deploy"
        assert run["status"] == "FAILED"


# =============================================================================
# Trainer: one child run per model, routed through TrainerConfig.tracking_*
# =============================================================================


def _train_xgboost(tmp_path: Path, **request_fields: Any) -> Any:
    from src.models.training.services.model_training import (
        ModelTrainingRequest,
        ModelTrainingService,
    )

    request = ModelTrainingRequest(
        model_name="xgboost",
        horizon=5,
        prepared_data=tiny_prepared_data(),
        output_dir=tmp_path / "models",
        use_feature_selection=False,
        use_calibration=False,
        **request_fields,
    )
    return ModelTrainingService().train_model(request)


class TestTrainerTracking:
    def test_no_tracking_by_default(self, tmp_path: Path) -> None:
        result = _train_xgboost(tmp_path)
        assert isinstance(result.trainer.tracker, DisabledTracker)
        assert not list(tmp_path.rglob("run_info.json"))

    def test_child_run_under_parent_mlflow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        store = install_fake_mlflow(monkeypatch)
        result = _train_xgboost(
            tmp_path,
            random_seed=11,
            tracking_backend="mlflow",
            tracking_uri="http://mlflow:5000",
            tracking_experiment="mes",
            tracking_parent_run_id="parent_run",
        )
        (run,) = store.runs.values()
        assert run["tags"][PARENT_RUN_TAG] == "parent_run"
        assert run["tags"]["model_name"] == "xgboost"
        assert run["status"] == "FINISHED"
        assert run["params"]["random_seed"] == "11"
        assert run["params"]["model_config.random_state"] == "11"
        assert "val_macro_f1" in run["metrics"]
        assert run["tags"]["artifact_path.model"].endswith("checkpoints")
        assert result.trainer.model.config["random_state"] == 11

    def test_failed_training_marks_run_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from src.models.training.trainer import Trainer

        def boom(self: Trainer, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("training exploded")

        monkeypatch.setattr(Trainer, "_run", boom)
        with pytest.raises(RuntimeError, match="exploded"):
            _train_xgboost(
                tmp_path, tracking_backend="local", tracking_uri=str(tmp_path / "tracking")
            )
        (run,) = LocalTracker.list_runs(tmp_path / "tracking")
        assert run["status"] == "FAILED"


class _FlakyTracker(DisabledTracker):
    """A tracker whose server fails on the chosen calls (review F4)."""

    def __init__(self, failing: set[str]) -> None:
        super().__init__(TrackerConfig())
        self.failing = failing
        self.calls: list[tuple[str, Any]] = []

    def _maybe_fail(self, name: str, payload: Any = None) -> None:
        self.calls.append((name, payload))
        if name in self.failing:
            raise ConnectionError(f"tracking server down during {name}")

    def start_run(self, run_name=None, run_id=None, tags=None) -> str:  # type: ignore[override]
        self._maybe_fail("start_run")
        return super().start_run(run_name, run_id, tags)

    def end_run(self, status: str = "FINISHED") -> None:
        self._maybe_fail("end_run", status)
        super().end_run(status)

    def log_params(self, params: dict[str, Any]) -> None:
        self._maybe_fail("log_params")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._maybe_fail("log_metrics")

    def log_artifact(self, path: Path | str, artifact_type: str | None = None) -> None:
        self._maybe_fail("log_artifact")


class TestTrackingFailuresNeverFailTraining:
    def _train_with(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tracker: Any) -> Any:
        import src.models.training.trainer as trainer_module

        monkeypatch.setattr(trainer_module, "get_tracker", lambda config: tracker)
        return _train_xgboost(tmp_path, tracking_backend="local")

    def test_every_call_failing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        tracker = _FlakyTracker(
            {"start_run", "log_params", "log_metrics", "log_artifact", "end_run"}
        )
        result = self._train_with(monkeypatch, tmp_path, tracker)
        assert result.metrics, "training completed"
        saved = list(result.trainer.output_path.joinpath("checkpoints").rglob("*"))
        assert saved, "the model was saved despite the tracker"

    def test_metrics_failing_still_ends_run_finished(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        tracker = _FlakyTracker({"log_params", "log_metrics", "log_artifact"})
        result = self._train_with(monkeypatch, tmp_path, tracker)
        assert result.metrics
        assert tracker.calls[0][0] == "start_run"
        assert tracker.calls[-1] == ("end_run", "FINISHED")
        assert list(result.trainer.output_path.joinpath("checkpoints").rglob("*"))

    def test_training_error_ends_run_failed_even_if_logging_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from src.models.training.trainer import Trainer

        def boom(self: Trainer, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("training exploded")

        monkeypatch.setattr(Trainer, "_run", boom)
        tracker = _FlakyTracker({"log_params"})
        with pytest.raises(RuntimeError, match="exploded"):
            self._train_with(monkeypatch, tmp_path, tracker)
        assert tracker.calls[-1] == ("end_run", "FAILED")

    def test_model_saved_before_first_metrics_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from src.models.training.trainer import Trainer

        order: list[str] = []
        real_save = Trainer._save_model

        def recording_save(self: Trainer) -> None:
            order.append("save_model")
            real_save(self)

        monkeypatch.setattr(Trainer, "_save_model", recording_save)
        tracker = _FlakyTracker(set())
        real_log = tracker.log_metrics

        def recording_log(metrics: dict[str, float], step: int | None = None) -> None:
            order.append("log_metrics")
            real_log(metrics, step)

        tracker.log_metrics = recording_log  # type: ignore[method-assign]
        self._train_with(monkeypatch, tmp_path, tracker)
        assert order.index("save_model") < order.index("log_metrics")


def test_params_redact_uri_credentials() -> None:
    flat = flatten_params(
        {"tracking": {"tracking_uri": "https://alice:s3cret@mlflow.example.com/api"}}
    )
    assert flat == {"tracking.tracking_uri": "https://***@mlflow.example.com/api"}
