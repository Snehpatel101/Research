"""MLFactory.run bookkeeping that needs no training: seeding before any work, the
run manifest's failure path, parent tracker runs, and the deploy manifest's
verification of the run manifest it references."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.config.experiment import ExperimentConfig, TrackingSection
from src.core.run_manifest import RUN_MANIFEST_FILE, RunManifest
from src.factory import MLFactory
from src.inference.deploy import DEPLOY_MANIFEST_FILE, DeployManifest, validate_deploy_artifact
from src.models.tracking import LocalTracker


def _config(tmp_path: Path, **tracking) -> ExperimentConfig:
    cfg = ExperimentConfig(run_id="prov_run", output_dir=tmp_path / "runs", random_seed=7)
    cfg.verbose = 0
    cfg.data.data_path = tmp_path / "missing.parquet"  # every run here fails on load
    if tracking:
        cfg.tracking = TrackingSection(**tracking)
    return cfg


def _manifest(cfg: ExperimentConfig) -> dict:
    return json.loads((Path(cfg.output_dir) / RUN_MANIFEST_FILE).read_text())


class TestSeedingAndFailureManifest:
    def test_seeds_before_data_and_records_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.core.reproducibility as reproducibility

        events: list[tuple[str, object]] = []
        real_set_all_seeds = reproducibility.set_all_seeds

        def recording_set_all_seeds(seed: int, deterministic: bool = False):
            events.append(("seed", (seed, deterministic)))
            return real_set_all_seeds(seed, deterministic)

        real_load = MLFactory._load_raw_bars

        def recording_load(self: MLFactory):
            events.append(("load", None))
            return real_load(self)

        monkeypatch.setattr(reproducibility, "set_all_seeds", recording_set_all_seeds)
        monkeypatch.setattr(MLFactory, "_load_raw_bars", recording_load)

        cfg = _config(tmp_path)
        with pytest.raises(FileNotFoundError):
            MLFactory(cfg, verbose=0, enable_checkpoints=False).run()

        assert events[0] == ("seed", (7, False)), "seeds must be set before any data work"
        assert ("load", None) in events

        manifest = _manifest(cfg)
        assert manifest["status"] == "failed"
        assert manifest["error"]["type"] == "FileNotFoundError"
        assert manifest["finished_at"] is not None
        assert manifest["provenance"]["config_hash"] == cfg.config_hash()
        assert manifest["provenance"]["seed"] == {"random_seed": 7, "deterministic": False}
        assert manifest["provenance"]["data_source"]["sha256"] is None
        assert manifest["tracking"] is None

    def test_local_parent_run_marked_failed(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, backend="local")
        with pytest.raises(FileNotFoundError):
            MLFactory(cfg, verbose=0, enable_checkpoints=False).run()

        tracking_root = tmp_path / "runs" / "tracking"
        (run,) = LocalTracker.list_runs(tracking_root)
        assert run["status"] == "FAILED"
        assert run["run_name"] == "prov_run"
        manifest = _manifest(cfg)
        assert manifest["tracking"]["run_id"] == run["run_id"]
        assert manifest["tracking"]["uri"] == str(tracking_root)
        run_dir = Path(run["run_dir"])
        tags = json.loads((run_dir / "tags.json").read_text())
        assert tags["config_hash"] == cfg.config_hash()
        params = json.loads((run_dir / "params.json").read_text())
        assert params["random_seed"] == 7
        assert params["training.models"] == ["xgboost"]

    def test_mlflow_not_installed_fails_before_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "mlflow", None)
        cfg = _config(tmp_path, backend="mlflow")
        loaded: list[bool] = []
        monkeypatch.setattr(MLFactory, "_load_raw_bars", lambda self: loaded.append(True))
        with pytest.raises(ImportError, match="mlflow"):
            MLFactory(cfg, verbose=0, enable_checkpoints=False).run()
        assert not loaded
        manifest = _manifest(cfg)
        assert manifest["status"] == "failed"
        assert manifest["error"]["type"] in ("ImportError", "ModuleNotFoundError")


class TestDeployManifestReference:
    def _deploy_dir(self, tmp_path: Path) -> tuple[Path, RunManifest]:
        run_dir = tmp_path / "run"
        manifest = RunManifest.begin(
            run_dir,
            run_id="r1",
            name="exp",
            config={"random_seed": 1},
            config_hash="h",
            seed={"random_seed": 1, "deterministic": False},
            data_path=None,
        )
        deploy_dir = run_dir / "deploy"
        DeployManifest(symbol="MES", run_manifest=manifest.reference(deploy_dir)).save(
            deploy_dir / DEPLOY_MANIFEST_FILE
        )
        return deploy_dir, manifest

    def test_round_trip_and_valid(self, tmp_path: Path) -> None:
        deploy_dir, manifest = self._deploy_dir(tmp_path)
        loaded = DeployManifest.load(deploy_dir / DEPLOY_MANIFEST_FILE)
        assert loaded.run_manifest["provenance_sha256"] == manifest.provenance_sha256
        assert validate_deploy_artifact(deploy_dir)["valid"]

    def test_tampered_run_manifest_detected(self, tmp_path: Path) -> None:
        deploy_dir, manifest = self._deploy_dir(tmp_path)
        content = json.loads(manifest.path.read_text())
        content["provenance"]["config_hash"] = "forged"
        manifest.path.write_text(json.dumps(content))
        report = validate_deploy_artifact(deploy_dir)
        assert not report["valid"]
        assert "digest" in report["issues"][0]

    def test_other_runs_manifest_detected(self, tmp_path: Path) -> None:
        deploy_dir, manifest = self._deploy_dir(tmp_path)
        other = RunManifest.begin(
            tmp_path / "other",
            run_id="r2",
            name="exp",
            config={"random_seed": 2},
            config_hash="h2",
            seed={"random_seed": 2, "deterministic": False},
            data_path=None,
        )
        manifest.path.write_text(other.path.read_text())
        report = validate_deploy_artifact(deploy_dir)
        assert not report["valid"]
        assert "different run" in report["issues"][0]

    def test_deploy_dir_shipped_alone_is_valid(self, tmp_path: Path) -> None:
        deploy_dir, manifest = self._deploy_dir(tmp_path)
        manifest.path.unlink()
        assert validate_deploy_artifact(deploy_dir)["valid"]


class TestFailureBookkeeping:
    def test_seeding_failure_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.core.reproducibility as reproducibility

        def broken(seed: int, deterministic: bool = False):
            raise RuntimeError("cannot seed")

        monkeypatch.setattr(reproducibility, "set_all_seeds", broken)
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="cannot seed"):
            MLFactory(cfg, verbose=0, enable_checkpoints=False).run()
        assert _manifest(cfg)["error"] == {"type": "RuntimeError", "message": "cannot seed"}

    def test_manifest_write_failure_keeps_the_original_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_finish(self: RunManifest, **kwargs) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(RunManifest, "finish", broken_finish)
        with pytest.raises(FileNotFoundError):
            MLFactory(_config(tmp_path), verbose=0, enable_checkpoints=False).run()

    def test_seed_bound(self, tmp_path: Path) -> None:
        from src.core.reproducibility import MAX_RANDOM_SEED

        with pytest.raises(ValueError, match="random_seed"):
            ExperimentConfig(run_id="r", output_dir=tmp_path, random_seed=2**32 - 1)
        ExperimentConfig(run_id="r", output_dir=tmp_path, random_seed=MAX_RANDOM_SEED)


class TestResume:
    """Checkpoints are keyed by config_hash() only (review F2)."""

    def _checkpointed(self, tmp_path: Path, stored_hash: str | None = None) -> ExperimentConfig:
        cfg = _config(tmp_path)
        factory = MLFactory(cfg, verbose=0, enable_checkpoints=True)
        assert factory._checkpoint_manager is not None
        factory._checkpoint_manager.save_checkpoint(
            stage_name="data_pipeline",
            stage_index=0,
            artifacts={},
            config_hash=stored_hash or cfg.config_hash(),
        )
        return cfg

    def _resume(self, cfg: ExperimentConfig, monkeypatch: pytest.MonkeyPatch, **kwargs):
        calls: list[bool] = []
        monkeypatch.setattr(MLFactory, "run", lambda self, resume=False: calls.append(resume))
        MLFactory(cfg, verbose=0, enable_checkpoints=True).resume_from_checkpoint(**kwargs)
        return calls

    def _reload(self, cfg: ExperimentConfig) -> ExperimentConfig:
        return ExperimentConfig.from_yaml(Path(cfg.output_dir) / "experiment_config.yaml")

    def test_tracking_only_edit_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = self._reload(self._checkpointed(tmp_path))
        cfg.tracking = TrackingSection(backend="local")
        cfg.verbose = 2
        assert self._resume(cfg, monkeypatch) == [True]

    def test_pre_117_checkpoint_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import hashlib

        cfg = _config(tmp_path)
        legacy = {k: v for k, v in cfg.to_dict().items() if k not in ("deterministic", "tracking")}
        old_hash = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
        cfg = self._checkpointed(tmp_path, stored_hash=old_hash)
        assert self._resume(self._reload(cfg), monkeypatch) == [True]

    def test_changed_experiment_refused_and_checkpoints_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = self._reload(self._checkpointed(tmp_path))
        cfg.training.models = ["xgboost", "logistic"]
        with pytest.raises(ValueError, match="Cannot resume"):
            self._resume(cfg, monkeypatch)
        checkpoints = list((Path(cfg.output_dir) / "checkpoints").glob("*.json"))
        assert checkpoints, "a refused resume must not delete checkpoints"

    def test_explicit_restart_discards_checkpoints(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = self._reload(self._checkpointed(tmp_path))
        cfg.training.models = ["logistic"]
        assert self._resume(cfg, monkeypatch, restart_on_config_change=True) == [False]
        assert not list((Path(cfg.output_dir) / "checkpoints").glob("*.json"))

    def test_resumed_run_keeps_original_provenance(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        with pytest.raises(FileNotFoundError):
            MLFactory(cfg, verbose=0, enable_checkpoints=True).run()
        original = _manifest(cfg)
        with pytest.raises(FileNotFoundError):
            MLFactory(self._reload(cfg), verbose=0, enable_checkpoints=True).run(resume=True)
        resumed = _manifest(cfg)
        assert resumed["provenance_sha256"] == original["provenance_sha256"]
        assert resumed["started_at"] == original["started_at"]
        assert len(resumed["resumes"]) == 1
        assert resumed["status"] == "failed"


def test_regime_importance_receives_the_run_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regime-conditional selection is seeded with random_state (review F3)."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    import src.optimization.feature_selection.regime_selection as regime_selection
    from src.models.training import UnifiedTrainingOrchestrator

    seen: dict[str, object] = {}

    def spy(df, feature_names, label_col, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(regime_selection, "compute_regime_importance", spy)
    cfg = ExperimentConfig(run_id="r", output_dir=tmp_path, random_seed=1234)
    cfg.training.models = ["xgboost"]
    cfg.training.horizons = [5]
    pipeline_config = cfg.to_pipeline_config(cv_gaps=(15, 30))
    pipeline_config.feature_selection = SimpleNamespace(  # type: ignore[attr-defined]
        regime_conditional=True, mtf_max_per_timeframe=8
    )
    orchestrator = UnifiedTrainingOrchestrator(pipeline_config)
    rng = np.random.default_rng(0)
    features = [f"f{i}" for i in range(6)]
    df = pd.DataFrame(rng.normal(size=(400, 6)), columns=features)
    df["label_h5"] = rng.integers(-1, 2, 400)
    monkeypatch.setattr(
        orchestrator,
        "_compute_mda_ranking",
        lambda frame, names: pd.Series(np.linspace(1, 0, len(names)), index=names),
    )
    orchestrator._run_feature_selection_pipeline(df, features)
    assert seen.get("random_state") == 1234
