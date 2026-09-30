"""run_manifest.json: provenance capture, lifecycle (running -> success/failed),
strict JSON, verification and the reference handed to the deploy manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.core.run_manifest import (
    RUN_MANIFEST_FILE,
    RunManifest,
    canonical_json_sha256,
    data_source_fingerprint,
    file_sha256,
    git_state,
    json_safe,
    package_versions,
    redact_secrets,
    verify_provenance,
)
from tests.helpers import REPO_ROOT


def _begin(tmp_path: Path, data_path: Path | None = None) -> RunManifest:
    return RunManifest.begin(
        tmp_path / "run",
        run_id="run_1",
        name="exp",
        config={"random_seed": 7, "data": {"symbol": "MES"}},
        config_hash="abc123",
        seed={"random_seed": 7, "deterministic": False},
        data_path=data_path,
    )


class TestFingerprints:
    def test_file_sha256_streams_large_files(self, tmp_path: Path) -> None:
        # Bigger than any single read chunk: the digest must cover every byte
        payload = np.random.default_rng(0).bytes(3 * 1024 * 1024 + 17)
        path = tmp_path / "blob.bin"
        path.write_bytes(payload)
        assert file_sha256(path) == hashlib.sha256(payload).hexdigest()

    def test_data_source_fingerprint(self, tmp_path: Path) -> None:
        path = tmp_path / "bars.parquet"
        path.write_bytes(b"ohlcv")
        fp = data_source_fingerprint(path)
        assert fp == {
            "path": str(path.resolve()),
            "size_bytes": 5,
            "sha256": hashlib.sha256(b"ohlcv").hexdigest(),
        }

    def test_missing_data_file_has_no_hash(self, tmp_path: Path) -> None:
        fp = data_source_fingerprint(tmp_path / "missing.parquet")
        assert fp["sha256"] is None and fp["size_bytes"] is None
        assert data_source_fingerprint(None)["path"] is None

    def test_package_versions(self) -> None:
        versions = package_versions()
        assert versions["numpy"] == np.__version__
        for dist in ("pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna"):
            assert versions[dist], dist
        assert "torch" in versions and "python" in versions

    def test_git_state_outside_a_repository(self, tmp_path: Path) -> None:
        assert git_state(tmp_path) == {
            "commit": None,
            "branch": None,
            "dirty": None,
            "diff_sha256": None,
        }

    def test_git_state_of_the_source_checkout(self) -> None:
        if not (REPO_ROOT / ".git").exists():
            pytest.skip("not running from a source checkout")
        state = git_state()
        assert state["commit"] is not None and len(state["commit"]) == 40
        assert isinstance(state["dirty"], bool)
        # A fingerprint of the uncommitted changes exactly when dirty
        assert (state["diff_sha256"] is not None) == state["dirty"]


class TestJsonSafe:
    def test_non_finite_and_numpy_values(self) -> None:
        value = {
            "nan": float("nan"),
            "inf": float("inf"),
            "np_float": np.float32(0.5),
            "np_int": np.int64(3),
            "path": Path("a/b"),
            "nested": [np.float64("nan"), (1, 2)],
        }
        assert json_safe(value) == {
            "nan": None,
            "inf": None,
            "np_float": 0.5,
            "np_int": 3,
            "path": "a/b",
            "nested": [None, [1, 2]],
        }


class TestLifecycle:
    def test_begin_writes_running_manifest(self, tmp_path: Path) -> None:
        data = tmp_path / "bars.csv"
        data.write_text("datetime,open\n")
        manifest = _begin(tmp_path, data)

        assert manifest.path == tmp_path / "run" / RUN_MANIFEST_FILE
        on_disk = json.loads(manifest.path.read_text())
        assert on_disk["status"] == "running"
        assert on_disk["run_id"] == "run_1"
        assert on_disk["finished_at"] is None
        provenance = on_disk["provenance"]
        assert provenance["config_hash"] == "abc123"
        assert provenance["config"]["random_seed"] == 7
        assert provenance["seed"] == {"random_seed": 7, "deterministic": False}
        assert provenance["data_source"]["sha256"] == file_sha256(data)
        assert set(provenance) >= {"git", "packages", "environment", "torch"}
        assert "PYTHONHASHSEED" in provenance["environment"]["env"]
        assert provenance["torch"]["available"] is True
        assert verify_provenance(on_disk)

    def test_finish_success_with_results(self, tmp_path: Path) -> None:
        manifest = _begin(tmp_path)
        manifest.record("data", {"n_rows": 100})
        manifest.record("data", {"bar_timeframe": "5min"})
        manifest.finish(success=True, results={"metrics": {"m": {"f1": float("nan")}}})

        on_disk = json.loads(manifest.path.read_text())
        assert on_disk["status"] == "success"
        assert on_disk["error"] is None
        assert on_disk["duration_seconds"] >= 0
        assert on_disk["finished_at"] is not None
        assert on_disk["data"] == {"n_rows": 100, "bar_timeframe": "5min"}
        # Strict JSON: NaN metrics are written as null
        assert on_disk["results"]["metrics"]["m"]["f1"] is None
        assert "NaN" not in manifest.path.read_text()

    def test_finish_failed_records_error(self, tmp_path: Path) -> None:
        manifest = _begin(tmp_path)
        manifest.finish(success=False, error=ValueError("bad data"))
        on_disk = json.loads(manifest.path.read_text())
        assert on_disk["status"] == "failed"
        assert on_disk["error"] == {"type": "ValueError", "message": "bad data"}

    def test_provenance_is_immutable(self, tmp_path: Path) -> None:
        manifest = _begin(tmp_path)
        with pytest.raises(ValueError, match="provenance"):
            manifest.record("provenance", {"config_hash": "forged"})

    def test_load_round_trip(self, tmp_path: Path) -> None:
        manifest = _begin(tmp_path)
        loaded = RunManifest.load(manifest.path)
        assert loaded.provenance_sha256 == manifest.provenance_sha256
        assert loaded.status == "running"
        assert not list(manifest.path.parent.glob(".*.tmp")), "atomic write left a temp file"


class TestVerificationAndReference:
    def test_tampering_is_detected(self, tmp_path: Path) -> None:
        manifest = _begin(tmp_path)
        content = json.loads(manifest.path.read_text())
        assert verify_provenance(content)
        content["provenance"]["config"]["random_seed"] = 8
        assert not verify_provenance(content)
        assert not verify_provenance({})

    def test_digest_is_canonical(self) -> None:
        assert canonical_json_sha256({"a": 1, "b": [1, 2]}) == canonical_json_sha256(
            {"b": [1, 2], "a": 1}
        )

    def test_reference_from_deploy_dir(self, tmp_path: Path) -> None:
        data = tmp_path / "bars.csv"
        data.write_text("x")
        manifest = _begin(tmp_path, data)
        ref = manifest.reference(tmp_path / "run" / "deploy")
        assert ref["path"] == "../run_manifest.json"
        assert (tmp_path / "run" / "deploy" / ref["path"]).resolve() == manifest.path.resolve()
        assert ref["run_id"] == "run_1"
        assert ref["provenance_sha256"] == manifest.provenance_sha256
        assert ref["config_hash"] == "abc123"
        assert ref["data_sha256"] == file_sha256(data)
        assert set(ref) >= {"git_commit", "git_dirty"}


class TestDirectoryData:
    def test_partitioned_dataset_hashed_in_sorted_order(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        (root / "year=2024").mkdir(parents=True)
        (root / "year=2024" / "part-1.parquet").write_bytes(b"one")
        (root / "year=2024" / "part-0.parquet").write_bytes(b"zero")
        (root / ".hidden").write_bytes(b"ignored")
        fp = data_source_fingerprint(root)
        assert fp["n_files"] == 2
        assert fp["size_bytes"] == 7
        assert fp["sha256"] is not None
        # Content change in one partition changes the digest
        (root / "year=2024" / "part-1.parquet").write_bytes(b"ONE")
        assert data_source_fingerprint(root)["sha256"] != fp["sha256"]

    def test_missing_path_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        fp = data_source_fingerprint(tmp_path / "nope.parquet")
        assert fp["sha256"] is None
        assert "does not exist" in caplog.text


class TestRedaction:
    def test_credentials_masked_everywhere(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            "sys.argv", ["ml", "run", "--tracking-uri", "https://bob:pw@mlflow.example.com"]
        )
        manifest = RunManifest.begin(
            tmp_path / "run",
            run_id="r",
            name="exp",
            config={"tracking": {"tracking_uri": "postgresql://u:secret@db:5432/mlflow"}},
            config_hash="h",
            seed={"random_seed": 1, "deterministic": False},
            data_path=None,
        )
        manifest.record("tracking", {"uri": "https://bob:pw@mlflow.example.com"})
        text = manifest.path.read_text()
        assert "secret" not in text and "bob:pw" not in text
        assert "postgresql://***@db:5432/mlflow" in text
        assert verify_provenance(json.loads(text))

    def test_redact_secrets_leaves_plain_values(self) -> None:
        value = {"a": ["file:///data/x.parquet", 3, None], "b": "no uri here"}
        assert redact_secrets(value) == value


class TestResume:
    def test_resume_keeps_provenance_and_appends_entry(self, tmp_path: Path) -> None:
        manifest = _begin(tmp_path)
        manifest.record("data", {"n_rows": 2500, "start": "2024-01-02T09:30:00"})
        manifest.finish(success=False, error=RuntimeError("crash"))
        original = json.loads(manifest.path.read_text())

        reopened = RunManifest.load(manifest.path)
        reopened.resume(from_stage=2, config_hash="abc123")
        content = json.loads(reopened.path.read_text())
        assert content["provenance"] == original["provenance"]
        assert content["provenance_sha256"] == original["provenance_sha256"]
        assert content["started_at"] == original["started_at"]
        assert content["data"] == {"n_rows": 2500, "start": "2024-01-02T09:30:00"}
        assert content["status"] == "running" and content["error"] is None
        (entry,) = content["resumes"]
        assert entry["from_stage"] == 2
        assert entry["previous_status"] == "failed"
        assert set(entry) >= {"started_at", "git", "packages", "config_hash"}
