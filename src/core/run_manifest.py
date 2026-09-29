"""
Run manifest: the provenance record every MLFactory run writes to its output dir.

``run_manifest.json`` answers "what exactly produced these artifacts?":

- ``provenance`` (written once, when the run starts, never modified): the full
  experiment config and its hash, the seed and determinism settings, the git
  commit (+ dirty flag and a hash of the uncommitted diff), package versions,
  the Python / torch / CUDA environment and the input data file's SHA-256.
  ``provenance_sha256`` is the digest of that block, so a deploy manifest can
  reference a run and anyone can verify the reference later
  (``verify_provenance``).
- ``data``: what the run loaded (rows, time range, bar timeframes).
- ``status`` / timing / ``error``: ``running`` while the run is in flight, then
  ``success`` or ``failed`` (with the exception type and message).
- ``results``: the final metrics summary and artifact paths.
- ``tracking``: where the run was logged (experiment tracker backend + run id).

Every write replaces the file atomically, so a crashed run leaves a valid
manifest with ``status: running``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

RUN_MANIFEST_FILE = "run_manifest.json"
RUN_MANIFEST_VERSION = 1

# Distributions whose versions decide what a run computes
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "numba",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "optuna",
    "pyarrow",
    "mlflow",
)

# Environment variables that change numeric results or their reproducibility
TRACKED_ENV_VARS = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
)

_GIT_TIMEOUT_SECONDS = 10
# Source checkout the running code came from (…/src/core/run_manifest.py -> repo root)
_REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# FINGERPRINTS
# =============================================================================


def canonical_json_sha256(value: Any) -> str:
    """SHA-256 of ``value`` as canonical JSON (sorted keys, no whitespace)."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, streamed in chunks (never loads the file into memory)."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def data_source_fingerprint(path: str | Path | None) -> dict[str, Any]:
    """Path, size and SHA-256 of the input data file (all None without a file)."""
    if path is None or not Path(path).is_file():
        return {"path": str(path) if path else None, "size_bytes": None, "sha256": None}
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _git(args: list[str], cwd: Path) -> str | None:
    """Output of a git command, or None when git or the repository is unavailable."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode(errors="replace")


def git_state(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """
    Commit, branch and working-tree state of the source checkout.

    ``dirty`` covers tracked files only (stray untracked files such as run
    outputs must not flag every run); when dirty, ``diff_sha256`` fingerprints
    the uncommitted changes so two dirty runs can still be told apart.
    """
    cwd = Path(repo_dir) if repo_dir is not None else _REPO_ROOT
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None, "diff_sha256": None}
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd) or ""
    dirty = bool(status.strip())
    diff = _git(["diff", "HEAD"], cwd) if dirty else None
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    return {
        "commit": commit.strip(),
        "branch": branch.strip() if branch else None,
        "dirty": dirty,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def package_versions() -> dict[str, str | None]:
    """Installed versions of the tracked packages (None when not installed)."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for dist in TRACKED_PACKAGES:
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def torch_environment() -> dict[str, Any]:
    """Torch build, device and determinism state (``available: False`` without torch)."""
    try:
        import torch
    except ImportError:
        return {"available": False}

    cuda = torch.cuda.is_available()
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "device_count": torch.cuda.device_count() if cuda else 0,
        "devices": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if cuda
            else []
        ),
        "num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def runtime_environment() -> dict[str, Any]:
    """Interpreter, OS, CPU and the environment variables that affect results."""
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "argv": list(sys.argv),
        "env": {name: os.environ.get(name) for name in TRACKED_ENV_VARS},
    }


def verify_provenance(manifest: dict[str, Any]) -> bool:
    """True when ``manifest['provenance']`` still hashes to its recorded digest."""
    provenance = manifest.get("provenance")
    recorded = manifest.get("provenance_sha256")
    return provenance is not None and canonical_json_sha256(provenance) == recorded


# =============================================================================
# RUN MANIFEST
# =============================================================================


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def json_safe(value: Any) -> Any:
    """Strict-JSON copy of ``value``: NaN/inf -> None, numpy scalars -> Python, Paths -> str."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item) and getattr(value, "ndim", None) == 0:
        value = value.item()  # numpy scalar / 0-d array
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str | int | bool) or value is None:
        return value
    return str(value)


class RunManifest:
    """
    ``run_manifest.json`` of one run: created by ``begin``, updated by
    ``record`` / ``set_results``, closed by ``finish``.

    Example:
        manifest = RunManifest.begin(output_dir, run_id=..., name=..., config=cfg.to_dict(),
                                     config_hash=cfg.config_hash(), seed={...},
                                     data_path=cfg.data.data_path)
        ...
        manifest.finish(success=True)
    """

    def __init__(self, path: Path, content: dict[str, Any]) -> None:
        self.path = path
        self.content = content

    @classmethod
    def begin(
        cls,
        output_dir: str | Path,
        *,
        run_id: str,
        name: str,
        config: dict[str, Any],
        config_hash: str,
        seed: dict[str, Any],
        data_path: str | Path | None,
    ) -> RunManifest:
        """Collect the provenance and write the manifest with ``status: running``."""
        provenance = json_safe(
            {
                "config_hash": config_hash,
                "config": config,
                "seed": seed,
                "git": git_state(),
                "packages": package_versions(),
                "environment": runtime_environment(),
                "torch": torch_environment(),
                "data_source": data_source_fingerprint(data_path),
            }
        )
        content: dict[str, Any] = {
            "manifest_version": RUN_MANIFEST_VERSION,
            "run_id": run_id,
            "name": name,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "duration_seconds": None,
            "error": None,
            "provenance_sha256": canonical_json_sha256(provenance),
            "provenance": provenance,
            "data": {},
            "tracking": None,
            "results": None,
        }
        manifest = cls(Path(output_dir) / RUN_MANIFEST_FILE, content)
        manifest.write()
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> RunManifest:
        """Read an existing manifest."""
        path = Path(path)
        with open(path) as f:
            return cls(path, json.load(f))

    # -- accessors ------------------------------------------------------------

    @property
    def provenance(self) -> dict[str, Any]:
        return self.content["provenance"]

    @property
    def provenance_sha256(self) -> str:
        return self.content["provenance_sha256"]

    @property
    def status(self) -> str:
        return self.content["status"]

    def reference(self, from_dir: str | Path) -> dict[str, Any]:
        """
        Pointer to this manifest for another artifact (e.g. the deploy manifest).

        Carries the identifying provenance inline, so the pointer stays
        meaningful when the referencing directory is shipped on its own, and
        ``provenance_sha256`` lets a reader verify the manifest it points at.
        """
        provenance = self.provenance
        return {
            "path": os.path.relpath(self.path, Path(from_dir)),
            "run_id": self.content["run_id"],
            "provenance_sha256": self.provenance_sha256,
            "config_hash": provenance["config_hash"],
            "git_commit": provenance["git"]["commit"],
            "git_dirty": provenance["git"]["dirty"],
            "data_sha256": provenance["data_source"]["sha256"],
        }

    # -- updates --------------------------------------------------------------

    def record(self, section: str, values: dict[str, Any]) -> None:
        """Merge ``values`` into a mutable section (``data``, ``tracking``) and write."""
        if section in ("provenance", "provenance_sha256"):
            raise ValueError("provenance is fixed when the run starts")
        current = self.content.get(section) or {}
        self.content[section] = {**current, **values}
        self.write()

    def finish(
        self,
        *,
        success: bool,
        error: BaseException | None = None,
        results: dict[str, Any] | None = None,
    ) -> None:
        """Close the run: final status, end time, duration, error and results."""
        finished = datetime.now().astimezone()
        started = datetime.fromisoformat(self.content["started_at"])
        self.content["status"] = "success" if success else "failed"
        self.content["finished_at"] = finished.isoformat()
        self.content["duration_seconds"] = (finished - started).total_seconds()
        if error is not None:
            self.content["error"] = {"type": type(error).__name__, "message": str(error)}
        if results is not None:
            self.content["results"] = results
        self.write()

    def write(self) -> None:
        """Atomically replace the manifest file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        with open(tmp, "w") as f:
            json.dump(json_safe(self.content), f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, self.path)


__all__ = [
    "RUN_MANIFEST_FILE",
    "RUN_MANIFEST_VERSION",
    "RunManifest",
    "canonical_json_sha256",
    "data_source_fingerprint",
    "file_sha256",
    "git_state",
    "json_safe",
    "package_versions",
    "runtime_environment",
    "torch_environment",
    "verify_provenance",
]
