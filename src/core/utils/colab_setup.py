"""
Google Colab Setup and Compatibility Module

This module provides utilities to run the ML Model Factory in Google Colab.
Run this at the start of your Colab notebook before importing any pipeline modules.

Usage in Colab:
    ```python
    # Cell 1: Clone repo and setup
    !git clone https://github.com/Snehpatel101/Research.git
    %cd Research
    !pip install -r requirements-colab.txt

    # Cell 2: Initialize Colab environment
    from src.core.utils.colab_setup import setup_colab_environment
    setup_colab_environment()

    # Cell 3: Run the pipeline on raw OHLCV bars
    from src.config.experiment import ExperimentConfig, DataSection
    from src.factory import MLFactory

    config = ExperimentConfig(data=DataSection(data_path="/content/drive/MyDrive/data/mes.parquet"))
    result = MLFactory(config).run()
    ```
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Check if running in Google Colab environment."""
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ


def _git_repo_root(start_dir: Path) -> Path | None:
    """Return git repo root for start_dir or None if not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(start_dir), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
        return Path(out)
    except Exception as e:
        logger.warning(f"Failed to find git repo root: {e}")
        return None


def setup_environment(
    runtime_target: str = "auto",
    repo_root: str | None = None,
    mount_drive: bool = True,
    use_gpu: bool = True,
    repo_url: str | None = None,
) -> dict[str, Any]:
    """
    Unified environment setup for Colab-hosted or local runtimes.

    Args:
        runtime_target: "auto", "colab_hosted", or "local".
        repo_root: Optional explicit repo root override.
        mount_drive: Mount Google Drive when in Colab-hosted runtime.
        use_gpu: Whether to check for GPU availability.
        repo_url: Optional git URL to clone into repo_root if missing.

    Returns:
        Dict with environment info and normalized paths.
    """
    if runtime_target not in {"auto", "colab_hosted", "local"}:
        raise ValueError("runtime_target must be one of: auto, colab_hosted, local")

    is_colab_runtime = is_colab() if runtime_target == "auto" else runtime_target == "colab_hosted"

    env_info = {
        "runtime_target": runtime_target,
        "is_colab": is_colab_runtime,
        "drive_mounted": False,
        "gpu_available": False,
        "gpu_name": None,
        "repo_root": None,
        "drive_root": None,
        "runtime_mode": "local_cpu",
    }

    # Drive mount (Colab-hosted only)
    if is_colab_runtime and mount_drive:
        try:
            from google.colab import drive  # type: ignore[import-not-found]

            drive.mount("/content/drive")
            env_info["drive_mounted"] = True
            env_info["drive_root"] = Path("/content/drive/MyDrive")
            print("[OK] Google Drive mounted at /content/drive")
        except Exception as e:
            print(f"[WARN] Could not mount Drive: {e}")

    # Resolve repo root
    if repo_root:
        resolved_root = Path(repo_root).expanduser()
    elif is_colab_runtime:
        drive_repo = Path("/content/drive/MyDrive/research")
        content_repo = Path("/content/research")
        resolved_root = drive_repo if drive_repo.exists() else content_repo
    else:
        resolved_root = _git_repo_root(Path.cwd()) or Path.cwd()

    # Clone repo if requested and missing
    if repo_url and not resolved_root.exists():
        resolved_root.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Clone] {repo_url} -> {resolved_root}")
        subprocess.run(
            ["git", "clone", repo_url, str(resolved_root)],
            check=True,
            capture_output=True,
        )

    repo_root_path = resolved_root.resolve()
    env_info["repo_root"] = repo_root_path

    # Add repo to path and chdir
    if str(repo_root_path) not in sys.path:
        sys.path.insert(0, str(repo_root_path))
    os.chdir(repo_root_path)

    # GPU detection
    if use_gpu:
        try:
            import torch

            env_info["gpu_available"] = torch.cuda.is_available()
            if env_info["gpu_available"]:
                env_info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass

    if env_info["is_colab"]:
        env_info["runtime_mode"] = "colab"
    else:
        env_info["runtime_mode"] = "local_gpu" if env_info["gpu_available"] else "local_cpu"

    return env_info


def setup_colab_environment(
    project_root: str | None = None,
    mount_drive: bool = True,
    use_gpu: bool = True,
    repo_url: str = "https://github.com/Snehpatel101/Research.git",
) -> dict[str, Any]:
    """
    Configure environment for Google Colab compatibility.

    Args:
        project_root: Path to the cloned project. If None, auto-detects.
        mount_drive: Whether to mount Google Drive for data access.
        use_gpu: Whether to configure GPU support (if available).
        repo_url: Git URL to clone if project is missing.

    Returns:
        Dict with environment info (gpu_available, drive_mounted, etc.)
    """
    if not is_colab():
        print("Not running in Colab - no setup needed")
        return {
            "is_colab": False,
            "gpu_available": False,
            "drive_mounted": False,
            "project_root": None,
        }

    print("=" * 60)
    print("Google Colab Environment Setup")
    print("=" * 60)

    env = setup_environment(
        runtime_target="colab_hosted",
        repo_root=project_root,
        mount_drive=mount_drive,
        use_gpu=use_gpu,
        repo_url=repo_url,
    )

    # Colab-specific defaults
    os.environ["COLAB_ENV"] = "1"
    os.environ["PYTORCH_NUM_WORKERS"] = "0"  # Avoid multiprocessing issues

    print("=" * 60)
    print("Environment setup complete")
    print("=" * 60)

    return {
        "is_colab": env["is_colab"],
        "gpu_available": env["gpu_available"],
        "drive_mounted": env["drive_mounted"],
        "project_root": str(env["repo_root"]) if env["repo_root"] else None,
    }


# Colab-specific DataLoader wrapper
def get_colab_dataloader_kwargs() -> dict[str, Any]:
    """
    Get DataLoader kwargs optimized for Colab.

    Colab has issues with multiprocessing in DataLoaders.
    Always use num_workers=0 to avoid crashes.

    Returns:
        Dict with DataLoader kwargs
    """
    return {
        "num_workers": 0,
        "pin_memory": False,  # Avoid memory issues
    }


def ensure_data_in_workspace(config, target_dir: str = "data/raw") -> None:
    """
    Ensure raw data file is present in the project workspace.

    If running in Colab and the file is in Drive but not in the project,
    it copies it to the specified target directory.

    Args:
        config: The notebook configuration object.
        target_dir: Relative path to target directory within project root.
    """
    import shutil

    if not config.is_colab or config.raw_data_file is None:
        return

    project_raw_dir = config.project_root / target_dir
    project_raw_dir.mkdir(parents=True, exist_ok=True)

    target_filename = f"{config.symbol}_1m{config.raw_data_file.suffix}"
    target_path = project_raw_dir / target_filename

    if (
        not str(config.raw_data_file).startswith(str(config.project_root))
        and not target_path.exists()
    ):
        print("\n  Copying data to project directory...")
        shutil.copy2(config.raw_data_file, target_path)
        print(f"  Done: {target_path.name}")


__all__ = [
    "is_colab",
    "setup_environment",
    "setup_colab_environment",
    "get_colab_dataloader_kwargs",
    "ensure_data_in_workspace",
]
