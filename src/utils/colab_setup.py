"""
Google Colab Setup and Compatibility Module

This module provides utilities to run the ML Model Factory in Google Colab.
Run this at the start of your Colab notebook before importing any pipeline modules.

Usage in Colab:
    ```python
    # Cell 1: Clone repo and setup
    !git clone https://github.com/YOUR_REPO/research.git
    %cd research
    !pip install -r requirements.txt

    # Cell 2: Initialize Colab environment
    from src.utils.colab_setup import setup_colab_environment, get_trainer_for_colab
    setup_colab_environment()

    # Cell 3: Run training
    trainer, results = get_trainer_for_colab(
        model_name="xgboost",
        horizon=20,
        data_path="/content/drive/MyDrive/data/splits/scaled"
    )
    ```
"""

import os
import sys
import subprocess
from pathlib import Path


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
    except Exception:
        return None


def setup_environment(
    runtime_target: str = "auto",
    repo_root: str | None = None,
    mount_drive: bool = True,
    use_gpu: bool = True,
    repo_url: str | None = None,
) -> dict:
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
            from google.colab import drive
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
        if drive_repo.exists():
            resolved_root = drive_repo
        else:
            resolved_root = content_repo
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

    env_info["repo_root"] = resolved_root.resolve()

    # Add repo to path and chdir
    if str(env_info["repo_root"]) not in sys.path:
        sys.path.insert(0, str(env_info["repo_root"]))
    os.chdir(env_info["repo_root"])

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
) -> dict:
    """
    Configure environment for Google Colab compatibility.

    Args:
        project_root: Path to the cloned project. If None, auto-detects.
        mount_drive: Whether to mount Google Drive for data access.
        use_gpu: Whether to configure GPU support (if available).

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


def get_trainer_for_colab(
    model_name: str,
    horizon: int = 20,
    data_path: str = "/content/drive/MyDrive/data/splits/scaled",
    feature_set: str | None = None,
    output_dir: str | None = None,
    **trainer_kwargs,
):
    """
    Create and run a trainer with Colab-compatible settings.

    This is the Python API alternative to the CLI scripts.

    Args:
        model_name: Model to train (e.g., "xgboost", "lstm", "stacking")
        horizon: Label horizon (default 20)
        data_path: Path to scaled splits directory
        feature_set: Optional feature set name
        output_dir: Output directory (defaults to /content/experiments)
        **trainer_kwargs: Additional TrainerConfig parameters

    Returns:
        Tuple of (trainer, results_dict)

    Example:
        ```python
        trainer, results = get_trainer_for_colab(
            model_name="xgboost",
            horizon=20,
            data_path="/content/drive/MyDrive/data/splits/scaled",
            feature_set="boosting_optimal",
        )
        print(f"Val F1: {results['evaluation_metrics']['macro_f1']:.4f}")
        ```
    """
    from src.models.trainer import Trainer
    from src.models.config import TrainerConfig
    from src.phase1.stages.datasets.container import TimeSeriesDataContainer

    # Set Colab-friendly defaults
    if output_dir is None:
        output_dir = Path("/content/experiments") if is_colab() else Path("experiments/runs")

    # Create config with Colab settings
    config = TrainerConfig(
        model_name=model_name,
        horizon=horizon,
        output_dir=Path(output_dir),
        feature_set=feature_set,
        evaluate_test_set=True,
        use_calibration=True,
        **trainer_kwargs,
    )

    # Load data
    print(f"Loading data from {data_path}...")
    container = TimeSeriesDataContainer.from_parquet_dir(
        data_path,
        horizon=horizon,
        exclude_invalid_labels=True,  # Critical for preventing leakage
    )

    # Create and run trainer
    print(f"Training {model_name}...")
    trainer = Trainer(config)
    results = trainer.run(container)

    print(f"\nTraining complete!")
    print(f"  Val F1: {results['evaluation_metrics']['macro_f1']:.4f}")
    print(f"  Val Accuracy: {results['evaluation_metrics']['accuracy']:.4f}")
    print(f"  Output: {results['output_path']}")

    return trainer, results


def train_ensemble_colab(
    base_models: list[str],
    meta_learner: str = "ridge_meta",
    horizon: int = 20,
    data_path: str = "/content/drive/MyDrive/data/splits/scaled",
    **kwargs,
):
    """
    Train a stacking ensemble with Colab-compatible settings.

    Args:
        base_models: List of base model names (e.g., ["xgboost", "lightgbm"])
        meta_learner: Meta-learner name (e.g., "ridge_meta", "mlp_meta")
        horizon: Label horizon
        data_path: Path to data
        **kwargs: Additional trainer parameters

    Returns:
        Tuple of (trainer, results_dict)
    """
    model_config = {
        "base_model_names": base_models,
        "meta_learner_name": meta_learner,
        "n_folds": 5,
        "use_probabilities": True,
        "use_default_configs_for_oof": True,  # Critical for preventing leakage
    }

    return get_trainer_for_colab(
        model_name="stacking",
        horizon=horizon,
        data_path=data_path,
        model_config=model_config,
        **kwargs,
    )


# Colab-specific DataLoader wrapper
def get_colab_dataloader_kwargs() -> dict:
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
    
    if not str(config.raw_data_file).startswith(str(config.project_root)):
        if not target_path.exists():
            print(f"\n  Copying data to project directory...")
            shutil.copy2(config.raw_data_file, target_path)
            print(f"  Done: {target_path.name}")


__all__ = [
    "is_colab",
    "setup_environment",
    "setup_colab_environment",
    "get_trainer_for_colab",
    "train_ensemble_colab",
    "get_colab_dataloader_kwargs",
    "ensure_data_in_workspace",
]
