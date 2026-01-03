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
    from notebooks.colab_setup import setup_colab_environment, get_trainer_for_colab
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
from pathlib import Path


def is_colab() -> bool:
    """Check if running in Google Colab environment."""
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ


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
    env_info = {
        "is_colab": is_colab(),
        "gpu_available": False,
        "drive_mounted": False,
        "project_root": None,
    }

    if not is_colab():
        print("Not running in Colab - no setup needed")
        return env_info

    print("=" * 60)
    print("Google Colab Environment Setup")
    print("=" * 60)

    # 1. Mount Google Drive (optional)
    if mount_drive:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            env_info["drive_mounted"] = True
            print("[OK] Google Drive mounted at /content/drive")
        except Exception as e:
            print(f"[WARN] Could not mount Drive: {e}")

    # 2. Set project root
    if project_root is None:
        # Auto-detect: look for common locations
        candidates = [
            Path("/content/research"),
            Path("/content/drive/MyDrive/research"),
            Path.cwd(),
        ]
        for candidate in candidates:
            if (candidate / "src" / "models").exists():
                project_root = str(candidate)
                break

    if project_root:
        # Add to Python path
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        os.chdir(project_root)
        env_info["project_root"] = project_root
        print(f"[OK] Project root: {project_root}")
    else:
        print("[WARN] Could not detect project root")

    # 3. Check GPU availability
    if use_gpu:
        try:
            import torch
            env_info["gpu_available"] = torch.cuda.is_available()
            if env_info["gpu_available"]:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[OK] GPU available: {gpu_name}")
            else:
                print("[INFO] No GPU detected - will use CPU")
        except ImportError:
            print("[INFO] PyTorch not imported yet - GPU check deferred")

    # 4. Set Colab-specific defaults
    os.environ["COLAB_ENV"] = "1"
    os.environ["PYTORCH_NUM_WORKERS"] = "0"  # Avoid multiprocessing issues

    print("=" * 60)
    print("Environment setup complete")
    print("=" * 60)

    return env_info


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


__all__ = [
    "is_colab",
    "setup_colab_environment",
    "get_trainer_for_colab",
    "train_ensemble_colab",
    "get_colab_dataloader_kwargs",
]
