#!/usr/bin/env python3
"""
One-click setup script for Google Colab.

This script handles all the setup required to run the ML Model Factory
in Google Colab, including:
- Dependency installation
- GPU detection and configuration
- Google Drive mounting (optional)
- Repository cloning or project path setup

Usage in Colab:
    !wget -q https://raw.githubusercontent.com/YOUR_REPO/main/colab_setup.py
    import colab_setup
    colab_setup.setup()

Or clone and run:
    !git clone https://github.com/YOUR_REPO.git
    %cd YOUR_REPO
    from colab_setup import setup
    setup()
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """Check if running in Kaggle."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    result = {"available": False, "name": None, "memory_gb": 0, "compute_capability": None}

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["available"] = True
            result["name"] = props.name
            result["memory_gb"] = props.total_memory / (1024 ** 3)
            result["compute_capability"] = (props.major, props.minor)
    except ImportError:
        pass

    return result


def install_packages(packages: List[str], quiet: bool = True) -> Dict[str, bool]:
    """
    Install packages using pip.

    Args:
        packages: List of package specifications
        quiet: Suppress pip output

    Returns:
        Dict mapping package names to success status
    """
    results = {}
    quiet_flag = ["-q"] if quiet else []

    for package in packages:
        package_name = package.split(">=")[0].split("==")[0].split("[")[0]
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + quiet_flag + [package],
                stdout=subprocess.DEVNULL if quiet else None,
                stderr=subprocess.DEVNULL if quiet else None,
            )
            results[package_name] = True
        except subprocess.CalledProcessError:
            results[package_name] = False
            print(f"[WARNING] Failed to install {package_name}")

    return results


def setup(
    mount_drive: bool = False,
    drive_path: str = "/content/drive",
    install_deps: bool = True,
    clone_repo: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Complete setup for Google Colab environment.

    Args:
        mount_drive: Whether to mount Google Drive
        drive_path: Path to mount Drive
        install_deps: Whether to install dependencies
        clone_repo: Optional GitHub repo URL to clone
        quiet: Suppress output

    Returns:
        Dict with setup results and environment info

    Example:
        >>> from colab_setup import setup
        >>> env = setup(mount_drive=True)
        >>> print(f"GPU: {env['gpu']['name']}")
    """
    results = {
        "is_colab": is_colab(),
        "is_kaggle": is_kaggle(),
        "gpu": get_gpu_info(),
        "drive_mounted": False,
        "packages_installed": {},
        "repo_cloned": False,
        "project_path": None,
    }

    print("=" * 60)
    print(" ML Model Factory - Colab Setup")
    print("=" * 60)

    # Environment detection
    if results["is_colab"]:
        print("[OK] Running in Google Colab")
    elif results["is_kaggle"]:
        print("[OK] Running in Kaggle")
    else:
        print("[INFO] Running in local environment")

    # GPU info
    if results["gpu"]["available"]:
        gpu = results["gpu"]
        cc = gpu["compute_capability"]
        print(f"[OK] GPU: {gpu['name']} ({gpu['memory_gb']:.1f} GB, SM {cc[0]}.{cc[1]})")

        # Recommend dtype based on compute capability
        if cc[0] >= 8:
            print("[OK] BFloat16 supported - optimal for mixed precision")
        elif cc[0] >= 7:
            print("[OK] Float16 supported - use with gradient scaler")
        else:
            print("[INFO] Mixed precision not recommended for this GPU")
    else:
        print("[WARNING] No GPU detected - training will be slow")

    # Mount Google Drive
    if mount_drive and results["is_colab"]:
        try:
            from google.colab import drive
            drive.mount(drive_path)
            results["drive_mounted"] = True
            print(f"[OK] Google Drive mounted at {drive_path}")
        except Exception as e:
            print(f"[WARNING] Failed to mount Drive: {e}")

    # Clone repository
    if clone_repo:
        try:
            repo_name = clone_repo.rstrip("/").split("/")[-1].replace(".git", "")
            if not Path(repo_name).exists():
                subprocess.check_call(["git", "clone", "-q", clone_repo])
            results["repo_cloned"] = True
            results["project_path"] = str(Path(repo_name).absolute())
            print(f"[OK] Cloned repository to {results['project_path']}")

            # Add to Python path
            sys.path.insert(0, results["project_path"])
        except Exception as e:
            print(f"[WARNING] Failed to clone repo: {e}")

    # Install dependencies
    if install_deps:
        print("\n[INFO] Installing dependencies...")

        core_packages = [
            "pandas>=1.5.0",
            "numpy>=1.22.0",
            "scipy>=1.9.0",
            "scikit-learn>=1.2.0",
            "matplotlib>=3.5.0",
            "tqdm>=4.64.0",
            "pyarrow>=10.0.0",
        ]

        boosting_packages = [
            "xgboost>=1.7.0",
            "lightgbm>=3.3.0",
            "catboost>=1.1.0",
        ]

        neural_packages = [
            "torch>=2.0.0",
        ]

        cv_packages = [
            "optuna>=3.0.0",
        ]

        all_packages = core_packages + boosting_packages + neural_packages + cv_packages
        results["packages_installed"] = install_packages(all_packages, quiet=quiet)

        success = sum(results["packages_installed"].values())
        total = len(results["packages_installed"])
        print(f"[OK] Installed {success}/{total} packages")

    # Set project path if in current directory
    if results["project_path"] is None:
        if Path("src/models").exists():
            results["project_path"] = str(Path.cwd())
            sys.path.insert(0, results["project_path"])
            print(f"[OK] Using current directory: {results['project_path']}")

    # Verify imports
    print("\n[INFO] Verifying imports...")
    try:
        from src.models import ModelRegistry
        models = ModelRegistry.list_models()
        total_models = sum(len(v) for v in models.values())
        print(f"[OK] ModelRegistry loaded: {total_models} models available")
        for family, model_list in models.items():
            print(f"     - {family}: {', '.join(model_list)}")
    except ImportError as e:
        print(f"[WARNING] Could not import ModelRegistry: {e}")
        print("[INFO] Make sure you're in the project directory")

    print("\n" + "=" * 60)
    print(" Setup Complete!")
    print("=" * 60)

    return results


def quick_test() -> bool:
    """
    Run a quick test to verify everything works.

    Returns:
        True if test passed
    """
    print("\n[INFO] Running quick test...")

    try:
        # Test imports
        from src.models import ModelRegistry, TrainerConfig
        from src.utils.notebook import setup_notebook, download_sample_data

        # Setup environment
        setup_notebook()

        # Generate sample data
        paths = download_sample_data(output_dir="/tmp/sample_data")

        # Test model creation
        model = ModelRegistry.create("xgboost")
        print(f"[OK] Created model: {model}")

        print("\n[OK] Quick test passed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Quick test failed: {e}")
        return False


# Colab-specific shortcuts
def mount_gdrive(path: str = "/content/drive") -> bool:
    """Shortcut to mount Google Drive."""
    if not is_colab():
        print("Not in Colab - skipping Drive mount")
        return False

    try:
        from google.colab import drive
        drive.mount(path)
        return True
    except Exception as e:
        print(f"Failed to mount: {e}")
        return False


def get_data_from_drive(
    source_path: str,
    dest_path: str = "data",
    drive_root: str = "/content/drive/MyDrive",
) -> bool:
    """
    Copy data from Google Drive to local storage.

    Args:
        source_path: Path relative to drive_root
        dest_path: Local destination path
        drive_root: Root path of mounted Drive

    Returns:
        True if successful
    """
    import shutil

    full_source = Path(drive_root) / source_path
    full_dest = Path(dest_path)

    if not full_source.exists():
        print(f"Source not found: {full_source}")
        return False

    full_dest.parent.mkdir(parents=True, exist_ok=True)

    if full_source.is_dir():
        shutil.copytree(full_source, full_dest, dirs_exist_ok=True)
    else:
        shutil.copy2(full_source, full_dest)

    print(f"Copied {full_source} -> {full_dest}")
    return True


if __name__ == "__main__":
    # When run directly, perform full setup
    results = setup(mount_drive=False, install_deps=True)

    if "--test" in sys.argv:
        quick_test()
