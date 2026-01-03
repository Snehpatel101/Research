"""
Automated setup for Google Colab training environment.

Handles:
- Google Drive mounting
- Repository cloning/updating
- Dependency installation
- W&B authentication
- GPU detection and configuration
- Environment variable setup

Usage:
    from utils.colab_setup import setup_colab_environment

    # Run at the start of every notebook
    setup_colab_environment(
        repo_url="https://github.com/yourusername/ml-factory.git",
        drive_mount_point="/content/drive",
        wandb_project="ohlcv-ml-factory"
    )
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional


def setup_colab_environment(
    repo_url: str,
    drive_mount_point: str = "/content/drive",
    wandb_project: Optional[str] = None,
    install_extra_deps: bool = False,
    gpu_memory_growth: bool = True,
) -> dict:
    """
    Complete Colab environment setup.

    Args:
        repo_url: GitHub repository URL
        drive_mount_point: Google Drive mount path
        wandb_project: W&B project name (prompts for API key if not set)
        install_extra_deps: Install optional dependencies (heavy libraries)
        gpu_memory_growth: Enable TensorFlow GPU memory growth

    Returns:
        Environment info dict (GPU type, paths, etc.)
    """
    print("🚀 Setting up Google Colab environment...")

    env_info = {}

    # 1. Mount Google Drive
    env_info["drive_path"] = mount_google_drive(drive_mount_point)

    # 2. Clone or update repository
    env_info["repo_path"] = clone_or_update_repo(repo_url)

    # 3. Install dependencies
    install_dependencies(env_info["repo_path"], install_extra_deps)

    # 4. Detect GPU
    env_info["gpu_info"] = detect_gpu()

    # 5. Configure GPU memory (if available)
    if env_info["gpu_info"]["available"] and gpu_memory_growth:
        configure_gpu_memory_growth()

    # 6. Setup W&B
    if wandb_project:
        env_info["wandb_run"] = setup_wandb(wandb_project)

    # 7. Set environment variables
    setup_environment_variables(env_info)

    print("\n✅ Colab environment ready!")
    print_environment_summary(env_info)

    return env_info


def mount_google_drive(mount_point: str = "/content/drive") -> Path:
    """Mount Google Drive and return path."""
    print("\n📁 Mounting Google Drive...")

    try:
        from google.colab import drive

        drive.mount(mount_point, force_remount=False)
        drive_path = Path(mount_point) / "MyDrive"
        print(f"✅ Drive mounted at: {drive_path}")
        return drive_path

    except ImportError:
        warnings.warn("Not running in Colab - skipping Drive mount")
        return Path.home() / "drive"


def clone_or_update_repo(repo_url: str, target_dir: str = "/content/ml-factory") -> Path:
    """Clone repository or pull latest changes if already exists."""
    print(f"\n📦 Setting up repository from {repo_url}...")

    repo_path = Path(target_dir)

    if repo_path.exists():
        print("Repository already exists - pulling latest changes...")
        subprocess.run(
            ["git", "-C", str(repo_path), "pull"],
            check=True,
            capture_output=True,
        )
    else:
        print("Cloning repository...")
        subprocess.run(
            ["git", "clone", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
        )

    # Add to Python path
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    print(f"✅ Repository ready at: {repo_path}")
    return repo_path


def install_dependencies(repo_path: Path, install_extra: bool = False) -> None:
    """Install Python dependencies from requirements.txt."""
    print("\n📚 Installing dependencies...")

    requirements_file = repo_path / "requirements.txt"

    if requirements_file.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
            check=True,
        )
        print("✅ Core dependencies installed")
    else:
        warnings.warn(f"requirements.txt not found at {requirements_file}")

    # Install extra dependencies (heavy libraries)
    if install_extra:
        extra_deps = [
            "pytorch-lightning",
            "transformers",
            "optuna",
            "wandb",
            "dvc[gs]",  # DVC with Google Cloud Storage support
        ]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + extra_deps,
            check=True,
        )
        print("✅ Extra dependencies installed")


def detect_gpu() -> dict:
    """Detect GPU availability and type."""
    print("\n🎮 Detecting GPU...")

    gpu_info = {"available": False, "name": None, "memory_gb": None}

    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
        else:
            print("⚠️  No GPU detected - using CPU")

    except ImportError:
        warnings.warn("PyTorch not installed - cannot detect GPU")

    return gpu_info


def configure_gpu_memory_growth() -> None:
    """Configure TensorFlow GPU memory growth to avoid OOM."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ TensorFlow GPU memory growth enabled")

    except ImportError:
        pass  # TensorFlow not installed


def setup_wandb(project_name: str) -> Optional[object]:
    """Setup Weights & Biases for experiment tracking."""
    print(f"\n☁️  Setting up Weights & Biases (project: {project_name})...")

    try:
        import wandb

        # Check if already logged in
        if wandb.api.api_key is None:
            print("W&B API key not found. Please authenticate:")
            wandb.login()

        print(f"✅ W&B ready (project: {project_name})")
        return wandb

    except ImportError:
        warnings.warn("W&B not installed. Install with: pip install wandb")
        return None


def setup_environment_variables(env_info: dict) -> None:
    """Set environment variables for reproducibility."""
    # Set random seeds
    os.environ["PYTHONHASHSEED"] = "42"

    # Disable TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Set number of threads (avoid oversubscription)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    # Add repo path to PYTHONPATH
    if "repo_path" in env_info:
        os.environ["PYTHONPATH"] = str(env_info["repo_path"])


def print_environment_summary(env_info: dict) -> None:
    """Print summary of environment setup."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT SUMMARY")
    print("=" * 60)

    print(f"\n📁 Google Drive: {env_info.get('drive_path', 'Not mounted')}")
    print(f"📦 Repository: {env_info.get('repo_path', 'Not cloned')}")

    if env_info.get("gpu_info", {}).get("available"):
        gpu = env_info["gpu_info"]
        print(f"🎮 GPU: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
    else:
        print("🎮 GPU: Not available (CPU mode)")

    if env_info.get("wandb_run"):
        print("☁️  W&B: Enabled")
    else:
        print("☁️  W&B: Disabled")

    print("=" * 60 + "\n")


# Convenience functions for common tasks

def check_disk_space() -> dict:
    """Check available disk space on Colab runtime."""
    result = subprocess.run(
        ["df", "-h", "/content"],
        capture_output=True,
        text=True,
    )

    lines = result.stdout.strip().split("\n")
    if len(lines) >= 2:
        parts = lines[1].split()
        return {
            "total": parts[1],
            "used": parts[2],
            "available": parts[3],
            "use_percent": parts[4],
        }
    return {}


def estimate_training_time_remaining() -> float:
    """
    Estimate remaining time before Colab session timeout.

    Returns:
        Hours remaining (approximate)
    """
    # Colab sessions have ~12 hour limit
    # This is a rough estimate - actual timeout depends on many factors
    try:
        uptime_result = subprocess.run(
            ["cat", "/proc/uptime"],
            capture_output=True,
            text=True,
        )
        uptime_seconds = float(uptime_result.stdout.split()[0])
        uptime_hours = uptime_seconds / 3600

        # Assume 12-hour limit
        remaining_hours = max(0, 12 - uptime_hours)
        return remaining_hours

    except Exception:
        return 12.0  # Default assumption


def send_completion_notification(message: str, use_colab_notification: bool = True) -> None:
    """
    Send notification when training completes.

    Args:
        message: Notification message
        use_colab_notification: Use Colab's built-in notification system
    """
    if use_colab_notification:
        try:
            from google.colab import output

            output.eval_js(f'alert("{message}")')
        except ImportError:
            print(f"\n🔔 {message}")
    else:
        print(f"\n🔔 {message}")


# Example usage
if __name__ == "__main__":
    env_info = setup_colab_environment(
        repo_url="https://github.com/yourusername/ml-factory.git",
        wandb_project="ohlcv-ml-factory",
    )

    # Check disk space
    disk = check_disk_space()
    print(f"\n💾 Disk space: {disk.get('available', 'unknown')} available")

    # Estimate remaining time
    remaining = estimate_training_time_remaining()
    print(f"⏱️  Estimated time remaining: {remaining:.1f} hours")
