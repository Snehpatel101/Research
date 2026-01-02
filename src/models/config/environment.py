"""Environment detection utilities."""

from enum import Enum


class Environment(Enum):
    """Execution environment types."""

    COLAB = "colab"
    LOCAL_CPU = "local_cpu"
    LOCAL_GPU = "local_gpu"
    UNKNOWN = "unknown"


def detect_environment() -> Environment:
    """Detect execution environment (Colab/GPU/CPU)."""
    try:
        import google.colab  # noqa: F401

        return Environment.COLAB
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return Environment.LOCAL_GPU
    except ImportError:
        pass
    return Environment.LOCAL_CPU


def is_colab() -> bool:
    """Check if running in Google Colab."""
    return detect_environment() == Environment.COLAB


def resolve_device(device_setting: str = "auto") -> str:
    """Resolve device setting to actual device ("cuda"/"cpu")."""
    if device_setting == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_setting
