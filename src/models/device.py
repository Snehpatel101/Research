"""
Device utilities for GPU detection and memory management.

Supports any NVIDIA GPU (GTX 10xx, RTX 20xx/30xx/40xx, Tesla T4/V100/A100/H100).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================


def is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """Detect if running in Kaggle."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def is_notebook() -> bool:
    """Detect if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except (ImportError, NameError):
        return False


def get_environment_info() -> dict[str, Any]:
    """Get information about the current environment."""
    return {
        "is_colab": is_colab(),
        "is_kaggle": is_kaggle(),
        "is_notebook": is_notebook(),
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
    }


def setup_colab(mount_drive: bool = False, install_packages: list | None = None) -> dict[str, Any]:
    """Setup Google Colab environment for ML training."""
    results = {
        "is_colab": is_colab(),
        "drive_mounted": False,
        "packages_installed": [],
        "gpu_info": None,
    }

    if not is_colab():
        logger.warning("setup_colab called but not running in Colab")
        return results

    if mount_drive:
        try:
            from google.colab import drive

            drive.mount("/content/drive")
            results["drive_mounted"] = True
        except Exception as e:
            logger.warning(f"Failed to mount Google Drive: {e}")

    if install_packages:
        import subprocess

        for package in install_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
                results["packages_installed"].append(package)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install {package}: {e}")

    gpu_info = get_best_gpu()
    if gpu_info:
        results["gpu_info"] = {
            "name": gpu_info.name,
            "memory_gb": gpu_info.total_memory_gb,
            "compute_capability": gpu_info.compute_capability,
            "supports_bf16": gpu_info.supports_bf16,
        }
    return results


# =============================================================================
# GPU DETECTION
# =============================================================================


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    compute_capability: tuple[int, int]
    supports_fp16: bool
    supports_bf16: bool

    @property
    def is_ampere_or_newer(self) -> bool:
        """Check if GPU is Ampere (SM 8.0) or newer."""
        return self.compute_capability[0] >= 8

    @property
    def is_volta_or_newer(self) -> bool:
        """Check if GPU is Volta (SM 7.0) or newer."""
        return self.compute_capability[0] >= 7

    @property
    def generation(self) -> str:
        """Get GPU generation name."""
        major, minor = self.compute_capability
        gens = {
            (6, 0): "Pascal",
            (6, 1): "Pascal",
            (6, 2): "Pascal",
            (7, 0): "Volta",
            (7, 2): "Volta",
            (7, 5): "Turing",
            (8, 0): "Ampere",
            (8, 6): "Ampere",
            (8, 7): "Ampere",
            (8, 9): "Ada Lovelace",
            (9, 0): "Hopper",
        }
        return gens.get((major, minor), f"Unknown (SM {major}.{minor})")


# Common GPU profiles for batch size recommendations
GPU_PROFILES = {
    "Tesla T4": {"vram_gb": 15, "cc": (7, 5), "batch_lstm": 256, "batch_tcn": 128},
    "Tesla V100": {"vram_gb": 16, "cc": (7, 0), "batch_lstm": 512, "batch_tcn": 256},
    "A100": {"vram_gb": 40, "cc": (8, 0), "batch_lstm": 1024, "batch_tcn": 512},
    "RTX 4090": {"vram_gb": 24, "cc": (8, 9), "batch_lstm": 512, "batch_tcn": 256},
    "RTX 4070 Ti": {"vram_gb": 12, "cc": (8, 9), "batch_lstm": 256, "batch_tcn": 128},
    "RTX 3090": {"vram_gb": 24, "cc": (8, 6), "batch_lstm": 512, "batch_tcn": 256},
    "RTX 3080": {"vram_gb": 10, "cc": (8, 6), "batch_lstm": 256, "batch_tcn": 128},
    "RTX 2080 Ti": {"vram_gb": 11, "cc": (7, 5), "batch_lstm": 256, "batch_tcn": 128},
    "GTX 1080 Ti": {"vram_gb": 11, "cc": (6, 1), "batch_lstm": 128, "batch_tcn": 64},
}


def detect_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_count() -> int:
    """Get number of available GPUs."""
    if not detect_cuda_available():
        return 0
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def get_gpu_info(device_index: int = 0) -> GPUInfo | None:
    """Get detailed information about a GPU."""
    if not detect_cuda_available():
        return None
    try:
        import torch

        if device_index >= torch.cuda.device_count():
            return None
        props = torch.cuda.get_device_properties(device_index)
        total_memory = props.total_memory / (1024**3)
        torch.cuda.set_device(device_index)
        free_memory = torch.cuda.mem_get_info(device_index)[0] / (1024**3)
        cc = (props.major, props.minor)
        return GPUInfo(
            index=device_index,
            name=props.name,
            total_memory_gb=total_memory,
            free_memory_gb=free_memory,
            compute_capability=cc,
            supports_fp16=cc >= (7, 0),
            supports_bf16=cc >= (8, 0),
        )
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return None


def get_best_gpu() -> GPUInfo | None:
    """Get the best available GPU (most free memory)."""
    n_gpus = get_gpu_count()
    if n_gpus == 0:
        return None
    best_gpu, best_free = None, -1
    for i in range(n_gpus):
        info = get_gpu_info(i)
        if info and info.free_memory_gb > best_free:
            best_free = info.free_memory_gb
            best_gpu = info
    return best_gpu


def get_device(prefer_gpu: bool = True) -> str:
    """Get best available device string."""
    if prefer_gpu and detect_cuda_available():
        best = get_best_gpu()
        return f"cuda:{best.index}" if best else "cuda:0"
    return "cpu"


# =============================================================================
# MIXED PRECISION CONFIGURATION
# =============================================================================


def get_amp_dtype(gpu_info: GPUInfo | None = None):
    """Get appropriate AMP dtype: bfloat16 for Ampere+, float16 for Volta/Turing, float32 otherwise."""
    import torch

    if gpu_info is None:
        gpu_info = get_best_gpu()
    if gpu_info is None:
        return torch.float32
    if gpu_info.supports_bf16:
        return torch.bfloat16
    if gpu_info.supports_fp16:
        return torch.float16
    return torch.float32


def get_mixed_precision_config(gpu_info: GPUInfo | None = None) -> dict[str, Any]:
    """Get mixed precision configuration based on GPU capabilities."""
    if gpu_info is None:
        gpu_info = get_best_gpu()
    if gpu_info is None:
        return {
            "enabled": False,
            "dtype": "float32",
            "use_amp": False,
            "grad_scaler": False,
            "reason": "No GPU",
        }

    if gpu_info.supports_bf16:
        return {
            "enabled": True,
            "dtype": "bfloat16",
            "use_amp": True,
            "grad_scaler": False,
            "reason": f"{gpu_info.generation} GPU supports native BF16",
        }
    if gpu_info.supports_fp16:
        return {
            "enabled": True,
            "dtype": "float16",
            "use_amp": True,
            "grad_scaler": True,
            "reason": f"{gpu_info.generation} GPU supports FP16 with gradient scaling",
        }
    return {
        "enabled": False,
        "dtype": "float32",
        "use_amp": False,
        "grad_scaler": False,
        "reason": f"{gpu_info.generation} GPU does not efficiently support mixed precision",
    }


# =============================================================================
# OPTIMAL GPU SETTINGS
# =============================================================================


def get_optimal_gpu_settings(model_family: str, gpu_info: GPUInfo | None = None) -> dict[str, Any]:
    """Get optimal training settings based on detected GPU capabilities."""
    if gpu_info is None:
        gpu_info = get_best_gpu()
    family = model_family.lower()

    if gpu_info is None:
        return {"batch_size": 32, "mixed_precision": False, "num_workers": 4, "pin_memory": False}

    mp_config = get_mixed_precision_config(gpu_info)
    vram = gpu_info.total_memory_gb
    vram_scale = vram / 12.0  # Reference: 12GB = 256 batch for LSTM

    if family == "boosting":
        return {
            "batch_size": "N/A",
            "mixed_precision": False,
            "num_workers": min(8, os.cpu_count() or 4),
        }
    elif family in ("lstm", "gru"):
        return {
            "batch_size": max(32, min(1024, int(256 * vram_scale))),
            "sequence_length": 60,
            "hidden_size": 256 if vram >= 10 else 128,
            "num_layers": 2,
            "mixed_precision": mp_config["enabled"],
            "amp_dtype": mp_config["dtype"],
            "grad_scaler": mp_config["grad_scaler"],
            "num_workers": 4,
            "pin_memory": True,
        }
    elif family == "tcn":
        return {
            "batch_size": max(16, min(512, int(128 * vram_scale))),
            "sequence_length": 120,
            "num_channels": [64, 64, 64, 64] if vram >= 10 else [32, 32, 32, 32],
            "mixed_precision": mp_config["enabled"],
            "amp_dtype": mp_config["dtype"],
            "grad_scaler": mp_config["grad_scaler"],
            "num_workers": 4,
            "pin_memory": True,
        }
    elif family in ("transformer", "patchtst", "informer"):
        return {
            "batch_size": max(8, min(256, int(64 * vram_scale))),
            "sequence_length": 128,
            "d_model": 256 if vram >= 16 else 128,
            "n_layers": 3 if vram >= 16 else 2,
            "n_heads": 8 if vram >= 16 else 4,
            "mixed_precision": mp_config["enabled"],
            "amp_dtype": mp_config["dtype"],
            "grad_scaler": mp_config["grad_scaler"],
            "num_workers": 4,
            "pin_memory": True,
        }
    return {
        "batch_size": max(16, min(128, int(64 * vram_scale))),
        "mixed_precision": mp_config["enabled"],
        "amp_dtype": mp_config["dtype"],
        "grad_scaler": mp_config["grad_scaler"],
        "num_workers": 4,
        "pin_memory": True,
    }


def estimate_memory_requirements(
    model_family: str,
    batch_size: int,
    sequence_length: int,
    n_features: int,
    hidden_size: int = 128,
    num_layers: int = 2,
) -> dict[str, float]:
    """Estimate GPU memory requirements for training."""
    family = model_family.lower()
    if family == "boosting":
        return {"total_memory_gb": 0.0, "recommended_batch_size": batch_size, "note": "CPU-based"}

    # Estimate parameters
    if family in ("lstm", "gru"):
        gate_mult = 4 if family == "lstm" else 3
        params = (
            gate_mult * hidden_size * (n_features + hidden_size + 1) * num_layers + hidden_size * 3
        )
    elif family == "tcn":
        params = (
            sum(
                n_features * hidden_size * 3 * 2 if i == 0 else hidden_size * hidden_size * 3 * 2
                for i in range(num_layers)
            )
            + hidden_size * 3
        )
    else:
        params = (
            12 * hidden_size * hidden_size * num_layers
            + sequence_length * hidden_size
            + hidden_size * 3
        )

    activation_mult = {"lstm": 4.0, "gru": 3.0, "tcn": 2.5, "transformer": 6.0}.get(family, 4.0)
    model_mem = (params * 4) / (1024**3)
    activation_mem = (batch_size * sequence_length * hidden_size * activation_mult * 4) / (1024**3)
    total = 1.2 * (model_mem * 4 + activation_mem)  # model + grad + optimizer (2x)

    gpu_info = get_best_gpu()
    target = (gpu_info.total_memory_gb * 0.8) if gpu_info else 8.0
    rec_batch = max(1, int(batch_size * target / total)) if total > target else batch_size

    return {
        "total_memory_gb": round(total, 3),
        "recommended_batch_size": rec_batch,
        "estimated_params": params,
    }


def get_optimal_batch_size(
    model_family: str,
    gpu_memory_gb: float | None = None,
    sequence_length: int = 60,
    n_features: int = 80,
    hidden_size: int = 128,
    target_utilization: float = 0.8,
) -> int:
    """Calculate optimal batch size for available GPU memory."""
    if gpu_memory_gb is None:
        gpu_info = get_best_gpu()
        gpu_memory_gb = gpu_info.total_memory_gb if gpu_info else 8.0
    target = gpu_memory_gb * target_utilization
    low, high, best = 1, 4096, 32
    while low <= high:
        mid = (low + high) // 2
        est = estimate_memory_requirements(
            model_family, mid, sequence_length, n_features, hidden_size
        )
        if est["total_memory_gb"] <= target:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def print_gpu_info() -> None:
    """Print GPU information to logger."""
    if not detect_cuda_available():
        logger.info("No CUDA GPU available")
        return
    n_gpus = get_gpu_count()
    logger.info(f"Found {n_gpus} CUDA GPU(s)")
    for i in range(n_gpus):
        info = get_gpu_info(i)
        if info:
            mp = get_mixed_precision_config(info)
            logger.info(
                f"  [{i}] {info.name}: {info.total_memory_gb:.1f}GB, "
                f"CC {info.compute_capability[0]}.{info.compute_capability[1]} ({info.generation}), "
                f"AMP: {mp['dtype']}"
            )


def get_training_device_config(model_family: str, prefer_gpu: bool = True) -> dict[str, Any]:
    """Get complete device configuration for training."""
    device = get_device(prefer_gpu)
    gpu_info = get_best_gpu() if "cuda" in device else None
    if gpu_info:
        settings = get_optimal_gpu_settings(model_family, gpu_info)
        mp = get_mixed_precision_config(gpu_info)
        return {
            "device": device,
            "gpu_name": gpu_info.name,
            "gpu_generation": gpu_info.generation,
            "gpu_memory_gb": gpu_info.total_memory_gb,
            "compute_capability": gpu_info.compute_capability,
            "batch_size": settings.get("batch_size", 64),
            "mixed_precision": mp,
            "num_workers": settings.get("num_workers", 4),
            "pin_memory": settings.get("pin_memory", True),
        }
    return {
        "device": "cpu",
        "gpu_name": None,
        "gpu_generation": None,
        "gpu_memory_gb": 0,
        "compute_capability": None,
        "batch_size": 64,
        "mixed_precision": {"enabled": False, "dtype": "float32"},
        "num_workers": 4,
        "pin_memory": False,
    }


class DeviceManager:
    """Centralized device management for training with automatic precision selection."""

    def __init__(self, prefer_gpu: bool = True):
        self._gpu_info = get_best_gpu() if prefer_gpu else None
        self._device_str = get_device(prefer_gpu)
        self._mp_config = get_mixed_precision_config(self._gpu_info)
        import torch

        self._device = torch.device(self._device_str)
        self._amp_dtype = get_amp_dtype(self._gpu_info)
        self._scaler = (
            torch.amp.GradScaler("cuda")
            if self._mp_config["grad_scaler"] and self._device.type == "cuda"
            else None
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def device_str(self) -> str:
        return self._device_str

    @property
    def gpu_info(self) -> GPUInfo | None:
        return self._gpu_info

    @property
    def amp_dtype(self):
        return self._amp_dtype

    @property
    def amp_enabled(self) -> bool:
        return self._mp_config["enabled"]

    @property
    def scaler(self) -> torch.amp.GradScaler | None:
        return self._scaler

    @property
    def mixed_precision_config(self) -> dict[str, Any]:
        return self._mp_config.copy()

    def autocast(self):
        """Get autocast context manager for mixed precision."""
        import torch

        if self._device.type == "cuda" and self._mp_config["enabled"]:
            return torch.amp.autocast("cuda", dtype=self._amp_dtype)
        return torch.amp.autocast("cpu", enabled=False)

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)

    def get_optimal_settings(self, model_family: str) -> dict[str, Any]:
        return get_optimal_gpu_settings(model_family, self._gpu_info)

    def __repr__(self) -> str:
        gpu_str = f"{self._gpu_info.name}" if self._gpu_info else "None"
        return (
            f"DeviceManager(device={self._device_str}, gpu={gpu_str}, amp_dtype={self._amp_dtype})"
        )


__all__ = [
    "is_colab",
    "is_kaggle",
    "is_notebook",
    "get_environment_info",
    "setup_colab",
    "GPUInfo",
    "GPU_PROFILES",
    "detect_cuda_available",
    "get_gpu_count",
    "get_gpu_info",
    "get_best_gpu",
    "get_device",
    "estimate_memory_requirements",
    "get_optimal_batch_size",
    "get_amp_dtype",
    "get_mixed_precision_config",
    "get_optimal_gpu_settings",
    "print_gpu_info",
    "get_training_device_config",
    "DeviceManager",
]
