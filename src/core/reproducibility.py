"""
Reproducibility controls for ML training.

Provides comprehensive seed management across all random number generators:
- Python's random module
- NumPy random
- PyTorch random (CPU and CUDA)
- CuDNN determinism settings

Usage:
    from src.core.reproducibility import set_all_seeds, get_reproducibility_info

    # Set seeds for reproducibility
    set_all_seeds(42, deterministic=True)

    # Get current reproducibility state for logging
    info = get_reproducibility_info()
    logger.info(f"Reproducibility: {info}")
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility settings."""

    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True  # Set to False for determinism (reduces performance)
    warn_only: bool = False  # Warn instead of error on non-deterministic ops

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "benchmark": self.benchmark,
            "warn_only": self.warn_only,
        }


@dataclass
class ReproducibilityInfo:
    """Information about current reproducibility state."""

    seed: int
    python_random_seed_set: bool
    numpy_seed_set: bool
    torch_seed_set: bool
    cuda_seed_set: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: str | None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "seed": self.seed,
            "python_random_seed_set": self.python_random_seed_set,
            "numpy_seed_set": self.numpy_seed_set,
            "torch_seed_set": self.torch_seed_set,
            "cuda_seed_set": self.cuda_seed_set,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "timestamp": self.timestamp,
        }


# Module-level tracking of seed state
_seed_state: dict[str, Any] = {
    "seed": None,
    "python_set": False,
    "numpy_set": False,
    "torch_set": False,
    "cuda_set": False,
}


def set_all_seeds(seed: int, deterministic: bool = False) -> ReproducibilityInfo:
    """
    Set seeds across all random number generators for full reproducibility.

    This function sets seeds for:
    - Python's `random` module
    - NumPy's random number generator
    - PyTorch's random number generator (CPU)
    - PyTorch's CUDA random number generator (if available)
    - Optionally enables deterministic mode for PyTorch operations

    Args:
        seed: Random seed to use (must be non-negative integer)
        deterministic: If True, enables deterministic behavior for PyTorch operations.
                      This may reduce performance but ensures reproducibility.
                      Sets `torch.use_deterministic_algorithms(True)` and
                      `torch.backends.cudnn.deterministic = True`.

    Returns:
        ReproducibilityInfo with details about what was set

    Example:
        >>> from src.core.reproducibility import set_all_seeds
        >>> info = set_all_seeds(42, deterministic=True)
        >>> print(f"Seed set: {info.seed}")
    """
    global _seed_state

    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    # Set Python random seed
    random.seed(seed)
    _seed_state["python_set"] = True

    # Set NumPy random seed
    np.random.seed(seed)
    _seed_state["numpy_set"] = True

    # Set PyTorch seeds
    torch_available = False
    cuda_available = False
    cuda_set = False
    torch_version = "not installed"
    cuda_version = None
    cudnn_version = None
    cudnn_deterministic = False
    cudnn_benchmark = True

    try:
        import torch

        torch_available = True
        torch_version = torch.__version__

        # Set PyTorch CPU seed
        torch.manual_seed(seed)
        _seed_state["torch_set"] = True

        # Set PyTorch CUDA seeds if available
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            cuda_set = True
            _seed_state["cuda_set"] = True

            # Get CUDA/cuDNN versions
            cuda_version = torch.version.cuda
            cudnn_version = (
                str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
            )

        # Set deterministic mode if requested
        if deterministic:
            # Enable deterministic algorithms
            torch.use_deterministic_algorithms(True, warn_only=True)

            # cuDNN deterministic settings
            if cuda_available:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                cudnn_deterministic = True
                cudnn_benchmark = False

            # Set environment variable for CUBLAS workspace config
            # Required for some deterministic operations on CUDA >= 10.2
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            logger.info(
                f"Deterministic mode enabled (seed={seed}). "
                "This may reduce performance but ensures reproducibility."
            )
        else:
            cudnn_deterministic = torch.backends.cudnn.deterministic if cuda_available else False
            cudnn_benchmark = torch.backends.cudnn.benchmark if cuda_available else True

    except ImportError:
        logger.debug("PyTorch not installed, skipping torch seed setting")

    _seed_state["seed"] = seed

    info = ReproducibilityInfo(
        seed=seed,
        python_random_seed_set=True,
        numpy_seed_set=True,
        torch_seed_set=torch_available,
        cuda_seed_set=cuda_set,
        cudnn_deterministic=cudnn_deterministic,
        cudnn_benchmark=cudnn_benchmark,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
    )

    logger.info(
        f"Seeds set: random={seed}, numpy={seed}, "
        f"torch={'yes' if torch_available else 'no'}, "
        f"cuda={'yes' if cuda_set else 'no'}, "
        f"deterministic={deterministic}"
    )

    return info


def get_reproducibility_info() -> ReproducibilityInfo:
    """
    Get current reproducibility state information.

    Returns information about:
    - What seeds have been set
    - PyTorch/CUDA versions
    - Determinism settings

    Returns:
        ReproducibilityInfo with current state
    """
    global _seed_state

    torch_version = "not installed"
    cuda_available = False
    cuda_version = None
    cudnn_version = None
    cudnn_deterministic = False
    cudnn_benchmark = True

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda
            cudnn_version = (
                str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
            )
            cudnn_deterministic = torch.backends.cudnn.deterministic
            cudnn_benchmark = torch.backends.cudnn.benchmark

    except ImportError:
        pass

    return ReproducibilityInfo(
        seed=_seed_state.get("seed", -1),
        python_random_seed_set=_seed_state.get("python_set", False),
        numpy_seed_set=_seed_state.get("numpy_set", False),
        torch_seed_set=_seed_state.get("torch_set", False),
        cuda_seed_set=_seed_state.get("cuda_set", False),
        cudnn_deterministic=cudnn_deterministic,
        cudnn_benchmark=cudnn_benchmark,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
    )


def ensure_reproducibility(seed: int | None = None) -> ReproducibilityInfo:
    """
    Ensure reproducibility is configured, optionally with a specific seed.

    If seed is None and seeds have already been set, does nothing.
    If seed is provided, sets all seeds to that value.
    If no seeds have been set and no seed is provided, uses default seed 42.

    Args:
        seed: Optional seed to set. If None, uses existing or default.

    Returns:
        ReproducibilityInfo with current state
    """
    global _seed_state

    if seed is not None:
        return set_all_seeds(seed)

    if _seed_state.get("seed") is None:
        # No seed set yet, use default
        return set_all_seeds(42)

    # Seeds already set, return current info
    return get_reproducibility_info()


def get_worker_init_fn(seed: int):
    """
    Get a worker_init_fn for DataLoader reproducibility.

    When using multiple workers in DataLoader, each worker needs its own
    seed to ensure reproducibility. This function returns a callable that
    properly seeds each worker.

    Args:
        seed: Base seed (each worker gets seed + worker_id)

    Returns:
        Callable for DataLoader worker_init_fn parameter

    Example:
        >>> from src.core.reproducibility import get_worker_init_fn, set_all_seeds
        >>> set_all_seeds(42)
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=get_worker_init_fn(42)
        ... )
    """

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        try:
            import torch

            torch.manual_seed(worker_seed)
        except ImportError:
            pass

    return worker_init_fn


__all__ = [
    "ReproducibilityConfig",
    "ReproducibilityInfo",
    "set_all_seeds",
    "get_reproducibility_info",
    "ensure_reproducibility",
    "get_worker_init_fn",
]
