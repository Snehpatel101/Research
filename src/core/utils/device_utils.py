"""
Device detection utilities for GPU/CPU selection.

This module provides simple, lightweight utilities for checking CUDA availability
and getting device strings. For more comprehensive GPU management (memory estimation,
mixed precision, optimal settings), use src.models.device.DeviceManager.
"""

import logging

logger = logging.getLogger(__name__)


def check_cuda_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.

    Returns
    -------
    bool
        True if CUDA is available, False otherwise

    Examples
    --------
    >>> if check_cuda_available():
    ...     print("GPU training available")
    ... else:
    ...     print("Using CPU")
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device_string(prefer_gpu: bool = True) -> str:
    """
    Get device string for PyTorch ('cuda' or 'cpu').

    This is a simple utility for basic device selection. For more advanced
    GPU selection (multi-GPU, memory-based selection), use DeviceManager
    from src.models.device.

    Parameters
    ----------
    prefer_gpu : bool, default True
        Whether to prefer GPU if available

    Returns
    -------
    str
        Device string: 'cuda' or 'cpu'

    Examples
    --------
    >>> device = get_device_string()
    >>> print(f"Using device: {device}")
    """
    if prefer_gpu and check_cuda_available():
        return "cuda"
    return "cpu"


def get_torch_device(prefer_gpu: bool = True):
    """
    Get a torch.device object.

    Parameters
    ----------
    prefer_gpu : bool, default True
        Whether to prefer GPU if available

    Returns
    -------
    torch.device
        Device object for tensor operations

    Examples
    --------
    >>> device = get_torch_device()
    >>> tensor = torch.zeros(10).to(device)
    """
    import torch

    return torch.device(get_device_string(prefer_gpu))


__all__ = [
    "check_cuda_available",
    "get_device_string",
    "get_torch_device",
]
