"""
OOM (Out of Memory) Recovery Manager.

Provides graceful handling of CUDA out-of-memory errors during training
by automatically reducing batch size and retrying operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class OOMConfig:
    """Configuration for OOM recovery."""

    enabled: bool = True
    max_retries: int = 6  # Enough to halve from 128 down to 2 (128→64→32→16→8→4→2)
    batch_reduction_factor: float = 0.5  # Reduce batch size by this factor
    min_batch_size: int = 2  # Don't reduce below this (lowered from 8 for small datasets)
    clear_cache_on_oom: bool = True
    force_gc_on_oom: bool = True
    log_memory_stats: bool = True


@dataclass
class OOMEvent:
    """Record of an OOM event."""

    original_batch_size: int
    reduced_batch_size: int
    retry_number: int
    memory_allocated: int | None = None
    memory_reserved: int | None = None
    recovered: bool = False


class OOMRecoveryManager:
    """
    Manages OOM recovery during PyTorch training.

    Provides automatic batch size reduction and memory cleanup when
    CUDA out-of-memory errors occur.

    Example:
        manager = OOMRecoveryManager(config)
        batch_size = 256

        while True:
            try:
                train_batch(batch_size)
                manager.mark_success()
                break
            except RuntimeError as e:
                if not manager.is_oom_error(e):
                    raise
                batch_size = manager.handle_oom(batch_size)
                if batch_size is None:
                    raise
    """

    def __init__(self, config: OOMConfig | None = None) -> None:
        """
        Initialize OOM recovery manager.

        Args:
            config: OOM recovery configuration
        """
        self.config = config or OOMConfig()
        self._events: list[OOMEvent] = []
        self._current_retries = 0

    @property
    def events(self) -> list[OOMEvent]:
        """Get all recorded OOM events."""
        return list(self._events)

    @property
    def total_oom_count(self) -> int:
        """Get total number of OOM events."""
        return len(self._events)

    @staticmethod
    def is_oom_error(error: Exception) -> bool:
        """
        Check if an exception is a CUDA OOM or related GPU memory error.

        Catches not just OOM but also cuBLAS/cuDNN internal errors that
        are often caused by insufficient GPU memory or dtype issues under
        memory pressure.

        Args:
            error: The exception to check

        Returns:
            True if this is a CUDA memory-related error
        """
        if not isinstance(error, RuntimeError):
            return False

        error_msg = str(error).lower()
        oom_indicators = [
            "cuda out of memory",
            "out of memory",
            "cudnn status bad allocation",
            "failed to allocate",
            "torch.cuda.outofmemoryerror",
            "cublas_status_internal_error",
            "cublas_status_alloc_failed",
            "cudnn_status_internal_error",
        ]
        return any(indicator in error_msg for indicator in oom_indicators)

    def handle_oom(self, current_batch_size: int) -> int | None:
        """
        Handle an OOM error by reducing batch size.

        Args:
            current_batch_size: Current batch size that caused OOM

        Returns:
            New batch size to try, or None if recovery is not possible
        """
        if not self.config.enabled:
            logger.warning("OOM recovery disabled")
            return None

        if self._current_retries >= self.config.max_retries:
            logger.error(f"OOM recovery failed: max retries ({self.config.max_retries}) exceeded")
            return None

        # Calculate new batch size
        new_batch_size = int(current_batch_size * self.config.batch_reduction_factor)

        if new_batch_size < self.config.min_batch_size:
            logger.error(
                f"OOM recovery failed: batch size {new_batch_size} below minimum "
                f"{self.config.min_batch_size}"
            )
            return None

        # Clear memory
        self._clear_memory()

        # Record event
        event = OOMEvent(
            original_batch_size=current_batch_size,
            reduced_batch_size=new_batch_size,
            retry_number=self._current_retries + 1,
            memory_allocated=self._get_memory_allocated(),
            memory_reserved=self._get_memory_reserved(),
            recovered=False,  # Will be updated if successful
        )
        self._events.append(event)
        self._current_retries += 1

        logger.warning(
            f"OOM detected. Reducing batch size: {current_batch_size} -> {new_batch_size} "
            f"(retry {self._current_retries}/{self.config.max_retries})"
        )

        if self.config.log_memory_stats:
            self._log_memory_stats()

        return new_batch_size

    def mark_success(self) -> None:
        """Mark the last OOM event as successfully recovered."""
        if self._events:
            self._events[-1].recovered = True
        self._current_retries = 0

    def reset(self) -> None:
        """Reset retry counter for a new training phase."""
        self._current_retries = 0

    def _clear_memory(self) -> None:
        """Clear GPU memory and run garbage collection."""
        from src.models.device import release_gpu_memory

        release_gpu_memory()

    def _get_memory_allocated(self) -> int | None:
        """Get currently allocated GPU memory in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return None

    def _get_memory_reserved(self) -> int | None:
        """Get currently reserved GPU memory in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved()
        return None

    def _log_memory_stats(self) -> None:
        """Log current GPU memory statistics."""
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        logger.info(
            f"GPU Memory: allocated={allocated:.2f}GB, "
            f"reserved={reserved:.2f}GB, max_allocated={max_allocated:.2f}GB"
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of OOM recovery activity."""
        return {
            "enabled": self.config.enabled,
            "total_oom_events": len(self._events),
            "recovered_events": sum(1 for e in self._events if e.recovered),
            "failed_events": sum(1 for e in self._events if not e.recovered),
            "max_retries": self.config.max_retries,
            "batch_reduction_factor": self.config.batch_reduction_factor,
            "min_batch_size": self.config.min_batch_size,
            "events": [
                {
                    "original_batch": e.original_batch_size,
                    "reduced_batch": e.reduced_batch_size,
                    "retry": e.retry_number,
                    "recovered": e.recovered,
                }
                for e in self._events
            ],
        }


def create_oom_manager(
    enabled: bool = True,
    max_retries: int = 6,
    batch_reduction_factor: float = 0.5,
    min_batch_size: int = 2,
) -> OOMRecoveryManager:
    """
    Factory function to create an OOM recovery manager.

    Args:
        enabled: Whether OOM recovery is enabled
        max_retries: Maximum retry attempts
        batch_reduction_factor: Factor to reduce batch size by
        min_batch_size: Minimum allowed batch size

    Returns:
        Configured OOMRecoveryManager
    """
    config = OOMConfig(
        enabled=enabled,
        max_retries=max_retries,
        batch_reduction_factor=batch_reduction_factor,
        min_batch_size=min_batch_size,
    )
    return OOMRecoveryManager(config)


__all__ = [
    "OOMConfig",
    "OOMEvent",
    "OOMRecoveryManager",
    "create_oom_manager",
]
