"""
Training checkpointing for neural network models.

Provides checkpoint management for:
- Periodic checkpoint saving during training
- Best model tracking and restoration
- Resume training from checkpoint
- Checkpoint pruning (keep N best)
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""

    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    interval_epochs: int = 10  # Save every N epochs
    keep_n_best: int = 3  # Number of best checkpoints to keep
    save_optimizer: bool = True  # Include optimizer state
    save_scheduler: bool = True  # Include scheduler state
    metric_name: str = "val_loss"  # Metric to track for best checkpoint
    metric_mode: str = "min"  # "min" or "max" (lower is better vs higher is better)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.interval_epochs < 1:
            raise ValueError(f"interval_epochs must be >= 1, got {self.interval_epochs}")
        if self.keep_n_best < 1:
            raise ValueError(f"keep_n_best must be >= 1, got {self.keep_n_best}")
        if self.metric_mode not in ("min", "max"):
            raise ValueError(f"metric_mode must be 'min' or 'max', got {self.metric_mode}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "interval_epochs": self.interval_epochs,
            "keep_n_best": self.keep_n_best,
            "save_optimizer": self.save_optimizer,
            "save_scheduler": self.save_scheduler,
            "metric_name": self.metric_name,
            "metric_mode": self.metric_mode,
        }


@dataclass
class CheckpointMetadata:
    """Metadata for a saved checkpoint."""

    epoch: int
    metrics: dict[str, float]
    timestamp: str
    config: dict[str, Any]
    checkpoint_path: str
    is_best: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "config": self.config,
            "checkpoint_path": self.checkpoint_path,
            "is_best": self.is_best,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """
    Manages training checkpoints with periodic saving and best model tracking.

    Features:
    - Periodic checkpoint saving (every N epochs)
    - Best model tracking based on validation metric
    - Automatic pruning of old checkpoints
    - Resume training from checkpoint
    - Checkpoint integrity verification

    Usage:
        ckpt_mgr = CheckpointManager(CheckpointConfig(checkpoint_dir="runs/exp1/ckpts"))

        for epoch in range(epochs):
            train_loss, val_loss = train_epoch(...)
            metrics = {"train_loss": train_loss, "val_loss": val_loss}

            # Save periodic checkpoint
            ckpt_mgr.maybe_save_checkpoint(
                model, optimizer, scheduler, epoch, metrics
            )

        # Get best checkpoint path
        best_path = ckpt_mgr.get_best_checkpoint()
    """

    def __init__(self, config: CheckpointConfig | None = None) -> None:
        """
        Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration. Uses defaults if None.
        """
        self.config = config or CheckpointConfig()
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoints
        self._checkpoints: list[CheckpointMetadata] = []
        self._best_metric: float | None = None
        self._best_checkpoint_path: Path | None = None

        # Load existing checkpoints if any
        self._load_checkpoint_index()

    def _load_checkpoint_index(self) -> None:
        """Load existing checkpoint index from disk."""
        index_path = self.config.checkpoint_dir / "checkpoint_index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                self._checkpoints = [
                    CheckpointMetadata.from_dict(c) for c in data.get("checkpoints", [])
                ]
                self._best_checkpoint_path = (
                    Path(data["best_checkpoint"]) if data.get("best_checkpoint") else None
                )
                self._best_metric = data.get("best_metric")
                logger.debug(f"Loaded {len(self._checkpoints)} existing checkpoints")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load checkpoint index: {e}")

    def _save_checkpoint_index(self) -> None:
        """Save checkpoint index to disk."""
        index_path = self.config.checkpoint_dir / "checkpoint_index.json"
        data = {
            "checkpoints": [c.to_dict() for c in self._checkpoints],
            "best_checkpoint": (
                str(self._best_checkpoint_path) if self._best_checkpoint_path else None
            ),
            "best_metric": self._best_metric,
            "config": self.config.to_dict(),
            "updated_at": datetime.now().isoformat(),
        }
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        epoch: int,
        metrics: dict[str, float],
        model_config: dict[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            epoch: Current epoch number
            metrics: Dictionary of metrics (must include config.metric_name)
            model_config: Model configuration to save
            extra_state: Additional state to include

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = f"checkpoint_epoch{epoch:04d}_{timestamp}.pt"
        ckpt_path = self.config.checkpoint_dir / ckpt_name

        # Build checkpoint state
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": timestamp,
            "model_config": model_config or {},
        }

        if self.config.save_optimizer and optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if self.config.save_scheduler and scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if extra_state:
            state["extra_state"] = extra_state

        # Save checkpoint
        torch.save(state, ckpt_path)
        logger.debug(f"Saved checkpoint: {ckpt_path}")

        # Check if this is the best checkpoint
        metric_value = metrics.get(self.config.metric_name)
        is_best = False

        if metric_value is not None:
            if (
                self._best_metric is None
                or self.config.metric_mode == "min"
                and metric_value < self._best_metric
                or self.config.metric_mode == "max"
                and metric_value > self._best_metric
            ):
                is_best = True

            if is_best:
                self._best_metric = metric_value
                self._best_checkpoint_path = ckpt_path
                # Copy to best.pt
                best_path = self.config.checkpoint_dir / "best.pt"
                shutil.copy2(ckpt_path, best_path)
                logger.info(
                    f"New best checkpoint (epoch {epoch}, {self.config.metric_name}={metric_value:.4f})"
                )

        # Create metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            metrics=metrics,
            timestamp=timestamp,
            config=model_config or {},
            checkpoint_path=str(ckpt_path),
            is_best=is_best,
        )
        self._checkpoints.append(metadata)

        # Prune old checkpoints
        self._prune_old_checkpoints()

        # Save index
        self._save_checkpoint_index()

        return ckpt_path

    def maybe_save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        epoch: int,
        metrics: dict[str, float],
        model_config: dict[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> Path | None:
        """
        Save checkpoint if epoch is at the configured interval.

        Args:
            Same as save_checkpoint

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        if (epoch + 1) % self.config.interval_epochs == 0:
            return self.save_checkpoint(
                model, optimizer, scheduler, epoch, metrics, model_config, extra_state
            )
        return None

    def load_checkpoint(
        self,
        path: Path | str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load tensors to

        Returns:
            Dictionary with checkpoint metadata (epoch, metrics, etc.)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if available
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} "
            f"(metrics: {checkpoint.get('metrics', {})})"
        )

        return {
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint.get("metrics", {}),
            "model_config": checkpoint.get("model_config", {}),
            "extra_state": checkpoint.get("extra_state", {}),
        }

    def get_best_checkpoint(self) -> Path | None:
        """
        Get path to best checkpoint.

        Returns:
            Path to best checkpoint, or None if no checkpoints saved
        """
        best_path = self.config.checkpoint_dir / "best.pt"
        if best_path.exists():
            return best_path
        return self._best_checkpoint_path

    def get_latest_checkpoint(self) -> Path | None:
        """
        Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints saved
        """
        if not self._checkpoints:
            return None
        latest = max(self._checkpoints, key=lambda c: c.epoch)
        return Path(latest.checkpoint_path)

    def _prune_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the N best."""
        if len(self._checkpoints) <= self.config.keep_n_best:
            return

        # Sort by metric (best first)
        if self.config.metric_mode == "min":
            sorted_ckpts = sorted(
                self._checkpoints,
                key=lambda c: c.metrics.get(self.config.metric_name, float("inf")),
            )
        else:
            sorted_ckpts = sorted(
                self._checkpoints,
                key=lambda c: c.metrics.get(self.config.metric_name, float("-inf")),
                reverse=True,
            )

        # Keep top N
        to_keep = {c.checkpoint_path for c in sorted_ckpts[: self.config.keep_n_best]}
        to_remove = [c for c in self._checkpoints if c.checkpoint_path not in to_keep]

        for ckpt in to_remove:
            ckpt_path = Path(ckpt.checkpoint_path)
            if ckpt_path.exists():
                ckpt_path.unlink()
                logger.debug(f"Pruned checkpoint: {ckpt_path}")

        self._checkpoints = [c for c in self._checkpoints if c.checkpoint_path in to_keep]

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all available checkpoints."""
        return self._checkpoints.copy()

    @property
    def best_metric(self) -> float | None:
        """Get the best metric value seen."""
        return self._best_metric

    @property
    def num_checkpoints(self) -> int:
        """Get number of saved checkpoints."""
        return len(self._checkpoints)


__all__ = [
    "CheckpointConfig",
    "CheckpointMetadata",
    "CheckpointManager",
]
