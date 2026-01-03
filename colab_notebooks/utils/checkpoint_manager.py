"""
Checkpoint manager for Google Colab training.

Handles automatic checkpointing to Google Drive and W&B to prevent data loss
on disconnections. Supports phase-level and epoch-level checkpointing.

Usage:
    from utils.checkpoint_manager import CheckpointManager

    # Initialize
    ckpt_mgr = CheckpointManager(
        drive_path="/content/drive/MyDrive/ml_factory/checkpoints",
        wandb_project="ohlcv-ml-factory",
        auto_save_interval=1800  # 30 minutes
    )

    # Save checkpoint
    ckpt_mgr.save_checkpoint(
        phase="train_xgboost",
        state={"model": model, "epoch": 10, "metrics": {...}},
        metadata={"symbol": "MES", "horizon": 20}
    )

    # Resume from checkpoint
    state = ckpt_mgr.load_latest_checkpoint(phase="train_xgboost")
"""

import json
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class CheckpointManager:
    """Manages checkpoints for Colab training with Drive/W&B sync."""

    def __init__(
        self,
        drive_path: str,
        wandb_project: Optional[str] = None,
        auto_save_interval: int = 1800,  # 30 minutes
        max_checkpoints: int = 3,  # Keep last 3 checkpoints
    ):
        """
        Initialize checkpoint manager.

        Args:
            drive_path: Google Drive path for checkpoint storage
            wandb_project: W&B project name (optional, but recommended)
            auto_save_interval: Auto-save interval in seconds
            max_checkpoints: Maximum checkpoints to keep per phase
        """
        self.drive_path = Path(drive_path)
        self.drive_path.mkdir(parents=True, exist_ok=True)

        self.wandb_project = wandb_project
        self.auto_save_interval = auto_save_interval
        self.max_checkpoints = max_checkpoints

        self.last_save_time = time.time()
        self.wandb_run = None

        # Initialize W&B if provided
        if self.wandb_project:
            try:
                import wandb

                self.wandb = wandb
            except ImportError:
                warnings.warn(
                    "W&B not installed. Install with: pip install wandb"
                )
                self.wandb_project = None

    def init_wandb_run(
        self,
        run_name: str,
        config: Dict[str, Any],
        tags: Optional[list] = None,
    ) -> None:
        """
        Initialize W&B run for experiment tracking.

        Args:
            run_name: W&B run name
            config: Experiment configuration
            tags: Optional tags for organization
        """
        if self.wandb_project and self.wandb:
            self.wandb_run = self.wandb.init(
                project=self.wandb_project,
                name=run_name,
                config=config,
                tags=tags or [],
                resume="allow",  # Allow resuming if run exists
            )
            print(f"✅ W&B run initialized: {self.wandb_run.url}")

    def save_checkpoint(
        self,
        phase: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Path:
        """
        Save checkpoint to Drive (and optionally W&B).

        Args:
            phase: Pipeline phase name (e.g., "train_xgboost", "phase_3_features")
            state: State dictionary to save (model, optimizer, epoch, etc.)
            metadata: Optional metadata (symbol, horizon, config, etc.)
            force: Force save even if auto_save_interval not elapsed

        Returns:
            Path to saved checkpoint
        """
        # Check if auto-save interval elapsed
        time_since_last_save = time.time() - self.last_save_time
        if not force and time_since_last_save < self.auto_save_interval:
            return None

        # Create checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{phase}_{timestamp}.pkl"
        checkpoint_path = self.drive_path / phase / checkpoint_name

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "phase": phase,
            "state": state,
            "metadata": metadata or {},
            "timestamp": timestamp,
        }

        # Save to Drive
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Save metadata as JSON (human-readable)
        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "phase": phase,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                },
                f,
                indent=2,
            )

        # Upload to W&B if enabled
        if self.wandb_run:
            artifact = self.wandb.Artifact(
                name=f"checkpoint_{phase}",
                type="checkpoint",
                metadata=metadata or {},
            )
            artifact.add_file(str(checkpoint_path))
            self.wandb_run.log_artifact(artifact)
            print(f"☁️  Checkpoint uploaded to W&B")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(phase)

        self.last_save_time = time.time()
        return checkpoint_path

    def load_latest_checkpoint(
        self, phase: str, prefer_wandb: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load latest checkpoint for a phase.

        Args:
            phase: Pipeline phase name
            prefer_wandb: Try to load from W&B first (slower but more reliable)

        Returns:
            Checkpoint state dict, or None if no checkpoint exists
        """
        if prefer_wandb and self.wandb_run:
            return self._load_from_wandb(phase)

        # Load from Drive
        phase_dir = self.drive_path / phase
        if not phase_dir.exists():
            print(f"⚠️  No checkpoints found for phase '{phase}'")
            return None

        checkpoints = sorted(phase_dir.glob(f"{phase}_*.pkl"))
        if not checkpoints:
            print(f"⚠️  No checkpoints found for phase '{phase}'")
            return None

        latest_checkpoint = checkpoints[-1]
        print(f"📂 Loading checkpoint: {latest_checkpoint}")

        with open(latest_checkpoint, "rb") as f:
            checkpoint = pickle.load(f)

        return checkpoint

    def _load_from_wandb(self, phase: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint from W&B artifacts."""
        try:
            artifact = self.wandb_run.use_artifact(
                f"checkpoint_{phase}:latest"
            )
            artifact_dir = artifact.download()
            checkpoint_files = list(Path(artifact_dir).glob("*.pkl"))
            if checkpoint_files:
                with open(checkpoint_files[0], "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load from W&B: {e}")
            return None

    def _cleanup_old_checkpoints(self, phase: str) -> None:
        """Keep only the N most recent checkpoints."""
        phase_dir = self.drive_path / phase
        checkpoints = sorted(phase_dir.glob(f"{phase}_*.pkl"))

        if len(checkpoints) > self.max_checkpoints:
            for old_checkpoint in checkpoints[: -self.max_checkpoints]:
                old_checkpoint.unlink()
                # Delete corresponding JSON metadata
                old_checkpoint.with_suffix(".json").unlink(missing_ok=True)
                print(f"🗑️  Deleted old checkpoint: {old_checkpoint.name}")

    def should_auto_save(self) -> bool:
        """Check if auto-save interval has elapsed."""
        return (
            time.time() - self.last_save_time
        ) >= self.auto_save_interval

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

    def finish_wandb_run(self) -> None:
        """Finish W&B run."""
        if self.wandb_run:
            self.wandb_run.finish()
            print("✅ W&B run finished")


# Example usage in training loop
if __name__ == "__main__":
    # Initialize checkpoint manager
    ckpt_mgr = CheckpointManager(
        drive_path="/content/drive/MyDrive/ml_factory/checkpoints",
        wandb_project="ohlcv-ml-factory",
        auto_save_interval=1800,  # 30 minutes
    )

    # Initialize W&B run
    ckpt_mgr.init_wandb_run(
        run_name="xgboost_MES_h20",
        config={"symbol": "MES", "horizon": 20, "model": "xgboost"},
        tags=["boosting", "MES"],
    )

    # Training loop
    for epoch in range(100):
        # ... train model ...

        # Log metrics
        ckpt_mgr.log_metrics(
            {"train_loss": 0.5, "val_loss": 0.6}, step=epoch
        )

        # Auto-save checkpoint
        if ckpt_mgr.should_auto_save():
            ckpt_mgr.save_checkpoint(
                phase="train_xgboost",
                state={
                    "epoch": epoch,
                    "model_state": "...",  # Actual model state
                    "optimizer_state": "...",
                },
                metadata={"symbol": "MES", "horizon": 20},
            )

    # Finish W&B run
    ckpt_mgr.finish_wandb_run()
