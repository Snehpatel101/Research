"""
State checkpointing for MLFactory pipeline recovery.

Enables resuming from failures by saving intermediate state after each stage.
This module provides robust checkpointing that allows long-running ML pipelines
to resume from the last successful stage rather than starting from scratch.

Key Features:
- Automatic checkpoint creation after each pipeline stage
- Config hash validation to detect configuration changes
- Artifact path tracking for stage outputs
- Simple resume logic with stage index tracking

Example:
    from src.core.checkpoint import PipelineCheckpointManager

    # Initialize for a specific run
    manager = PipelineCheckpointManager(run_dir="experiments/run_001")

    # Save after each stage
    manager.save_checkpoint(
        stage_name="feature_engineering",
        stage_index=1,
        artifacts={"features": Path("features.parquet")},
        config_hash="abc123",
        n_features=150,
    )

    # Resume from checkpoint
    if manager.has_checkpoint():
        state = manager.load_latest()
        resume_from = state.stage_index + 1
        print(f"Resuming from stage {resume_from}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """
    State saved at each checkpoint.

    Attributes:
        stage_name: Human-readable name of the completed stage
        stage_index: Zero-based index of the completed stage
        completed_at: Timestamp when checkpoint was created
        artifacts: Mapping of artifact name to file path
        config_hash: Hash of config to detect changes
        metadata: Additional stage-specific metadata
    """

    stage_name: str
    stage_index: int
    completed_at: datetime
    artifacts: dict[str, Path]
    config_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "stage_name": self.stage_name,
            "stage_index": self.stage_index,
            "completed_at": self.completed_at.isoformat(),
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "config_hash": self.config_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        """Create CheckpointState from dictionary."""
        return cls(
            stage_name=data["stage_name"],
            stage_index=data["stage_index"],
            completed_at=datetime.fromisoformat(data["completed_at"]),
            artifacts={k: Path(v) for k, v in data["artifacts"].items()},
            config_hash=data["config_hash"],
            metadata=data.get("metadata", {}),
        )


class PipelineCheckpointManager:
    """
    Manages checkpoints for pipeline recovery.

    The PipelineCheckpointManager saves state after each pipeline stage, enabling
    recovery from failures. Checkpoints include:
    - Stage name and index
    - Paths to saved artifacts
    - Configuration hash (to detect config changes)
    - Custom metadata

    Usage:
        manager = PipelineCheckpointManager(run_dir="experiments/run_001")

        # Save after each stage
        manager.save_checkpoint("features", 0, artifacts={"features": feature_path})
        manager.save_checkpoint("training", 1, artifacts={"models": model_path})

        # Resume from checkpoint
        if manager.has_checkpoint():
            state = manager.load_latest()
            if state.config_hash == current_config_hash:
                resume_from = state.stage_index + 1
            else:
                manager.clear_checkpoints()  # Config changed, start fresh

    Checkpoint Structure:
        run_dir/
        checkpoints/
            checkpoint_000_data_pipeline.json
            checkpoint_001_feature_engineering.json
            checkpoint_002_training.json
            latest.json  (symlink or copy of most recent)

    Attributes:
        run_dir: Root directory for this run
        checkpoint_dir: Directory where checkpoints are stored
    """

    # Pipeline stages in order
    STAGES = [
        "data_pipeline",
        "feature_engineering",
        "training",
        "ensemble",
        "evaluation",
        "bundling",
    ]

    def __init__(
        self,
        run_dir: str | Path,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Initialize PipelineCheckpointManager.

        Args:
            run_dir: Root directory for the experiment run
            checkpoint_dir: Subdirectory name for checkpoints (default: "checkpoints")
        """
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"PipelineCheckpointManager initialized: {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        stage_name: str,
        stage_index: int,
        artifacts: dict[str, Path],
        config_hash: str,
        **metadata: Any,
    ) -> Path:
        """
        Save checkpoint after stage completion.

        Args:
            stage_name: Name of the completed stage
            stage_index: Zero-based index of the stage
            artifacts: Dict mapping artifact names to their file paths
            config_hash: Hash of the current configuration
            **metadata: Additional metadata to store (e.g., n_features, n_samples)

        Returns:
            Path to the saved checkpoint file

        Example:
            manager.save_checkpoint(
                stage_name="training",
                stage_index=2,
                artifacts={
                    "models": Path("models/"),
                    "metrics": Path("metrics.json"),
                },
                config_hash="abc123def456",
                n_models=5,
                best_f1=0.85,
            )
        """
        state = CheckpointState(
            stage_name=stage_name,
            stage_index=stage_index,
            completed_at=datetime.now(),
            artifacts=artifacts,
            config_hash=config_hash,
            metadata=metadata,
        )

        # Save numbered checkpoint file
        checkpoint_filename = f"checkpoint_{stage_index:03d}_{stage_name}.json"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename

        with open(checkpoint_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        # Update latest pointer
        latest_path = self.checkpoint_dir / "latest.json"
        with open(latest_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        logger.info(f"Checkpoint saved: {stage_name} (stage {stage_index})")
        logger.debug(f"  Artifacts: {list(artifacts.keys())}")
        logger.debug(f"  Path: {checkpoint_path}")

        return checkpoint_path

    def load_latest(self) -> CheckpointState | None:
        """
        Load most recent checkpoint.

        Returns:
            CheckpointState if checkpoint exists, None otherwise

        Example:
            state = manager.load_latest()
            if state:
                print(f"Can resume from stage {state.stage_index + 1}")
                print(f"Last completed: {state.stage_name}")
        """
        latest_path = self.checkpoint_dir / "latest.json"

        if not latest_path.exists():
            logger.debug("No checkpoint found")
            return None

        try:
            with open(latest_path) as f:
                data = json.load(f)
            state = CheckpointState.from_dict(data)
            logger.info(f"Loaded checkpoint: {state.stage_name} (stage {state.stage_index})")
            return state
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def load_checkpoint(self, stage_index: int) -> CheckpointState | None:
        """
        Load a specific checkpoint by stage index.

        Args:
            stage_index: Zero-based index of the stage to load

        Returns:
            CheckpointState if found, None otherwise
        """
        # Find checkpoint file for this stage
        pattern = f"checkpoint_{stage_index:03d}_*.json"
        matches = list(self.checkpoint_dir.glob(pattern))

        if not matches:
            logger.debug(f"No checkpoint found for stage {stage_index}")
            return None

        checkpoint_path = matches[0]

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            return CheckpointState.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None

    def has_checkpoint(self) -> bool:
        """
        Check if any checkpoint exists.

        Returns:
            True if at least one checkpoint exists
        """
        latest_path = self.checkpoint_dir / "latest.json"
        return latest_path.exists()

    def get_resume_stage(self) -> int:
        """
        Get stage index to resume from.

        Returns the stage index that should be run next, which is
        one past the last completed checkpoint.

        Returns:
            Stage index to resume from (0 if no checkpoint exists)

        Example:
            resume_from = manager.get_resume_stage()
            for stage_idx in range(resume_from, len(stages)):
                run_stage(stage_idx)
        """
        state = self.load_latest()
        if state is None:
            return 0
        return state.stage_index + 1

    def validate_config(self, current_hash: str) -> bool:
        """
        Check if current config matches checkpoint config.

        Args:
            current_hash: Hash of current configuration

        Returns:
            True if configs match, False if different or no checkpoint

        Note:
            If configs don't match, you should typically clear checkpoints
            and start fresh to avoid mixing incompatible states.
        """
        state = self.load_latest()
        if state is None:
            return True  # No checkpoint, any config is valid

        matches = state.config_hash == current_hash
        if not matches:
            logger.warning(
                f"Config mismatch: checkpoint has {state.config_hash[:8]}..., "
                f"current is {current_hash[:8]}..."
            )
        return matches

    def clear_checkpoints(self) -> None:
        """
        Remove all checkpoints (for fresh runs).

        Use this when:
        - Configuration has changed
        - You want to force a fresh run
        - Checkpoints are corrupted
        """
        if self.checkpoint_dir.exists():
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                checkpoint_file.unlink()
            logger.info("Cleared all checkpoints")
        else:
            logger.debug("No checkpoints to clear")

    def get_all_checkpoints(self) -> list[CheckpointState]:
        """
        Load all checkpoints in order.

        Returns:
            List of CheckpointState objects, sorted by stage_index
        """
        checkpoints = []

        for checkpoint_file in sorted(self.checkpoint_dir.glob("checkpoint_*.json")):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)
                checkpoints.append(CheckpointState.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load {checkpoint_file}: {e}")

        return sorted(checkpoints, key=lambda c: c.stage_index)

    def get_artifact(self, stage_name: str, artifact_name: str) -> Path | None:
        """
        Get path to a specific artifact from a checkpoint.

        Args:
            stage_name: Name of the stage that produced the artifact
            artifact_name: Name of the artifact

        Returns:
            Path to artifact if found, None otherwise

        Example:
            features_path = manager.get_artifact("feature_engineering", "features")
            if features_path and features_path.exists():
                df = pd.read_parquet(features_path)
        """
        for checkpoint in self.get_all_checkpoints():
            if checkpoint.stage_name == stage_name:
                return checkpoint.artifacts.get(artifact_name)
        return None


def compute_config_hash(config: Any) -> str:
    """
    Compute a hash of a configuration object.

    Args:
        config: Configuration object (must be pickle-able or have to_dict())

    Returns:
        SHA256 hash of the configuration

    Example:
        hash1 = compute_config_hash(config)
        # ... later ...
        if compute_config_hash(config) != hash1:
            print("Config has changed!")
    """
    try:
        # Try to use to_dict if available (for dataclasses)
        if hasattr(config, "to_dict"):
            config_bytes = json.dumps(config.to_dict(), sort_keys=True).encode()
        elif hasattr(config, "__dict__"):
            # For regular objects, use __dict__
            config_bytes = json.dumps(
                {k: str(v) for k, v in config.__dict__.items()},
                sort_keys=True,
            ).encode()
        else:
            # Fall back to pickle
            config_bytes = pickle.dumps(config)

        return hashlib.sha256(config_bytes).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute config hash: {e}")
        # Return a random hash to force fresh run
        return hashlib.sha256(str(datetime.now()).encode()).hexdigest()


__all__ = [
    "CheckpointState",
    "PipelineCheckpointManager",
    "compute_config_hash",
]
