"""
Pipeline utilities and data classes.

This module contains shared data structures and helper functions used across
the pipeline stages.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StageStatus(StrEnum):
    """Status of a pipeline stage execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of executing a pipeline stage."""

    stage_name: str
    status: StageStatus
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    artifacts: list[Path] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "artifacts": [str(p) for p in self.artifacts],
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageResult":
        """Create StageResult from dictionary."""
        return cls(
            stage_name=data["stage_name"],
            status=StageStatus(data["status"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            duration_seconds=data.get("duration_seconds", 0.0),
            artifacts=[Path(p) for p in data.get("artifacts", [])],
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


def create_stage_result(
    stage_name: str,
    start_time: datetime,
    artifacts: list[Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> StageResult:
    """
    Create a successful StageResult.

    Args:
        stage_name: Name of the stage
        start_time: When the stage started
        artifacts: List of artifact paths produced
        metadata: Additional metadata

    Returns:
        StageResult with COMPLETED status
    """
    end_time = datetime.now()
    return StageResult(
        stage_name=stage_name,
        status=StageStatus.COMPLETED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=(end_time - start_time).total_seconds(),
        artifacts=artifacts or [],
        metadata=metadata or {},
    )


def create_failed_result(
    stage_name: str, start_time: datetime, error: str, artifacts: list[Path] | None = None
) -> StageResult:
    """
    Create a failed StageResult.

    Args:
        stage_name: Name of the stage
        start_time: When the stage started
        error: Error message
        artifacts: Any partial artifacts produced

    Returns:
        StageResult with FAILED status
    """
    end_time = datetime.now()
    return StageResult(
        stage_name=stage_name,
        status=StageStatus.FAILED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=(end_time - start_time).total_seconds(),
        artifacts=artifacts or [],
        error=error,
    )


# =============================================================================
# CONFIG SNAPSHOT UTILITIES (DATA-004)
# =============================================================================

# Keys to exclude from config snapshots (secrets, credentials, absolute paths)
CONFIG_SNAPSHOT_EXCLUDE_KEYS = {
    "password",
    "secret",
    "token",
    "api_key",
    "credential",
    "auth",
}

# Path-related keys to sanitize (convert to relative or basename only)
CONFIG_SNAPSHOT_PATH_KEYS = {
    "project_root",
    "data_dir",
    "raw_data_dir",
    "clean_data_dir",
    "features_dir",
    "splits_dir",
    "output_dir",
    "model_path",
    "checkpoint_dir",
}


def _serialize_config_value(value: Any) -> Any:
    """
    Serialize a config value to JSON-compatible format.

    Handles common types: primitives, lists, dicts, Paths, dataclasses.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        return [_serialize_config_value(v) for v in value]

    if isinstance(value, dict):
        return {k: _serialize_config_value(v) for k, v in value.items()}

    if isinstance(value, Path):
        return value.name  # Return basename only

    if hasattr(value, "__dict__"):
        # Recursively serialize objects with __dict__
        return {
            k: _serialize_config_value(v) for k, v in vars(value).items() if not k.startswith("_")
        }

    # Fallback: convert to string
    return str(value)
