"""
Pipeline State Management.

Robust state tracking for ML pipeline execution with:
- Schema versioning for forward/backward compatibility
- Detailed phase tracking with IN_PROGRESS, COMPLETED, FAILED, SKIPPED states
- Thread-safe state updates
- Type-preserving JSON serialization
- State validation and rollback capabilities
- State comparison for debugging
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StateVersion(Enum):
    """Schema version for state serialization.

    Enables forward/backward compatibility as state schema evolves.
    """
    V1 = "v1"  # Original simple state (legacy)
    V2 = "v2"  # Current robust state with PhaseResult


class PhaseState(Enum):
    """Execution state of a pipeline phase."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PhaseResult:
    """
    Detailed result of a pipeline phase execution.

    Tracks timing, status, metrics, and checkpoints for each phase.
    Immutable once created (use factory methods to create new states).
    """
    status: PhaseState
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    error_message: str | None = None
    skip_reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with type markers for reconstruction."""
        return {
            "__type__": "PhaseResult",
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "skip_reason": self.skip_reason,
            "metrics": self.metrics,
            "checkpoints": self.checkpoints,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseResult:
        """Deserialize from dictionary with type reconstruction."""
        return cls(
            status=PhaseState(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            duration_seconds=data.get("duration_seconds"),
            error_message=data.get("error_message"),
            skip_reason=data.get("skip_reason"),
            metrics=data.get("metrics", {}),
            checkpoints=data.get("checkpoints", []),
            artifacts=data.get("artifacts", []),
        )

    @classmethod
    def not_started(cls) -> PhaseResult:
        """Factory for NOT_STARTED phase."""
        return cls(status=PhaseState.NOT_STARTED)

    @classmethod
    def in_progress(cls, started_at: datetime | None = None) -> PhaseResult:
        """Factory for IN_PROGRESS phase."""
        return cls(
            status=PhaseState.IN_PROGRESS,
            started_at=started_at or datetime.now(),
        )

    @classmethod
    def completed(
        cls,
        started_at: datetime,
        metrics: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
    ) -> PhaseResult:
        """Factory for COMPLETED phase."""
        completed_at = datetime.now()
        return cls(
            status=PhaseState.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            metrics=metrics or {},
            artifacts=artifacts or [],
        )

    @classmethod
    def failed(
        cls,
        started_at: datetime,
        error: str,
        metrics: dict[str, Any] | None = None,
    ) -> PhaseResult:
        """Factory for FAILED phase."""
        completed_at = datetime.now()
        return cls(
            status=PhaseState.FAILED,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            error_message=error,
            metrics=metrics or {},
        )

    @classmethod
    def skipped(cls, reason: str) -> PhaseResult:
        """Factory for SKIPPED phase."""
        return cls(
            status=PhaseState.SKIPPED,
            skip_reason=reason,
        )


@dataclass
class StateDiff:
    """Difference between two pipeline states for debugging."""
    phases_added: list[str]
    phases_removed: list[str]
    phases_changed: dict[str, dict[str, tuple[Any, Any]]]  # phase -> field -> (old, new)
    metadata_changed: dict[str, tuple[Any, Any]]

    def is_empty(self) -> bool:
        """Check if there are no differences."""
        return (
            not self.phases_added
            and not self.phases_removed
            and not self.phases_changed
            and not self.metadata_changed
        )

    def to_summary(self) -> str:
        """Human-readable summary of differences."""
        lines = []
        if self.phases_added:
            lines.append(f"Phases added: {', '.join(self.phases_added)}")
        if self.phases_removed:
            lines.append(f"Phases removed: {', '.join(self.phases_removed)}")
        if self.phases_changed:
            for phase, changes in self.phases_changed.items():
                for field_name, (old, new) in changes.items():
                    lines.append(f"Phase '{phase}' {field_name}: {old} -> {new}")
        if self.metadata_changed:
            for key, (old, new) in self.metadata_changed.items():
                lines.append(f"Metadata '{key}': {old} -> {new}")
        return "\n".join(lines) if lines else "No differences"


class StateValidationError(Exception):
    """Raised when state validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"State validation failed: {'; '.join(errors)}")


class PipelineState:
    """
    Robust pipeline execution state with checkpoint management.

    Features:
    - Thread-safe state updates via lock
    - Schema versioning for compatibility
    - Config hash for drift detection
    - Detailed phase tracking with metrics
    - Rollback capability
    - State comparison for debugging

    Usage:
        # Create new state
        state = PipelineState(run_id="20260116_120000_abc123")

        # Start a phase
        state.start_phase("data_generation")

        # Complete with metrics
        state.complete_phase("data_generation", metrics={"rows": 10000})

        # Check dependencies before starting
        if state.can_start_phase("feature_engineering"):
            state.start_phase("feature_engineering")

        # Save and load
        state.save(Path("state.json"))
        loaded = PipelineState.load(Path("state.json"))
    """

    def __init__(
        self,
        run_id: str,
        version: StateVersion = StateVersion.V2,
        config_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize pipeline state.

        Args:
            run_id: Unique identifier for this pipeline run
            version: State schema version
            config_hash: Hash of config for drift detection
            metadata: Additional metadata to track
        """
        if not run_id:
            raise ValueError("run_id cannot be empty")

        self._lock = threading.RLock()
        self.run_id = run_id
        self.version = version
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.config_hash = config_hash
        self.phases: dict[str, PhaseResult] = {}
        self.current_phase: str | None = None
        self.metadata: dict[str, Any] = metadata or {}
        self._history: list[dict[str, Any]] = []  # For rollback

    @property
    def output_dir(self) -> Path:
        """Get output directory for this run."""
        return Path(f"experiments/runs/{self.run_id}")

    @property
    def state_file(self) -> Path:
        """Get default path to state file."""
        return self.output_dir / "pipeline_state.json"

    def _record_history(self) -> None:
        """Record current state for rollback (internal use)."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "phases": {name: result.to_dict() for name, result in self.phases.items()},
            "current_phase": self.current_phase,
        }
        self._history.append(snapshot)
        # Keep last 20 snapshots
        if len(self._history) > 20:
            self._history = self._history[-20:]

    def start_phase(self, name: str) -> None:
        """
        Mark phase as IN_PROGRESS.

        Args:
            name: Phase name to start

        Raises:
            ValueError: If phase is already in progress or completed
        """
        with self._lock:
            existing = self.phases.get(name)
            if existing and existing.status == PhaseState.IN_PROGRESS:
                raise ValueError(f"Phase '{name}' is already in progress")
            if existing and existing.status == PhaseState.COMPLETED:
                raise ValueError(f"Phase '{name}' is already completed")

            self._record_history()
            self.phases[name] = PhaseResult.in_progress()
            self.current_phase = name
            self.updated_at = datetime.now()

            logger.info(f"Phase '{name}' started")

    def complete_phase(
        self,
        name: str,
        metrics: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
    ) -> None:
        """
        Mark phase as COMPLETED with metrics.

        Args:
            name: Phase name to complete
            metrics: Optional metrics from phase execution
            artifacts: Optional list of artifact paths produced

        Raises:
            ValueError: If phase was not in progress
        """
        with self._lock:
            existing = self.phases.get(name)
            if not existing or existing.status != PhaseState.IN_PROGRESS:
                raise ValueError(f"Phase '{name}' is not in progress")

            self._record_history()
            self.phases[name] = PhaseResult.completed(
                started_at=existing.started_at,
                metrics=metrics,
                artifacts=artifacts,
            )
            if self.current_phase == name:
                self.current_phase = None
            self.updated_at = datetime.now()

            duration = self.phases[name].duration_seconds
            logger.info(f"Phase '{name}' completed in {duration:.2f}s")

    def fail_phase(self, name: str, error: str, metrics: dict[str, Any] | None = None) -> None:
        """
        Mark phase as FAILED with error.

        Args:
            name: Phase name that failed
            error: Error message describing the failure
            metrics: Optional partial metrics before failure

        Raises:
            ValueError: If phase was not in progress
        """
        with self._lock:
            existing = self.phases.get(name)
            if not existing or existing.status != PhaseState.IN_PROGRESS:
                raise ValueError(f"Phase '{name}' is not in progress")

            self._record_history()
            self.phases[name] = PhaseResult.failed(
                started_at=existing.started_at,
                error=error,
                metrics=metrics,
            )
            if self.current_phase == name:
                self.current_phase = None
            self.updated_at = datetime.now()

            logger.error(f"Phase '{name}' failed: {error}")

    def skip_phase(self, name: str, reason: str) -> None:
        """
        Mark phase as SKIPPED.

        Args:
            name: Phase name to skip
            reason: Reason for skipping
        """
        with self._lock:
            self._record_history()
            self.phases[name] = PhaseResult.skipped(reason)
            self.updated_at = datetime.now()

            logger.info(f"Phase '{name}' skipped: {reason}")

    def add_checkpoint(self, phase: str, checkpoint_path: str) -> None:
        """Add a checkpoint path to a phase."""
        with self._lock:
            if phase not in self.phases:
                raise ValueError(f"Phase '{phase}' not found")

            result = self.phases[phase]
            new_checkpoints = result.checkpoints + [checkpoint_path]
            # Create new PhaseResult with updated checkpoints
            self.phases[phase] = PhaseResult(
                status=result.status,
                started_at=result.started_at,
                completed_at=result.completed_at,
                duration_seconds=result.duration_seconds,
                error_message=result.error_message,
                skip_reason=result.skip_reason,
                metrics=result.metrics,
                checkpoints=new_checkpoints,
                artifacts=result.artifacts,
            )
            self.updated_at = datetime.now()

    def can_start_phase(self, name: str, dependencies: list[str] | None = None) -> bool:
        """
        Check if a phase can be started (dependencies met).

        Args:
            name: Phase to check
            dependencies: List of required dependency phase names

        Returns:
            True if all dependencies are COMPLETED
        """
        with self._lock:
            # Check if already started or completed
            existing = self.phases.get(name)
            if existing and existing.status in (PhaseState.IN_PROGRESS, PhaseState.COMPLETED):
                return False

            # Check dependencies
            if dependencies:
                for dep in dependencies:
                    dep_result = self.phases.get(dep)
                    if not dep_result or dep_result.status != PhaseState.COMPLETED:
                        return False

            return True

    def is_phase_complete(self, name: str) -> bool:
        """Check if a specific phase is complete."""
        with self._lock:
            result = self.phases.get(name)
            return result is not None and result.status == PhaseState.COMPLETED

    def get_phase_result(self, name: str) -> PhaseResult | None:
        """Get the result of a specific phase."""
        with self._lock:
            return self.phases.get(name)

    def get_completed_phases(self) -> list[str]:
        """Get list of completed phase names."""
        with self._lock:
            return [
                name for name, result in self.phases.items()
                if result.status == PhaseState.COMPLETED
            ]

    def get_failed_phases(self) -> list[str]:
        """Get list of failed phase names."""
        with self._lock:
            return [
                name for name, result in self.phases.items()
                if result.status == PhaseState.FAILED
            ]

    def rollback_to_phase(self, name: str) -> bool:
        """
        Rollback state to before a specific phase.

        Removes the phase and all phases that depend on it (must be handled by caller).

        Args:
            name: Phase to rollback to (this phase and after will be reset)

        Returns:
            True if rollback succeeded
        """
        with self._lock:
            if name not in self.phases:
                logger.warning(f"Cannot rollback: phase '{name}' not found")
                return False

            self._record_history()

            # Reset this phase to NOT_STARTED
            self.phases[name] = PhaseResult.not_started()

            # Clear current phase if it was the rolled back phase
            if self.current_phase == name:
                self.current_phase = None

            self.updated_at = datetime.now()
            logger.info(f"Rolled back phase '{name}' to NOT_STARTED")
            return True

    def validate(self) -> list[str]:
        """
        Validate state consistency.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        with self._lock:
            # Check for orphaned current_phase
            if self.current_phase:
                result = self.phases.get(self.current_phase)
                if not result:
                    errors.append(f"current_phase '{self.current_phase}' not in phases dict")
                elif result.status != PhaseState.IN_PROGRESS:
                    errors.append(
                        f"current_phase '{self.current_phase}' status is "
                        f"{result.status.value}, expected IN_PROGRESS"
                    )

            # Check for multiple IN_PROGRESS phases
            in_progress = [
                name for name, result in self.phases.items()
                if result.status == PhaseState.IN_PROGRESS
            ]
            if len(in_progress) > 1:
                errors.append(f"Multiple phases in progress: {in_progress}")

            # Check COMPLETED phases have required fields
            for name, result in self.phases.items():
                if result.status == PhaseState.COMPLETED:
                    if result.started_at is None:
                        errors.append(f"Phase '{name}' COMPLETED but started_at is None")
                    if result.completed_at is None:
                        errors.append(f"Phase '{name}' COMPLETED but completed_at is None")
                    if result.duration_seconds is None:
                        errors.append(f"Phase '{name}' COMPLETED but duration_seconds is None")

                if result.status == PhaseState.FAILED:
                    if result.error_message is None:
                        errors.append(f"Phase '{name}' FAILED but error_message is None")

                if result.status == PhaseState.SKIPPED:
                    if result.skip_reason is None:
                        errors.append(f"Phase '{name}' SKIPPED but skip_reason is None")

            # Check timestamps are reasonable
            if self.updated_at < self.created_at:
                errors.append("updated_at is before created_at")

        return errors

    def diff(self, other: PipelineState) -> StateDiff:
        """
        Compare two states for debugging.

        Args:
            other: State to compare against

        Returns:
            StateDiff with all differences
        """
        with self._lock:
            phases_added = [name for name in other.phases if name not in self.phases]
            phases_removed = [name for name in self.phases if name not in other.phases]

            phases_changed: dict[str, dict[str, tuple[Any, Any]]] = {}
            for name in self.phases:
                if name in other.phases:
                    self_result = self.phases[name]
                    other_result = other.phases[name]

                    changes: dict[str, tuple[Any, Any]] = {}
                    if self_result.status != other_result.status:
                        changes["status"] = (self_result.status.value, other_result.status.value)
                    if self_result.error_message != other_result.error_message:
                        changes["error_message"] = (self_result.error_message, other_result.error_message)

                    if changes:
                        phases_changed[name] = changes

            metadata_changed: dict[str, tuple[Any, Any]] = {}
            all_keys = set(self.metadata.keys()) | set(other.metadata.keys())
            for key in all_keys:
                self_val = self.metadata.get(key)
                other_val = other.metadata.get(key)
                if self_val != other_val:
                    metadata_changed[key] = (self_val, other_val)

            return StateDiff(
                phases_added=phases_added,
                phases_removed=phases_removed,
                phases_changed=phases_changed,
                metadata_changed=metadata_changed,
            )

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        with self._lock:
            lines = [
                f"PipelineState: {self.run_id}",
                f"  Version: {self.version.value}",
                f"  Created: {self.created_at.isoformat()}",
                f"  Updated: {self.updated_at.isoformat()}",
                f"  Current Phase: {self.current_phase or 'None'}",
                f"  Config Hash: {self.config_hash or 'None'}",
                "",
                "  Phases:",
            ]

            for name, result in self.phases.items():
                status_str = result.status.value.upper()
                extra = ""
                if result.status == PhaseState.COMPLETED and result.duration_seconds:
                    extra = f" ({result.duration_seconds:.2f}s)"
                elif result.status == PhaseState.FAILED and result.error_message:
                    extra = f" - {result.error_message[:50]}..."
                elif result.status == PhaseState.SKIPPED and result.skip_reason:
                    extra = f" - {result.skip_reason}"

                lines.append(f"    {name}: {status_str}{extra}")

            if not self.phases:
                lines.append("    (no phases recorded)")

            return "\n".join(lines)

    def save(self, path: Path | None = None) -> None:
        """
        Save state to disk with type-preserving serialization.

        Args:
            path: Path to save to (defaults to self.state_file)
        """
        path = path or self.state_file
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = {
                "__version__": self.version.value,
                "run_id": self.run_id,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "config_hash": self.config_hash,
                "current_phase": self.current_phase,
                "phases": {name: result.to_dict() for name, result in self.phases.items()},
                "metadata": self.metadata,
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"State saved: {path}")

    @classmethod
    def load(cls, path: Path | str) -> PipelineState:
        """
        Load state from disk with type reconstruction.

        Args:
            path: Path to load from (or run_id string)

        Returns:
            Loaded PipelineState

        Raises:
            FileNotFoundError: If state file doesn't exist
            StateValidationError: If loaded state is invalid
        """
        # Handle run_id string
        if isinstance(path, str) and "/" not in path and "\\" not in path:
            path = Path(f"experiments/runs/{path}/pipeline_state.json")
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        # Detect version
        version_str = data.get("__version__", "v1")
        version = StateVersion(version_str)

        if version == StateVersion.V1:
            return cls._load_v1(data)
        else:
            return cls._load_v2(data)

    @classmethod
    def _load_v1(cls, data: dict[str, Any]) -> PipelineState:
        """Load legacy V1 state format."""
        state = cls(
            run_id=data["run_id"],
            version=StateVersion.V1,
        )

        # Convert old boolean phases to PhaseResult
        phases_completed = data.get("phases_completed", {})
        phase_start_times = data.get("phase_start_times", {})
        phase_end_times = data.get("phase_end_times", {})
        phase_results = data.get("phase_results", {})

        for phase, completed in phases_completed.items():
            if completed:
                started = (
                    datetime.fromisoformat(phase_start_times[phase])
                    if phase in phase_start_times
                    else None
                )
                ended = (
                    datetime.fromisoformat(phase_end_times[phase])
                    if phase in phase_end_times
                    else None
                )
                duration = (ended - started).total_seconds() if started and ended else None

                state.phases[phase] = PhaseResult(
                    status=PhaseState.COMPLETED,
                    started_at=started,
                    completed_at=ended,
                    duration_seconds=duration,
                    metrics=phase_results.get(phase, {}),
                )
            else:
                state.phases[phase] = PhaseResult(
                    status=PhaseState.FAILED,
                    error_message=data.get("error_message"),
                )

        state.created_at = datetime.fromisoformat(data.get("start_time", datetime.now().isoformat()))
        state.updated_at = datetime.fromisoformat(data.get("end_time", datetime.now().isoformat())) if data.get("end_time") else datetime.now()

        return state

    @classmethod
    def _load_v2(cls, data: dict[str, Any]) -> PipelineState:
        """Load V2 state format."""
        state = cls(
            run_id=data["run_id"],
            version=StateVersion.V2,
            config_hash=data.get("config_hash"),
            metadata=data.get("metadata", {}),
        )

        state.created_at = datetime.fromisoformat(data["created_at"])
        state.updated_at = datetime.fromisoformat(data["updated_at"])
        state.current_phase = data.get("current_phase")

        for name, result_data in data.get("phases", {}).items():
            state.phases[name] = PhaseResult.from_dict(result_data)

        # Validate loaded state
        errors = state.validate()
        if errors:
            raise StateValidationError(errors)

        return state

    @classmethod
    def load_or_create(cls, run_id: str, config_hash: str | None = None) -> PipelineState:
        """Load existing state or create new one."""
        try:
            state = cls.load(run_id)
            # Check config hash if provided
            if config_hash and state.config_hash and state.config_hash != config_hash:
                logger.warning(
                    f"Config hash mismatch: expected {config_hash[:8]}..., "
                    f"got {state.config_hash[:8]}... (config drift detected)"
                )
            return state
        except FileNotFoundError:
            return cls(run_id=run_id, config_hash=config_hash)

    # ========================================================================
    # Backward Compatibility Methods (deprecated, use new API)
    # ========================================================================

    def mark_start(self, phase: str) -> None:
        """Deprecated: Use start_phase() instead."""
        logger.warning("mark_start() is deprecated, use start_phase()")
        self.start_phase(phase)

    def mark_complete(self, phase: str, result: Any = None) -> None:
        """Deprecated: Use complete_phase() instead."""
        logger.warning("mark_complete() is deprecated, use complete_phase()")
        metrics = result if isinstance(result, dict) else {"result": result}

        # Ensure phase is in progress
        if phase not in self.phases or self.phases[phase].status != PhaseState.IN_PROGRESS:
            self.start_phase(phase)

        self.complete_phase(phase, metrics=metrics)
        self.save()

    def mark_failed(self, phase: str, error: str) -> None:
        """Deprecated: Use fail_phase() instead."""
        logger.warning("mark_failed() is deprecated, use fail_phase()")

        # Ensure phase is in progress
        if phase not in self.phases or self.phases[phase].status != PhaseState.IN_PROGRESS:
            self.start_phase(phase)

        self.fail_phase(phase, error)
        self.save()

    def is_complete(self, phase: str) -> bool:
        """Deprecated: Use is_phase_complete() instead."""
        return self.is_phase_complete(phase)

    def get_result(self, phase: str) -> Any:
        """Deprecated: Use get_phase_result() instead."""
        result = self.get_phase_result(phase)
        if result and result.status == PhaseState.COMPLETED:
            return result.metrics
        return None

    def set_checkpoint(self, phase: str, path: Path | str) -> None:
        """Deprecated: Use add_checkpoint() instead."""
        logger.warning("set_checkpoint() is deprecated, use add_checkpoint()")
        self.add_checkpoint(phase, str(path))
        self.save()

    def get_checkpoint(self, phase: str) -> Path | None:
        """Deprecated: Get first checkpoint for phase."""
        result = self.get_phase_result(phase)
        if result and result.checkpoints:
            return Path(result.checkpoints[0])
        return None

    def get_summary(self) -> dict[str, Any]:
        """Get summary as dict (backward compat)."""
        with self._lock:
            completed_phases = self.get_completed_phases()
            failed_phases = self.get_failed_phases()
            pending_phases = [
                name for name, result in self.phases.items()
                if result.status in (PhaseState.NOT_STARTED, PhaseState.IN_PROGRESS)
            ]

            total_duration = None
            completed_results = [
                r for r in self.phases.values()
                if r.status == PhaseState.COMPLETED and r.duration_seconds
            ]
            if completed_results:
                total_duration = sum(r.duration_seconds for r in completed_results)

            return {
                "run_id": self.run_id,
                "status": "completed" if not failed_phases else "failed",
                "completed_phases": completed_phases,
                "failed_phases": failed_phases,
                "pending_phases": pending_phases,
                "start_time": self.created_at.isoformat(),
                "end_time": self.updated_at.isoformat(),
                "total_duration_seconds": total_duration,
                "error_message": next(
                    (r.error_message for r in self.phases.values() if r.error_message),
                    None
                ),
            }

    def __repr__(self) -> str:
        completed = len(self.get_completed_phases())
        total = len(self.phases)
        failed = len(self.get_failed_phases())
        status = "failed" if failed > 0 else "running"
        return (
            f"PipelineState(run_id={self.run_id}, "
            f"status={status}, "
            f"completed={completed}/{total})"
        )


def compute_config_hash(config: Any) -> str:
    """
    Compute a hash of configuration for drift detection.

    Args:
        config: Configuration object (dict, dataclass, or object with to_dict)

    Returns:
        SHA-256 hash string (first 16 chars)
    """
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif hasattr(config, "__dict__"):
        config_dict = vars(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {"value": str(config)}

    # Remove non-deterministic fields
    excluded = {"run_id", "created_at", "updated_at", "timestamp"}
    filtered = {k: v for k, v in config_dict.items() if k not in excluded}

    # Sort keys for determinism
    json_str = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]
