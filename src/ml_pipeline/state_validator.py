"""
Pipeline State Validator.

Validates pipeline state consistency including:
- Phase dependency validation
- Schema version validation
- Config hash matching for drift detection
- State consistency checks

Usage:
    from src.ml_pipeline.state_validator import StateValidator

    validator = StateValidator(state, config_hash="abc123")
    errors = validator.validate_all()

    if errors:
        for error in errors:
            print(f"Validation error: {error}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .phase_registry import PHASE_REGISTRY, get_all_dependencies, get_phase
from .state import PhaseState, PipelineState, StateVersion

if TYPE_CHECKING:
    from .state import PhaseResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of state validation."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]

    def __bool__(self) -> bool:
        """ValidationResult is truthy if valid."""
        return self.is_valid

    def to_summary(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.is_valid:
            lines.append("State is VALID")
        else:
            lines.append("State is INVALID")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        return "\n".join(lines)


class StateValidator:
    """
    Validates pipeline state consistency.

    Performs various checks to ensure the state is valid:
    - Schema version compatibility
    - Phase dependency satisfaction
    - Config drift detection
    - State consistency (no orphans, valid transitions)

    Usage:
        validator = StateValidator(state)
        result = validator.validate_all()

        if not result:
            raise StateValidationError(result.errors)
    """

    def __init__(
        self,
        state: PipelineState,
        expected_config_hash: str | None = None,
    ):
        """
        Initialize validator.

        Args:
            state: PipelineState to validate
            expected_config_hash: Expected config hash for drift detection
        """
        self.state = state
        self.expected_config_hash = expected_config_hash

    def validate_all(self) -> ValidationResult:
        """
        Run all validations.

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Schema version
        version_errors = self.validate_schema_version()
        errors.extend(version_errors)

        # Config hash
        hash_errors, hash_warnings = self.validate_config_hash()
        errors.extend(hash_errors)
        warnings.extend(hash_warnings)

        # Phase dependencies
        dep_errors = self.validate_phase_dependencies()
        errors.extend(dep_errors)

        # State consistency (from state.validate())
        consistency_errors = self.state.validate()
        errors.extend(consistency_errors)

        # Phase state transitions
        transition_warnings = self.validate_phase_transitions()
        warnings.extend(transition_warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_schema_version(self) -> list[str]:
        """
        Validate state schema version is supported.

        Returns:
            List of errors (empty if valid)
        """
        errors: list[str] = []

        supported_versions = {StateVersion.V1, StateVersion.V2}
        if self.state.version not in supported_versions:
            errors.append(
                f"Unsupported state version: {self.state.version}. "
                f"Supported: {[v.value for v in supported_versions]}"
            )

        # V1 is legacy, warn about upgrade
        if self.state.version == StateVersion.V1:
            logger.warning(
                "State uses legacy V1 format. Consider saving to upgrade to V2."
            )

        return errors

    def validate_config_hash(self) -> tuple[list[str], list[str]]:
        """
        Validate config hash matches expected value.

        Returns:
            Tuple of (errors, warnings)
        """
        errors: list[str] = []
        warnings: list[str] = []

        if self.expected_config_hash is None:
            # No expected hash provided, skip validation
            return errors, warnings

        if self.state.config_hash is None:
            warnings.append(
                "State has no config_hash stored. Cannot verify config drift."
            )
        elif self.state.config_hash != self.expected_config_hash:
            warnings.append(
                f"Config drift detected: state hash {self.state.config_hash[:8]}... "
                f"!= expected {self.expected_config_hash[:8]}..."
            )

        return errors, warnings

    def validate_phase_dependencies(self) -> list[str]:
        """
        Validate that completed phases have their dependencies satisfied.

        Returns:
            List of dependency errors
        """
        errors: list[str] = []

        for phase_name, result in self.state.phases.items():
            # Skip phases not in registry (custom phases)
            if phase_name not in PHASE_REGISTRY:
                continue

            # Only check completed/in_progress phases
            if result.status not in (PhaseState.COMPLETED, PhaseState.IN_PROGRESS):
                continue

            # Get required dependencies
            try:
                deps = get_all_dependencies(phase_name)
            except KeyError:
                continue

            # Check each dependency is completed
            for dep in deps:
                dep_result = self.state.phases.get(dep)
                if dep_result is None:
                    errors.append(
                        f"Phase '{phase_name}' ({result.status.value}) "
                        f"has missing dependency '{dep}'"
                    )
                elif dep_result.status != PhaseState.COMPLETED:
                    # Exception: IN_PROGRESS phases can have IN_PROGRESS deps if they're the same
                    if not (
                        result.status == PhaseState.IN_PROGRESS
                        and dep_result.status == PhaseState.IN_PROGRESS
                    ):
                        errors.append(
                            f"Phase '{phase_name}' ({result.status.value}) "
                            f"has unsatisfied dependency '{dep}' ({dep_result.status.value})"
                        )

        return errors

    def validate_phase_transitions(self) -> list[str]:
        """
        Check for unusual phase state transitions.

        Returns:
            List of warnings about unusual transitions
        """
        warnings: list[str] = []

        for phase_name, result in self.state.phases.items():
            # Completed without metrics
            if (
                result.status == PhaseState.COMPLETED
                and not result.metrics
            ):
                warnings.append(
                    f"Phase '{phase_name}' completed with no metrics recorded"
                )

            # Failed with very short duration
            if (
                result.status == PhaseState.FAILED
                and result.duration_seconds is not None
                and result.duration_seconds < 0.1
            ):
                warnings.append(
                    f"Phase '{phase_name}' failed very quickly ({result.duration_seconds:.3f}s) "
                    "- possible validation failure before execution"
                )

            # Completed with very long duration
            if (
                result.status == PhaseState.COMPLETED
                and result.duration_seconds is not None
                and result.duration_seconds > 3600  # > 1 hour
            ):
                warnings.append(
                    f"Phase '{phase_name}' took {result.duration_seconds/60:.1f} minutes "
                    "- unusually long"
                )

        return warnings

    def can_start_phase(self, phase_name: str) -> tuple[bool, list[str]]:
        """
        Check if a phase can be started based on dependencies.

        Args:
            phase_name: Name of phase to check

        Returns:
            Tuple of (can_start, list of blocking reasons)
        """
        blocking_reasons: list[str] = []

        # Check if phase is already in progress or completed
        existing = self.state.phases.get(phase_name)
        if existing:
            if existing.status == PhaseState.IN_PROGRESS:
                blocking_reasons.append(f"Phase '{phase_name}' is already in progress")
            elif existing.status == PhaseState.COMPLETED:
                blocking_reasons.append(f"Phase '{phase_name}' is already completed")

        # Check dependencies
        if phase_name in PHASE_REGISTRY:
            try:
                deps = get_all_dependencies(phase_name)
                for dep in deps:
                    dep_result = self.state.phases.get(dep)
                    if dep_result is None:
                        blocking_reasons.append(f"Dependency '{dep}' has not started")
                    elif dep_result.status != PhaseState.COMPLETED:
                        blocking_reasons.append(
                            f"Dependency '{dep}' is {dep_result.status.value}"
                        )
            except KeyError:
                pass

        return len(blocking_reasons) == 0, blocking_reasons

    def get_runnable_phases(self) -> list[str]:
        """
        Get list of phases that can be started.

        Returns:
            List of phase names that have all dependencies satisfied
        """
        runnable: list[str] = []

        for phase_name in PHASE_REGISTRY:
            can_start, _ = self.can_start_phase(phase_name)
            if can_start:
                runnable.append(phase_name)

        return runnable

    def get_next_phase(self) -> str | None:
        """
        Get the next phase to run in order.

        Returns:
            Name of next runnable phase, or None if all complete/blocked
        """
        from .phase_registry import get_phases_in_order

        for phase in get_phases_in_order():
            result = self.state.phases.get(phase.name)

            # Skip completed and skipped
            if result and result.status in (PhaseState.COMPLETED, PhaseState.SKIPPED):
                continue

            # Check if can start
            can_start, _ = self.can_start_phase(phase.name)
            if can_start:
                return phase.name

        return None

    def get_completion_status(self) -> dict[str, int]:
        """
        Get counts of phases by status.

        Returns:
            Dict mapping status to count
        """
        counts: dict[str, int] = {
            "not_started": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
        }

        for phase_name in PHASE_REGISTRY:
            result = self.state.phases.get(phase_name)
            if result is None:
                counts["not_started"] += 1
            else:
                counts[result.status.value] += 1

        return counts


def validate_state(
    state: PipelineState,
    expected_config_hash: str | None = None,
) -> ValidationResult:
    """
    Convenience function to validate a state.

    Args:
        state: PipelineState to validate
        expected_config_hash: Expected config hash for drift detection

    Returns:
        ValidationResult
    """
    validator = StateValidator(state, expected_config_hash)
    return validator.validate_all()


def can_resume_pipeline(state: PipelineState) -> tuple[bool, str | None, list[str]]:
    """
    Check if a pipeline can be resumed from its current state.

    Args:
        state: PipelineState to check

    Returns:
        Tuple of (can_resume, next_phase, blocking_reasons)
    """
    validator = StateValidator(state)
    result = validator.validate_all()

    if not result.is_valid:
        return False, None, result.errors

    next_phase = validator.get_next_phase()
    if next_phase is None:
        # Check if all phases complete or if blocked
        status = validator.get_completion_status()
        if status["failed"] > 0:
            return False, None, ["Pipeline has failed phases that must be resolved"]
        elif status["not_started"] == 0 and status["in_progress"] == 0:
            return False, None, ["Pipeline is already complete"]

    return True, next_phase, []
