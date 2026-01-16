"""
Pipeline Phase Registry.

Defines the canonical phase definitions with dependencies, descriptions,
and estimated durations for the ML pipeline.

This module provides:
- PhaseDefinition dataclass for phase metadata
- PHASE_REGISTRY with all pipeline phases
- Utility functions for phase lookup and dependency resolution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class PhaseDefinition:
    """
    Immutable definition of a pipeline phase.

    Attributes:
        name: Unique phase identifier (snake_case)
        dependencies: List of phase names that must complete first
        description: Human-readable description
        stage_number: Numeric ordering (supports decimals like 7.5)
        estimated_duration: Expected duration in seconds (None if unknown)
        required: Whether phase failure should halt the pipeline
        can_skip: Whether phase can be skipped via configuration
        retry_count: Default number of retries on failure
    """
    name: str
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    stage_number: float = 0.0
    estimated_duration: float | None = None
    required: bool = True
    can_skip: bool = False
    retry_count: int = 0

    def __post_init__(self):
        """Validate phase definition."""
        if not self.name:
            raise ValueError("Phase name cannot be empty")
        if self.stage_number < 0:
            raise ValueError(f"Stage number must be non-negative: {self.stage_number}")
        if self.retry_count < 0:
            raise ValueError(f"Retry count must be non-negative: {self.retry_count}")


# =============================================================================
# PHASE DEFINITIONS
# =============================================================================
# All phases are defined here with their dependencies.
# Dependencies form a DAG - no cycles allowed.
# Stage numbers provide ordering within groups.

# Phase 1: Data Pipeline (Phases 1-5 in CLAUDE.md)
_DATA_GENERATION = PhaseDefinition(
    name="data_generation",
    dependencies=(),
    description="Stage 1: Generate or validate raw OHLCV data files",
    stage_number=1.0,
    estimated_duration=30.0,  # ~30 seconds for typical dataset
    required=True,
)

_DATA_CLEANING = PhaseDefinition(
    name="data_cleaning",
    dependencies=("data_generation",),
    description="Stage 2: Clean and resample OHLCV data (1min -> target TF)",
    stage_number=2.0,
    estimated_duration=60.0,  # ~1 minute
    required=True,
)

_FEATURE_ENGINEERING = PhaseDefinition(
    name="feature_engineering",
    dependencies=("data_cleaning",),
    description="Stage 3: Generate technical features (~180 indicators)",
    stage_number=3.0,
    estimated_duration=300.0,  # ~5 minutes (compute intensive)
    required=True,
)

_INITIAL_LABELING = PhaseDefinition(
    name="initial_labeling",
    dependencies=("feature_engineering",),
    description="Stage 4: Apply initial triple-barrier labeling",
    stage_number=4.0,
    estimated_duration=60.0,
    required=True,
)

_GA_OPTIMIZE = PhaseDefinition(
    name="ga_optimize",
    dependencies=("initial_labeling",),
    description="Stage 5: Optuna optimization of barrier parameters",
    stage_number=5.0,
    estimated_duration=600.0,  # ~10 minutes (Optuna trials)
    required=True,
    can_skip=True,  # Can skip if using default parameters
)

_FINAL_LABELS = PhaseDefinition(
    name="final_labels",
    dependencies=("ga_optimize",),
    description="Stage 6: Apply optimized labels with quality scores",
    stage_number=6.0,
    estimated_duration=60.0,
    required=True,
)

_CREATE_SPLITS = PhaseDefinition(
    name="create_splits",
    dependencies=("final_labels",),
    description="Stage 7: Create train/val/test splits with purge/embargo",
    stage_number=7.0,
    estimated_duration=30.0,
    required=True,
)

_FEATURE_SCALING = PhaseDefinition(
    name="feature_scaling",
    dependencies=("create_splits",),
    description="Stage 7.5: Train-only feature scaling (robust scaler)",
    stage_number=7.5,
    estimated_duration=30.0,
    required=True,
)

_BUILD_DATASETS = PhaseDefinition(
    name="build_datasets",
    dependencies=("feature_scaling",),
    description="Stage 7.6: Build TimeSeriesDataContainer for training",
    stage_number=7.6,
    estimated_duration=30.0,
    required=True,
)

_VALIDATE_SCALED = PhaseDefinition(
    name="validate_scaled",
    dependencies=("build_datasets",),
    description="Stage 7.7: Post-scale drift validation",
    stage_number=7.7,
    estimated_duration=30.0,
    required=True,
)

_VALIDATE = PhaseDefinition(
    name="validate",
    dependencies=("validate_scaled",),
    description="Stage 8: Comprehensive data validation (features, labels, splits)",
    stage_number=8.0,
    estimated_duration=60.0,
    required=True,
)

_GENERATE_REPORT = PhaseDefinition(
    name="generate_report",
    dependencies=("validate",),
    description="Stage 9: Generate completion report with metrics",
    stage_number=9.0,
    estimated_duration=10.0,
    required=True,
)

# Phase 6: Training (from unified.py)
_TRAINING = PhaseDefinition(
    name="training",
    dependencies=("generate_report",),  # Requires data pipeline complete
    description="Phase 6: Model training (TrainingOrchestrator)",
    stage_number=10.0,
    estimated_duration=None,  # Highly variable
    required=True,
)

# Phase 7: Evaluation (from unified.py)
_EVALUATION = PhaseDefinition(
    name="evaluation",
    dependencies=("training",),
    description="Phase 7: Model evaluation (CV, walk-forward, CPCV-PBO)",
    stage_number=11.0,
    estimated_duration=None,  # Highly variable
    required=False,  # Optional
    can_skip=True,
)


# =============================================================================
# PHASE REGISTRY
# =============================================================================

PHASE_REGISTRY: dict[str, PhaseDefinition] = {
    # Data Pipeline (Phases 1-5)
    "data_generation": _DATA_GENERATION,
    "data_cleaning": _DATA_CLEANING,
    "feature_engineering": _FEATURE_ENGINEERING,
    "initial_labeling": _INITIAL_LABELING,
    "ga_optimize": _GA_OPTIMIZE,
    "final_labels": _FINAL_LABELS,
    "create_splits": _CREATE_SPLITS,
    "feature_scaling": _FEATURE_SCALING,
    "build_datasets": _BUILD_DATASETS,
    "validate_scaled": _VALIDATE_SCALED,
    "validate": _VALIDATE,
    "generate_report": _GENERATE_REPORT,
    # Training & Evaluation (Phases 6-7)
    "training": _TRAINING,
    "evaluation": _EVALUATION,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_phase(name: str) -> PhaseDefinition:
    """
    Get phase definition by name.

    Args:
        name: Phase name

    Returns:
        PhaseDefinition

    Raises:
        KeyError: If phase not found
    """
    if name not in PHASE_REGISTRY:
        raise KeyError(f"Unknown phase: '{name}'. Available: {list(PHASE_REGISTRY.keys())}")
    return PHASE_REGISTRY[name]


def get_phase_dependencies(name: str) -> list[str]:
    """
    Get direct dependencies for a phase.

    Args:
        name: Phase name

    Returns:
        List of dependency phase names
    """
    return list(get_phase(name).dependencies)


def get_all_dependencies(name: str) -> list[str]:
    """
    Get all transitive dependencies for a phase (topologically sorted).

    Args:
        name: Phase name

    Returns:
        List of all dependency phase names in execution order
    """
    visited: set[str] = set()
    result: list[str] = []

    def visit(phase_name: str) -> None:
        if phase_name in visited:
            return
        visited.add(phase_name)

        phase = get_phase(phase_name)
        for dep in phase.dependencies:
            visit(dep)

        # Don't include the original phase in its own dependencies
        if phase_name != name:
            result.append(phase_name)

    visit(name)
    return result


def get_phases_in_order() -> list[PhaseDefinition]:
    """
    Get all phases sorted by stage_number.

    Returns:
        List of PhaseDefinition in execution order
    """
    return sorted(PHASE_REGISTRY.values(), key=lambda p: p.stage_number)


def get_phase_names_in_order() -> list[str]:
    """
    Get all phase names sorted by stage_number.

    Returns:
        List of phase names in execution order
    """
    return [p.name for p in get_phases_in_order()]


def iter_phases() -> Iterator[PhaseDefinition]:
    """Iterate over phases in execution order."""
    yield from get_phases_in_order()


def get_data_pipeline_phases() -> list[str]:
    """Get phase names for data pipeline only (stages 1-9)."""
    return [p.name for p in get_phases_in_order() if p.stage_number < 10]


def get_training_phases() -> list[str]:
    """Get phase names for training and evaluation (stages 10+)."""
    return [p.name for p in get_phases_in_order() if p.stage_number >= 10]


def validate_phase_graph() -> list[str]:
    """
    Validate that phase dependencies form a valid DAG.

    Returns:
        List of validation errors (empty if valid)
    """
    errors: list[str] = []

    # Check all dependencies exist
    for name, phase in PHASE_REGISTRY.items():
        for dep in phase.dependencies:
            if dep not in PHASE_REGISTRY:
                errors.append(f"Phase '{name}' depends on unknown phase '{dep}'")

    # Check for cycles using DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {name: WHITE for name in PHASE_REGISTRY}

    def dfs(name: str, path: list[str]) -> None:
        if color[name] == GRAY:
            cycle = " -> ".join(path + [name])
            errors.append(f"Cycle detected: {cycle}")
            return
        if color[name] == BLACK:
            return

        color[name] = GRAY
        path.append(name)

        for dep in PHASE_REGISTRY[name].dependencies:
            if dep in PHASE_REGISTRY:
                dfs(dep, path)

        path.pop()
        color[name] = BLACK

    for name in PHASE_REGISTRY:
        if color[name] == WHITE:
            dfs(name, [])

    # Check stage numbers are consistent with dependencies
    for name, phase in PHASE_REGISTRY.items():
        for dep in phase.dependencies:
            dep_phase = PHASE_REGISTRY.get(dep)
            if dep_phase and dep_phase.stage_number >= phase.stage_number:
                errors.append(
                    f"Phase '{name}' (stage {phase.stage_number}) depends on "
                    f"'{dep}' (stage {dep_phase.stage_number}) which has equal or later stage number"
                )

    return errors


def estimate_total_duration(phase_names: list[str] | None = None) -> float:
    """
    Estimate total duration for a set of phases.

    Args:
        phase_names: List of phases to estimate (None for all)

    Returns:
        Total estimated duration in seconds (excludes None estimates)
    """
    if phase_names is None:
        phases = PHASE_REGISTRY.values()
    else:
        phases = [PHASE_REGISTRY[name] for name in phase_names if name in PHASE_REGISTRY]

    return sum(p.estimated_duration or 0.0 for p in phases)


def get_next_phase(current: str | None) -> str | None:
    """
    Get the next phase after the current one.

    Args:
        current: Current phase name (None for first phase)

    Returns:
        Next phase name, or None if at end
    """
    ordered = get_phase_names_in_order()

    if current is None:
        return ordered[0] if ordered else None

    try:
        idx = ordered.index(current)
        return ordered[idx + 1] if idx + 1 < len(ordered) else None
    except ValueError:
        return None
