"""
Pipeline Stage Registry.

Defines the PipelineStage dataclass and provides stage registration utilities.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class StageName(str, Enum):
    """
    Canonical stage names for the pipeline.

    Phase 12.5G: Replaces magic strings with type-safe enum values.
    Using str as base allows direct string comparison while providing type safety.

    Usage:
        stage_name = StageName.FEATURE_ENGINEERING
        if stage_name == "feature_engineering":  # Works due to str inheritance
            ...
    """

    DATA_GENERATION = "data_generation"
    DATA_CLEANING = "data_cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    INITIAL_LABELING = "initial_labeling"
    GA_OPTIMIZE = "ga_optimize"
    FINAL_LABELS = "final_labels"
    CREATE_SPLITS = "create_splits"
    FEATURE_SCALING = "feature_scaling"
    BUILD_DATASETS = "build_datasets"
    VALIDATE_SCALED = "validate_scaled"
    VALIDATE = "validate"
    GENERATE_REPORT = "generate_report"


@dataclass
class PipelineStage:
    """Definition of a pipeline stage."""

    name: str
    function: Callable
    dependencies: list[str] = field(default_factory=list)
    description: str = ""
    required: bool = True
    can_run_parallel: bool = False
    stage_number: float = 0.0

    def __post_init__(self):
        """Validate stage definition."""
        if not self.name:
            raise ValueError("Stage name cannot be empty")
        if not callable(self.function):
            raise ValueError(f"Stage function for '{self.name}' must be callable")


def get_stage_definitions() -> list[dict]:
    """
    Get the ordered list of stage definitions.

    Returns a list of dictionaries with stage metadata (without functions).
    Functions are bound in PipelineRunner.

    Returns:
        List of stage definition dictionaries

    Note on Validation Order (PIPE-005):
        Stage 8 (validate) runs AFTER splits and scaling. This is intentional:

        1. Some validations require scaled data (e.g., distribution checks on final features)
        2. Split validation (train/val/test label distribution) requires splits to exist
        3. Drift validation (validate_scaled at 7.7) checks post-scaling data quality

        This ordering is correct for comprehensive data quality validation.
        Earlier stages have their own inline validation (e.g., feature_engineering
        validates feature counts, labeling validates label distributions).

        If pre-split validation is needed, add checks within the relevant stage
        or create a dedicated validation stage between feature_engineering and
        initial_labeling (stage 3.5).
    """
    return [
        {
            "name": StageName.DATA_GENERATION.value,
            "dependencies": [],
            "description": "Stage 1: Generate or validate raw data files",
            "required": True,
            "stage_number": 1,
        },
        {
            "name": StageName.DATA_CLEANING.value,
            "dependencies": [StageName.DATA_GENERATION.value],
            "description": "Stage 2: Clean and resample OHLCV data",
            "required": True,
            "stage_number": 2,
        },
        {
            "name": StageName.FEATURE_ENGINEERING.value,
            "dependencies": [StageName.DATA_CLEANING.value],
            "description": "Stage 3: Generate technical features",
            "required": True,
            "stage_number": 3,
        },
        {
            "name": StageName.INITIAL_LABELING.value,
            "dependencies": [StageName.FEATURE_ENGINEERING.value],
            "description": "Stage 4: Apply initial triple-barrier labeling",
            "required": True,
            "stage_number": 4,
        },
        {
            "name": StageName.GA_OPTIMIZE.value,
            "dependencies": [StageName.INITIAL_LABELING.value],
            "description": "Stage 5: GA optimization of barrier parameters",
            "required": True,
            "stage_number": 5,
        },
        {
            "name": StageName.FINAL_LABELS.value,
            "dependencies": [StageName.GA_OPTIMIZE.value],
            "description": "Stage 6: Apply optimized labels with quality scores",
            "required": True,
            "stage_number": 6,
        },
        {
            "name": StageName.CREATE_SPLITS.value,
            "dependencies": [StageName.FINAL_LABELS.value],
            "description": "Stage 7: Create train/val/test splits",
            "required": True,
            "stage_number": 7,
        },
        {
            "name": StageName.FEATURE_SCALING.value,
            "dependencies": [StageName.CREATE_SPLITS.value],
            "description": "Stage 7.5: Train-only feature scaling",
            "required": True,
            "stage_number": 7.5,
        },
        {
            "name": StageName.BUILD_DATASETS.value,
            "dependencies": [StageName.FEATURE_SCALING.value],
            "description": "Stage 7.6: Build dataset splits and manifests",
            "required": True,
            "stage_number": 7.6,
        },
        {
            "name": StageName.VALIDATE_SCALED.value,
            "dependencies": [StageName.BUILD_DATASETS.value],
            "description": "Stage 7.7: Post-scale drift validation",
            "required": True,
            "stage_number": 7.7,
        },
        {
            "name": StageName.VALIDATE.value,
            "dependencies": [StageName.VALIDATE_SCALED.value],
            "description": "Stage 8: Comprehensive data validation",
            "required": True,
            "stage_number": 8,
        },
        {
            "name": StageName.GENERATE_REPORT.value,
            "dependencies": [StageName.VALIDATE.value],
            "description": "Stage 9: Generate completion report",
            "required": True,
            "stage_number": 9,
        },
    ]


def get_stage_order() -> list[str]:
    """Get ordered list of stage names."""
    return [s["name"] for s in get_stage_definitions()]


def get_stage_by_name(name: str) -> dict | None:
    """Get stage definition by name."""
    for stage in get_stage_definitions():
        if stage["name"] == name:
            return stage
    return None
