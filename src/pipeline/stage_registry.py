"""
Pipeline Stage Registry.

Defines the PipelineStage dataclass and provides stage registration utilities.
"""

from collections.abc import Callable
from dataclasses import dataclass, field


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
            "name": "data_generation",
            "dependencies": [],
            "description": "Stage 1: Generate or validate raw data files",
            "required": True,
            "stage_number": 1,
        },
        {
            "name": "data_cleaning",
            "dependencies": ["data_generation"],
            "description": "Stage 2: Clean and resample OHLCV data",
            "required": True,
            "stage_number": 2,
        },
        {
            "name": "feature_engineering",
            "dependencies": ["data_cleaning"],
            "description": "Stage 3: Generate technical features",
            "required": True,
            "stage_number": 3,
        },
        {
            "name": "initial_labeling",
            "dependencies": ["feature_engineering"],
            "description": "Stage 4: Apply initial triple-barrier labeling",
            "required": True,
            "stage_number": 4,
        },
        {
            "name": "ga_optimize",
            "dependencies": ["initial_labeling"],
            "description": "Stage 5: GA optimization of barrier parameters",
            "required": True,
            "stage_number": 5,
        },
        {
            "name": "final_labels",
            "dependencies": ["ga_optimize"],
            "description": "Stage 6: Apply optimized labels with quality scores",
            "required": True,
            "stage_number": 6,
        },
        {
            "name": "create_splits",
            "dependencies": ["final_labels"],
            "description": "Stage 7: Create train/val/test splits",
            "required": True,
            "stage_number": 7,
        },
        {
            "name": "feature_scaling",
            "dependencies": ["create_splits"],
            "description": "Stage 7.5: Train-only feature scaling",
            "required": True,
            "stage_number": 7.5,
        },
        {
            "name": "build_datasets",
            "dependencies": ["feature_scaling"],
            "description": "Stage 7.6: Build dataset splits and manifests",
            "required": True,
            "stage_number": 7.6,
        },
        {
            "name": "validate_scaled",
            "dependencies": ["build_datasets"],
            "description": "Stage 7.7: Post-scale drift validation",
            "required": True,
            "stage_number": 7.7,
        },
        {
            "name": "validate",
            "dependencies": ["validate_scaled"],
            "description": "Stage 8: Comprehensive data validation",
            "required": True,
            "stage_number": 8,
        },
        {
            "name": "generate_report",
            "dependencies": ["validate"],
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
