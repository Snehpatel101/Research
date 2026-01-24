"""
Pipeline Package - Data Preparation Pipeline.

This package provides a modular, stage-based pipeline for processing
raw OHLCV data into ML-ready features and labels.

Stage Flow:
1. data_generation    - Generate/validate raw data
2. data_cleaning      - Clean and resample OHLCV data
3. feature_engineering - Generate technical features
4. initial_labeling   - Apply initial triple-barrier labels
5. ga_optimize        - Genetic algorithm optimization of barrier params
6. final_labels       - Apply optimized labels with quality scores
7. create_splits      - Create train/val/test splits
8. validate           - Comprehensive data validation
9. generate_report    - Generate completion report

Usage:
    from src.data.pipeline import PipelineRunner, DataConfig

    config = DataConfig(
        symbols=['MES'],
        start_date='2020-01-01',
        end_date='2024-12-31'
    )

    runner = PipelineRunner(config)
    success = runner.run()
"""

from .feature_manifest import FeatureManifest
from .schemas import (
    STAGE_SCHEMAS,
    StageSchema,
    StageValidationError,
    get_stage_schema,
    register_stage_schema,
    validate_stage_output,
)
from .stage_registry import PipelineStage, get_stage_definitions, get_stage_order
from .utils import StageResult, StageStatus

__all__ = [
    "PipelineRunner",
    "DataConfig",
    "StageStatus",
    "StageResult",
    "PipelineStage",
    "get_stage_definitions",
    "get_stage_order",
    # Phase 7B: Schema validation
    "StageSchema",
    "StageValidationError",
    "STAGE_SCHEMAS",
    "validate_stage_output",
    "get_stage_schema",
    "register_stage_schema",
    # Phase 7D: Feature manifest
    "FeatureManifest",
]

__version__ = "1.0.0"


def __getattr__(name: str):
    """Lazy import for circular dependency avoidance."""
    if name == "PipelineRunner":
        from .runner import PipelineRunner

        return PipelineRunner
    if name == "DataConfig":
        from .data_config import DataConfig

        return DataConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
