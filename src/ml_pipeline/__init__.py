"""
Unified ML Pipeline - Single cohesive pipeline from raw data to deployed model.

This module provides the MLPipeline class that orchestrates all phases:
- Phase 1-5: Data pipeline (raw OHLCV -> labeled features)
- Phase 6: Model training (baseline + ensembles)
- Phase 7: Evaluation (CV, walk-forward, CPCV-PBO)
- Phase 8: Deployment (model serving API)

State Management:
- PipelineState: Robust state tracking with thread-safety
- PhaseState: Phase status enum (NOT_STARTED, IN_PROGRESS, COMPLETED, FAILED, SKIPPED)
- PhaseResult: Detailed phase execution result
- StateValidator: Validates state consistency and dependencies

Phase Registry:
- PhaseDefinition: Immutable phase metadata
- PHASE_REGISTRY: All pipeline phases with dependencies

Usage:
    from src.ml_pipeline import MLPipeline, MLConfig

    config = MLConfig(symbol="MES", horizons=[20], models=["xgboost", "lstm"])
    pipeline = MLPipeline(config)
    results = pipeline.run()
"""

from .config import MLConfig, ModelConfig
from .phase_registry import (
    PHASE_REGISTRY,
    PhaseDefinition,
    estimate_total_duration,
    get_all_dependencies,
    get_data_pipeline_phases,
    get_next_phase,
    get_phase,
    get_phase_dependencies,
    get_phase_names_in_order,
    get_phases_in_order,
    get_training_phases,
    iter_phases,
    validate_phase_graph,
)
from .state import (
    PhaseResult,
    PhaseState,
    PipelineState,
    StateDiff,
    StateValidationError,
    StateVersion,
    compute_config_hash,
)
from .state_validator import (
    StateValidator,
    ValidationResult,
    can_resume_pipeline,
    validate_state,
)
from .unified import MLPipeline

__all__ = [
    # Main pipeline
    "MLPipeline",
    "MLConfig",
    "ModelConfig",
    # State management
    "PipelineState",
    "PhaseState",
    "PhaseResult",
    "StateDiff",
    "StateVersion",
    "StateValidationError",
    "compute_config_hash",
    # Phase registry
    "PhaseDefinition",
    "PHASE_REGISTRY",
    "get_phase",
    "get_phase_dependencies",
    "get_all_dependencies",
    "get_phases_in_order",
    "get_phase_names_in_order",
    "get_data_pipeline_phases",
    "get_training_phases",
    "get_next_phase",
    "iter_phases",
    "estimate_total_duration",
    "validate_phase_graph",
    # Validation
    "StateValidator",
    "ValidationResult",
    "validate_state",
    "can_resume_pipeline",
]
