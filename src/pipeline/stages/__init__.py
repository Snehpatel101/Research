"""
Pipeline Stages - Unified Entry Point for All Stage Functions.

ARCHITECTURE: This module is THE canonical interface for pipeline stage functions.
PipelineRunner and MLFactory import from here, NOT from internal implementations.

This re-export layer provides:
1. Single import location for all stages
2. Clean abstraction boundary between pipeline orchestration and stage logic
3. Stable API while implementations can be refactored

Stage implementations are in src/pipeline/_phase1_impl/stages/.

Usage:
    from src.pipeline.stages import run_feature_engineering  # Always use this!
"""

# =============================================================================
# STAGE 1: DATA INGESTION
# =============================================================================
from src.pipeline._phase1_impl.stages.ingest.run import run_data_generation

# =============================================================================
# STAGE 2: DATA CLEANING
# =============================================================================
from src.pipeline._phase1_impl.stages.clean.run import run_data_cleaning

# =============================================================================
# STAGE 3: FEATURE ENGINEERING
# =============================================================================
from src.pipeline._phase1_impl.stages.features.run import run_feature_engineering

# =============================================================================
# STAGE 4: INITIAL LABELING (Triple-Barrier)
# =============================================================================
from src.pipeline._phase1_impl.stages.labeling.run import run_initial_labeling

# =============================================================================
# STAGE 5: BARRIER OPTIMIZATION (Optuna/GA)
# =============================================================================
from src.pipeline._phase1_impl.stages.ga_optimize.run import run_ga_optimization

# =============================================================================
# STAGE 6: FINAL LABELS (Quality Scores, Sample Weights)
# =============================================================================
from src.pipeline._phase1_impl.stages.final_labels.run import run_final_labels

# =============================================================================
# STAGE 7: TRAIN/VAL/TEST SPLITS
# =============================================================================
from src.pipeline._phase1_impl.stages.splits.run import run_create_splits

# =============================================================================
# STAGE 8: FEATURE SCALING
# =============================================================================
from src.pipeline._phase1_impl.stages.scaling.run import run_feature_scaling

# =============================================================================
# STAGE 9: DATASET BUILDING (TimeSeriesDataContainer)
# =============================================================================
from src.pipeline._phase1_impl.stages.datasets.run import run_build_datasets

# =============================================================================
# STAGE 10: SCALED DATA VALIDATION
# =============================================================================
from src.pipeline._phase1_impl.stages.scaled_validation.run import run_scaled_validation

# =============================================================================
# STAGE 11: COMPREHENSIVE VALIDATION
# =============================================================================
from src.pipeline._phase1_impl.stages.validation.run import run_validation

# =============================================================================
# STAGE 12: REPORT GENERATION
# =============================================================================
from src.pipeline._phase1_impl.stages.reporting.run import run_generate_report


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    # Data stages
    "run_data_generation",
    "run_data_cleaning",
    # Feature stages
    "run_feature_engineering",
    # Labeling stages
    "run_initial_labeling",
    "run_ga_optimization",
    "run_final_labels",
    # Split/scale stages
    "run_create_splits",
    "run_feature_scaling",
    "run_build_datasets",
    # Validation stages
    "run_scaled_validation",
    "run_validation",
    # Reporting
    "run_generate_report",
]
