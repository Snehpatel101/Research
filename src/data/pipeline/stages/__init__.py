"""
Production-ready data pipeline stages for ensemble trading system.

Stage 1: Data Ingestion - Load and standardize raw data
Stage 2: Data Cleaning - Clean, validate, and fill gaps
Stage 3: Feature Engineering - Generate 50+ technical indicators
Stage 4: Labeling - Generate target labels
Stage 5: GA Optimization - Genetic algorithm for label optimization
Stage 6: Final Labels - Generate final optimized labels
Stage 7: Data Splitting - Chronological train/val/test splits with purging and embargo
Stage 7.5: Feature Scaling - Fit scalers on train, transform all splits
Stage 7.6: Dataset Building - Create model-ready datasets
Stage 8: Validation - Comprehensive data, label, and feature quality checks
Stage 9: Report Generation - Comprehensive Phase 1 summary with charts
Stage 10: Evaluation - Post-training model evaluation and metrics
"""

from .clean import DataCleaner
from .clean.run import run_data_cleaning
from .datasets.run import run_build_datasets
from .evaluation.run import run_evaluation
from .features import FeatureEngineer
from .features.run import run_feature_engineering
from .final_labels.run import run_final_labels
from .ga_optimize.run import run_ga_optimization
from .ingest import DataIngestor
from .ingest.run import run_data_generation
from .labeling.run import run_initial_labeling
from .reporting.run import run_generate_report
from .scaled_validation.run import run_scaled_validation
from .scaling import (
    FeatureScaler,
    FeatureScalingConfig,
    scale_splits,
)
from .scaling.run import run_feature_scaling
from .splits.core import create_chronological_splits
from .splits.run import run_create_splits
from .validation.run import run_validation

__all__ = [
    "DataIngestor",
    "DataCleaner",
    "FeatureEngineer",
    "create_chronological_splits",
    "run_validation",
    "FeatureScalingConfig",
    "FeatureScaler",
    "scale_splits",
    "run_data_cleaning",
    "run_build_datasets",
    "run_evaluation",
    "run_feature_engineering",
    "run_final_labels",
    "run_ga_optimization",
    "run_data_generation",
    "run_initial_labeling",
    "run_generate_report",
    "run_scaled_validation",
    "run_feature_scaling",
    "run_create_splits",
]

__version__ = "1.0.0"
