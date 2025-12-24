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
"""

# Core stage imports - redirect to phase1 modules
from src.phase1.stages.ingest import DataIngestor
from src.phase1.stages.clean import DataCleaner
from src.phase1.stages.features import FeatureEngineer

# Extended stage imports
from src.phase1.stages.splits.core import create_chronological_splits
from src.phase1.stages.validation.run import run_validation

# Feature scaler imports
from src.phase1.stages.scaling import (
    FeatureScalingConfig,
    FeatureScaler,
    scale_splits,
)

__all__ = [
    # Core stages
    'DataIngestor',
    'DataCleaner',
    'FeatureEngineer',
    # Extended stages
    'create_chronological_splits',
    'run_validation',
    # Scaling
    'FeatureScalingConfig',
    'FeatureScaler',
    'scale_splits',
]

__version__ = '1.0.0'
