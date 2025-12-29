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

# Core stage imports
from .clean import DataCleaner
from .features import FeatureEngineer
from .ingest import DataIngestor

# Feature scaler imports
from .scaling import (
    FeatureScaler,
    FeatureScalingConfig,
    scale_splits,
)

# Extended stage imports
from .splits.core import create_chronological_splits
from .validation.run import run_validation

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
