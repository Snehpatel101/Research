"""
Production-ready data pipeline stages for ensemble trading system.

Stage 1: Data Ingestion - Load and standardize raw data
Stage 2: Data Cleaning - Clean, validate, and fill gaps
Stage 3: Feature Engineering - Generate 50+ technical indicators
Stage 4: Labeling - Generate target labels
Stage 5: GA Optimization - Genetic algorithm for label optimization
Stage 6: Final Labels - Generate final optimized labels
Stage 7: Data Splitting - Chronological train/val/test splits with purging and embargo
Stage 8: Validation - Comprehensive data, label, and feature quality checks
Baseline Backtest - Simple strategy to verify label signal
Report Generation - Comprehensive Phase 1 summary with charts
"""

from .stage1_ingest import DataIngestor
from .stage2_clean import DataCleaner
from .stage3_features import FeatureEngineer

# Import other stages if available
try:
    from .stage7_splits import create_splits
    from .stage8_validate import validate_data
    from .baseline_backtest import run_baseline_backtest
    from .generate_report import generate_phase1_report
    _extended_imports = True
except ImportError:
    _extended_imports = False

__all__ = [
    'DataIngestor',
    'DataCleaner',
    'FeatureEngineer'
]

if _extended_imports:
    __all__.extend([
        'create_splits',
        'validate_data',
        'run_baseline_backtest',
        'generate_phase1_report'
    ])

__version__ = '1.0.0'
