"""
Ensemble Price Prediction Pipeline

A comprehensive ML pipeline for financial price prediction using ensemble methods
with triple-barrier labeling and genetic algorithm optimization.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

# Package-level imports for convenience
from src.pipeline_config import PipelineConfig, create_default_config
from src.pipeline.runner import PipelineRunner

__all__ = [
    "__version__",
    "__author__",
    "PipelineConfig",
    "create_default_config",
    "PipelineRunner",
]
