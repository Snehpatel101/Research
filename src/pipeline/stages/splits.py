"""
Stage 7: Create Train/Val/Test Splits.

Wrapper that imports from src.stages.splits.run
"""
from src.stages.splits.run import run_create_splits

__all__ = ['run_create_splits']
