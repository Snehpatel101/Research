"""
Stage 2: Data Cleaning.

Wrapper that delegates to src/stages/clean/run.py
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Import orchestration logic from stage subdirectory
from src.phase1.stages.clean.run import run_data_cleaning

__all__ = ["run_data_cleaning"]
