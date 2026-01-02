"""
Stage 3: Feature Engineering.

Wrapper that delegates to src/stages/features/run.py
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Import orchestration logic from stage subdirectory
from src.phase1.stages.features.run import run_feature_engineering

__all__ = ["run_feature_engineering"]
