"""
Stage 2: Data Cleaning.

Wrapper that delegates to src/stages/clean/run.py
"""
from typing import TYPE_CHECKING

from ..utils import StageResult

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

# Import orchestration logic from stage subdirectory
from src.stages.clean.run import run_data_cleaning

__all__ = ['run_data_cleaning']
