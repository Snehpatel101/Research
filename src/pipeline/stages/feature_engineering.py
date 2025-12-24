"""
Stage 3: Feature Engineering.

Wrapper that delegates to src/stages/features/run.py
"""
from typing import TYPE_CHECKING

from ..utils import StageResult

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

# Import orchestration logic from stage subdirectory
from src.stages.features.run import run_feature_engineering

__all__ = ['run_feature_engineering']
