"""
Stage 1: Data Generation and Ingestion with Validation.

Wrapper that delegates to src/stages/ingest/run.py
"""
from typing import TYPE_CHECKING

from ..utils import StageResult

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

# Import orchestration logic from stage subdirectory
from src.stages.ingest.run import run_data_generation

__all__ = ['run_data_generation']
