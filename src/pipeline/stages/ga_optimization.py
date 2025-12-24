"""
Stage 5: Genetic Algorithm Optimization.

Optimizes triple-barrier labeling parameters using genetic algorithms.
"""
import logging
from typing import TYPE_CHECKING

from ..utils import StageResult, create_stage_result, create_failed_result

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

logger = logging.getLogger(__name__)


def run_ga_optimization(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
    """
    Stage 5: Genetic Algorithm Optimization.

    Delegates to src/stages/ga_optimize/run.py for implementation.

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    from src.phase1.stages.ga_optimize.run import run_ga_optimization_stage

    return run_ga_optimization_stage(
        config, manifest, create_stage_result, create_failed_result
    )
