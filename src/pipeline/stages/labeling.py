"""
Stage 4 & 6: Labeling Stages.

Contains both initial triple-barrier labeling and final optimized labeling.
"""
import logging
from typing import TYPE_CHECKING

from ..utils import StageResult, create_failed_result, create_stage_result

if TYPE_CHECKING:
    from manifest import ArtifactManifest
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def run_initial_labeling(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
    """
    Stage 4: Initial Triple-Barrier Labeling.

    Delegates to src/stages/labeling/run.py for implementation.

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    from src.phase1.stages.labeling.run import run_initial_labeling as _run_initial_labeling

    return _run_initial_labeling(
        config, manifest, create_stage_result, create_failed_result
    )


def run_final_labels(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
    """
    Stage 6: Apply optimized labels with quality scores and sample weights.

    Delegates to src/stages/final_labels/run.py for implementation.

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    from src.phase1.stages.final_labels.run import run_final_labels as _run_final_labels

    return _run_final_labels(
        config, manifest, create_stage_result, create_failed_result
    )
