"""
Stage 8: Data Validation.

Comprehensive validation of pipeline outputs including feature quality,
label distribution, and data integrity checks.
"""
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import StageResult, StageStatus, create_stage_result, create_failed_result

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

logger = logging.getLogger(__name__)


def run_validation(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
    """
    Stage 8: Comprehensive data validation.

    Validates:
    - Data completeness and integrity
    - Feature quality (missing values, correlations, variance)
    - Label distribution and balance
    - Split integrity and leakage prevention
    - Feature selection for high-quality features

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 8: Data Validation")
    logger.info("=" * 70)

    try:
        from stages.stage8_validate import validate_data

        # Path to combined labeled data
        combined_path = config.final_data_dir / "combined_final_labeled.parquet"

        if not combined_path.exists():
            raise FileNotFoundError(f"Combined labeled data not found: {combined_path}")

        # Output paths
        validation_report_path = (
            config.results_dir / f"validation_report_{config.run_id}.json"
        )
        feature_selection_path = (
            config.results_dir / f"feature_selection_{config.run_id}.json"
        )

        logger.info(f"Validating combined dataset: {combined_path}")

        # Run validation with feature selection
        summary, feature_selection_result = validate_data(
            data_path=combined_path,
            output_path=validation_report_path,
            horizons=config.label_horizons,
            run_feature_selection=True,
            correlation_threshold=0.85,
            variance_threshold=0.01,
            feature_selection_output_path=feature_selection_path
        )

        artifacts = [validation_report_path]
        if feature_selection_path.exists():
            artifacts.append(feature_selection_path)

        # Log summary
        logger.info(f"\n{'='*50}")
        logger.info("VALIDATION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Status: {summary['status']}")
        logger.info(f"Issues: {summary['issues_count']}")
        logger.info(f"Warnings: {summary['warnings_count']}")

        if summary['issues']:
            logger.error("Critical Issues Found:")
            for issue in summary['issues'][:10]:  # Show first 10
                logger.error(f"  - {issue}")
            if len(summary['issues']) > 10:
                logger.error(f"  ... and {len(summary['issues'])-10} more")

        if summary['warnings']:
            logger.warning("Warnings:")
            for warning in summary['warnings'][:10]:
                logger.warning(f"  - {warning}")

        # Feature selection results
        if feature_selection_result:
            logger.info("\nFeature Selection:")
            logger.info(f"  Original features: {feature_selection_result.original_count}")
            logger.info(f"  Selected features: {feature_selection_result.final_count}")
            logger.info(f"  Removed: {len(feature_selection_result.removed_features)}")

        # Add to manifest
        manifest.add_artifact(
            name="validation_report",
            file_path=validation_report_path,
            stage="validate",
            metadata={
                'status': summary['status'],
                'issues_count': summary['issues_count'],
                'warnings_count': summary['warnings_count']
            }
        )

        # Determine if validation passed or failed
        if summary['status'] == 'FAILED':
            logger.error(
                f"\nValidation FAILED with {summary['issues_count']} critical issues"
            )
            end_time = datetime.now()
            return StageResult(
                stage_name="validate",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=artifacts,
                error=f"Validation failed with {summary['issues_count']} critical issues",
                metadata=summary
            )

        logger.info(f"\nValidation PASSED (with {summary['warnings_count']} warnings)")

        return create_stage_result(
            stage_name="validate",
            start_time=start_time,
            artifacts=artifacts,
            metadata=summary
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="validate",
            start_time=start_time,
            error=str(e)
        )
