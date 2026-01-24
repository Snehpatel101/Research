"""
Stage 8: Data Validation.

Pipeline wrapper for comprehensive data validation.
"""

import logging
import traceback
from datetime import datetime
from typing import TYPE_CHECKING

from src.data.pipeline.stages.validation import validate_data
from src.data.pipeline.utils import (
    StageResult,
    StageStatus,
    create_failed_result,
    create_stage_result,
)

if TYPE_CHECKING:
    from src.core.common.manifest import ArtifactManifest
    from src.data.pipeline.data_config import DataConfig as PipelineConfig
    from src.optimization.feature_selection.result import FeatureSelectionResult

logger = logging.getLogger(__name__)


def _validate_single_timeframe(
    config: "PipelineConfig",
    tf: str,
    is_multi_tf: bool,
) -> tuple[dict, "FeatureSelectionResult | None", list]:
    """
    Validate data for a single timeframe.

    Args:
        config: Pipeline configuration
        tf: Timeframe to validate
        is_multi_tf: Whether running in multi-TF mode

    Returns:
        Tuple of (summary dict, feature_selection_result, artifact paths)
    """

    logger.info(f"\n{'='*50}")
    logger.info(f"Validating timeframe: {tf}")
    logger.info(f"{'='*50}")

    # Determine path based on multi-TF mode
    if is_multi_tf:
        combined_path = config.final_data_dir / f"combined_{tf}_final_labeled.parquet"
        tf_suffix = f"_{tf}"
    else:
        combined_path = config.final_data_dir / "combined_final_labeled.parquet"
        tf_suffix = ""

    if not combined_path.exists():
        raise FileNotFoundError(f"Combined labeled data not found for {tf}: {combined_path}")

    # Run-scoped output paths for reproducibility
    validation_report_path = (
        config.run_artifacts_dir / f"validation_report{tf_suffix}_{config.run_id}.json"
    )
    feature_selection_path = (
        config.run_artifacts_dir / f"feature_selection{tf_suffix}_{config.run_id}.json"
    )

    logger.info(f"Validating combined dataset: {combined_path}")

    summary, feature_selection_result = validate_data(
        data_path=combined_path,
        output_path=validation_report_path,
        horizons=config.label_horizons,
        run_feature_selection=True,
        correlation_threshold=0.85,
        variance_threshold=0.01,
        feature_selection_output_path=feature_selection_path,
    )

    artifacts = [validation_report_path]
    if feature_selection_path.exists():
        artifacts.append(feature_selection_path)

    logger.info(f"\n{'='*50}")
    logger.info(f"VALIDATION SUMMARY for {tf}")
    logger.info(f"{'='*50}")
    logger.info(f"Status: {summary['status']}")
    logger.info(f"Issues: {summary['issues_count']}")
    logger.info(f"Warnings: {summary['warnings_count']}")

    if summary["issues"]:
        logger.error("Critical Issues Found:")
        for issue in summary["issues"][:10]:
            logger.error(f"  - {issue}")
        if len(summary["issues"]) > 10:
            logger.error(f"  ... and {len(summary['issues'])-10} more")

    if summary["warnings"]:
        logger.warning("Warnings:")
        for warning in summary["warnings"][:10]:
            logger.warning(f"  - {warning}")

    if feature_selection_result:
        logger.info("\nFeature Selection:")
        logger.info(f"  Original features: {feature_selection_result.original_count}")
        logger.info(f"  Selected features: {feature_selection_result.final_count}")
        logger.info(f"  Removed: {len(feature_selection_result.removed_features)}")

    return summary, feature_selection_result, artifacts


def run_validation(config: "PipelineConfig", manifest: "ArtifactManifest") -> StageResult:
    """
    Stage 8: Comprehensive data validation.

    Validates:
    - Data completeness and integrity
    - Feature quality (missing values, correlations, variance)
    - Label distribution and balance
    - Split integrity and leakage prevention
    - Feature selection for high-quality features

    Supports multi-TF mode by validating each timeframe's combined file separately.

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
        # Get effective output timeframes (supports multi-TF mode)
        output_timeframes = config.effective_output_timeframes
        is_multi_tf = len(output_timeframes) > 1

        logger.info(f"Output timeframes: {output_timeframes}")
        logger.info(f"Multi-TF mode: {is_multi_tf}")

        all_artifacts = []
        all_summaries = {}
        any_failed = False
        total_issues = 0
        total_warnings = 0

        # Validate each timeframe
        for tf in output_timeframes:
            summary, feature_selection_result, artifacts = _validate_single_timeframe(
                config, tf, is_multi_tf
            )
            all_artifacts.extend(artifacts)
            all_summaries[tf] = summary

            # Track overall status
            if summary["status"] == "FAILED":
                any_failed = True
            total_issues += summary["issues_count"]
            total_warnings += summary["warnings_count"]

            # Add manifest entry for this timeframe
            manifest.add_artifact(
                name=f"validation_report_{tf}" if is_multi_tf else "validation_report",
                file_path=artifacts[0],  # validation_report path
                stage="validate",
                metadata={
                    "timeframe": tf,
                    "status": summary["status"],
                    "issues_count": summary["issues_count"],
                    "warnings_count": summary["warnings_count"],
                },
            )

        # Build combined metadata
        combined_metadata = {
            "multi_tf_mode": is_multi_tf,
            "output_timeframes": output_timeframes,
            "total_issues": total_issues,
            "total_warnings": total_warnings,
            "timeframe_summaries": all_summaries,
        }

        if any_failed:
            logger.error(
                f"\nValidation FAILED with {total_issues} critical issues across timeframes"
            )
            end_time = datetime.now()
            return StageResult(
                stage_name="validate",
                status=StageStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                artifacts=all_artifacts,
                error=f"Validation failed with {total_issues} critical issues",
                metadata=combined_metadata,
            )

        logger.info(f"\nValidation PASSED (with {total_warnings} warnings across timeframes)")

        return create_stage_result(
            stage_name="validate",
            start_time=start_time,
            artifacts=all_artifacts,
            metadata=combined_metadata,
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(stage_name="validate", start_time=start_time, error=str(e))
