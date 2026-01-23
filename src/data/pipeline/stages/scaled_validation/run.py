"""
Stage 7.7: Post-Scale Validation with Drift Checks.

Pipeline wrapper for scaled data validation.
"""

import json
import logging
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.data.pipeline.config.feature_sets import (
    get_feature_set_definitions,
    resolve_feature_set_names,
)
from src.data.pipeline.config.features import get_drift_config
from src.data.pipeline.stages.validation.drift import check_feature_drift
from src.data.pipeline.utils import StageResult, create_failed_result, create_stage_result
from src.data.pipeline.utils.feature_sets import resolve_feature_set

if TYPE_CHECKING:
    from src.core.common.manifest import ArtifactManifest
    from src.data.pipeline.data_config import DataConfig as PipelineConfig

logger = logging.getLogger(__name__)


def _validate_single_timeframe(
    config: "PipelineConfig",
    tf: str,
    is_multi_tf: bool,
    drift_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate scaled splits for a single timeframe.

    Args:
        config: Pipeline configuration
        tf: Timeframe to validate
        is_multi_tf: Whether running in multi-TF mode
        drift_config: Drift checking configuration

    Returns:
        Drift report for this timeframe
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Validating scaled data for timeframe: {tf}")
    logger.info(f"{'='*50}")

    # Determine paths based on multi-TF mode
    scaled_dir = config.splits_dir / tf / "scaled" if is_multi_tf else config.splits_dir / "scaled"

    train_path = scaled_dir / "train_scaled.parquet"
    val_path = scaled_dir / "val_scaled.parquet"
    test_path = scaled_dir / "test_scaled.parquet"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Scaled split not found: {path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    definitions = get_feature_set_definitions()
    feature_set_names = resolve_feature_set_names(config.feature_generation)

    tf_drift_report: dict[str, Any] = {
        "timeframe": tf,
        "feature_sets": {},
    }

    for set_name in feature_set_names:
        definition = definitions[set_name]
        feature_cols = resolve_feature_set(train_df, definition)
        if not feature_cols:
            raise ValueError(f"No features resolved for feature set '{set_name}' in {tf}")

        logger.info(f"\nChecking drift for feature set: {set_name} ({len(feature_cols)} features)")

        val_drift = check_feature_drift(
            train_df,
            val_df,
            feature_cols,
            bins=drift_config.get("bins", 10),
            psi_threshold=drift_config.get("psi_threshold", 0.2),
            max_features=drift_config.get("max_features", 200),
        )
        test_drift = check_feature_drift(
            train_df,
            test_df,
            feature_cols,
            bins=drift_config.get("bins", 10),
            psi_threshold=drift_config.get("psi_threshold", 0.2),
            max_features=drift_config.get("max_features", 200),
        )

        tf_drift_report["feature_sets"][set_name] = {
            "feature_count": len(feature_cols),
            "val_drift": val_drift,
            "test_drift": test_drift,
        }

        val_drifted = val_drift.get("drifted_feature_count", 0)
        val_total = val_drift.get("feature_count", 0)
        test_drifted = test_drift.get("drifted_feature_count", 0)
        test_total = test_drift.get("feature_count", 0)
        logger.info(f"  Val drift - drifted: {val_drifted}/{val_total} features")
        logger.info(f"  Test drift - drifted: {test_drifted}/{test_total} features")

    return tf_drift_report


def run_scaled_validation(config: "PipelineConfig", manifest: "ArtifactManifest") -> StageResult:
    """
    Stage 7.7: Validate scaled splits and compute drift metrics.

    Checks for feature drift between train/val and train/test splits
    using Population Stability Index (PSI).

    Supports multi-TF mode by validating each timeframe's scaled splits separately.

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 7.7: Post-Scale Validation (Drift Checks)")
    logger.info("=" * 70)

    try:
        drift_config = get_drift_config()
        if not drift_config.get("enabled", True):
            # Run-scoped output for reproducibility
            report_path = config.run_artifacts_dir / f"scaled_drift_report_{config.run_id}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            drift_report = {
                "run_id": config.run_id,
                "created_at": datetime.now().isoformat(),
                "enabled": False,
                "reason": "disabled_in_config",
                "config": drift_config,
            }
            with open(report_path, "w") as f:
                json.dump(drift_report, f, indent=2)

            manifest.add_artifact(
                name="scaled_drift_report",
                file_path=report_path,
                stage="validate_scaled",
                metadata={"enabled": False},
            )

            return create_stage_result(
                stage_name="validate_scaled",
                start_time=start_time,
                artifacts=[report_path],
                metadata={
                    "enabled": False,
                    "report_path": str(report_path),
                },
            )

        # Get effective output timeframes (supports multi-TF mode)
        output_timeframes = config.effective_output_timeframes
        is_multi_tf = len(output_timeframes) > 1

        logger.info(f"Output timeframes: {output_timeframes}")
        logger.info(f"Multi-TF mode: {is_multi_tf}")

        feature_set_names = resolve_feature_set_names(config.feature_generation)

        drift_report_full: dict[str, Any] = {
            "run_id": config.run_id,
            "created_at": datetime.now().isoformat(),
            "enabled": True,
            "config": drift_config,
            "multi_tf_mode": is_multi_tf,
            "output_timeframes": output_timeframes,
        }

        if is_multi_tf:
            # Multi-TF mode: validate each timeframe separately
            drift_report_full["timeframes"] = {}
            for tf in output_timeframes:
                tf_report = _validate_single_timeframe(config, tf, is_multi_tf, drift_config)
                drift_report_full["timeframes"][tf] = tf_report
        else:
            # Single-TF mode: use existing structure for backward compatibility
            tf = output_timeframes[0]
            tf_report = _validate_single_timeframe(config, tf, is_multi_tf, drift_config)
            drift_report_full["feature_sets"] = tf_report["feature_sets"]

        # Run-scoped output for reproducibility
        report_path = config.run_artifacts_dir / f"scaled_drift_report_{config.run_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(drift_report_full, f, indent=2)

        manifest.add_artifact(
            name="scaled_drift_report",
            file_path=report_path,
            stage="validate_scaled",
            metadata={
                "feature_sets": feature_set_names,
                "multi_tf_mode": is_multi_tf,
                "output_timeframes": output_timeframes,
            },
        )

        logger.info(f"\nDrift report saved to: {report_path}")

        return create_stage_result(
            stage_name="validate_scaled",
            start_time=start_time,
            artifacts=[report_path],
            metadata={
                "feature_sets": feature_set_names,
                "report_path": str(report_path),
                "drift_config": drift_config,
                "multi_tf_mode": is_multi_tf,
                "output_timeframes": output_timeframes,
            },
        )

    except Exception as e:
        logger.error(f"Scaled validation failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="validate_scaled", start_time=start_time, error=str(e)
        )
