"""
Stage 7.7: Post-Scale Validation with Drift Checks.

Pipeline wrapper for scaled data validation.
"""

import json
import logging
import traceback
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from src.phase1.config.feature_sets import get_feature_set_definitions, resolve_feature_set_names
from src.phase1.config.features import get_drift_config
from src.phase1.stages.validation.drift import check_feature_drift
from src.phase1.utils.feature_sets import resolve_feature_set
from src.pipeline.utils import StageResult, create_failed_result, create_stage_result

if TYPE_CHECKING:
    from manifest import ArtifactManifest
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def run_scaled_validation(config: "PipelineConfig", manifest: "ArtifactManifest") -> StageResult:
    """
    Stage 7.7: Validate scaled splits and compute drift metrics.

    Checks for feature drift between train/val and train/test splits
    using Population Stability Index (PSI).

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

        scaled_dir = config.splits_dir / "scaled"
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
        feature_set_names = resolve_feature_set_names(config.feature_set)

        drift_report: dict[str, dict] = {
            "run_id": config.run_id,
            "created_at": datetime.now().isoformat(),
            "enabled": True,
            "config": drift_config,
            "feature_sets": {},
        }

        for set_name in feature_set_names:
            definition = definitions[set_name]
            feature_cols = resolve_feature_set(train_df, definition)
            if not feature_cols:
                raise ValueError(f"No features resolved for feature set '{set_name}'")

            logger.info(
                f"\nChecking drift for feature set: {set_name} ({len(feature_cols)} features)"
            )

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

            drift_report["feature_sets"][set_name] = {
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

        # Run-scoped output for reproducibility
        report_path = config.run_artifacts_dir / f"scaled_drift_report_{config.run_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(drift_report, f, indent=2)

        manifest.add_artifact(
            name="scaled_drift_report",
            file_path=report_path,
            stage="validate_scaled",
            metadata={"feature_sets": feature_set_names},
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
            },
        )

    except Exception as e:
        logger.error(f"Scaled validation failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="validate_scaled", start_time=start_time, error=str(e)
        )
