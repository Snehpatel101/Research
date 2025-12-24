"""
Stage 7.6: Dataset Build.

Pipeline wrapper for building dataset splits and manifests.
"""
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import pandas as pd

from src.pipeline.utils import StageResult, create_stage_result, create_failed_result
from src.config.feature_sets import get_feature_set_definitions, resolve_feature_set_names
from src.config.labels import REQUIRED_LABEL_TEMPLATES, OPTIONAL_LABEL_TEMPLATES
from src.utils.feature_sets import (
    METADATA_COLUMNS,
    build_feature_set_manifest,
    resolve_feature_set,
    validate_feature_set_columns,
)

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

logger = logging.getLogger(__name__)


def _select_label_columns(df: pd.DataFrame, horizon: int) -> List[str]:
    """Select label columns for a given horizon."""
    required = [t.format(h=horizon) for t in REQUIRED_LABEL_TEMPLATES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required label columns for h{horizon}: {missing}"
        )
    optional = [
        t.format(h=horizon) for t in OPTIONAL_LABEL_TEMPLATES
        if t.format(h=horizon) in df.columns
    ]
    return required + optional


def _select_metadata_columns(df: pd.DataFrame) -> List[str]:
    """Select metadata columns present in DataFrame."""
    return [col for col in df.columns if col in METADATA_COLUMNS]


def run_build_datasets(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
    """
    Stage 7.6: Build dataset splits and manifests.

    Creates:
    - Feature set manifest (all available features organized by category)
    - Dataset splits for each (feature_set, horizon) combination
    - Dataset manifest with paths and metadata

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 7.6: Dataset Build")
    logger.info("=" * 70)

    try:
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

        feature_set_names = resolve_feature_set_names(config.feature_set)
        definitions = get_feature_set_definitions()

        feature_set_manifest = build_feature_set_manifest(train_df, definitions)
        feature_set_manifest_path = config.run_artifacts_dir / "feature_set_manifest.json"
        feature_set_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(feature_set_manifest_path, "w") as f:
            json.dump(feature_set_manifest, f, indent=2)

        artifacts: List[Path] = [feature_set_manifest_path]
        dataset_manifest: Dict[str, Dict] = {
            "run_id": config.run_id,
            "created_at": datetime.now().isoformat(),
            "feature_sets": {},
        }

        metadata_cols = _select_metadata_columns(train_df)
        datasets_root = config.splits_dir / "datasets"
        datasets_root.mkdir(parents=True, exist_ok=True)

        for set_name in feature_set_names:
            definition = definitions[set_name]
            features = resolve_feature_set(train_df, definition)
            validate_feature_set_columns(train_df, features, set_name)

            dataset_manifest["feature_sets"][set_name] = {
                "description": definition.description,
                "features": features,
                "horizons": {},
            }

            for horizon in config.label_horizons:
                label_cols = _select_label_columns(train_df, horizon)
                columns = metadata_cols + features + label_cols

                output_dir = datasets_root / set_name / f"h{horizon}"
                output_dir.mkdir(parents=True, exist_ok=True)

                train_out = output_dir / "train.parquet"
                val_out = output_dir / "val.parquet"
                test_out = output_dir / "test.parquet"

                train_df[columns].to_parquet(train_out, index=False)
                val_df[columns].to_parquet(val_out, index=False)
                test_df[columns].to_parquet(test_out, index=False)

                artifacts.extend([train_out, val_out, test_out])

                dataset_manifest["feature_sets"][set_name]["horizons"][str(horizon)] = {
                    "feature_count": len(features),
                    "label_columns": label_cols,
                    "metadata_columns": metadata_cols,
                    "train_path": str(train_out),
                    "val_path": str(val_out),
                    "test_path": str(test_out),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                }

                logger.info(
                    f"Created dataset: {set_name}/h{horizon} "
                    f"({len(features)} features, {len(label_cols)} labels)"
                )

        dataset_manifest_path = config.run_artifacts_dir / "dataset_manifest.json"
        with open(dataset_manifest_path, "w") as f:
            json.dump(dataset_manifest, f, indent=2)
        artifacts.append(dataset_manifest_path)

        manifest.add_artifact(
            name="feature_set_manifest",
            file_path=feature_set_manifest_path,
            stage="build_datasets",
            metadata={"feature_sets": feature_set_names},
        )
        manifest.add_artifact(
            name="dataset_manifest",
            file_path=dataset_manifest_path,
            stage="build_datasets",
            metadata={"feature_sets": feature_set_names},
        )

        logger.info(f"\nDataset build complete:")
        logger.info(f"  Feature sets: {len(feature_set_names)}")
        logger.info(f"  Horizons: {len(config.label_horizons)}")
        logger.info(f"  Total datasets: {len(feature_set_names) * len(config.label_horizons)}")
        logger.info(f"  Output directory: {datasets_root}")

        return create_stage_result(
            stage_name="build_datasets",
            start_time=start_time,
            artifacts=artifacts,
            metadata={
                "feature_sets": feature_set_names,
                "dataset_manifest_path": str(dataset_manifest_path),
            },
        )

    except Exception as e:
        logger.error(f"Dataset build failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="build_datasets",
            start_time=start_time,
            error=str(e)
        )
