"""
Stage 7.6: Dataset Build.

Pipeline wrapper for building dataset splits and manifests with multi-timeframe support.

PIPE-001/PIPE-002 Fix: This stage now handles multi-TF mode by processing
each timeframe's scaled data separately.
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.phase1.config.feature_sets import get_feature_set_definitions, resolve_feature_set_names
from src.phase1.config.labels import OPTIONAL_LABEL_TEMPLATES, REQUIRED_LABEL_TEMPLATES
from src.phase1.utils.feature_sets import (
    METADATA_COLUMNS,
    build_feature_set_manifest,
    resolve_feature_set,
    validate_feature_set_columns,
)
from src.pipeline.utils import StageResult, create_failed_result, create_stage_result


class FeatureSchemaError(ValueError):
    """Raised when feature schemas are inconsistent across DataFrames."""

    pass


def validate_feature_schema(
    dfs: dict[str, pd.DataFrame],
    context: str,
    *,
    warn_on_order_mismatch: bool = True,
) -> None:
    """
    Validate that all DataFrames have identical feature columns.

    PIPE-004: This validation ensures consistent feature schemas across splits
    and timeframes to prevent silent model failures from NaN features.

    Args:
        dfs: Dictionary mapping split/file names to DataFrames
            Example: {"train": train_df, "val": val_df, "test": test_df}
        context: Description of what is being validated (for error messages)
            Example: "timeframe 5min" or "symbol MES"
        warn_on_order_mismatch: If True, log warning when column order differs
            but sets match. Defaults to True.

    Raises:
        FeatureSchemaError: If DataFrames have different column sets, with
            detailed message showing which files have missing/extra columns.

    Example:
        >>> validate_feature_schema(
        ...     {"train": train_df, "val": val_df, "test": test_df},
        ...     context="timeframe 5min"
        ... )
    """
    if len(dfs) < 2:
        logger.debug(f"validate_feature_schema: Only {len(dfs)} DataFrame(s), skipping")
        return

    # Use first DataFrame as reference
    names = list(dfs.keys())
    ref_name = names[0]
    ref_df = dfs[ref_name]
    ref_columns = set(ref_df.columns)

    # Track all differences for comprehensive error message
    missing_by_file: dict[str, list[str]] = {}
    extra_by_file: dict[str, list[str]] = {}
    order_mismatches: list[str] = []

    for name in names[1:]:
        df = dfs[name]
        current_columns = set(df.columns)

        missing = ref_columns - current_columns
        extra = current_columns - ref_columns

        if missing:
            missing_by_file[name] = sorted(missing)
        if extra:
            extra_by_file[name] = sorted(extra)

        # Check column order if sets match
        if not missing and not extra:
            if list(ref_df.columns) != list(df.columns):
                order_mismatches.append(name)

    # Log order mismatches as warnings (not errors)
    if order_mismatches and warn_on_order_mismatch:
        logger.warning(
            f"[{context}] Column order differs from '{ref_name}' in: {order_mismatches}. "
            f"Column sets match but order is inconsistent."
        )

    # Build comprehensive error message if schemas differ
    if missing_by_file or extra_by_file:
        error_lines = [
            f"Feature schema mismatch in {context}.",
            f"Reference: '{ref_name}' with {len(ref_columns)} columns.",
        ]

        if missing_by_file:
            error_lines.append("")
            error_lines.append("MISSING columns (present in reference, absent in target):")
            for name, cols in missing_by_file.items():
                preview = cols[:10]
                suffix = f" ... and {len(cols) - 10} more" if len(cols) > 10 else ""
                error_lines.append(f"  - '{name}': {preview}{suffix}")

        if extra_by_file:
            error_lines.append("")
            error_lines.append("EXTRA columns (absent in reference, present in target):")
            for name, cols in extra_by_file.items():
                preview = cols[:10]
                suffix = f" ... and {len(cols) - 10} more" if len(cols) > 10 else ""
                error_lines.append(f"  - '{name}': {preview}{suffix}")

        error_lines.append("")
        error_lines.append(
            "This may cause NaN features in combined data and silent model failures."
        )
        error_lines.append(
            "Check upstream stages (features, scaling) for consistency issues."
        )

        raise FeatureSchemaError("\n".join(error_lines))

if TYPE_CHECKING:
    from manifest import ArtifactManifest
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def _select_label_columns(df: pd.DataFrame, horizon: int) -> list[str]:
    """Select label columns for a given horizon."""
    required = [t.format(h=horizon) for t in REQUIRED_LABEL_TEMPLATES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required label columns for h{horizon}: {missing}")
    optional = [
        t.format(h=horizon) for t in OPTIONAL_LABEL_TEMPLATES if t.format(h=horizon) in df.columns
    ]
    return required + optional


def _select_metadata_columns(df: pd.DataFrame) -> list[str]:
    """Select metadata columns present in DataFrame."""
    return [col for col in df.columns if col in METADATA_COLUMNS]


def _build_datasets_for_timeframe(
    config: "PipelineConfig",
    manifest: "ArtifactManifest",
    tf: str,
    is_multi_tf: bool,
    definitions: dict,
    feature_set_names: list[str],
) -> tuple[list[Path], dict]:
    """
    Build datasets for a single timeframe.

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest
        tf: Timeframe to process
        is_multi_tf: Whether running in multi-TF mode
        definitions: Feature set definitions
        feature_set_names: List of feature set names to process

    Returns:
        Tuple of (artifacts list, dataset manifest for this timeframe)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Building datasets for timeframe: {tf}")
    logger.info(f"{'='*50}")

    # Determine paths based on multi-TF mode
    if is_multi_tf:
        scaled_dir = config.splits_dir / tf / "scaled"
        datasets_root = config.splits_dir / tf / "datasets"
    else:
        scaled_dir = config.splits_dir / "scaled"
        datasets_root = config.splits_dir / "datasets"

    train_path = scaled_dir / "train_scaled.parquet"
    val_path = scaled_dir / "val_scaled.parquet"
    test_path = scaled_dir / "test_scaled.parquet"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Scaled split not found for {tf}: {path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    logger.info(f"Loaded scaled data for {tf}:")
    logger.info(f"  Train: {len(train_df):,} rows")
    logger.info(f"  Val: {len(val_df):,} rows")
    logger.info(f"  Test: {len(test_df):,} rows")

    # PIPE-004: Validate feature schema consistency across splits
    # This catches column mismatches early before they cause NaN issues
    validate_feature_schema(
        {"train": train_df, "val": val_df, "test": test_df},
        context=f"timeframe {tf}",
    )
    logger.debug(f"Feature schema validation passed for {tf}")

    artifacts: list[Path] = []
    tf_manifest: dict = {
        "timeframe": tf,
        "feature_sets": {},
    }

    metadata_cols = _select_metadata_columns(train_df)
    datasets_root.mkdir(parents=True, exist_ok=True)

    for set_name in feature_set_names:
        definition = definitions[set_name]
        features = resolve_feature_set(train_df, definition)
        validate_feature_set_columns(train_df, features, set_name)

        tf_manifest["feature_sets"][set_name] = {
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

            tf_manifest["feature_sets"][set_name]["horizons"][str(horizon)] = {
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
                f"Created dataset: {tf}/{set_name}/h{horizon} "
                f"({len(features)} features, {len(label_cols)} labels)"
            )

    return artifacts, tf_manifest


def run_build_datasets(config: "PipelineConfig", manifest: "ArtifactManifest") -> StageResult:
    """
    Stage 7.6: Build dataset splits and manifests.

    Creates:
    - Feature set manifest (all available features organized by category)
    - Dataset splits for each (feature_set, horizon, timeframe) combination
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
        # Get effective output timeframes (supports multi-TF mode)
        output_timeframes = config.effective_output_timeframes
        is_multi_tf = len(output_timeframes) > 1

        logger.info(f"Output timeframes: {output_timeframes}")
        logger.info(f"Multi-TF mode: {is_multi_tf}")

        feature_set_names = resolve_feature_set_names(config.feature_set)
        definitions = get_feature_set_definitions()

        all_artifacts: list[Path] = []
        dataset_manifest: dict = {
            "run_id": config.run_id,
            "created_at": datetime.now().isoformat(),
            "output_timeframes": output_timeframes,
            "multi_tf_mode": is_multi_tf,
            "timeframes": {},
        }

        # Build feature set manifest from the primary timeframe
        # (feature definitions are the same across timeframes)
        primary_tf = config.target_timeframe
        if is_multi_tf:
            primary_scaled_dir = config.splits_dir / primary_tf / "scaled"
        else:
            primary_scaled_dir = config.splits_dir / "scaled"

        primary_train_path = primary_scaled_dir / "train_scaled.parquet"
        if not primary_train_path.exists():
            raise FileNotFoundError(f"Primary train data not found: {primary_train_path}")

        primary_train_df = pd.read_parquet(primary_train_path)
        feature_set_manifest = build_feature_set_manifest(primary_train_df, definitions)
        feature_set_manifest_path = config.run_artifacts_dir / "feature_set_manifest.json"
        feature_set_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(feature_set_manifest_path, "w") as f:
            json.dump(feature_set_manifest, f, indent=2)
        all_artifacts.append(feature_set_manifest_path)

        # PIPE-004: Cross-timeframe schema validation in multi-TF mode
        # Ensures all timeframes have consistent feature columns
        if is_multi_tf and len(output_timeframes) > 1:
            logger.info("Validating feature schema consistency across timeframes...")
            tf_train_dfs: dict[str, pd.DataFrame] = {}
            for tf in output_timeframes:
                tf_train_path = config.splits_dir / tf / "scaled" / "train_scaled.parquet"
                if tf_train_path.exists():
                    tf_train_dfs[f"train_{tf}"] = pd.read_parquet(tf_train_path)
                else:
                    logger.warning(f"Skipping {tf} in cross-TF validation: {tf_train_path} not found")

            if len(tf_train_dfs) > 1:
                validate_feature_schema(
                    tf_train_dfs,
                    context="cross-timeframe comparison",
                )
                logger.info(
                    f"Cross-timeframe schema validation passed for {len(tf_train_dfs)} timeframes"
                )

        # Process each timeframe
        for tf in output_timeframes:
            artifacts, tf_manifest = _build_datasets_for_timeframe(
                config, manifest, tf, is_multi_tf, definitions, feature_set_names
            )
            all_artifacts.extend(artifacts)
            dataset_manifest["timeframes"][tf] = tf_manifest

        # Save dataset manifest
        dataset_manifest_path = config.run_artifacts_dir / "dataset_manifest.json"
        with open(dataset_manifest_path, "w") as f:
            json.dump(dataset_manifest, f, indent=2)
        all_artifacts.append(dataset_manifest_path)

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
            metadata={
                "feature_sets": feature_set_names,
                "output_timeframes": output_timeframes,
            },
        )

        # Summary
        total_datasets = len(feature_set_names) * len(config.label_horizons) * len(output_timeframes)
        logger.info("\nDataset build complete:")
        logger.info(f"  Timeframes: {len(output_timeframes)}")
        logger.info(f"  Feature sets: {len(feature_set_names)}")
        logger.info(f"  Horizons: {len(config.label_horizons)}")
        logger.info(f"  Total datasets: {total_datasets}")

        return create_stage_result(
            stage_name="build_datasets",
            start_time=start_time,
            artifacts=all_artifacts,
            metadata={
                "feature_sets": feature_set_names,
                "output_timeframes": output_timeframes,
                "multi_tf_mode": is_multi_tf,
                "dataset_manifest_path": str(dataset_manifest_path),
            },
        )

    except Exception as e:
        logger.error(f"Dataset build failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="build_datasets", start_time=start_time, error=str(e)
        )
