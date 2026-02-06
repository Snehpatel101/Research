"""
Pipeline Runner - Main Orchestrator.

Manages stage execution, dependency tracking, and artifact management for
the data preparation pipeline.
"""

import json
import logging
import signal
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .schemas import StageValidationError, validate_stage_transition
from .stage_registry import PipelineStage, get_stage_definitions
from .stages import (
    run_build_datasets,
    run_create_splits,
    run_data_cleaning,
    run_data_generation,
    run_feature_engineering,
    run_feature_scaling,
    run_final_labels,
    run_ga_optimization,
    run_generate_report,
    run_initial_labeling,
    run_scaled_validation,
    run_validation,
)
from .utils import StageResult, StageStatus

# Note: run_evaluation is exported from stages but not used in default pipeline
# Stage 10 is optional and runs post-training


class StageTimeoutError(Exception):
    """Raised when a pipeline stage exceeds its timeout."""

    def __init__(self, stage_name: str, timeout_seconds: int):
        self.stage_name = stage_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Stage '{stage_name}' timed out after {timeout_seconds} seconds")


def _run_with_timeout(
    func: Callable[[], Any],
    timeout_seconds: int,
    stage_name: str,
) -> Any:
    """Execute a function with a timeout using signal.SIGALRM.

    Note: Only works on Unix-like systems. On Windows, timeout is disabled.

    Args:
        func: Function to execute
        timeout_seconds: Maximum execution time in seconds (0 = no timeout)
        stage_name: Name of the stage (for error messages)

    Returns:
        Result of the function call

    Raises:
        StageTimeoutError: If the function exceeds the timeout
    """
    if timeout_seconds <= 0:
        return func()

    # Check if we're on a system that supports SIGALRM
    if not hasattr(signal, "SIGALRM"):
        # Windows doesn't support SIGALRM, run without timeout
        logging.getLogger(__name__).warning(
            f"Timeout not supported on this platform, running {stage_name} without timeout"
        )
        return func()

    def timeout_handler(signum: int, frame: Any) -> None:
        raise StageTimeoutError(stage_name, timeout_seconds)

    # Set up the timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        result = func()
        return result
    finally:
        # Cancel the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

if TYPE_CHECKING:
    from src.data.pipeline.data_config import DataConfig


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class PipelineRunner:
    """Orchestrates the data pipeline execution."""

    def __init__(self, config: "DataConfig", resume: bool = False):
        """
        Initialize pipeline runner.

        Args:
            config: Pipeline configuration
            resume: Whether to resume from last successful stage
        """
        from src.core.common.manifest import ArtifactManifest

        self.config = config
        self.resume = resume
        # Ensure project_root is set before using it
        if config.project_root is None:
            raise ValueError("config.project_root must be set before creating PipelineRunner")
        self.manifest = ArtifactManifest(config.run_id, config.project_root)

        # Set up logging
        self.config.create_directories()
        self.log_file = self.config.run_logs_dir / "pipeline.log"
        self._setup_logging()

        self.logger = logging.getLogger(__name__)

        # Stage tracking
        self.stages: list[PipelineStage] = []
        self.stage_results: dict[str, StageResult] = {}
        self.completed_stages: set[str] = set()

        # Define pipeline stages
        self._define_stages()

        # Load previous state if resuming
        if self.resume:
            self._load_state()

    def _setup_logging(self) -> None:
        """Configure logging for the pipeline."""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Check if handlers already exist to avoid duplicates
        if root_logger.hasHandlers():
            # Clear existing handlers only if they're generic (not from other loggers)
            # This prevents duplicate logging in repeated runs
            for handler in root_logger.handlers[:]:
                if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
                    root_logger.removeHandler(handler)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def _define_stages(self) -> None:
        """Define all pipeline stages and their dependencies."""
        # Map stage names to their execution functions
        stage_functions = {
            "data_generation": lambda: run_data_generation(self.config, self.manifest),
            "data_cleaning": lambda: run_data_cleaning(self.config, self.manifest),
            "feature_engineering": lambda: run_feature_engineering(self.config, self.manifest),
            "initial_labeling": lambda: run_initial_labeling(self.config, self.manifest),
            "ga_optimize": lambda: run_ga_optimization(self.config, self.manifest),
            "final_labels": lambda: run_final_labels(self.config, self.manifest),
            "create_splits": lambda: run_create_splits(self.config, self.manifest),
            "feature_scaling": lambda: run_feature_scaling(self.config, self.manifest),
            "build_datasets": lambda: run_build_datasets(self.config, self.manifest),
            "validate_scaled": lambda: run_scaled_validation(self.config, self.manifest),
            "validate": lambda: run_validation(self.config, self.manifest),
            "generate_report": lambda: run_generate_report(
                self.config, self.manifest, self.stage_results
            ),
        }

        # Build stages from definitions
        self.stages = []
        for stage_def in get_stage_definitions():
            func = stage_functions.get(stage_def["name"])
            if func is None:
                raise ValueError(f"No function defined for stage: {stage_def['name']}")

            self.stages.append(
                PipelineStage(
                    name=stage_def["name"],
                    function=func,
                    dependencies=stage_def["dependencies"],
                    description=stage_def["description"],
                    required=stage_def["required"],
                    stage_number=stage_def["stage_number"],
                )
            )

    def run(self, from_stage: str | None = None) -> bool:
        """
        Run the complete pipeline.

        Args:
            from_stage: Stage name to resume from (None to run all)

        Returns:
            True if all stages completed successfully
        """
        pipeline_start = datetime.now()

        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: DATA PREPARATION PIPELINE")
        self.logger.info(f"Run ID: {self.config.run_id}")
        self.logger.info("=" * 70)

        # Save configuration
        self.config.save_config()
        self.logger.info(f"Configuration saved to {self.config.run_config_dir / 'config.json'}")

        # Determine which stages to run
        stages_to_run = self._get_stages_to_run(from_stage)

        # Execute stages
        all_success = True
        for stage in stages_to_run:
            # Check dependencies
            if not self._check_dependencies(stage):
                self.logger.error(f"Dependencies not met for stage: {stage.name}")
                all_success = False
                break

            # Execute stage (with optional timeout - Phase 43)
            self.logger.info(f"\nExecuting stage: {stage.name}")
            try:
                if self.config.enable_stage_timeouts and self.config.stage_timeout_seconds > 0:
                    self.logger.debug(
                        f"Stage timeout enabled: {self.config.stage_timeout_seconds}s"
                    )
                    result = _run_with_timeout(
                        stage.function,
                        self.config.stage_timeout_seconds,
                        stage.name,
                    )
                else:
                    result = stage.function()
            except StageTimeoutError as e:
                self.logger.error(f"[FAIL] Stage timed out: {stage.name}")
                self.logger.error(f"Timeout: {e.timeout_seconds} seconds exceeded")
                from .utils import create_failed_result

                result = create_failed_result(
                    stage_name=stage.name,
                    start_time=datetime.now(),
                    error=str(e),
                )

            # Store result
            self.stage_results[stage.name] = result

            if result.status == StageStatus.COMPLETED:
                self.completed_stages.add(stage.name)
                self.logger.info(
                    f"[PASS] Stage completed: {stage.name} ({result.duration_seconds:.2f}s)"
                )

                # Phase 7B: Validate stage output schema
                try:
                    self._validate_stage_output(stage.name, result)
                except StageValidationError as e:
                    self.logger.error(f"[FAIL] Schema validation failed for {stage.name}: {e}")
                    all_success = False
                    if stage.required:
                        self.logger.error(
                            "Required stage schema validation failed. Stopping pipeline."
                        )
                        break

                # Phase 43: Validate stage transition to next stage
                if getattr(self.config, "enable_transition_validation", True):
                    try:
                        self._validate_stage_transition(stage, result, stages_to_run)
                    except StageValidationError as e:
                        self.logger.error(
                            f"[FAIL] Stage transition validation failed after {stage.name}: {e}"
                        )
                        all_success = False
                        if stage.required:
                            self.logger.error(
                                "Required stage transition validation failed. Stopping pipeline."
                            )
                            break
            else:
                self.logger.error(f"[FAIL] Stage failed: {stage.name}")
                if result.error:
                    self.logger.error(f"Error: {result.error}")
                all_success = False
                if stage.required:
                    self.logger.error("Required stage failed. Stopping pipeline.")
                    break

            # Save state after each stage
            self._save_state()

        # Save final manifest
        self.manifest.save()
        self.logger.info(f"\nManifest saved to {self.manifest.manifest_path}")

        # Save lineage metadata if pipeline completed successfully
        if all_success:
            self._save_lineage()

        # Final summary
        pipeline_end = datetime.now()
        total_duration = (pipeline_end - pipeline_start).total_seconds()

        self.logger.info("\n" + "=" * 70)
        if all_success:
            self.logger.info("[PASS] PIPELINE COMPLETED SUCCESSFULLY")
        else:
            self.logger.info("[FAIL] PIPELINE FAILED")
        self.logger.info("=" * 70)
        self.logger.info(f"Total duration: {total_duration:.2f} seconds")
        self.logger.info(f"Completed stages: {len(self.completed_stages)}/{len(self.stages)}")
        self.logger.info(f"Run ID: {self.config.run_id}")
        self.logger.info(f"Logs: {self.log_file}")

        return all_success

    def _get_stages_to_run(self, from_stage: str | None) -> list[PipelineStage]:
        """Determine which stages to run based on resume point."""
        if from_stage is None:
            return self.stages

        # Find the index of the from_stage
        start_idx = None
        for idx, stage in enumerate(self.stages):
            if stage.name == from_stage:
                start_idx = idx
                break

        if start_idx is None:
            raise ValueError(f"Stage not found: {from_stage}")

        return self.stages[start_idx:]

    def _check_dependencies(self, stage: PipelineStage) -> bool:
        """Check if all dependencies for a stage are completed."""
        return all(dep in self.completed_stages for dep in stage.dependencies)

    def _save_state(self) -> None:
        """Save current pipeline state."""
        state = {
            "run_id": self.config.run_id,
            "completed_stages": list(self.completed_stages),
            "stage_results": {
                name: result.to_dict() for name, result in self.stage_results.items()
            },
            "saved_at": datetime.now().isoformat(),
        }

        state_path = self.config.run_artifacts_dir / "pipeline_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, cls=NumpyEncoder)

    def _load_state(self) -> None:
        """Load previous pipeline state for resuming."""
        from src.core.common.manifest import ArtifactManifest

        state_path = self.config.run_artifacts_dir / "pipeline_state.json"

        if not state_path.exists():
            self.logger.warning("No previous state found. Starting from beginning.")
            return

        with open(state_path) as f:
            state = json.load(f)

        self.completed_stages = set(state.get("completed_stages", []))
        self.logger.info(f"Loaded state with {len(self.completed_stages)} completed stages")

        # Load manifest if exists
        try:
            if self.config.project_root is None:
                self.logger.warning("project_root is None, cannot load previous manifest")
            else:
                self.manifest = ArtifactManifest.load(self.config.run_id, self.config.project_root)
        except FileNotFoundError:
            self.logger.warning("No previous manifest found.")

    def get_stage_status(self, stage_name: str) -> StageStatus | None:
        """Get the status of a specific stage."""
        if stage_name in self.stage_results:
            return self.stage_results[stage_name].status
        elif stage_name in self.completed_stages:
            return StageStatus.COMPLETED
        return StageStatus.PENDING

    def get_completed_stages(self) -> list[str]:
        """Get list of completed stage names."""
        return list(self.completed_stages)

    def get_stage_result(self, stage_name: str) -> StageResult | None:
        """Get the result of a specific stage."""
        return self.stage_results.get(stage_name)

    def _validate_stage_output(self, stage_name: str, result: StageResult) -> None:
        """
        Validate stage output against its schema (Phase 7B).

        Reads output artifacts and validates them against the stage schema.
        Only validates parquet files; skips other artifact types.

        Args:
            stage_name: Name of the stage
            result: StageResult containing artifact paths

        Raises:
            StageValidationError: If validation fails
        """
        import pandas as pd

        from .schemas import get_stage_schema, validate_stage_output

        schema = get_stage_schema(stage_name)
        if schema is None:
            self.logger.debug(f"No schema for stage '{stage_name}', skipping validation")
            return

        # Validate each parquet artifact
        for artifact_path in result.artifacts:
            if not artifact_path.exists():
                self.logger.warning(f"Artifact not found: {artifact_path}")
                continue

            if artifact_path.suffix != ".parquet":
                continue  # Only validate parquet files

            try:
                df = pd.read_parquet(artifact_path)
                is_valid, issues = validate_stage_output(
                    df=df,
                    stage_name=stage_name,
                    schema=schema,
                    raise_on_failure=True,
                )
                if is_valid:
                    self.logger.info(
                        f"  Schema validation passed for {artifact_path.name}: "
                        f"{len(df)} rows, {len(df.columns)} columns"
                    )
            except Exception as e:
                self.logger.error(f"Schema validation error for {artifact_path}: {e}")
                raise

    def _validate_stage_transition(
        self,
        current_stage: PipelineStage,
        result: StageResult,
        stages_to_run: list[PipelineStage],
    ) -> None:
        """
        Validate data when transitioning between pipeline stages (Phase 43).

        Reads the current stage's output artifacts and validates them against
        the requirements for the next stage in the pipeline.

        Args:
            current_stage: The stage that just completed
            result: StageResult containing artifact paths
            stages_to_run: List of stages being run in this execution

        Raises:
            StageValidationError: If transition validation fails
        """
        import pandas as pd

        # Find the next stage
        current_idx = None
        for idx, stage in enumerate(stages_to_run):
            if stage.name == current_stage.name:
                current_idx = idx
                break

        if current_idx is None or current_idx >= len(stages_to_run) - 1:
            # No next stage, skip transition validation
            return

        next_stage = stages_to_run[current_idx + 1]

        # Validate each parquet artifact for the transition
        validated_any = False
        for artifact_path in result.artifacts:
            if not artifact_path.exists():
                continue

            if artifact_path.suffix != ".parquet":
                continue  # Only validate parquet files

            try:
                df = pd.read_parquet(artifact_path)
                warnings = validate_stage_transition(
                    output_df=df,
                    from_stage=current_stage.name,
                    to_stage=next_stage.name,
                    strict=True,
                )
                validated_any = True

                if warnings:
                    for warning in warnings:
                        self.logger.warning(f"  Transition warning: {warning}")
                else:
                    self.logger.debug(
                        f"  Transition validation passed: "
                        f"{current_stage.name} -> {next_stage.name} "
                        f"({artifact_path.name})"
                    )
            except StageValidationError:
                # Re-raise validation errors
                raise
            except Exception as e:
                self.logger.warning(
                    f"Could not validate transition for {artifact_path.name}: {e}"
                )

        if not validated_any:
            self.logger.debug(
                f"No parquet artifacts to validate for transition "
                f"{current_stage.name} -> {next_stage.name}"
            )

    def _save_lineage(self) -> None:
        """Save pipeline lineage metadata for dataset validation."""
        from datetime import datetime

        # Import from canonical location (src/core/)
        from src.core.lineage import PipelineLineage, create_dataset_checksum

        if self.config.project_root is None:
            self.logger.warning("project_root is None, cannot save lineage")
            return

        lineage_dir = self.config.project_root / "data" / "lineage"
        lineage_dir.mkdir(parents=True, exist_ok=True)

        dataset_checksums = {}

        for tf in getattr(self.config, "output_timeframes", [self.config.target_timeframe]):
            tf_suffix = f"_{tf}" if len(getattr(self.config, "output_timeframes", [])) > 1 else ""
            scaled_dir = self.config.splits_dir / "scaled" / (tf if tf_suffix else "")

            dataset_files = {
                f"train{tf_suffix}": scaled_dir / "train_scaled.parquet",
                f"val{tf_suffix}": scaled_dir / "val_scaled.parquet",
                f"test{tf_suffix}": scaled_dir / "test_scaled.parquet",
            }

            for name, path in dataset_files.items():
                if path.exists():
                    try:
                        dataset_checksums[name] = create_dataset_checksum(path, name)
                    except Exception as e:
                        self.logger.warning(f"Failed to create checksum for {name}: {e}")

        lineage = PipelineLineage(
            pipeline_run_id=self.config.run_id,
            target_timeframe=self.config.target_timeframe,
            output_timeframes=getattr(
                self.config, "output_timeframes", [self.config.target_timeframe]
            ),
            symbols=self.config.symbols,
            feature_generation=self.config.feature_generation,
            label_horizons=self.config.label_horizons,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
            random_seed=self.config.random_seed,
            dataset_checksums=dataset_checksums,
            created_at=datetime.now().isoformat(),
        )

        lineage_path = lineage_dir / f"{self.config.run_id}.json"
        lineage.save(lineage_path)
        self.logger.info(f"Pipeline lineage saved to {lineage_path}")
