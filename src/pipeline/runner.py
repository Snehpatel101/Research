"""
Pipeline Runner - Main Orchestrator.

Manages stage execution, dependency tracking, and artifact management for
the Phase 1 data preparation pipeline.
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


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

from .stage_registry import PipelineStage, get_stage_definitions

# Import stage execution functions
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


class PipelineRunner:
    """Orchestrates the Phase 1 pipeline execution."""

    def __init__(self, config: 'PipelineConfig', resume: bool = False):
        """
        Initialize pipeline runner.

        Args:
            config: Pipeline configuration
            resume: Whether to resume from last successful stage
        """
        from src.common.manifest import ArtifactManifest

        self.config = config
        self.resume = resume
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
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
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

            self.stages.append(PipelineStage(
                name=stage_def["name"],
                function=func,
                dependencies=stage_def["dependencies"],
                description=stage_def["description"],
                required=stage_def["required"],
                stage_number=stage_def["stage_number"]
            ))

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
        self.logger.info(
            f"Configuration saved to {self.config.run_config_dir / 'config.json'}"
        )

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

            # Execute stage
            self.logger.info(f"\nExecuting stage: {stage.name}")
            result = stage.function()

            # Store result
            self.stage_results[stage.name] = result

            if result.status == StageStatus.COMPLETED:
                self.completed_stages.add(stage.name)
                self.logger.info(
                    f"[PASS] Stage completed: {stage.name} ({result.duration_seconds:.2f}s)"
                )
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
        for dep in stage.dependencies:
            if dep not in self.completed_stages:
                return False
        return True

    def _save_state(self) -> None:
        """Save current pipeline state."""
        state = {
            'run_id': self.config.run_id,
            'completed_stages': list(self.completed_stages),
            'stage_results': {
                name: result.to_dict()
                for name, result in self.stage_results.items()
            },
            'saved_at': datetime.now().isoformat()
        }

        state_path = self.config.run_artifacts_dir / "pipeline_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, cls=NumpyEncoder)

    def _load_state(self) -> None:
        """Load previous pipeline state for resuming."""
        from src.common.manifest import ArtifactManifest

        state_path = self.config.run_artifacts_dir / "pipeline_state.json"

        if not state_path.exists():
            self.logger.warning("No previous state found. Starting from beginning.")
            return

        with open(state_path) as f:
            state = json.load(f)

        self.completed_stages = set(state.get('completed_stages', []))
        self.logger.info(
            f"Loaded state with {len(self.completed_stages)} completed stages"
        )

        # Load manifest if exists
        try:
            self.manifest = ArtifactManifest.load(
                self.config.run_id,
                self.config.project_root
            )
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
