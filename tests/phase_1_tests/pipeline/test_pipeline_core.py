"""Tests for pipeline core classes."""
import json
from datetime import datetime
from pathlib import Path

import pytest

from src.pipeline.runner import PipelineRunner, PipelineStage, StageResult, StageStatus


class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_pending_value(self):
        """Test PENDING status has correct value."""
        assert StageStatus.PENDING.value == "pending"

    def test_in_progress_value(self):
        """Test IN_PROGRESS status has correct value."""
        assert StageStatus.IN_PROGRESS.value == "in_progress"

    def test_completed_value(self):
        """Test COMPLETED status has correct value."""
        assert StageStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        """Test FAILED status has correct value."""
        assert StageStatus.FAILED.value == "failed"

    def test_skipped_value(self):
        """Test SKIPPED status has correct value."""
        assert StageStatus.SKIPPED.value == "skipped"

    def test_all_statuses_exist(self):
        """Test all expected statuses are defined."""
        expected_statuses = ['PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'SKIPPED']
        for status_name in expected_statuses:
            assert hasattr(StageStatus, status_name)

    def test_status_comparison(self):
        """Test status enum comparisons work correctly."""
        assert StageStatus.COMPLETED == StageStatus.COMPLETED
        assert StageStatus.PENDING != StageStatus.COMPLETED

    def test_status_from_value(self):
        """Test creating status from string value."""
        assert StageStatus("completed") == StageStatus.COMPLETED
        assert StageStatus("pending") == StageStatus.PENDING


# =============================================================================
# STAGE RESULT TESTS
# =============================================================================


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_creation_minimal(self):
        """Test creating StageResult with minimal parameters."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.PENDING,
            start_time=datetime.now()
        )

        assert result.stage_name == "test_stage"
        assert result.status == StageStatus.PENDING
        assert result.end_time is None
        assert result.duration_seconds == 0.0
        assert result.artifacts == []
        assert result.error is None
        assert result.metadata == {}

    def test_creation_full(self):
        """Test creating StageResult with all parameters."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 5, 0)
        artifacts = [Path("/tmp/test.parquet"), Path("/tmp/test2.csv")]
        metadata = {'key': 'value', 'count': 100}

        result = StageResult(
            stage_name="full_stage",
            status=StageStatus.COMPLETED,
            start_time=start,
            end_time=end,
            duration_seconds=300.0,
            artifacts=artifacts,
            error=None,
            metadata=metadata
        )

        assert result.stage_name == "full_stage"
        assert result.status == StageStatus.COMPLETED
        assert result.start_time == start
        assert result.end_time == end
        assert result.duration_seconds == 300.0
        assert len(result.artifacts) == 2
        assert result.metadata == metadata

    def test_creation_with_error(self):
        """Test creating StageResult with error."""
        result = StageResult(
            stage_name="error_stage",
            status=StageStatus.FAILED,
            start_time=datetime.now(),
            error="Something went wrong"
        )

        assert result.status == StageStatus.FAILED
        assert result.error == "Something went wrong"

    def test_to_dict_minimal(self):
        """Test converting minimal StageResult to dictionary."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        result = StageResult(
            stage_name="test",
            status=StageStatus.PENDING,
            start_time=start
        )

        d = result.to_dict()

        assert d['stage_name'] == "test"
        assert d['status'] == "pending"
        assert d['start_time'] == start.isoformat()
        assert d['end_time'] is None
        assert d['duration_seconds'] == 0.0
        assert d['artifacts'] == []
        assert d['error'] is None
        assert d['metadata'] == {}

    def test_to_dict_full(self):
        """Test converting full StageResult to dictionary."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 5, 0)

        result = StageResult(
            stage_name="full",
            status=StageStatus.COMPLETED,
            start_time=start,
            end_time=end,
            duration_seconds=300.0,
            artifacts=[Path("/tmp/test.parquet")],
            error=None,
            metadata={'key': 'value'}
        )

        d = result.to_dict()

        assert d['stage_name'] == "full"
        assert d['status'] == "completed"
        assert d['start_time'] == start.isoformat()
        assert d['end_time'] == end.isoformat()
        assert d['duration_seconds'] == 300.0
        assert '/tmp/test.parquet' in d['artifacts'][0]
        assert d['metadata'] == {'key': 'value'}

    def test_to_dict_with_error(self):
        """Test converting StageResult with error to dictionary."""
        result = StageResult(
            stage_name="error",
            status=StageStatus.FAILED,
            start_time=datetime.now(),
            error="Test error message"
        )

        d = result.to_dict()

        assert d['status'] == "failed"
        assert d['error'] == "Test error message"

    def test_to_dict_serializable(self):
        """Test that to_dict() output is JSON serializable."""
        result = StageResult(
            stage_name="test",
            status=StageStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            artifacts=[Path("/tmp/test.parquet")],
            metadata={'nested': {'key': 'value'}}
        )

        d = result.to_dict()

        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Should be parseable back
        parsed = json.loads(json_str)
        assert parsed['stage_name'] == "test"


# =============================================================================
# PIPELINE STAGE TESTS
# =============================================================================


class TestPipelineStage:
    """Tests for PipelineStage dataclass."""

    def test_creation_minimal(self):
        """Test creating PipelineStage with minimal parameters."""
        stage = PipelineStage(
            name="test_stage",
            function=lambda: None
        )

        assert stage.name == "test_stage"
        assert callable(stage.function)
        assert stage.dependencies == []
        assert stage.description == ""
        assert stage.required is True
        assert stage.can_run_parallel is False

    def test_creation_full(self):
        """Test creating PipelineStage with all parameters."""
        def test_func():
            return "result"

        stage = PipelineStage(
            name="full_stage",
            function=test_func,
            dependencies=["dep1", "dep2"],
            description="A test stage",
            required=False,
            can_run_parallel=True
        )

        assert stage.name == "full_stage"
        assert stage.function() == "result"
        assert stage.dependencies == ["dep1", "dep2"]
        assert stage.description == "A test stage"
        assert stage.required is False
        assert stage.can_run_parallel is True

    def test_dependencies_list(self):
        """Test stage dependencies are stored correctly."""
        stage = PipelineStage(
            name="dependent_stage",
            function=lambda: None,
            dependencies=["stage_a", "stage_b", "stage_c"]
        )

        assert len(stage.dependencies) == 3
        assert "stage_a" in stage.dependencies
        assert "stage_b" in stage.dependencies
        assert "stage_c" in stage.dependencies


# =============================================================================
# PIPELINE CONFIG TESTS
# =============================================================================


class TestPipelineRunnerInit:
    """Tests for PipelineRunner initialization."""

    def test_runner_creation(self, sample_config):
        """Test basic PipelineRunner creation."""
        runner = PipelineRunner(sample_config)

        assert runner.config == sample_config
        assert runner.resume is False
        assert len(runner.stages) > 0
        assert len(runner.stage_results) == 0
        assert len(runner.completed_stages) == 0

    def test_runner_creation_with_resume(self, sample_config):
        """Test PipelineRunner creation with resume flag."""
        runner = PipelineRunner(sample_config, resume=True)

        assert runner.resume is True

    def test_runner_stages_defined(self, sample_config):
        """Test that all expected stages are defined."""
        runner = PipelineRunner(sample_config)

        expected_stages = [
            'data_generation',
            'data_cleaning',
            'feature_engineering',
            'initial_labeling',
            'ga_optimize',
            'final_labels',
            'create_splits',
            'feature_scaling',
            'build_datasets',
            'validate_scaled',
            'validate',
            'generate_report'
        ]

        stage_names = [s.name for s in runner.stages]

        for expected in expected_stages:
            assert expected in stage_names, f"Stage {expected} not found"

    def test_runner_stages_have_functions(self, sample_config):
        """Test that all stages have callable functions."""
        runner = PipelineRunner(sample_config)

        for stage in runner.stages:
            assert callable(stage.function), f"Stage {stage.name} function not callable"

    def test_runner_log_file_created(self, sample_config):
        """Test that log file is created."""
        runner = PipelineRunner(sample_config)

        assert runner.log_file.parent.exists()


# =============================================================================
# STAGE DEPENDENCY TESTS
# =============================================================================
