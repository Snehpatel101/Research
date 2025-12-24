"""
Unit tests for Pipeline State Persistence.

Run with: pytest tests/phase_1_tests/pipeline/test_pipeline_state.py -v
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.pipeline.runner import PipelineRunner
from src.pipeline.utils import StageResult, StageStatus


class TestStatePersistence:
    """Tests for state save/load functionality."""

    def test_save_state_creates_file(self, sample_config):
        """Test _save_state creates state file."""
        runner = PipelineRunner(sample_config)
        runner._save_state()

        state_path = sample_config.run_artifacts_dir / "pipeline_state.json"
        assert state_path.exists()

    def test_save_state_content(self, sample_config):
        """Test _save_state saves correct content."""
        runner = PipelineRunner(sample_config)
        runner.completed_stages.add('test_stage')
        runner.stage_results['test_stage'] = StageResult(
            stage_name='test_stage',
            status=StageStatus.COMPLETED,
            start_time=datetime.now()
        )

        runner._save_state()

        state_path = sample_config.run_artifacts_dir / "pipeline_state.json"
        with open(state_path) as f:
            state = json.load(f)

        assert 'run_id' in state
        assert 'completed_stages' in state
        assert 'stage_results' in state
        assert 'saved_at' in state
        assert 'test_stage' in state['completed_stages']
        assert 'test_stage' in state['stage_results']

    def test_save_state_multiple_stages(self, sample_config):
        """Test _save_state with multiple stages."""
        runner = PipelineRunner(sample_config)

        for stage_name in ['stage1', 'stage2', 'stage3']:
            runner.completed_stages.add(stage_name)
            runner.stage_results[stage_name] = StageResult(
                stage_name=stage_name,
                status=StageStatus.COMPLETED,
                start_time=datetime.now()
            )

        runner._save_state()

        state_path = sample_config.run_artifacts_dir / "pipeline_state.json"
        with open(state_path) as f:
            state = json.load(f)

        assert len(state['completed_stages']) == 3
        assert len(state['stage_results']) == 3

    def test_load_state_restores_completed_stages(self, sample_config):
        """Test _load_state restores completed stages."""
        # Create initial runner and save state
        runner1 = PipelineRunner(sample_config)
        runner1.completed_stages.add('stage1')
        runner1.completed_stages.add('stage2')
        runner1._save_state()

        # Create new runner with resume=True
        runner2 = PipelineRunner(sample_config, resume=True)

        assert 'stage1' in runner2.completed_stages
        assert 'stage2' in runner2.completed_stages

    def test_load_state_no_state_file(self, sample_config):
        """Test _load_state handles missing state file gracefully."""
        # Delete state file if it exists
        state_path = sample_config.run_artifacts_dir / "pipeline_state.json"
        if state_path.exists():
            state_path.unlink()

        # Should not raise, just log warning
        runner = PipelineRunner(sample_config, resume=True)

        assert len(runner.completed_stages) == 0


# =============================================================================
# GET STAGES TO RUN TESTS
# =============================================================================


class TestArtifactHandling:
    """Tests for artifact handling in stage results."""

    def test_artifacts_as_paths(self):
        """Test that artifacts are stored as Path objects."""
        result = StageResult(
            stage_name="test",
            status=StageStatus.COMPLETED,
            start_time=datetime.now(),
            artifacts=[Path("/tmp/test1.parquet"), Path("/tmp/test2.csv")]
        )

        assert all(isinstance(a, Path) for a in result.artifacts)
        assert len(result.artifacts) == 2

    def test_artifacts_to_dict_conversion(self):
        """Test that artifacts are converted to strings in to_dict."""
        result = StageResult(
            stage_name="test",
            status=StageStatus.COMPLETED,
            start_time=datetime.now(),
            artifacts=[Path("/tmp/test.parquet")]
        )

        d = result.to_dict()

        # Artifacts should be strings in dict
        assert all(isinstance(a, str) for a in d['artifacts'])

    def test_empty_artifacts_list(self):
        """Test handling of empty artifacts list."""
        result = StageResult(
            stage_name="test",
            status=StageStatus.COMPLETED,
            start_time=datetime.now(),
            artifacts=[]
        )

        assert result.artifacts == []

        d = result.to_dict()
        assert d['artifacts'] == []


# =============================================================================
# METADATA HANDLING TESTS
# =============================================================================


class TestMetadataHandling:
    """Tests for metadata handling in stage results."""

    def test_metadata_dict(self):
        """Test metadata as dictionary."""
        metadata = {
            'key1': 'value1',
            'key2': 100,
            'nested': {'inner': 'value'}
        }

        result = StageResult(
            stage_name="test",
            status=StageStatus.COMPLETED,
            start_time=datetime.now(),
            metadata=metadata
        )

        assert result.metadata == metadata

    def test_metadata_to_dict_preserved(self):
        """Test metadata preserved in to_dict."""
        metadata = {'key': 'value', 'count': 42}

        result = StageResult(
            stage_name="test",
            status=StageStatus.COMPLETED,
            start_time=datetime.now(),
            metadata=metadata
        )

        d = result.to_dict()
        assert d['metadata'] == metadata

    def test_empty_metadata(self):
        """Test empty metadata handling."""
        result = StageResult(
            stage_name="test",
            status=StageStatus.COMPLETED,
            start_time=datetime.now()
        )

        assert result.metadata == {}

        d = result.to_dict()
        assert d['metadata'] == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

