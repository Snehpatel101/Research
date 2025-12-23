"""
Unit tests for Pipeline Execution and Stage Dependencies.

Run with: pytest tests/phase_1_tests/pipeline/test_pipeline_execution.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from pipeline.runner import PipelineRunner
from pipeline.stage_registry import PipelineStage


class TestStageDependencies:
    """Tests for pipeline stage dependencies."""

    def test_data_generation_has_no_dependencies(self, sample_config):
        """Test data_generation has no dependencies."""
        runner = PipelineRunner(sample_config)

        stage = next(s for s in runner.stages if s.name == 'data_generation')
        assert stage.dependencies == []

    def test_data_cleaning_depends_on_generation(self, sample_config):
        """Test data_cleaning depends on data_generation."""
        runner = PipelineRunner(sample_config)

        stage = next(s for s in runner.stages if s.name == 'data_cleaning')
        assert 'data_generation' in stage.dependencies

    def test_feature_engineering_depends_on_cleaning(self, sample_config):
        """Test feature_engineering depends on data_cleaning."""
        runner = PipelineRunner(sample_config)

        stage = next(s for s in runner.stages if s.name == 'feature_engineering')
        assert 'data_cleaning' in stage.dependencies

    def test_initial_labeling_depends_on_features(self, sample_config):
        """Test initial_labeling depends on feature_engineering."""
        runner = PipelineRunner(sample_config)

        stage = next(s for s in runner.stages if s.name == 'initial_labeling')
        assert 'feature_engineering' in stage.dependencies

    def test_ga_optimize_depends_on_initial_labeling(self, sample_config):
        """Test ga_optimize depends on initial_labeling."""
        runner = PipelineRunner(sample_config)

        stage = next(s for s in runner.stages if s.name == 'ga_optimize')
        assert 'initial_labeling' in stage.dependencies

    def test_final_labels_depends_on_ga(self, sample_config):
        """Test final_labels depends on ga_optimize."""
        runner = PipelineRunner(sample_config)

        stage = next(s for s in runner.stages if s.name == 'final_labels')
        assert 'ga_optimize' in stage.dependencies

    def test_create_splits_depends_on_final_labels(self, sample_config):
        """Test create_splits depends on final_labels."""
        runner = PipelineRunner(sample_config)

        stage = next(s for s in runner.stages if s.name == 'create_splits')
        assert 'final_labels' in stage.dependencies

    def test_validate_depends_on_feature_scaling(self, sample_config):
        """Test validate depends on feature_scaling."""
        runner = PipelineRunner(sample_config)

        stage = next(s for s in runner.stages if s.name == 'validate')
        assert 'feature_scaling' in stage.dependencies

    def test_generate_report_depends_on_validate(self, sample_config):
        """Test generate_report depends on validate."""
        runner = PipelineRunner(sample_config)

        stage = next(s for s in runner.stages if s.name == 'generate_report')
        assert 'validate' in stage.dependencies

    def test_check_dependencies_unmet(self, sample_config):
        """Test _check_dependencies returns False for unmet dependencies."""
        runner = PipelineRunner(sample_config)

        # Create a stage with unmet dependencies
        test_stage = PipelineStage(
            name="test",
            function=lambda: None,
            dependencies=["nonexistent_stage"]
        )

        assert runner._check_dependencies(test_stage) is False

    def test_check_dependencies_met(self, sample_config):
        """Test _check_dependencies returns True when dependencies are met."""
        runner = PipelineRunner(sample_config)

        # Mark dependency as completed
        runner.completed_stages.add("prerequisite_stage")

        # Create a stage with that dependency
        test_stage = PipelineStage(
            name="test",
            function=lambda: None,
            dependencies=["prerequisite_stage"]
        )

        assert runner._check_dependencies(test_stage) is True

    def test_check_dependencies_no_dependencies(self, sample_config):
        """Test _check_dependencies returns True for stages with no dependencies."""
        runner = PipelineRunner(sample_config)

        test_stage = PipelineStage(
            name="test",
            function=lambda: None,
            dependencies=[]
        )

        assert runner._check_dependencies(test_stage) is True

    def test_check_dependencies_multiple(self, sample_config):
        """Test _check_dependencies with multiple dependencies."""
        runner = PipelineRunner(sample_config)

        # Mark some dependencies as completed
        runner.completed_stages.add("dep1")
        runner.completed_stages.add("dep2")

        # Stage with all dependencies met
        test_stage = PipelineStage(
            name="test",
            function=lambda: None,
            dependencies=["dep1", "dep2"]
        )
        assert runner._check_dependencies(test_stage) is True

        # Stage with some dependencies unmet
        test_stage2 = PipelineStage(
            name="test2",
            function=lambda: None,
            dependencies=["dep1", "dep3"]  # dep3 not completed
        )
        assert runner._check_dependencies(test_stage2) is False


# =============================================================================
# STATE PERSISTENCE TESTS
# =============================================================================


class TestGetStagesToRun:
    """Tests for _get_stages_to_run method."""

    def test_get_all_stages_when_no_from_stage(self, sample_config):
        """Test all stages returned when from_stage is None."""
        runner = PipelineRunner(sample_config)

        stages = runner._get_stages_to_run(None)

        assert len(stages) == len(runner.stages)

    def test_get_stages_from_specific_stage(self, sample_config):
        """Test getting stages from a specific starting point."""
        runner = PipelineRunner(sample_config)

        stages = runner._get_stages_to_run('feature_engineering')

        # Should include feature_engineering and all subsequent stages
        stage_names = [s.name for s in stages]
        assert 'feature_engineering' in stage_names
        assert 'initial_labeling' in stage_names
        assert 'ga_optimize' in stage_names

        # Should not include earlier stages
        assert 'data_generation' not in stage_names
        assert 'data_cleaning' not in stage_names

    def test_get_stages_from_last_stage(self, sample_config):
        """Test getting stages from the last stage."""
        runner = PipelineRunner(sample_config)

        stages = runner._get_stages_to_run('generate_report')

        assert len(stages) == 1
        assert stages[0].name == 'generate_report'

    def test_get_stages_invalid_stage_raises(self, sample_config):
        """Test ValueError raised for invalid stage name."""
        runner = PipelineRunner(sample_config)

        with pytest.raises(ValueError, match="Stage not found"):
            runner._get_stages_to_run('nonexistent_stage')


# =============================================================================
# SPLITS VALIDATION TESTS
# =============================================================================

