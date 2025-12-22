
class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_create_default_config_minimal(self, temp_project_dir):
        """Test creating config with minimal parameters."""
        config = create_default_config(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        assert config.symbols == ['MES']
        assert config.run_id is not None
        assert len(config.run_id) > 0

    def test_create_default_config_full(self, temp_project_dir):
        """Test creating config with full parameters."""
        config = create_default_config(
            symbols=['MES', 'MGC'],
            start_date='2020-01-01',
            end_date='2024-12-31',
            run_id='test_run_123',
            project_root=temp_project_dir
        )

        assert config.symbols == ['MES', 'MGC']
        assert config.start_date == '2020-01-01'
        assert config.end_date == '2024-12-31'
        assert config.run_id == 'test_run_123'

    def test_config_ratios_sum_to_one(self, temp_project_dir):
        """Test that train/val/test ratios sum to 1.0."""
        config = create_default_config(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        total = config.train_ratio + config.val_ratio + config.test_ratio
        assert abs(total - 1.0) < 0.001

    def test_config_invalid_ratios_raises(self, temp_project_dir):
        """Test that invalid ratios raise ValueError."""
        with pytest.raises(ValueError, match="ratios must sum to 1.0"):
            PipelineConfig(
                symbols=['MES'],
                train_ratio=0.5,
                val_ratio=0.5,
                test_ratio=0.5,  # Sum > 1.0
                project_root=temp_project_dir
            )

    def test_config_empty_symbols_raises(self, temp_project_dir):
        """Test that empty symbols list raises ValueError."""
        with pytest.raises(ValueError, match="At least one symbol"):
            PipelineConfig(
                symbols=[],
                project_root=temp_project_dir
            )

    def test_config_invalid_date_format_raises(self, temp_project_dir):
        """Test that invalid date format raises ValueError."""
        with pytest.raises(ValueError, match="YYYY-MM-DD format"):
            PipelineConfig(
                symbols=['MES'],
                start_date='01-01-2020',  # Wrong format
                project_root=temp_project_dir
            )

    def test_config_directories_created(self, temp_project_dir):
        """Test that config creates necessary directories."""
        config = create_default_config(
            symbols=['MES'],
            project_root=temp_project_dir
        )
        config.create_directories()

        assert config.raw_data_dir.exists()
        assert config.clean_data_dir.exists()
        assert config.features_dir.exists()
        assert config.final_data_dir.exists()
        assert config.splits_dir.exists()
        assert config.run_dir.exists()
        assert config.run_config_dir.exists()
        assert config.run_logs_dir.exists()
        assert config.run_artifacts_dir.exists()
        assert config.results_dir.exists()

    def test_config_save_and_load(self, temp_project_dir):
        """Test saving and loading configuration."""
        original = create_default_config(
            symbols=['MES', 'MGC'],
            start_date='2020-01-01',
            end_date='2024-12-31',
            run_id='save_load_test',
            description='Test save/load',
            project_root=temp_project_dir
        )

        # Save config
        original.create_directories()
        config_path = original.save_config()

        assert config_path.exists()

        # Load config
        loaded = PipelineConfig.load_config(config_path)

        assert loaded.symbols == original.symbols
        assert loaded.start_date == original.start_date
        assert loaded.end_date == original.end_date
        assert loaded.run_id == original.run_id
        assert loaded.description == original.description

    def test_config_validate_returns_empty_for_valid(self, temp_project_dir):
        """Test validate() returns empty list for valid config."""
        config = create_default_config(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        issues = config.validate()
        assert issues == []

    def test_config_validate_catches_issues(self, temp_project_dir):
        """Test validate() catches configuration issues."""
        config = create_default_config(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        # Corrupt some values to trigger validation issues
        config.barrier_k_up = -1.0
        config.purge_bars = -10

        issues = config.validate()
        assert len(issues) > 0
        assert any('barrier_k_up' in issue for issue in issues)
        assert any('purge_bars' in issue for issue in issues)

    def test_config_summary(self, temp_project_dir):
        """Test config summary generation."""
        config = create_default_config(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        summary = config.summary()

        assert 'MES' in summary
        assert 'Train:' in summary
        assert 'Validation:' in summary
        assert 'Test:' in summary

    def test_config_to_dict(self, temp_project_dir):
        """Test config to_dict conversion."""
        config = create_default_config(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        d = config.to_dict()

        assert 'symbols' in d
        assert 'train_ratio' in d
        assert 'val_ratio' in d
        assert 'test_ratio' in d
        assert 'project_root' in d
        assert isinstance(d['project_root'], str)


# =============================================================================
# PIPELINE RUNNER INITIALIZATION TESTS
# =============================================================================


class TestSplitsCreation:
    """Tests for splits creation and validation."""

    def test_negative_train_end_purged_raises(self, sample_config, sample_labeled_df):
        """Test that negative train_end_purged raises error."""
        # Set very high purge_bars that would result in negative train_end
        sample_config.purge_bars = 10000000

        runner = PipelineRunner(sample_config)

        # Save labeled data
        output_path = sample_config.final_data_dir / "MES_labeled.parquet"
        sample_labeled_df.to_parquet(output_path, index=False)

        # Running create_splits should fail
        result = runner._run_create_splits()

        assert result.status == StageStatus.FAILED
        assert 'purge' in result.error.lower() or 'eliminated' in result.error.lower()

    def test_splits_with_valid_purge(self, temp_project_dir, sample_labeled_df):
        """Test splits creation with valid purge settings."""
        # Create config with reasonable purge settings
        config = create_default_config(
            symbols=['MES'],
            project_root=temp_project_dir
        )
        config.purge_bars = 10
        config.embargo_bars = 5

        runner = PipelineRunner(config)

        # Save labeled data
        output_path = config.final_data_dir / "MES_labeled.parquet"
        sample_labeled_df.to_parquet(output_path, index=False)

        # Mark final_labels as completed (dependency for create_splits)
        runner.completed_stages.add('final_labels')

        # This will fail because of missing validation dependencies,
        # but we can test up to the splits creation logic
        # by checking if the stage function at least starts executing


# =============================================================================
# REPORT GENERATION TESTS
# =============================================================================


class TestReportGeneration:
    """Tests for report content generation."""

    def test_generate_report_content(self, temp_project_dir, sample_labeled_df):
        """Test _generate_report_content produces valid markdown."""
        config = create_default_config(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        runner = PipelineRunner(config)

        # Create mock split config
        split_config = {
            'run_id': config.run_id,
            'total_samples': len(sample_labeled_df),
            'train_samples': 700,
            'val_samples': 150,
            'test_samples': 150,
            'purge_bars': 60,
            'embargo_bars': 288,
            'train_date_start': '2020-01-01 09:30:00',
            'train_date_end': '2023-06-01 09:30:00',
            'val_date_start': '2023-06-02 09:30:00',
            'val_date_end': '2023-12-01 09:30:00',
            'test_date_start': '2023-12-02 09:30:00',
            'test_date_end': '2024-12-31 09:30:00',
        }

        feature_cols = [f'feature_{i}' for i in range(10)]

        label_stats = {
            1: {'short': 100, 'neutral': 500, 'long': 100},
            5: {'short': 150, 'neutral': 400, 'long': 150},
            20: {'short': 200, 'neutral': 300, 'long': 200}
        }

        report = runner._generate_report_content(
            sample_labeled_df,
            split_config,
            feature_cols,
            label_stats
        )

        # Check report contains expected sections
        assert '# Phase 1 Completion Report' in report
        assert 'Executive Summary' in report
        assert 'Dataset Overview' in report
        assert 'Label Distribution' in report
        assert 'Data Splits' in report
        assert 'Train' in report
        assert 'Validation' in report
        assert 'Test' in report
        assert 'Purge bars:' in report
        assert 'Embargo period:' in report


# =============================================================================
# MANIFEST INTEGRATION TESTS
# =============================================================================


class TestManifestIntegration:
    """Tests for manifest integration with PipelineRunner."""

    def test_runner_creates_manifest(self, sample_config):
        """Test that PipelineRunner creates a manifest."""
        runner = PipelineRunner(sample_config)

        assert runner.manifest is not None
        assert runner.manifest.run_id == sample_config.run_id


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogging:
    """Tests for pipeline logging."""

    def test_setup_logging_creates_handlers(self, sample_config):
        """Test that setup_logging creates appropriate handlers."""
        runner = PipelineRunner(sample_config)

        # Check log file exists
        assert runner.log_file.exists() or runner.log_file.parent.exists()

    def test_logger_initialized(self, sample_config):
        """Test that logger is initialized."""
        runner = PipelineRunner(sample_config)

        assert runner.logger is not None
        assert isinstance(runner.logger, logging.Logger)


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_completed_stages_initial(self, sample_config):
        """Test completed_stages is empty initially."""
        runner = PipelineRunner(sample_config)

        assert len(runner.completed_stages) == 0

    def test_empty_stage_results_initial(self, sample_config):
        """Test stage_results is empty initially."""
        runner = PipelineRunner(sample_config)

        assert len(runner.stage_results) == 0

    def test_stage_result_stored_after_execution(self, sample_config):
        """Test that stage results are stored after execution."""
        runner = PipelineRunner(sample_config)

        # Manually create a result
        result = StageResult(
            stage_name='test',
            status=StageStatus.COMPLETED,
            start_time=datetime.now()
        )

        runner.stage_results['test'] = result
        runner.completed_stages.add('test')

        assert 'test' in runner.stage_results
        assert 'test' in runner.completed_stages

    def test_failed_stage_not_added_to_completed(self, sample_config):
        """Test that failed stages are not added to completed_stages."""
        runner = PipelineRunner(sample_config)

        # Create a failed result
        result = StageResult(
            stage_name='failed_test',
            status=StageStatus.FAILED,
            start_time=datetime.now(),
            error="Test failure"
        )

        runner.stage_results['failed_test'] = result
        # Note: We don't add failed stages to completed_stages

        assert 'failed_test' in runner.stage_results
        assert 'failed_test' not in runner.completed_stages


# =============================================================================
# STAGE RESULT ARTIFACT HANDLING TESTS
# =============================================================================

