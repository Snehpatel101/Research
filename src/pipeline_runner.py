"""
Pipeline Runner - Backward Compatibility Wrapper.

This module has been refactored into src/pipeline/
This file remains for backward compatibility with existing code.

The implementation has been split into:
- src/pipeline/runner.py       - Main PipelineRunner class
- src/pipeline/utils.py        - StageStatus, StageResult classes
- src/pipeline/stage_registry.py - Stage definitions
- src/pipeline/stages/         - Individual stage modules

For new code, import directly from the pipeline package:
    from pipeline import PipelineRunner, StageStatus, StageResult
"""
# Re-export from new location for backward compatibility
from pipeline import PipelineRunner, StageStatus, StageResult, PipelineStage

__all__ = ['PipelineRunner', 'StageStatus', 'StageResult', 'PipelineStage']


if __name__ == "__main__":
    # Example usage - preserved from original
    from pipeline_config import create_default_config

    config = create_default_config(
        symbols=['MES', 'MGC'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        description='Test pipeline run'
    )

    runner = PipelineRunner(config)
    success = runner.run()

    if success:
        print("\n[PASS] Pipeline completed successfully!")
    else:
        print("\n[FAIL] Pipeline failed. Check logs for details.")
