"""
CLI Run Commands - run and rerun pipeline commands.

Provides fully configurable CLI for the Phase 1 data pipeline and
Phase 2+ model training with support for:
- MTF (Multi-Timeframe) settings
- Feature toggles (wavelets, microstructure, etc.)
- Labeling parameters (barriers, horizons)
- Split ratios (train/val/test)
- Scaling options
- Model type selection (for Phase 2+)
- Ensemble configuration

This module serves as the main entry point, re-exporting commands
from specialized submodules for better organization.
"""

# Re-export all commands from submodules
from .run_commands_core import _create_config_from_args
from .run_commands_info import models_command
from .run_commands_pipeline import rerun_command, run_command

__all__ = [
    'run_command',
    'rerun_command',
    'models_command',
    '_create_config_from_args',
]
