"""
Stage 7.7: Post-Scale Validation with Drift Checks.

Wrapper that imports from src.phase1.stages.scaled_validation.run
"""
from src.phase1.stages.scaled_validation.run import run_scaled_validation

__all__ = ['run_scaled_validation']
