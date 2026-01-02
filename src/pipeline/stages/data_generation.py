"""
Stage 1: Data Generation and Ingestion with Validation.

Wrapper that delegates to src/stages/ingest/run.py
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Import orchestration logic from stage subdirectory
from src.phase1.stages.ingest.run import run_data_generation

__all__ = ["run_data_generation"]
