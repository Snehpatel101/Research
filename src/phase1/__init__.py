"""Phase 1: Data Preparation Pipeline.

This package contains the complete Phase 1 data preparation pipeline including:
- Data ingestion and cleaning
- Feature engineering
- Triple-barrier labeling with GA optimization
- Train/val/test splitting with purging and embargo
- Feature scaling
- Data validation

Subpackages are imported lazily to avoid circular dependencies.
Use explicit imports like:
    from src.phase1.stages import DataIngestor
    from src.phase1.config import BARRIER_PARAMS
    from src.phase1.utils import select_features
"""

__all__ = ["stages", "config", "utils"]

__version__ = "1.0.0"


def __getattr__(name: str):
    """Lazy import subpackages to avoid circular dependencies."""
    import importlib

    if name == "stages":
        return importlib.import_module("src.phase1.stages")
    elif name == "config":
        return importlib.import_module("src.phase1.config")
    elif name == "utils":
        return importlib.import_module("src.phase1.utils")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
