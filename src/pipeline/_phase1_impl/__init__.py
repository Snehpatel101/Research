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
    from src.pipeline._phase1_impl.stages import DataIngestor
    from src.pipeline._phase1_impl.config import BARRIER_PARAMS
    from src.pipeline._phase1_impl.utils import select_features
"""

__all__ = ["stages", "config", "utils"]

__version__ = "1.0.0"


def __getattr__(name: str):
    """Lazy import subpackages to avoid circular dependencies."""
    import importlib

    if name == "stages":
        return importlib.import_module("src.pipeline._phase1_impl.stages")
    elif name == "config":
        return importlib.import_module("src.pipeline._phase1_impl.config")
    elif name == "utils":
        return importlib.import_module("src.pipeline._phase1_impl.utils")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
