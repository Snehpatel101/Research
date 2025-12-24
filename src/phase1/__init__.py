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

__all__ = ['stages', 'config', 'utils']

__version__ = '1.0.0'


def __getattr__(name: str):
    """Lazy import subpackages to avoid circular dependencies."""
    if name == 'stages':
        from src.phase1 import stages as _stages
        return _stages
    elif name == 'config':
        from src.phase1 import config as _config
        return _config
    elif name == 'utils':
        from src.phase1 import utils as _utils
        return _utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
