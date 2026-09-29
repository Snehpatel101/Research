"""
Pipeline Package - feature engineering building blocks for MLFactory.

MLFactory (``src/factory.py``) is the one pipeline: raw OHLCV bars ->
``FeatureEngineer`` -> triple-barrier labels -> models. This package holds the
pieces it uses:

- ``stages/features``: FeatureEngineer and the feature functions
- ``stages/mtf``: multi-timeframe features (shift(1) anti-lookahead)
- ``stages/clean/utils``: OHLCV resampling
- ``stages/sessions``: CME trading calendar
- ``stages/regime``: regime detection
- ``config``: barrier parameters, feature sets, runtime defaults
- ``feature_manifest``: feature lineage manifest
"""

from .feature_manifest import FeatureManifest

__all__ = ["FeatureManifest"]

__version__ = "1.0.0"
