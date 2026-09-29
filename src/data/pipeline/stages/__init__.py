"""
Data pipeline stages used by MLFactory.

- features: FeatureEngineer (technical indicators, entropy, wavelets, MTF)
- mtf: multi-timeframe feature generation
- clean: OHLCV resampling utilities
- sessions: CME trading calendar
- regime: market regime detection
"""

from .features import FeatureEngineer

__all__ = ["FeatureEngineer"]

__version__ = "1.0.0"
