"""
Unified regime detection entry point.

CANONICAL PATH: All regime detection should go through ``get_regime_labels()``
defined in this module. It delegates to ``CompositeRegimeDetector`` which
combines volatility, trend, and market-structure detectors.

Other regime modules in the codebase:
    - ``src/data/pipeline/stages/regime/composite.py``  (implementation)
    - ``src/models/training/regime_detector.py``         (training-time detector)
    - ``src/inference/regime_detector.py``               (inference-time detector)
    - ``src/models/training/modes/regime_aware.py``      (mode-level wrapper)
    - ``src/data/pipeline/stages/regime/hmm.py``         (HMM-based, experimental)

Usage::

    from src.data.pipeline.stages.regime.unified import get_regime_labels
    regime_series = get_regime_labels(df)                # defaults
    regime_series = get_regime_labels(df, config=cfg)    # custom config
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .composite import CompositeRegimeDetector


def get_regime_labels(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.Series:
    """Return a single composite regime label per row.

    Delegates to ``CompositeRegimeDetector.create_composite_label`` which
    combines volatility, trend, and structure regimes into one string
    (e.g. ``"high_uptrend_trending"``).

    All regime columns are internally shifted by 1 bar (anti-lookahead).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns (at minimum ``close``; ``high`` and
        ``low`` required for ADX-based trend detection).
    config : dict, optional
        Regime configuration dict matching ``CompositeRegimeDetector.from_config``
        schema. When *None*, default parameters are used.

    Returns
    -------
    pd.Series
        Composite regime label per row (NaN for first bar due to shift).
    """
    if config is not None:
        detector = CompositeRegimeDetector.from_config(config)
    else:
        detector = CompositeRegimeDetector.with_defaults()

    return detector.create_composite_label(df)


__all__ = ["get_regime_labels"]
