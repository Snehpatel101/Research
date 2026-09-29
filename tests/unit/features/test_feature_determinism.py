"""D11: Feature engineering determinism test.

Verifies that live feature engine functions (src/data/pipeline/stages/features/)
produce identical output when called twice on the same input data. Any
nondeterminism in feature computation would break reproducibility guarantees.

Covers five feature families:
- Entropy: add_shannon_entropy
- Volatility: add_atr, add_bollinger_bands, add_historical_volatility
- Trend/mean-reversion: add_adx (includes bb z-score via add_bollinger_bands)
"""

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline.stages.features.entropy import add_shannon_entropy
from src.data.pipeline.stages.features.trend import add_adx
from src.data.pipeline.stages.features.volatility import (
    add_atr,
    add_bollinger_bands,
    add_historical_volatility,
)


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create fixed OHLCV data for determinism testing."""
    rng = np.random.RandomState(seed)
    close = 5000.0 + rng.randn(n).cumsum() * 10
    high = close + rng.uniform(1, 10, n)
    low = close - rng.uniform(1, 10, n)
    open_ = close + rng.randn(n) * 3
    volume = rng.randint(100, 10000, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


FEATURE_FUNCTIONS = [
    pytest.param(add_atr, {"periods": [14]}, id="atr_14"),
    pytest.param(add_bollinger_bands, {"period": 20}, id="bollinger_bands_20"),
    pytest.param(add_historical_volatility, {"periods": [20], "timeframe": "5min"}, id="hvol_20"),
    pytest.param(add_adx, {"period": 14}, id="adx_14"),
    pytest.param(add_shannon_entropy, {"windows": [10]}, id="entropy_shannon_10"),
]


def _run_feature(feature_fn, kwargs) -> pd.DataFrame:
    """Run a live add_* feature function and return only the added columns."""
    df = _make_ohlcv()
    base_cols = set(df.columns)
    out = feature_fn(df.copy(), {}, **kwargs)
    added_cols = [c for c in out.columns if c not in base_cols]
    assert len(added_cols) > 0, f"{feature_fn.__name__} added no columns"
    return out[added_cols]


@pytest.mark.parametrize("feature_fn,kwargs", FEATURE_FUNCTIONS)
def test_feature_determinism(feature_fn, kwargs):
    """Running a feature function twice on identical data must produce identical results."""
    result_1 = _run_feature(feature_fn, kwargs)
    result_2 = _run_feature(feature_fn, kwargs)

    pd.testing.assert_frame_equal(result_1, result_2, check_exact=True)
