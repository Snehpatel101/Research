"""D11: Feature engineering determinism test.

Verifies that feature compute functions produce identical output when called
twice on the same input data. Any nondeterminism in feature computation would
break reproducibility guarantees.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features.compute.entropy import compute_entropy_shannon_10
from src.data.features.compute.mean_reversion import compute_mr_zscore_20
from src.data.features.compute.volatility import (
    compute_atr_14,
    compute_bb_width,
    compute_hvol_20,
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
    pytest.param(compute_atr_14, id="atr_14"),
    pytest.param(compute_bb_width, id="bb_width"),
    pytest.param(compute_hvol_20, id="hvol_20"),
    pytest.param(compute_mr_zscore_20, id="mr_zscore_20"),
    pytest.param(compute_entropy_shannon_10, id="entropy_shannon_10"),
]


@pytest.mark.parametrize("feature_fn", FEATURE_FUNCTIONS)
def test_feature_determinism(feature_fn):
    """Running a feature function twice on identical data must produce identical results."""
    df = _make_ohlcv()

    result_1 = feature_fn(df.copy())
    result_2 = feature_fn(df.copy())

    pd.testing.assert_series_equal(result_1, result_2, check_names=False)
