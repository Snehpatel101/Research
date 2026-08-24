"""Control datasets with KNOWN ground truth — the falsifiability instrument.

Phase 0 established that this repo's failure mode is not crashes but confident
wrong answers: 593 tests passed while the end-to-end pipeline certified a
worse-than-random model. Counting passing tests cannot detect that. Controls
can.

Two datasets with ground truth known by construction:

  ``make_noise``  -- driftless random walk. Future returns are independent of
                     the past, so NO model may beat the majority baseline.
                     A model that "wins" here has found leakage, not signal.
  ``make_signal`` -- an AR(1) momentum relationship. EVERY competent model
                     must beat the majority baseline. A model that loses here
                     is broken.

Any metric that cannot separate these two is not measuring skill. This is why
Phase 0 ruled that headline claims use accuracy-vs-majority and MCC rather
than macro-F1, which gains +0.168 on pure noise (it beats a single-class
baseline simply by spreading its guesses).

Kept import-light (numpy/pandas only) so it can be used without the ML stack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_N = 6000
DEFAULT_SEED = 7


def _ohlcv_from_close(close: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Wrap a close series in a plausible OHLCV frame."""
    spread = np.abs(rng.normal(0, 0.05, len(close)))
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=len(close), freq="1min"),
            "open": close + rng.normal(0, 0.02, len(close)),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            # float, not int: integer volume is what crashes the lookahead
            # auditor under pandas 3 (Phase 0 F13).
            "volume": rng.integers(100, 1000, len(close)).astype("float64"),
        }
    )


def make_noise(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Driftless random walk — unpredictable by construction."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return _ohlcv_from_close(close, rng)


def make_signal(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """AR(1) momentum — genuinely learnable from past returns alone."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.6 * r[t - 1] + rng.normal(0, 0.05)
    close = 100 + np.cumsum(r)
    return _ohlcv_from_close(close, rng)


def featurize(df: pd.DataFrame, horizon: int = 5):
    """Minimal, strictly causal features plus a 3-class forward label.

    Deliberately hand-rolled rather than routed through ``FeatureEngineer``:
    a control instrument must not depend on the machinery it is used to
    audit.

    Returns:
        (X, y, n_rows) with X float32 (n, 5) and y in {-1, 0, 1}.
    """
    c = df["close"]
    X = pd.DataFrame(
        {
            "ret1": c.pct_change(1),
            "ret5": c.pct_change(5),
            "ret10": c.pct_change(10),
            "vol10": c.pct_change().rolling(10).std(),
            "mom20": c / c.rolling(20).mean() - 1.0,
        }
    )
    fwd = c.shift(-horizon) / c - 1.0
    thr = fwd.std() * 0.5
    y = pd.Series(0, index=df.index, dtype=int)
    y[fwd > thr] = 1
    y[fwd < -thr] = -1
    ok = X.notna().all(axis=1) & fwd.notna()
    return X[ok].to_numpy(np.float32), y[ok].to_numpy(), int(ok.sum())


def temporal_split(X: np.ndarray, y: np.ndarray, train_frac: float = 0.7):
    """Strict chronological split. Never shuffles — that is the whole point."""
    cut = int(len(X) * train_frac)
    return X[:cut], X[cut:], y[:cut], y[cut:]


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Predict the training majority class everywhere.

    This is the bar every model and every ensemble must clear. Phase 0 found
    no such baseline existed anywhere in ``src/`` (finding F6).
    """
    majority = np.bincount(y_train + 1).argmax() - 1
    return np.full_like(y_test, majority)
