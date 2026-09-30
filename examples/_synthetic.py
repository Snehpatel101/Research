"""Shared helpers for the examples: synthetic OHLCV bars and quiet logging.

The bars are a seeded random walk (same recipe as ``scripts/mix_match.py``),
so every example runs offline in a few minutes on a CPU. A random walk has no
edge to find: expect metrics near chance. The point is the workflow.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Examples write here (gitignored); each run gets its own <run_id> subdirectory.
OUTPUT_ROOT = REPO_ROOT / "experiments" / "examples"


def use_this_checkout() -> None:
    """Import ``src`` from the checkout holding this file, not another install."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def quiet_logging() -> None:
    """Warnings and above only; single-threaded BLAS for predictable CPU runs."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    # Expected on a CPU-only machine: "CUDA not available, falling back to CPU"
    logging.getLogger("src.models.boosting").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
    warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
    # Sequence models have no OOF prediction for their first seq_len - 1 bars;
    # the ensemble drops those rows (logged as "OOF predictions ... contain NaN")
    logging.getLogger("src.models.training.services.ensemble_service").setLevel(logging.ERROR)
    # One-shot test-split banner and short-tail label notices: expected in a demo
    logging.getLogger("src.models.training.evaluation").setLevel(logging.ERROR)
    logging.getLogger("src.data.adapters.preparation").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def make_synthetic_ohlcv(
    path: Path, n_rows: int = 4000, seed: int = 7, start: str = "2024-01-02 09:30"
) -> Path:
    """Write ``n_rows`` synthetic 5-minute OHLCV bars to ``path`` (parquet) and return it."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_rows, freq="5min")
    close = 5000.0 + np.cumsum(rng.normal(0, 2.0, n_rows))
    open_ = np.roll(close, 1) + rng.normal(0, 0.5, n_rows)
    open_[0] = close[0]
    wick = np.abs(rng.normal(0, 0.5, n_rows)) + 0.25
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": rng.integers(100, 5000, n_rows).astype(float),
        },
        index=pd.DatetimeIndex(idx, name="datetime"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path)
    return path
