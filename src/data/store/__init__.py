"""
Raw multi-timeframe OHLCV store for 4D model training.

Storage and retrieval of raw OHLCV per timeframe, read by the multi-stream adapter's
store fallback (``MultiStreamAdapter.from_store``). MLFactory hands multi-stream models
their timeframes directly (``additional_dfs``) and does not use the store.

>>> from src.data.store import save_raw_mtf, load_raw_mtf
>>> save_raw_mtf("MES", "5min", "train", df)
>>> df = load_raw_mtf("MES", "5min", "train")
"""

from src.core.exceptions import (
    InvalidSplitError,
    InvalidTimeframeError,
    RawMTFStoreError,
    TimeframeNotFoundError,
)

from .raw_mtf_store import (
    TIMEFRAMES,
    VALID_SPLITS,
    get_mtf_path,
    load_all_timeframes,
    load_raw_mtf,
    save_raw_mtf,
)

__all__ = [
    "TIMEFRAMES",
    "VALID_SPLITS",
    "RawMTFStoreError",
    "TimeframeNotFoundError",
    "InvalidTimeframeError",
    "InvalidSplitError",
    "get_mtf_path",
    "save_raw_mtf",
    "load_raw_mtf",
    "load_all_timeframes",
]
