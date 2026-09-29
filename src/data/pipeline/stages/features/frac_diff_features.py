"""
Fractionally differentiated price features (AFML ch. 5).

Log prices are non-stationary; ordinary returns (d = 1) are stationary but
erase all memory of the price level. A fixed-width-window fractional
difference with the smallest sufficient order ``d`` keeps as much of that
memory as stationarity allows.

The window depends only on ``d``, the weight threshold and the window cap —
never on how many bars are supplied — so training and inference compute
identical values, and the value at bar ``t`` never depends on later bars.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Price-level columns FFD features can be built from
FRAC_DIFF_PRICE_COLUMNS = ("open", "high", "low", "close")
DEFAULT_FRAC_DIFF_COLUMNS = ("close", "open", "high", "low")


def frac_diff_feature_name(column: str) -> str:
    """Feature column holding the fractional difference of ``log(column)``."""
    return f"ffd_log_{column}"


def add_frac_diff_features(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    columns: list[str],
    d: float,
    max_window: int,
    threshold: float,
) -> pd.DataFrame:
    """Add ``ffd_log_<col>``: FFD(order ``d``) of the log price of each column.

    ANTI-LOOKAHEAD: shifted by one bar like every other feature, so the value
    at bar ``t`` uses bars up to ``t - 1`` (the first ``max_window`` rows are
    NaN warmup).
    """
    from src.data.features.frac_diff import frac_diff_ffd

    unknown = [c for c in columns if c not in FRAC_DIFF_PRICE_COLUMNS]
    if unknown:
        raise ValueError(
            f"frac_diff columns must be among {FRAC_DIFF_PRICE_COLUMNS}, got {unknown}"
        )
    logger.info(f"Adding fractional-differentiation features {columns} (d={d})")
    for column in columns:
        log_price = pd.Series(np.log(df[column].to_numpy(dtype=np.float64)), index=df.index)
        ffd = frac_diff_ffd(log_price, d=d, threshold=threshold, max_window=max_window)
        name = frac_diff_feature_name(column)
        df[name] = ffd.shift(1).to_numpy()
        feature_metadata[name] = (
            f"Fixed-window fractional difference (d={d:g}, window<={max_window}) "
            f"of log {column} (lagged)"
        )
    return df
