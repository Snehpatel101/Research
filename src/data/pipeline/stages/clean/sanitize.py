"""
Raw-bar sanitizing: the one canonical cleaning step for OHLCV bars.

``MLFactory`` (training) and ``PreprocessingGraph`` (inference) both run their raw
bars through ``sanitize_bars`` before resampling and feature engineering, so training
and serving see identically cleaned data:

1. timestamps to naive UTC (tz-aware input is converted to UTC, then the zone dropped;
   naive input is taken to be UTC already), unparsable timestamps dropped;
2. chronological order (stable, so file order decides between equal timestamps);
3. non-numeric / NaN / inf prices dropped, non-positive prices dropped;
4. high/low made consistent with open/close (``high = max(o,h,l,c)``, ``low = min(o,h,l,c)``);
5. exact-duplicate timestamps collapsed (the last row wins);
6. volume: NaN / inf / negative treated as no trades (0).

Every correction is counted and logged; a frame that is already clean passes through
unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close"]
OHLCV_COLUMNS = [*PRICE_COLUMNS, "volume"]


@dataclass
class SanitizeReport:
    """What ``sanitize_bars`` changed."""

    rows_in: int = 0
    rows_out: int = 0
    bad_timestamps: int = 0
    non_finite_prices: int = 0
    non_positive_prices: int = 0
    duplicate_timestamps: int = 0
    high_low_fixed: int = 0
    bad_volume: int = 0
    reordered: bool = False
    tz_converted: bool = False

    @property
    def changed(self) -> bool:
        return bool(
            self.rows_in != self.rows_out
            or self.high_low_fixed
            or self.bad_volume
            or self.reordered
            or self.tz_converted
        )

    def summary(self) -> str:
        parts = [
            (self.bad_timestamps, "unparsable timestamps dropped"),
            (self.non_finite_prices, "rows with NaN/inf/non-numeric prices dropped"),
            (self.non_positive_prices, "rows with non-positive prices dropped"),
            (self.duplicate_timestamps, "duplicate timestamps collapsed (last kept)"),
            (self.high_low_fixed, "rows with inconsistent high/low fixed"),
            (self.bad_volume, "bad volumes set to 0"),
        ]
        items = [f"{n} {label}" for n, label in parts if n]
        if self.reordered:
            items.append("rows re-sorted chronologically")
        if self.tz_converted:
            items.append("timestamps converted to naive UTC")
        body = "; ".join(items) if items else "no changes"
        return f"Sanitized bars: {self.rows_in} -> {self.rows_out} rows ({body})"


def to_naive_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    """A timestamp (e.g. ``data.start_date``) as naive UTC, the zone sanitized bars use."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def sanitize_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, SanitizeReport]:
    """
    Clean raw OHLCV bars (see module docstring).

    Args:
        df: Frame indexed by bar timestamp with lower-case ``open, high, low, close,
            volume`` columns (extra columns are kept and follow their rows).

    Returns:
        ``(clean frame with a sorted, unique, naive-UTC DatetimeIndex named
        "datetime", SanitizeReport)``

    Raises:
        ValueError: If OHLCV columns are missing or no valid row remains.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    report = SanitizeReport(rows_in=len(df))
    out = df.copy()

    # 1. timestamps -> naive UTC
    was_aware = isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None
    index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True, errors="coerce"))
    report.tz_converted = was_aware
    index = index.tz_localize(None)
    bad_time = np.asarray(index.isna())
    if bad_time.any():
        report.bad_timestamps = int(bad_time.sum())
        out, index = out.loc[~bad_time], index[~bad_time]
    out.index = index

    # 2. chronological order (stable)
    if not out.index.is_monotonic_increasing:
        order = np.argsort(out.index.to_numpy(), kind="stable")
        out = out.iloc[order]
        report.reordered = True

    # 3. prices: numeric, finite, positive
    prices = out[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)
    finite = prices.notna().all(axis=1).to_numpy()
    positive = (prices > 0).all(axis=1).to_numpy()
    report.non_finite_prices = int((~finite).sum())
    report.non_positive_prices = int((finite & ~positive).sum())
    keep = finite & positive
    out = out.loc[keep].copy()
    prices = prices.loc[keep]

    # 4. high/low consistent with open/close
    high = prices.max(axis=1)
    low = prices.min(axis=1)
    inconsistent = ((prices["high"] != high) | (prices["low"] != low)).to_numpy()
    report.high_low_fixed = int(inconsistent.sum())
    for column in PRICE_COLUMNS:
        if out[column].dtype == object:
            out[column] = prices[column]
    if report.high_low_fixed:
        out["high"] = high
        out["low"] = low

    # 6. volume: no trades instead of NaN / inf / negative
    volume = pd.to_numeric(out["volume"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad_volume = (volume.isna() | (volume < 0)).to_numpy()
    report.bad_volume = int(bad_volume.sum())
    if report.bad_volume or out["volume"].dtype == object:
        out["volume"] = volume.where(~bad_volume, 0.0)

    # 5. exact-duplicate timestamps: last row wins
    duplicated = out.index.duplicated(keep="last")
    report.duplicate_timestamps = int(duplicated.sum())
    if report.duplicate_timestamps:
        out = out.loc[~duplicated]

    out.index.name = "datetime"
    report.rows_out = len(out)
    if out.empty:
        raise ValueError(f"No valid bars left after sanitizing ({report.summary()})")

    if report.changed:
        logger.warning(report.summary())
    else:
        logger.debug(report.summary())
    return out, report
