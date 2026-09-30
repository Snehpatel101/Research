"""
Event sampling for triple-barrier labels (AFML ch. 2).

Instead of labeling every bar, only bars where a symmetric CUSUM filter fires
carry a label; the rest are marked invalid (-99), which every downstream
consumer (adapters, CV, feature selection, OOF, tuning) already drops. Features
are still computed on every bar, and label spans stay in bar coordinates.

The resolved spec (method + absolute threshold) is recorded with the
deployment bundle so serving reproduces the same event definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.features.cusum_filter import auto_cusum_threshold, cusum_events_mask, log_returns

EVENT_SAMPLING_NONE = "none"
EVENT_SAMPLING_CUSUM = "cusum"
EVENT_SAMPLING_METHODS = (EVENT_SAMPLING_NONE, EVENT_SAMPLING_CUSUM)


@dataclass(frozen=True)
class EventSamplingSpec:
    """A fully resolved event definition: method plus absolute threshold.

    Attributes:
        method: ``"cusum"``.
        threshold: CUSUM threshold in log-return units (already resolved from
            ``"auto"`` on the training rows).
    """

    method: str
    threshold: float

    def __post_init__(self) -> None:
        if self.method != EVENT_SAMPLING_CUSUM:
            raise ValueError(
                f"Unknown event sampling method {self.method!r}; "
                f"expected one of {EVENT_SAMPLING_METHODS}"
            )
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError(f"Event threshold must be positive and finite, got {self.threshold}")

    def mask(self, close: pd.Series) -> np.ndarray:
        """Boolean event mask over the bars of ``close``.

        The CUSUM runs forward over the log returns: the decision for bar ``t``
        depends on bars ``<= t`` only, so the mask of a prefix equals the prefix
        of the mask. It is path dependent (the sums reset at every event), so
        replaying it needs the same history start.
        """
        return cusum_events_mask(log_returns(close), self.threshold).to_numpy()

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "threshold": float(self.threshold)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventSamplingSpec:
        return cls(method=str(data["method"]), threshold=float(data["threshold"]))


def resolve_event_sampling(
    method: str,
    threshold: float | str,
    vol_multiple: float,
    train_close: pd.Series,
) -> EventSamplingSpec | None:
    """Resolve the configured event sampling into a frozen spec.

    Args:
        method: ``"none"`` (returns None) or ``"cusum"``.
        threshold: Absolute log-return threshold, or ``"auto"``.
        vol_multiple: For ``"auto"``: threshold = ``vol_multiple`` x the
            per-bar log-return volatility of ``train_close``.
        train_close: Close prices of the TRAINING rows only. Nothing after the
            training period may reach the auto threshold.
    """
    if method == EVENT_SAMPLING_NONE:
        return None
    if method != EVENT_SAMPLING_CUSUM:
        raise ValueError(
            f"Unknown event sampling method {method!r}; expected one of {EVENT_SAMPLING_METHODS}"
        )
    if isinstance(threshold, str):
        if threshold != "auto":
            raise ValueError(f"cusum_threshold must be a number or 'auto', got {threshold!r}")
        value = auto_cusum_threshold(log_returns(train_close), vol_multiple)
    else:
        value = float(threshold)
    return EventSamplingSpec(method=method, threshold=value)


def apply_event_mask(
    labels: np.ndarray, label_ends: np.ndarray, events: np.ndarray, invalid_label: int
) -> tuple[np.ndarray, np.ndarray]:
    """Invalidate the label (and label end) of every non-event bar.

    Returns copies; ``labels`` keeps its dtype (which must hold ``invalid_label``).
    """
    labels = np.where(events, labels, invalid_label).astype(labels.dtype)
    label_ends = np.where(events, label_ends, -1).astype(label_ends.dtype)
    return labels, label_ends
