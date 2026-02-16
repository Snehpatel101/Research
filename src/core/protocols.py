"""
Core protocols for the ML Factory inference system.

Defines structural typing contracts that enable protocol-aware
component extraction without requiring inheritance changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.models.base import PredictionResult


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol for trainer instances that produce inference artifacts.

    Trainers satisfying this protocol expose all components needed
    for bundle creation without duck-typing fallback chains.
    """

    @property
    def model(self) -> Any: ...

    @property
    def scaler(self) -> Any | None: ...

    @property
    def feature_columns(self) -> list[str]: ...

    @property
    def calibrator(self) -> Any | None: ...

    @property
    def model_name(self) -> str: ...

    @property
    def model_key(self) -> str:
        """Return a unique key like 'xgboost_h20' for bundle identification."""
        ...


@runtime_checkable
class InferenceBundle(Protocol):
    """Protocol for inference bundles (single model or ensemble).

    Provides a unified interface for loading artifacts and making
    predictions from raw OHLCV data.
    """

    def predict(self, X: pd.DataFrame | np.ndarray, calibrate: bool = True) -> PredictionResult: ...

    def predict_from_raw(
        self, raw_df: pd.DataFrame, calibrate: bool = True
    ) -> PredictionResult: ...

    @classmethod
    def load(cls, path: str | ...) -> InferenceBundle: ...


__all__ = [
    "TrainerProtocol",
    "InferenceBundle",
]
