"""
Regression tests for 4D model integration (bundle + evaluation + stacking CV indexing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.inference.bundle import ModelBundle
from src.models.base import PredictionOutput
from src.models.ensemble.stacking import StackingEnsemble
from src.models.training_utils import evaluate_model


@dataclass
class _Dummy4DDataset:
    n_samples: int = 8
    n_timeframes: int = 3
    seq_len: int = 60
    n_features: int = 5

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, float]:
        X = np.zeros((self.n_timeframes, self.seq_len, self.n_features), dtype=np.float32)
        # Cycle labels through [-1, 0, 1]
        y = [-1, 0, 1][index % 3]
        w = 1.0
        return X, y, w


class _Dummy4DContainer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_multi_resolution_4d(  # type: ignore[no-untyped-def]
        self,
        split: str,
        *,
        seq_len: int,
        timeframes: list[str] | None = None,
        features_per_timeframe: list[str] | None = None,
        symbol_isolated: bool = True,
        include_base_features: bool = True,
        stride: int = 1,
    ) -> _Dummy4DDataset:
        self.calls.append(
            {
                "split": split,
                "seq_len": seq_len,
                "timeframes": timeframes,
                "features_per_timeframe": features_per_timeframe,
                "symbol_isolated": symbol_isolated,
                "include_base_features": include_base_features,
                "stride": stride,
            }
        )
        return _Dummy4DDataset(seq_len=seq_len)


class _Dummy4DModel:
    model_family = "transformer"
    requires_sequences = True
    requires_4d = True
    _is_fitted = True

    def __init__(self) -> None:
        self._config = {"sequence_length": 60}

    def _get_model_type(self) -> str:
        return "patchtst"

    def predict(self, X: np.ndarray) -> PredictionOutput:
        n = len(X)
        probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
        preds = np.zeros(n, dtype=np.int64)
        conf = np.full(n, 1.0 / 3.0, dtype=np.float32)
        return PredictionOutput(
            class_predictions=preds,
            class_probabilities=probs,
            confidence=conf,
            metadata={},
        )


def test_model_bundle_validates_and_predicts_with_4d_input() -> None:
    model = _Dummy4DModel()
    bundle = ModelBundle.from_training(
        model=model,
        scaler=None,
        feature_columns=["unused_a", "unused_b"],
        horizon=20,
    )

    X_ok = np.zeros((2, 3, 60, 5), dtype=np.float32)
    out = bundle.predict(X_ok, calibrate=False)
    assert out.class_probabilities.shape == (2, 3)

    with pytest.raises(ValueError):
        bundle.predict(pd.DataFrame({"unused_a": [1.0]}), calibrate=False)

    with pytest.raises(ValueError):
        bundle.predict(np.zeros((2, 2, 60, 5), dtype=np.float32), calibrate=False)


def test_training_utils_evaluate_model_routes_requires_4d() -> None:
    model = _Dummy4DModel()
    container = _Dummy4DContainer()

    metrics = evaluate_model(model=model, container=container, split="test")
    assert "accuracy" in metrics
    assert len(container.calls) == 1
    assert container.calls[0]["split"] == "test"


@pytest.mark.parametrize(
    "X_train",
    [
        np.zeros((20, 5, 2), dtype=np.float32),  # 3D sequences
        np.zeros((20, 3, 5, 2), dtype=np.float32),  # 4D sequences
    ],
)
def test_stacking_oof_generation_handles_prewindowed_inputs(X_train: np.ndarray) -> None:
    n_samples = X_train.shape[0]
    seq_len = X_train.shape[2] if X_train.ndim == 4 else X_train.shape[1]

    # Provide bar-level label_end_times longer than pre-windowed inputs
    n_bars = n_samples + (seq_len - 1)
    idx = pd.date_range("2026-02-01", periods=n_bars, freq="min")
    # Keep label_end_times non-overlapping for small synthetic data to avoid
    # PurgedKFold dropping below min_train_size after overlap purging.
    label_end_times = pd.Series(idx, index=idx)

    ensemble = StackingEnsemble()
    ensemble._n_folds = 2  # keep CV feasible for small arrays

    oof, fold_models, oof_per_model = ensemble._generate_oof_predictions(
        X_train=X_train,
        y_train=np.zeros(n_samples, dtype=np.int64),
        base_model_names=[],
        base_model_configs={},
        sample_weights=None,
        use_probabilities=True,
        label_end_times=label_end_times,
        purge_bars=0,
        embargo_bars=0,
        X_train_seq=None,
    )

    assert oof.shape == (n_samples, 0)
    assert fold_models == []
    assert oof_per_model == {}
