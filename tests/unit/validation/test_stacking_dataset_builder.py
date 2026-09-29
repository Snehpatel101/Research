"""
StackingDatasetBuilder: the first model's frame is authoritative for fold_id.

Later models contribute ONLY their own prefixed columns (predictions/probabilities);
copying their ``fold_id`` (or ``y_true``) over the base frame would corrupt the CV
structure the meta-learner is trained on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.cv.oof_core import OOFPrediction
from src.validation.cv.oof_stacking import StackingDatasetBuilder

N = 50


def _oof(name: str, fold_ids: np.ndarray, rng: np.random.Generator) -> OOFPrediction:
    frame = pd.DataFrame(
        {
            f"{name}_pred": rng.choice([-1, 0, 1], size=N).astype(float),
            f"{name}_prob_short": rng.random(N),
            f"{name}_prob_neutral": rng.random(N),
            f"{name}_prob_long": rng.random(N),
            "fold_id": fold_ids,
            "y_true": rng.choice([-1, 0, 1], size=N),
            "datetime": pd.date_range("2024-01-01", periods=N, freq="h"),
        }
    )
    return OOFPrediction(model_name=name, predictions=frame, fold_info=[], coverage=1.0)


def test_fold_ids_come_from_the_first_model_and_labels_from_the_caller() -> None:
    rng = np.random.default_rng(0)
    oof_a = _oof("model_a", np.array([0] * 25 + [1] * 25), rng)
    oof_b = _oof("model_b", np.array([2] * 25 + [3] * 25), rng)  # deliberately different

    y_true = pd.Series(rng.choice([-1, 0, 1], size=N))
    dataset = StackingDatasetBuilder().build_stacking_dataset(
        oof_predictions={"model_a": oof_a, "model_b": oof_b},
        y_true=y_true,
        horizon=20,
        add_derived_features=False,
        drop_nan_samples=False,
    )

    np.testing.assert_array_equal(
        dataset.data["fold_id"].to_numpy(), oof_a.predictions["fold_id"].to_numpy()
    )
    np.testing.assert_array_equal(dataset.data["y_true"].to_numpy(), y_true.to_numpy())
    for col in ("model_b_prob_short", "model_b_prob_neutral", "model_b_prob_long"):
        np.testing.assert_allclose(
            dataset.data[col].to_numpy(), oof_b.predictions[col].to_numpy(), err_msg=col
        )
