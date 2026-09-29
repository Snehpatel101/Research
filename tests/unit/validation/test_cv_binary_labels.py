"""`ml cv` on binary labels (LabelingConfig.binary_mode).

The evaluation container carries the label class count; the per-fold feature
selection loop (the `ml cv` default), the plain OOF path, the tuner and the
stacking datasets all build 2-class models and 2-class probability columns
from it. Both paths emit the standard OOF schema, so their outputs stack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.container import TimeSeriesDataContainer
from src.core.label_spans import label_end_column
from src.validation.cv.cv_runner import CrossValidationRunner
from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

HORIZON = 5
N = 600
MODELS = ["logistic", "random_forest"]


def _container(n_classes: int) -> TimeSeriesDataContainer:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(N, 3))
    signal = features[:, 0] + rng.normal(scale=0.5, size=N)
    if n_classes == 2:
        y = (np.abs(signal) > 0.8).astype(int)  # {0: no move, 1: move}
    else:
        y = np.where(signal > 0.5, 1, np.where(signal < -0.5, -1, 0))
    df = pd.DataFrame(features, columns=["f1", "f2", "f3"])
    df["datetime"] = pd.date_range("2024-01-02 09:30", periods=N, freq="5min")
    df[f"label_h{HORIZON}"] = y
    df[f"sample_weight_h{HORIZON}"] = 1.0
    df[label_end_column(f"label_h{HORIZON}")] = np.minimum(np.arange(N) + 6, N - 1)
    return TimeSeriesDataContainer.from_dataframes(
        train_df=df, horizon=HORIZON, n_classes=n_classes
    )


def _runner(select_features: bool) -> CrossValidationRunner:
    cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=6, embargo_bars=6))
    return CrossValidationRunner(
        cv=cv,
        models=MODELS,
        horizons=[HORIZON],
        tune_hyperparams=False,
        select_features=select_features,
        n_features_to_select=2,
    )


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("select_features", [True, False])
def test_cv_and_stacking_follow_the_container_class_count(
    n_classes: int, select_features: bool
) -> None:
    container = _container(n_classes)
    runner = _runner(select_features)
    results = runner.run(container)
    assert set(results) == {(m, HORIZON) for m in MODELS}

    X, y, _ = container.get_sklearn_arrays("train", return_df=True)
    assert isinstance(X, pd.DataFrame)
    labels = set(np.unique(y))
    for model in MODELS:
        oof = results[(model, HORIZON)].oos_predictions
        prob_cols = [c for c in oof.columns if c.startswith(f"{model}_prob_")]
        assert len(prob_cols) == n_classes
        predicted = oof[f"{model}_pred"].dropna()
        assert len(predicted) > N // 2
        assert set(np.unique(predicted)) <= labels
        probs = oof.loc[predicted.index, prob_cols].to_numpy()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert (oof["datetime"] == X.index).all()

    stacking = runner.build_stacking_datasets(results, container)[HORIZON]
    assert stacking.model_names == MODELS
    assert len(stacking.data) > N // 2
    assert np.isfinite(stacking.data["prediction_entropy"]).all()


def test_container_rejects_unknown_class_count() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        _container(4)
