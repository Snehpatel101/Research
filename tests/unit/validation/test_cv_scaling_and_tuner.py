"""`ml cv` scales per fold; the tuner leaves the shared CV config alone.

The evaluation container holds UNSCALED features (each fold scales with statistics
from its own fit rows). Two paths used to feed them raw to the model: the per-fold
feature-selection loop (the `ml cv` default) and the tuner objective. Logistic regression
on a feature 1e7 times larger is the canary: with per-fold scaling the out-of-fold
predictions do not depend on the feature's units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.container import TimeSeriesDataContainer
from src.core.label_spans import label_end_column
from src.validation.cv.cv_runner import CrossValidationRunner
from src.validation.cv.cv_tuner import TimeSeriesOptunaTuner
from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

HORIZON = 5
N = 700


def _container(unit_scale: float) -> TimeSeriesDataContainer:
    rng = np.random.default_rng(11)
    f1 = rng.normal(size=N)
    f2 = rng.normal(size=N)
    f3 = rng.normal(size=N)
    signal = f1 + 0.5 * f2 + rng.normal(scale=0.7, size=N)
    y = np.where(signal > 0.5, 1, np.where(signal < -0.5, -1, 0))
    df = pd.DataFrame(
        {
            "f1": f1 * unit_scale,  # the informative feature, in different units
            "f2": f2,
            "f3": f3,
            f"label_h{HORIZON}": y,
            f"sample_weight_h{HORIZON}": 1.0,
            label_end_column(f"label_h{HORIZON}"): np.minimum(np.arange(N) + 6, N - 1),
        }
    )
    return TimeSeriesDataContainer.from_dataframes(train_df=df, horizon=HORIZON)


def _oof(unit_scale: float, *, select_features: bool) -> np.ndarray:
    cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=6, embargo_bars=6))
    runner = CrossValidationRunner(
        cv=cv,
        models=["logistic"],
        horizons=[HORIZON],
        tune_hyperparams=False,
        select_features=select_features,
        n_features_to_select=3,
    )
    result = runner.run(_container(unit_scale))[("logistic", HORIZON)]
    return result.oos_predictions["logistic_pred"].to_numpy(dtype=float)


@pytest.mark.parametrize("select_features", [True, False])
def test_cv_predictions_do_not_depend_on_feature_units(select_features: bool) -> None:
    base = _oof(1.0, select_features=select_features)
    huge = _oof(1e7, select_features=select_features)
    valid = ~np.isnan(base) & ~np.isnan(huge)
    assert valid.sum() > N // 2
    agreement = float(np.mean(base[valid] == huge[valid]))
    assert agreement > 0.97, f"predictions changed with feature units (agreement {agreement:.3f})"


def test_runner_uses_the_callers_n_splits() -> None:
    cv = PurgedKFold(PurgedKFoldConfig(n_splits=2, purge_bars=6, embargo_bars=6))
    runner = CrossValidationRunner(
        cv=cv, models=["logistic"], horizons=[HORIZON], tune_hyperparams=False, select_features=True
    )
    result = runner.run(_container(1.0))[("logistic", HORIZON)]
    assert result.n_folds == 2  # the caller's --n-splits, not a per-family default


class TestTunerLeavesCvAlone:
    def _tune(self, cv: PurgedKFold, unit_scale: float = 1.0) -> dict:
        c = _container(unit_scale)
        X, y, w = c.get_sklearn_arrays("train", return_df=True)
        tuner = TimeSeriesOptunaTuner(
            model_name="logistic",
            cv=cv,
            n_trials=2,
            max_samples=300,  # force strided subsampling (embargo scaled down)
            scale_per_fold=True,
        )
        return tuner.tune(X, y, w, label_spans=c.get_label_spans("train"))

    def test_tuning_twice_keeps_embargo(self) -> None:
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=6, embargo_bars=40))
        self._tune(cv)
        assert cv.config.embargo_bars == 40
        self._tune(cv)
        assert cv.config.embargo_bars == 40

    def test_tuner_scores_do_not_depend_on_feature_units(self) -> None:
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=6, embargo_bars=6))
        base = self._tune(cv, 1.0)["best_value"]
        huge = self._tune(cv, 1e7)["best_value"]
        assert base is not None and huge is not None
        assert abs(base - huge) < 0.05
