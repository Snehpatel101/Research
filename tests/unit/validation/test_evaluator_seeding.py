"""The standalone evaluators (``ml cv``, ``ml walk-forward``, ``ml cpcv-pbo``) take
the run seed: mutual-information ranking, per-fold tuning and fold models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli.unified_cli import app
from src.validation.cv import cv_feature_selection
from src.validation.cv.cv_runner import CrossValidationRunner
from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig


def test_per_fold_selection_is_seeded(monkeypatch: pytest.MonkeyPatch) -> None:
    mi_seeds: list[Any] = []
    real_mi = cv_feature_selection.mutual_info_classif

    def spy_mi(*args: Any, **kwargs: Any) -> Any:
        mi_seeds.append(kwargs.get("random_state"))
        return real_mi(*args, **kwargs)

    monkeypatch.setattr(cv_feature_selection, "mutual_info_classif", spy_mi)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=list("abcde"))
    y = pd.Series(rng.choice([-1, 0, 1], size=400))
    cv = PurgedKFold(PurgedKFoldConfig(n_splits=2, purge_bars=2, embargo_bars=2))
    splits = list(cv.split(X, y))
    result = cv_feature_selection.run_cv_with_per_fold_feature_selection(
        X=X,
        y=y,
        weights=None,
        cv_splits=splits,
        model_name="logistic",
        config={"random_state": 77, "random_seed": 77},
        n_features_to_select=3,
        cv=cv,
        seed=77,
    )
    assert mi_seeds == [77, 77]
    assert result["oof_prediction"] is not None


class _Captured(Exception):
    pass


def test_runner_seeds_fold_models_over_the_model_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model's default config carries random_state=42; the run seed wins."""
    from src.validation.cv import cv_runner

    captured: dict[str, Any] = {}

    def spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _Captured

    monkeypatch.setattr(cv_runner, "run_cv_with_per_fold_feature_selection", spy)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = pd.Series(rng.choice([-1, 0, 1], size=300))

    class Container:
        def get_sklearn_arrays(self, split: str, return_df: bool = False):
            return X, y, None

        def get_label_spans(self, split: str):
            return None

    runner = CrossValidationRunner(
        cv=PurgedKFold(PurgedKFoldConfig(n_splits=2, purge_bars=2, embargo_bars=2)),
        models=["logistic"],
        horizons=[5],
        tune_hyperparams=False,
        seed=9,
    )
    with pytest.raises(_Captured):
        runner.run(Container())
    assert captured["seed"] == 9
    assert captured["config"]["random_state"] == 9
    assert captured["config"]["random_seed"] == 9


@pytest.mark.parametrize("command", ["cv", "walk-forward", "cpcv-pbo"])
def test_evaluator_commands_take_a_seed(command: str) -> None:
    result = CliRunner().invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert "--seed" in result.output
