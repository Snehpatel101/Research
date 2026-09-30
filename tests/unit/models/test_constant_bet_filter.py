"""Meta-labeling with a primary whose bets cannot train a classifier."""

from __future__ import annotations

import numpy as np

from src.core.utils.safe_pickle import safe_pickle_dump, safe_pickle_load
from src.inference.meta_labeling_bundle import ConstantBetFilter, build_meta_features


def test_constant_filter_returns_the_win_rate_for_every_bar() -> None:
    proba = (
        ConstantBetFilter(0.25).fit(np.zeros((3, 2)), np.zeros(3)).predict_proba(np.zeros((4, 7)))
    )
    np.testing.assert_allclose(proba, [[0.75, 0.25]] * 4)


def test_no_bets_passes_every_bet_through() -> None:
    """With no sided bets the filter must not veto anything the primary does later."""
    assert (ConstantBetFilter(1.0).predict_proba(np.zeros((5, 3)))[:, 1] >= 0.99).all()


def test_constant_filter_survives_the_bundle_pickle_round_trip(tmp_path) -> None:
    safe_pickle_dump(ConstantBetFilter(0.6), tmp_path / "meta_model.pkl")
    loaded = safe_pickle_load(tmp_path / "meta_model.pkl")
    np.testing.assert_allclose(loaded.predict_proba(np.zeros((2, 1)))[:, 1], 0.6)


def test_meta_features_for_a_primary_that_never_bets() -> None:
    """Zero sided rows (e.g. a 4D primary that always predicts neutral) keep their width."""
    for shape in [(0, 7), (0, 16, 5), (0, 3, 16, 5)]:
        features = build_meta_features(np.zeros(shape), np.zeros((0, 3)))
        assert features.shape == (0, int(np.prod(shape[1:])) + 3 + 1)


def test_meta_labeling_bet_filter_is_saved_with_the_models(tmp_path) -> None:
    """The bet filter is a plain estimator (no .save()); it must still be persisted."""
    from sklearn.linear_model import LogisticRegression

    from src.models.training.services.artifact_persistence import ArtifactManager

    X, y = np.random.default_rng(0).normal(size=(40, 3)), np.tile([0, 1], 20)
    models = {
        "meta_labeling_h5_meta": LogisticRegression().fit(X, y),
        "const": ConstantBetFilter(0.4),
    }
    ArtifactManager(tmp_path).save_models(models)
    for key, model in models.items():
        loaded = safe_pickle_load(tmp_path / "models" / f"{key}.pkl")
        np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
