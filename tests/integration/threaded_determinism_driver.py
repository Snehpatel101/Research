"""
Multi-threaded model fits and feature ranking for ``test_threaded_determinism.py``.

Runs in its own interpreter (own PYTHONHASHSEED) and writes ``out.npz`` with
random-forest and LightGBM probabilities (both fitted with several threads)
and the clustered-MDA importances with their ranking.

Usage: python -m tests.integration.threaded_determinism_driver <output.npz> <n_threads>
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd


def main(output: str, n_threads: int) -> None:
    warnings.simplefilter("ignore")
    import src.models  # noqa: F401 - registers every model
    from src.models.registry import ModelRegistry
    from src.optimization.feature_selection.ranking import rank_by_importance
    from src.optimization.feature_selection.walk_forward import WalkForwardFeatureSelector

    rng = np.random.default_rng(0)
    n, n_features = 3000, 12
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
    y = np.digitize(signal + rng.normal(size=n), [-0.6, 0.6]) - 1
    fit, val = slice(0, 2400), slice(2400, n)

    arrays: dict[str, np.ndarray] = {}
    for name, config in (
        ("random_forest", {"n_estimators": 60, "n_jobs": n_threads, "random_state": 11}),
        ("lightgbm", {"n_estimators": 60, "n_jobs": n_threads, "random_state": 11}),
    ):
        model = ModelRegistry.create(name, config=config)
        model.fit(X[fit], y[fit], X[val], y[val])
        arrays[f"proba__{name}"] = model.predict(X).class_probabilities

    columns = [f"f{i:02d}" for i in range(n_features)]
    frame = pd.DataFrame(X, columns=columns)
    labels = pd.Series(y)
    selector = WalkForwardFeatureSelector(n_estimators=30, mda_n_repeats=2, random_state=5)
    # Clustered MDA is what the live pipeline ranks with (plain MDA's process
    # pool start-up would triple this test's runtime)
    scores = selector._clustered_mda_importance(
        frame.iloc[fit], labels.iloc[fit], X_test=frame.iloc[val], y_test=labels.iloc[val]
    )
    arrays["importance__clustered"] = scores.reindex(columns).to_numpy()
    arrays["ranking__clustered"] = np.array(list(rank_by_importance(scores).index))
    np.savez(output, **arrays)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
