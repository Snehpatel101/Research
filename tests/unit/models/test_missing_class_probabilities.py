"""A model fitted without one of the classes still predicts class labels.

scikit-learn-style classifiers return one probability column per class they
saw. Taking argmax over those columns gave a COLUMN position: trained on labels
{-1, +1} only, a "+1" prediction came out as column 1 -> label 0 (review F6).
Every model now returns ``n_classes`` columns in class-index order.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.models  # noqa: F401 - registers every model
from src.models.common import full_class_probabilities
from src.models.registry import ModelRegistry


def test_full_class_probabilities_places_columns_by_class() -> None:
    proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    full = full_class_probabilities(proba, np.array([0, 2]), 3)
    assert full.tolist() == [[0.2, 0.0, 0.8], [0.9, 0.0, 0.1]]
    complete = np.array([[0.1, 0.2, 0.7]])
    assert np.array_equal(full_class_probabilities(complete, np.arange(3), 3), complete)


@pytest.mark.parametrize(
    "model_name",
    ["random_forest", "logistic", "svm", "catboost", "calibrated_meta", "mlp_meta", "ridge_meta"],
)
def test_training_without_neutral_class(model_name: str) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    y = np.where(X[:, 0] + 0.3 * rng.normal(size=600) > 0, 1, -1)  # no neutral label
    config = {"n_classes": 3}
    if model_name == "random_forest":
        config.update(n_estimators=20, oob_score=False)
    if model_name == "catboost":
        config.update(iterations=30, verbose=0)
    model = ModelRegistry.create(model_name, config=config)
    model.fit(X[:450], y[:450], X[450:], y[450:])

    output = model.predict(X[450:])
    assert output.class_probabilities.shape == (150, 3)
    if model_name != "mlp_meta":  # mlp_meta is fitted with every class declared
        assert np.all(output.class_probabilities[:, 1] == 0.0), "unseen class must get 0"
    assert set(np.unique(output.class_predictions)) <= {-1, 1}
    # A learnable signal: predictions must agree with the labels well above chance
    assert (output.class_predictions == y[450:]).mean() > 0.7
    if hasattr(model, "predict_proba"):
        assert np.array_equal(model.predict_proba(X[450:]), output.class_probabilities)
