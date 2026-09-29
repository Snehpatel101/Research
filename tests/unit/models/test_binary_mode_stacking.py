"""Test D5: Binary mode (n_classes=2) pipeline test.

Verifies that StackingEnsemble works with binary labels {0, 1}
and n_classes=2 without crashing, and that output predictions
stay in {0, 1} (not {-1, 0, 1}).
"""

import numpy as np

from src.models.ensemble.stacking import StackingEnsemble


def _make_binary_data(n_train: int = 200, n_val: int = 50, n_features: int = 10, seed: int = 42):
    """Generate synthetic binary classification data."""
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, n_features).astype(np.float32)
    X_val = rng.randn(n_val, n_features).astype(np.float32)
    y_train = rng.randint(0, 2, size=n_train)
    y_val = rng.randint(0, 2, size=n_val)
    return X_train, y_train, X_val, y_val


def test_stacking_binary_mode_runs():
    """StackingEnsemble with n_classes=2 trains and predicts without error."""
    X_train, y_train, X_val, y_val = _make_binary_data()

    ensemble = StackingEnsemble(
        config={
            "base_model_names": ["logistic", "random_forest"],
            "meta_learner_name": "logistic",
            "n_classes": 2,
            "n_folds": 2,
            "use_probabilities": True,
            "analyze_diversity": False,
            "purge_bars": 0,
            "embargo_bars": 0,
        }
    )

    metrics = ensemble.fit(X_train, y_train, X_val, y_val)
    assert metrics.val_f1 >= 0.0

    result = ensemble.predict(X_val)
    assert result.class_predictions.shape == (len(X_val),)
    assert result.class_probabilities.shape[0] == len(X_val)


def test_stacking_binary_predictions_in_binary_range():
    """Output predictions from binary stacking are in {0, 1}, not {-1, 0, 1}."""
    X_train, y_train, X_val, y_val = _make_binary_data()

    ensemble = StackingEnsemble(
        config={
            "base_model_names": ["logistic", "random_forest"],
            "meta_learner_name": "logistic",
            "n_classes": 2,
            "n_folds": 2,
            "use_probabilities": True,
            "analyze_diversity": False,
            "purge_bars": 0,
            "embargo_bars": 0,
        }
    )

    ensemble.fit(X_train, y_train, X_val, y_val)
    result = ensemble.predict(X_val)

    unique_preds = set(result.class_predictions.tolist())
    assert unique_preds.issubset(
        {0, 1}
    ), f"Binary mode predictions should be in {{0, 1}}, got {unique_preds}"
    assert -1 not in unique_preds, "Binary mode should never predict -1"


def test_stacking_binary_probabilities_shape():
    """Binary stacking should produce 2-column probability matrix."""
    X_train, y_train, X_val, y_val = _make_binary_data()

    ensemble = StackingEnsemble(
        config={
            "base_model_names": ["logistic", "random_forest"],
            "meta_learner_name": "logistic",
            "n_classes": 2,
            "n_folds": 2,
            "use_probabilities": True,
            "analyze_diversity": False,
            "purge_bars": 0,
            "embargo_bars": 0,
        }
    )

    ensemble.fit(X_train, y_train, X_val, y_val)
    result = ensemble.predict(X_val)

    assert result.class_probabilities.shape == (
        len(X_val),
        2,
    ), f"Expected (n_samples, 2) probabilities, got {result.class_probabilities.shape}"
    # Probabilities should sum to ~1
    row_sums = result.class_probabilities.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
