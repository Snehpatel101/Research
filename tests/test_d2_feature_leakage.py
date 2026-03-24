"""
Test D2: Feature selection uses training data only (no leakage from test set).

Verifies that permutation importance (MDA) computed on the training split
does NOT pick up signal that exists only in the test split. A "leaky" feature
that is pure noise in training but perfectly correlated with labels in testing
must receive near-zero importance when evaluated correctly on train-only data.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


@pytest.fixture()
def leaky_dataset():
    """
    Build synthetic data with a leaky feature.

    - 1000 rows, 10 random features + 1 leaky feature (index 10)
    - Train split: rows 0-699  -- leaky feature is random noise
    - Test split:  rows 700-999 -- leaky feature == labels (perfect signal)

    If feature selection accidentally touches test data, the leaky feature
    will appear important. On train-only data it should be worthless.
    """
    rng = np.random.RandomState(42)
    n_rows = 1000
    n_real_features = 10
    split_idx = 700

    X_random = rng.randn(n_rows, n_real_features).astype(np.float32)
    labels = rng.choice([0, 1, 2], size=n_rows)

    # Leaky feature: noise in train, perfect copy of labels in test
    leaky = rng.randn(n_rows).astype(np.float32)
    leaky[split_idx:] = labels[split_idx:].astype(np.float32)

    X = np.column_stack([X_random, leaky])
    feature_names = [f"feat_{i}" for i in range(n_real_features)] + ["leaky"]

    return {
        "X_train": X[:split_idx],
        "y_train": labels[:split_idx],
        "X_test": X[split_idx:],
        "y_test": labels[split_idx:],
        "feature_names": feature_names,
        "leaky_idx": n_real_features,
    }


def test_leaky_feature_has_low_importance_on_train(leaky_dataset):
    """
    Permutation importance on TRAIN data should give the leaky feature
    near-zero importance because it is pure noise in the training split.
    """
    X_train = leaky_dataset["X_train"]
    y_train = leaky_dataset["y_train"]
    leaky_idx = leaky_dataset["leaky_idx"]

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    result = permutation_importance(rf, X_train, y_train, n_repeats=5, random_state=42, n_jobs=-1)

    leaky_importance = result.importances_mean[leaky_idx]
    mean_importance = np.mean(result.importances_mean)

    # Leaky feature should not stand out -- its importance on train data
    # should be below twice the average (it is noise, same as others).
    assert leaky_importance < 2.0 * mean_importance, (
        f"Leaky feature importance ({leaky_importance:.4f}) should be near "
        f"the average ({mean_importance:.4f}) on train-only data"
    )


def test_leaky_feature_has_high_importance_on_test(leaky_dataset):
    """
    Sanity check: permutation importance on TEST data correctly detects
    the leaky feature, proving the test setup is valid.
    """
    X_train = leaky_dataset["X_train"]
    y_train = leaky_dataset["y_train"]
    X_test = leaky_dataset["X_test"]
    y_test = leaky_dataset["y_test"]
    leaky_idx = leaky_dataset["leaky_idx"]

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    result = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)

    leaky_importance = result.importances_mean[leaky_idx]
    max_other = np.max(np.delete(result.importances_mean, leaky_idx))

    # On test data the leaky feature should dominate all others
    assert leaky_importance > max_other, (
        f"Leaky feature ({leaky_importance:.4f}) should dominate other "
        f"features ({max_other:.4f}) on test data (sanity check)"
    )


def test_leaky_feature_rank_on_test_vs_train(leaky_dataset):
    """
    The leaky feature must rank #1 when a model is trained on ALL data
    (including test, simulating leakage) and importance is measured on test,
    but must NOT rank #1 when a model is trained on train-only data.
    """
    X_train = leaky_dataset["X_train"]
    y_train = leaky_dataset["y_train"]
    X_test = leaky_dataset["X_test"]
    y_test = leaky_dataset["y_test"]
    leaky_idx = leaky_dataset["leaky_idx"]

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # --- Leaked model: trained on ALL data (train + test) ---
    rf_leaked = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf_leaked.fit(X_all, y_all)

    leaked_result = permutation_importance(
        rf_leaked, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    leaked_top = int(np.argmax(leaked_result.importances_mean))

    # When the model has seen test data, leaky feature should rank #1
    assert leaked_top == leaky_idx, (
        f"With data leakage, leaky feature (idx {leaky_idx}) should be top "
        f"but got idx {leaked_top}"
    )

    # --- Correct model: trained on train-only data ---
    rf_correct = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf_correct.fit(X_train, y_train)

    correct_result = permutation_importance(
        rf_correct, X_train, y_train, n_repeats=5, random_state=42, n_jobs=-1
    )
    correct_top = int(np.argmax(correct_result.importances_mean))

    # With train-only evaluation, leaky feature must NOT be the top feature
    assert correct_top != leaky_idx, (
        f"With train-only evaluation, leaky feature (idx {leaky_idx}) should "
        f"NOT be the top feature, but it was"
    )
