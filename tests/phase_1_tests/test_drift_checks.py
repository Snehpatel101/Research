import numpy as np
import pandas as pd

from src.stages.validators.drift import compute_psi, check_feature_drift


def test_compute_psi_detects_shift():
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = pd.Series(np.random.normal(1.5, 1, 1000))
    psi = compute_psi(expected, actual, bins=10)
    assert psi > 0.1


def test_check_feature_drift_flags_shift():
    train = pd.DataFrame({"feat": np.random.normal(0, 1, 1000)})
    val = pd.DataFrame({"feat": np.random.normal(2.0, 1, 1000)})
    result = check_feature_drift(train, val, ["feat"], psi_threshold=0.2)
    assert result["drifted_feature_count"] == 1
