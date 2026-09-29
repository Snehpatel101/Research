"""
AdapterScaler save/load persistence tests (Phase 73 format fix coverage).

Verifies behavioral parity between an in-memory fitted AdapterScaler and one
round-tripped through save()/load():

1. loaded.transform(X) matches original.transform(X)
2. float32 dtype is preserved end-to-end (fit/transform/inverse, both paths)
3. inverse_transform round-trips back to the original data within tolerance
4. Format self-consistency: files written by the current save() are readable
   by the current load(), and metadata/statistics survive the round trip

KNOWN GAP (documented, not fixed here): save() persists only the sklearn
scaler pickle + JSON config. The manual float32 stats computed in fit()
(_f32_center/_f32_scale/_f32_data_min/_f32_data_range and the _input_f32
flag) are NOT persisted, so a loaded scaler routes float32 input through
sklearn's float64 statistics instead of the manual float32 path. Results
agree only to float32 precision (~2e-6 on unit-scale data), not bitwise.
Tests below assert parity at a tolerance the current behavior supports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.adapters.scaling import AdapterScaler, ScalerConfig

METHODS = ["robust", "standard", "minmax"]

# Observed float32-vs-float64-stats divergence on unit-scale data is ~2.4e-6.
# 5e-5 gives a comfortable margin while still catching real format breakage
# (a wrong pickle format or stale stats produce errors orders of magnitude larger).
F32_PARITY_ATOL = 5e-5


@pytest.fixture
def x_f32() -> np.ndarray:
    """Small float32 2D matrix with injected outliers (600 x 8)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=(600, 8)).astype(np.float32)
    x[::50] += 25.0  # outliers so robust vs standard actually differ
    return x


@pytest.fixture
def x3d_f32() -> np.ndarray:
    """Small float32 3D sequence tensor (50 x 10 x 4)."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, size=(50, 10, 4)).astype(np.float32)
    x[::10, 0, :] -= 15.0
    return x


def _round_trip(scaler: AdapterScaler, tmp_path: Path) -> AdapterScaler:
    """Save scaler to tmp_path and load it back with the current format."""
    save_dir = tmp_path / "scaler_artifacts"
    scaler.save(save_dir)
    return AdapterScaler.load(save_dir)


# ---------------------------------------------------------------------------
# 1. Transform parity: original vs save/load round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_transform_parity_float32(method: str, x_f32: np.ndarray, tmp_path: Path) -> None:
    """Loaded scaler transforms float32 data to (near-)identical values."""
    scaler = AdapterScaler(ScalerConfig(method=method, clip_value=0.0))
    scaler.fit(x_f32)
    expected = scaler.transform(x_f32)

    loaded = _round_trip(scaler, tmp_path)
    actual = loaded.transform(x_f32)

    assert actual.shape == expected.shape
    # Not bitwise: loaded scaler lacks the f32 stats (see module docstring).
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=F32_PARITY_ATOL)


@pytest.mark.parametrize("method", METHODS)
def test_transform_parity_float64_exact(method: str, x_f32: np.ndarray, tmp_path: Path) -> None:
    """Float64 input takes the sklearn path on both sides -> bitwise identical."""
    x64 = x_f32.astype(np.float64)
    scaler = AdapterScaler(ScalerConfig(method=method, clip_value=0.0))
    scaler.fit(x64)
    expected = scaler.transform(x64)

    loaded = _round_trip(scaler, tmp_path)
    actual = loaded.transform(x64)

    assert np.array_equal(actual, expected), "pickled sklearn scaler must round-trip exactly"


def test_transform_parity_3d_float32(x3d_f32: np.ndarray, tmp_path: Path) -> None:
    """Save/load parity holds for 3D sequence tensors (adapter usage)."""
    scaler = AdapterScaler(ScalerConfig(method="robust", clip_value=0.0))
    scaler.fit(x3d_f32)
    expected = scaler.transform(x3d_f32)

    loaded = _round_trip(scaler, tmp_path)
    actual = loaded.transform(x3d_f32)

    assert actual.shape == x3d_f32.shape
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=F32_PARITY_ATOL)


# ---------------------------------------------------------------------------
# 2. dtype preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_float32_dtype_preserved_end_to_end(method: str, x_f32: np.ndarray, tmp_path: Path) -> None:
    """float32 stays float32 through fit/transform/inverse on both scalers."""
    scaler = AdapterScaler(ScalerConfig(method=method, clip_value=0.0))
    scaled = scaler.fit_transform(x_f32)
    assert scaled.dtype == np.float32

    inverse = scaler.inverse_transform(scaled)
    assert inverse.dtype == np.float32

    loaded = _round_trip(scaler, tmp_path)
    loaded_scaled = loaded.transform(x_f32)
    assert loaded_scaled.dtype == np.float32

    loaded_inverse = loaded.inverse_transform(loaded_scaled)
    assert loaded_inverse.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. inverse_transform round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_inverse_transform_round_trip(method: str, x_f32: np.ndarray, tmp_path: Path) -> None:
    """inverse(transform(X)) ~= X for both the original and the loaded scaler.

    clip_value=0 disables clipping — clipped values cannot round-trip by design.
    """
    scaler = AdapterScaler(ScalerConfig(method=method, clip_value=0.0))
    scaler.fit(x_f32)

    recovered = scaler.inverse_transform(scaler.transform(x_f32))
    np.testing.assert_allclose(recovered, x_f32, rtol=0.0, atol=1e-4)

    loaded = _round_trip(scaler, tmp_path)
    recovered_loaded = loaded.inverse_transform(loaded.transform(x_f32))
    np.testing.assert_allclose(recovered_loaded, x_f32, rtol=0.0, atol=1e-4)


def test_clipping_applied_consistently(x_f32: np.ndarray, tmp_path: Path) -> None:
    """Default clip_value=5.0 bounds outputs identically before and after load."""
    scaler = AdapterScaler(ScalerConfig(method="robust", clip_value=5.0))
    scaler.fit(x_f32)
    expected = scaler.transform(x_f32)
    assert expected.max() <= 5.0 and expected.min() >= -5.0
    # The injected +25 outliers must actually hit the clip boundary
    assert np.isclose(expected.max(), 5.0)

    loaded = _round_trip(scaler, tmp_path)
    actual = loaded.transform(x_f32)
    assert actual.max() <= 5.0 and actual.min() >= -5.0
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=F32_PARITY_ATOL)


# ---------------------------------------------------------------------------
# 4. Format self-consistency
# ---------------------------------------------------------------------------


def test_save_writes_expected_files(x_f32: np.ndarray, tmp_path: Path) -> None:
    """Current save() format: scaler.pkl (pickle) + scaler_config.json."""
    scaler = AdapterScaler(ScalerConfig(method="robust"))
    scaler.fit(x_f32)
    save_dir = tmp_path / "artifacts"
    scaler.save(save_dir)

    assert (save_dir / "scaler.pkl").exists()
    assert (save_dir / "scaler_config.json").exists()


@pytest.mark.parametrize("method", METHODS)
def test_metadata_survives_round_trip(method: str, x_f32: np.ndarray, tmp_path: Path) -> None:
    """Config and fitted statistics are identical after save/load."""
    config = ScalerConfig(method=method, clip_value=3.5)
    scaler = AdapterScaler(config)
    scaler.fit(x_f32)

    loaded = _round_trip(scaler, tmp_path)

    assert loaded.is_fitted is True
    assert loaded.n_features == x_f32.shape[1]
    assert loaded.config.method == method
    assert loaded.config.clip_value == 3.5
    assert loaded.config.quantile_range == config.quantile_range
    assert loaded.config.feature_range == config.feature_range

    # Fitted sklearn statistics (center/scale, mean/std, min/max) match exactly
    assert loaded.get_statistics() == scaler.get_statistics()


def test_method_none_round_trip(x_f32: np.ndarray, tmp_path: Path) -> None:
    """method='none' is a pass-through and survives save/load (no scaler.pkl)."""
    scaler = AdapterScaler(ScalerConfig(method="none"))
    scaler.fit(x_f32)
    assert scaler.transform(x_f32) is x_f32  # identity pass-through

    save_dir = tmp_path / "none_scaler"
    scaler.save(save_dir)
    assert not (save_dir / "scaler.pkl").exists()  # nothing to pickle

    loaded = AdapterScaler.load(save_dir)
    assert loaded.is_fitted is True
    np.testing.assert_array_equal(loaded.transform(x_f32), x_f32)


def test_load_missing_config_raises(tmp_path: Path) -> None:
    """Loading from a directory without scaler_config.json raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        AdapterScaler.load(tmp_path / "does_not_exist")


def test_loaded_scaler_rejects_feature_mismatch(x_f32: np.ndarray, tmp_path: Path) -> None:
    """n_features metadata is enforced on the loaded scaler."""
    scaler = AdapterScaler(ScalerConfig(method="standard"))
    scaler.fit(x_f32)
    loaded = _round_trip(scaler, tmp_path)

    wrong = x_f32[:, :5]
    with pytest.raises(ValueError, match="Feature count mismatch"):
        loaded.transform(wrong)


# ---------------------------------------------------------------------------
# Known gap documentation (current behavior, see module docstring)
# ---------------------------------------------------------------------------


def test_float32_stats_persisted(x_f32: np.ndarray, tmp_path: Path) -> None:
    """Manual float32 scaling stats survive save/load.

    Regression: save() used to drop _input_f32/_f32_center/_f32_scale, so a
    loaded scaler routed float32 input through the sklearn float64 path and
    produced slightly different transforms.
    """
    scaler = AdapterScaler(ScalerConfig(method="robust", clip_value=0.0))
    scaler.fit(x_f32)
    assert scaler._input_f32 is True
    assert scaler._f32_scale is not None

    loaded = _round_trip(scaler, tmp_path)
    assert loaded._input_f32 is True
    assert loaded._f32_scale is not None
    np.testing.assert_array_equal(loaded._f32_center, scaler._f32_center)
    np.testing.assert_array_equal(loaded._f32_scale, scaler._f32_scale)
    # With the f32 stats restored, float32 transforms are bitwise identical.
    np.testing.assert_array_equal(loaded.transform(x_f32), scaler.transform(x_f32))
