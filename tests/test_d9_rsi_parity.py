"""
D9: Verify RSI computed via Numba matches a reference pandas implementation.

Tests both the _rsi_numba (momentum.py) and calculate_rsi_numba (numba_functions.py)
against a hand-rolled pure-Python reference using Wilder's classic smoothing:
  1. SMA of the first `period` gains/losses as the seed
  2. Exponential update: avg = (1-alpha)*avg + alpha*val, where alpha = 1/period
"""

import numpy as np
import pandas as pd
import pytest


def _rsi_reference_pandas(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Reference RSI using Wilder's smoothing (SMA seed + exponential decay).

    This matches the classic RSI formula used by both Numba implementations:
      - Initialize avg_gain/avg_loss as SMA of first `period` changes
      - Update: avg = alpha * current + (1 - alpha) * prev, alpha = 1/period
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    # Price changes
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # SMA seed over first `period` changes
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    alpha = 1.0 / period

    # First RSI value at index=period
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    # Subsequent values using Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = alpha * gains[i - 1] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i - 1] + (1 - alpha) * avg_loss
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


@pytest.fixture
def synthetic_close():
    """200 bars of synthetic close prices via random walk."""
    rng = np.random.RandomState(42)
    returns = rng.normal(0, 0.01, 200)
    close = 100.0 * np.exp(np.cumsum(returns))
    return close


class TestRSINumbaParityMomentum:
    """Test _rsi_numba from momentum.py matches pandas reference."""

    def test_rsi_14_parity(self, synthetic_close):
        from src.data.features.compute.momentum import _rsi_numba

        period = 14
        numba_rsi = _rsi_numba(synthetic_close.astype(np.float64), period)
        ref_rsi = _rsi_reference_pandas(synthetic_close, period)

        mask = ~np.isnan(numba_rsi) & ~np.isnan(ref_rsi)
        assert mask.sum() > 0, "No valid RSI values produced"
        np.testing.assert_allclose(
            numba_rsi[mask],
            ref_rsi[mask],
            atol=1e-6,
            err_msg="_rsi_numba (momentum.py) does not match pandas reference",
        )

    def test_rsi_7_parity(self, synthetic_close):
        from src.data.features.compute.momentum import _rsi_numba

        period = 7
        numba_rsi = _rsi_numba(synthetic_close.astype(np.float64), period)
        ref_rsi = _rsi_reference_pandas(synthetic_close, period)

        mask = ~np.isnan(numba_rsi) & ~np.isnan(ref_rsi)
        assert mask.sum() > 0
        np.testing.assert_allclose(
            numba_rsi[mask],
            ref_rsi[mask],
            atol=1e-6,
            err_msg="_rsi_numba period=7 mismatch",
        )

    def test_rsi_21_parity(self, synthetic_close):
        from src.data.features.compute.momentum import _rsi_numba

        period = 21
        numba_rsi = _rsi_numba(synthetic_close.astype(np.float64), period)
        ref_rsi = _rsi_reference_pandas(synthetic_close, period)

        mask = ~np.isnan(numba_rsi) & ~np.isnan(ref_rsi)
        assert mask.sum() > 0
        np.testing.assert_allclose(
            numba_rsi[mask],
            ref_rsi[mask],
            atol=1e-6,
            err_msg="_rsi_numba period=21 mismatch",
        )


class TestRSINumbaParityPipeline:
    """Test calculate_rsi_numba from numba_functions.py matches pandas reference."""

    def test_rsi_14_parity(self, synthetic_close):
        from src.data.pipeline.stages.features.numba_functions import (
            calculate_rsi_numba,
        )

        period = 14
        numba_rsi = calculate_rsi_numba(synthetic_close.astype(np.float64), period)
        ref_rsi = _rsi_reference_pandas(synthetic_close, period)

        mask = ~np.isnan(numba_rsi) & ~np.isnan(ref_rsi)
        assert mask.sum() > 0, "No valid RSI values produced"
        np.testing.assert_allclose(
            numba_rsi[mask],
            ref_rsi[mask],
            atol=1e-6,
            err_msg="calculate_rsi_numba does not match pandas reference",
        )

    def test_rsi_7_parity(self, synthetic_close):
        from src.data.pipeline.stages.features.numba_functions import (
            calculate_rsi_numba,
        )

        period = 7
        numba_rsi = calculate_rsi_numba(synthetic_close.astype(np.float64), period)
        ref_rsi = _rsi_reference_pandas(synthetic_close, period)

        mask = ~np.isnan(numba_rsi) & ~np.isnan(ref_rsi)
        assert mask.sum() > 0
        np.testing.assert_allclose(
            numba_rsi[mask],
            ref_rsi[mask],
            atol=1e-6,
            err_msg="calculate_rsi_numba period=7 mismatch",
        )


class TestRSIComputeRSIWrapper:
    """Test _compute_rsi wrapper dispatches correctly and matches reference."""

    def test_compute_rsi_14(self, synthetic_close):
        from src.data.features.compute.momentum import _compute_rsi

        series = pd.Series(synthetic_close, name="close")
        result = _compute_rsi(series, period=14)
        ref_rsi = _rsi_reference_pandas(synthetic_close, period=14)

        mask = ~np.isnan(result.values) & ~np.isnan(ref_rsi)
        assert mask.sum() > 0
        np.testing.assert_allclose(
            result.values[mask],
            ref_rsi[mask],
            atol=1e-6,
            err_msg="_compute_rsi wrapper does not match pandas reference",
        )


class TestRSIBothImplementationsAgree:
    """Verify both Numba implementations produce identical results."""

    def test_momentum_vs_pipeline_rsi_14(self, synthetic_close):
        from src.data.features.compute.momentum import _rsi_numba
        from src.data.pipeline.stages.features.numba_functions import (
            calculate_rsi_numba,
        )

        period = 14
        arr = synthetic_close.astype(np.float64)
        rsi_momentum = _rsi_numba(arr, period)
        rsi_pipeline = calculate_rsi_numba(arr, period)

        mask = ~np.isnan(rsi_momentum) & ~np.isnan(rsi_pipeline)
        assert mask.sum() > 0
        np.testing.assert_allclose(
            rsi_momentum[mask],
            rsi_pipeline[mask],
            atol=1e-6,
            err_msg="momentum._rsi_numba and numba_functions.calculate_rsi_numba differ",
        )


class TestRSIEdgeCases:
    """Edge cases for RSI Numba implementations."""

    def test_constant_prices(self):
        """Constant prices should yield RSI = NaN or 100 (no losses)."""
        from src.data.features.compute.momentum import _rsi_numba

        close = np.full(50, 100.0)
        rsi = _rsi_numba(close, 14)
        valid = rsi[~np.isnan(rsi)]
        # With zero gains and zero losses, RSI is either 100 or NaN
        assert all((v == 100.0) or np.isnan(v) for v in valid)

    def test_monotonically_rising(self):
        """Monotonically rising prices should yield RSI near 100."""
        from src.data.features.compute.momentum import _rsi_numba

        close = np.arange(1.0, 201.0)
        rsi = _rsi_numba(close, 14)
        valid = rsi[~np.isnan(rsi)]
        assert len(valid) > 0
        assert np.all(valid > 95.0), "Rising prices should have RSI > 95"

    def test_monotonically_falling(self):
        """Monotonically falling prices should yield RSI near 0."""
        from src.data.features.compute.momentum import _rsi_numba

        close = np.arange(200.0, 0.0, -1.0)
        rsi = _rsi_numba(close, 14)
        valid = rsi[~np.isnan(rsi)]
        assert len(valid) > 0
        assert np.all(valid < 5.0), "Falling prices should have RSI < 5"

    def test_short_array(self):
        """Array shorter than period should return all NaN."""
        from src.data.features.compute.momentum import _rsi_numba

        close = np.array([100.0, 101.0, 99.0])
        rsi = _rsi_numba(close, 14)
        assert np.all(np.isnan(rsi))
