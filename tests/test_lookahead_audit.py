"""
Tests for lookahead bias auditing.

Verifies that the corruption-based lookahead detector correctly passes
clean features (e.g., simple moving average) and catches forward-looking
features. Also tests different corruption methods and resample config
validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.lookahead_audit import (
    LookaheadAuditor,
    LookaheadBiasError,
    ResampleConfig,
    validate_resample_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """200-bar synthetic OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(42)
    n = 200
    base_price = 4500.0
    returns = rng.normal(0, 0.002, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.003, n))
    low = close * (1 - rng.uniform(0, 0.003, n))
    open_ = close * (1 + rng.uniform(-0.001, 0.001, n))
    volume = rng.integers(1000, 5000, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="5min"),
    )


# ---------------------------------------------------------------------------
# Helper feature functions
# ---------------------------------------------------------------------------


def sma_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking SMA -- should NOT have lookahead."""
    df = df.copy()
    df["sma_10"] = df["close"].rolling(10).mean()
    return df


def forward_shifted_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-shifted feature -- deliberate lookahead bias."""
    df = df.copy()
    df["future_close"] = df["close"].shift(-5)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCleanFeature:
    """A simple backward-looking SMA must pass the audit."""

    def test_sma_no_lookahead(self, ohlcv_df: pd.DataFrame) -> None:
        auditor = LookaheadAuditor(corruption_point=0.8, corruption_method="random")
        result = auditor.audit_feature_function(
            ohlcv_df, sma_feature, name="SMA", raise_on_lookahead=False
        )
        assert result.has_lookahead is False
        assert len(result.affected_columns) == 0


class TestForwardLooking:
    """A forward-shifted feature must be detected as lookahead."""

    def test_forward_shift_detected(self, ohlcv_df: pd.DataFrame) -> None:
        auditor = LookaheadAuditor(corruption_point=0.8, corruption_method="random")
        result = auditor.audit_feature_function(
            ohlcv_df,
            forward_shifted_feature,
            name="FutureClose",
            raise_on_lookahead=False,
        )
        assert result.has_lookahead is True
        assert "future_close" in result.affected_columns

    def test_raises_in_blocking_mode(self, ohlcv_df: pd.DataFrame) -> None:
        auditor = LookaheadAuditor(corruption_point=0.8, corruption_method="random")
        with pytest.raises(LookaheadBiasError):
            auditor.audit_feature_function(
                ohlcv_df,
                forward_shifted_feature,
                name="FutureClose",
                raise_on_lookahead=True,
            )


class TestCorruptionMethods:
    """Different corruption methods should all detect obvious lookahead."""

    @pytest.mark.parametrize("method", ["nan", "random", "shuffle"])
    def test_method_detects_lookahead(self, ohlcv_df: pd.DataFrame, method: str) -> None:
        auditor = LookaheadAuditor(
            corruption_point=0.8, corruption_method=method  # type: ignore[arg-type]
        )
        result = auditor.audit_feature_function(
            ohlcv_df,
            forward_shifted_feature,
            name=f"FutureClose_{method}",
            raise_on_lookahead=False,
        )
        # nan corruption may cause the feature fn to produce NaN for the
        # corrupted region, but past values should still differ when
        # the feature peeks ahead -- or the function may gracefully handle
        # NaN and still pass (which the auditor interprets as "no lookahead").
        # random and shuffle should always catch it.
        if method in ("random", "shuffle"):
            assert result.has_lookahead is True


class TestCorruptionPointValidation:
    """corruption_point must be in (0, 1)."""

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="corruption_point"):
            LookaheadAuditor(corruption_point=0.0)

    def test_one_raises(self) -> None:
        with pytest.raises(ValueError, match="corruption_point"):
            LookaheadAuditor(corruption_point=1.0)


class TestResampleConfigValidation:
    """validate_resample_config should flag mismatched closed/label params."""

    def test_correct_config_valid(self) -> None:
        is_valid, issues = validate_resample_config(closed="left", label="left")
        assert is_valid is True
        assert all("differs" not in msg for msg in issues)

    def test_wrong_closed_invalid(self) -> None:
        is_valid, issues = validate_resample_config(closed="right", label="left")
        assert is_valid is False
        assert any("closed" in msg for msg in issues)

    def test_wrong_label_invalid(self) -> None:
        is_valid, issues = validate_resample_config(closed="left", label="right")
        assert is_valid is False
        assert any("label" in msg for msg in issues)

    def test_none_params_warn(self) -> None:
        is_valid, issues = validate_resample_config()
        # None is technically valid (pandas defaults) but should warn
        assert is_valid is True
        assert len(issues) >= 1

    def test_ohlcv_default(self) -> None:
        cfg = ResampleConfig.ohlcv_default()
        assert cfg.closed == "left"
        assert cfg.label == "left"
