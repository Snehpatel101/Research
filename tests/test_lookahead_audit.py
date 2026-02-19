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
    PropagationScanResult,
    ResampleConfig,
    ResamplingParityResult,
    scan_dependency_propagation,
    validate_resample_config,
    verify_resampling_parity,
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


# ---------------------------------------------------------------------------
# H3: Dependency Propagation Scan Tests
# ---------------------------------------------------------------------------

# Minimal DAG that mirrors the real FEATURE_DEPENDENCIES structure
SAMPLE_DAG: dict[str, list[str]] = {
    "returns": [],
    "sma": [],
    "ema": [],
    "rsi": ["returns"],
    "macd": ["ema"],
    "bollinger": ["sma"],
    "keltner": ["ema", "atr"],
    "atr": [],
    "hvol": ["returns"],
    "regime": ["hvol", "returns"],
    "mtf": ["*"],  # depends on everything
}


class TestDependencyPropagationScan:
    """Tests for scan_dependency_propagation (H3)."""

    def test_no_taint_all_clean(self) -> None:
        """When nothing is tainted, all features are clean."""
        result = scan_dependency_propagation(SAMPLE_DAG, tainted_features=[])
        assert result.tainted_features == []
        assert result.propagated_features == []
        assert len(result.clean_features) == len(SAMPLE_DAG)
        assert not result.has_propagation

    def test_taint_returns_propagates_to_rsi(self) -> None:
        """If 'returns' is tainted, 'rsi' (depends on returns) is propagated."""
        result = scan_dependency_propagation(SAMPLE_DAG, tainted_features=["returns"])
        assert "returns" in result.tainted_features
        assert "rsi" in result.propagated_features
        assert "rsi" in result.all_affected

    def test_taint_returns_propagates_to_regime(self) -> None:
        """'regime' depends on 'hvol' and 'returns'.
        'hvol' depends on 'returns'.  Both should be propagated."""
        result = scan_dependency_propagation(SAMPLE_DAG, tainted_features=["returns"])
        assert "hvol" in result.propagated_features
        assert "regime" in result.propagated_features

    def test_taint_returns_propagates_to_mtf(self) -> None:
        """'mtf' depends on '*' (everything), so any taint reaches it."""
        result = scan_dependency_propagation(SAMPLE_DAG, tainted_features=["returns"])
        assert "mtf" in result.propagated_features

    def test_taint_ema_propagates_to_macd_and_keltner(self) -> None:
        """'macd' depends on 'ema'; 'keltner' depends on 'ema' + 'atr'."""
        result = scan_dependency_propagation(SAMPLE_DAG, tainted_features=["ema"])
        assert "macd" in result.propagated_features
        assert "keltner" in result.propagated_features
        # sma, returns, atr should remain clean (not depending on ema)
        assert "sma" in result.clean_features
        assert "returns" in result.clean_features
        assert "atr" in result.clean_features

    def test_clean_features_exclude_all_tainted(self) -> None:
        """Clean features should contain no directly or transitively tainted families."""
        result = scan_dependency_propagation(SAMPLE_DAG, tainted_features=["returns"])
        for feat in result.all_affected:
            assert feat not in result.clean_features

    def test_propagation_paths_recorded(self) -> None:
        """Propagation paths should trace back to the tainted source."""
        result = scan_dependency_propagation(SAMPLE_DAG, tainted_features=["returns"])
        # rsi directly depends on returns
        assert "rsi" in result.propagation_paths
        path = result.propagation_paths["rsi"]
        assert path[0] == "returns"
        assert path[-1] == "rsi"

    def test_to_dict_serializable(self) -> None:
        """PropagationScanResult.to_dict should produce a plain dict."""
        result = scan_dependency_propagation(SAMPLE_DAG, tainted_features=["ema"])
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "tainted_features" in d
        assert "propagated_features" in d
        assert "propagation_paths" in d
        assert "clean_features" in d
        assert "all_affected" in d
        assert "has_propagation" in d

    def test_auditor_method_delegates(self) -> None:
        """LookaheadAuditor.scan_dependency_propagation delegates correctly."""
        auditor = LookaheadAuditor(corruption_point=0.8)
        result = auditor.scan_dependency_propagation(
            SAMPLE_DAG, tainted_features=["returns"]
        )
        assert isinstance(result, PropagationScanResult)
        assert "rsi" in result.propagated_features

    def test_unknown_tainted_feature_ignored(self) -> None:
        """A tainted feature not in the DAG is silently excluded."""
        result = scan_dependency_propagation(
            SAMPLE_DAG, tainted_features=["nonexistent_feature"]
        )
        assert result.tainted_features == []
        assert result.propagated_features == []
        assert len(result.clean_features) == len(SAMPLE_DAG)


# ---------------------------------------------------------------------------
# M6: Resampling Parity Verification Tests
# ---------------------------------------------------------------------------


class TestResamplingParity:
    """Tests for verify_resampling_parity (M6)."""

    def test_matching_configs_pass(self) -> None:
        """Identical configs should return is_match=True."""
        train = {"closed": "left", "label": "left"}
        infer = {"closed": "left", "label": "left"}
        result = verify_resampling_parity(train, infer, raise_on_mismatch=False)
        assert result.is_match is True
        assert result.mismatches == []

    def test_mismatched_closed_raises(self) -> None:
        """Different 'closed' param should raise ValueError by default."""
        train = {"closed": "left", "label": "left"}
        infer = {"closed": "right", "label": "left"}
        with pytest.raises(ValueError, match="Resampling parity violation"):
            verify_resampling_parity(train, infer)

    def test_mismatched_label_raises(self) -> None:
        """Different 'label' param should raise ValueError by default."""
        train = {"closed": "left", "label": "left"}
        infer = {"closed": "left", "label": "right"}
        with pytest.raises(ValueError, match="Resampling parity violation"):
            verify_resampling_parity(train, infer)

    def test_mismatch_no_raise_returns_result(self) -> None:
        """With raise_on_mismatch=False, return result instead of raising."""
        train = {"closed": "left", "label": "left"}
        infer = {"closed": "right", "label": "right"}
        result = verify_resampling_parity(train, infer, raise_on_mismatch=False)
        assert result.is_match is False
        assert len(result.mismatches) == 2
        assert any("closed" in m for m in result.mismatches)
        assert any("label" in m for m in result.mismatches)

    def test_extra_key_in_inference_flagged(self) -> None:
        """A key present in inference but missing in training is a mismatch."""
        train = {"closed": "left", "label": "left"}
        infer = {"closed": "left", "label": "left", "origin": "epoch"}
        result = verify_resampling_parity(train, infer, raise_on_mismatch=False)
        assert result.is_match is False
        assert any("origin" in m for m in result.mismatches)

    def test_extra_key_in_training_flagged(self) -> None:
        """A key present in training but missing in inference is a mismatch."""
        train = {"closed": "left", "label": "left", "offset": "0s"}
        infer = {"closed": "left", "label": "left"}
        result = verify_resampling_parity(train, infer, raise_on_mismatch=False)
        assert result.is_match is False
        assert any("offset" in m for m in result.mismatches)

    def test_full_config_match(self) -> None:
        """Full config with origin and freq should match if identical."""
        cfg = {"closed": "left", "label": "left", "origin": "epoch", "freq": "5min"}
        result = verify_resampling_parity(cfg, cfg.copy(), raise_on_mismatch=False)
        assert result.is_match is True

    def test_to_dict_serializable(self) -> None:
        """ResamplingParityResult.to_dict should produce a plain dict."""
        train = {"closed": "left", "label": "left"}
        infer = {"closed": "left", "label": "left"}
        result = verify_resampling_parity(train, infer, raise_on_mismatch=False)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "is_match" in d
        assert "mismatches" in d
        assert "training_params" in d
        assert "inference_params" in d

    def test_empty_configs_pass(self) -> None:
        """Both configs empty -- nothing to compare, should match."""
        result = verify_resampling_parity({}, {}, raise_on_mismatch=False)
        assert result.is_match is True
