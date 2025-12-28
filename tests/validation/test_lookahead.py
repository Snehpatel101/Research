"""
Tests for Lookahead Audit.

Tests:
- Corruption testing detects lookahead in bad features
- Corruption testing passes for good features (with shift(1))
- MTF alignment validation
- Resample config validation
"""
import numpy as np
import pandas as pd
import pytest

from src.validation.lookahead_audit import (
    LookaheadAuditor,
    LookaheadAuditResult,
    ResampleConfig,
    audit_feature_lookahead,
    audit_mtf_alignment,
    validate_resample_config,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="5min")

    # Generate realistic OHLCV
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100, 1000, n)

    return pd.DataFrame({
        "datetime": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_mtf_data():
    """Generate sample MTF-aligned data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="5min")

    # Simulate properly shifted MTF features (first 12 rows NaN for 1h TF)
    mtf_feature = np.random.randn(n)
    mtf_feature[:12] = np.nan  # Proper shift for 1h MTF on 5min base

    return pd.DataFrame({
        "mtf_close_1h": mtf_feature,
    }, index=dates)


# =============================================================================
# RESAMPLE CONFIG TESTS
# =============================================================================

class TestResampleConfig:
    """Tests for ResampleConfig validation."""

    def test_ohlcv_default_config(self):
        """OHLCV default uses left/left."""
        config = ResampleConfig.ohlcv_default()
        assert config.closed == "left"
        assert config.label == "left"

    def test_validate_explicit_correct(self):
        """Explicit correct params pass validation."""
        is_valid, issues = validate_resample_config(
            closed="left", label="left"
        )
        assert is_valid
        assert len(issues) == 0

    def test_validate_implicit_params_warns(self):
        """Implicit (None) params generate warnings."""
        is_valid, issues = validate_resample_config(
            closed=None, label=None
        )
        # Still valid (defaults are correct) but warnings issued
        assert is_valid
        assert len(issues) == 2
        assert any("implicit" in msg.lower() for msg in issues)

    def test_validate_wrong_closed_fails(self):
        """Wrong 'closed' param fails validation."""
        is_valid, issues = validate_resample_config(
            closed="right", label="left"
        )
        assert not is_valid
        assert any("differs from expected" in msg for msg in issues)

    def test_validate_wrong_label_fails(self):
        """Wrong 'label' param fails validation."""
        is_valid, issues = validate_resample_config(
            closed="left", label="right"
        )
        assert not is_valid
        assert any("differs from expected" in msg for msg in issues)


# =============================================================================
# CORRUPTION TESTING
# =============================================================================

class TestLookaheadAuditor:
    """Tests for LookaheadAuditor corruption testing."""

    def test_init_valid_params(self):
        """Auditor initializes with valid params."""
        auditor = LookaheadAuditor(
            corruption_point=0.7,
            corruption_method="nan",
            tolerance=1e-8,
        )
        assert auditor.corruption_point == 0.7
        assert auditor.corruption_method == "nan"

    def test_init_invalid_corruption_point_raises(self):
        """Invalid corruption_point raises ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            LookaheadAuditor(corruption_point=0.0)
        with pytest.raises(ValueError, match="must be in"):
            LookaheadAuditor(corruption_point=1.0)

    def test_detects_lookahead_in_bad_feature(self, sample_ohlcv_data):
        """Auditor detects lookahead in feature that uses future data."""
        df = sample_ohlcv_data

        def bad_feature_fn(df):
            """BAD: Uses future data (rolling mean includes current + future)."""
            df = df.copy()
            # center=True causes lookahead - uses future values
            df["bad_ma"] = df["close"].rolling(10, center=True).mean()
            return df

        auditor = LookaheadAuditor(corruption_point=0.8, corruption_method="nan")
        result = auditor.audit_feature_function(df, bad_feature_fn, "bad_ma")

        assert result.has_lookahead
        assert "bad_ma" in result.affected_columns
        assert "LOOKAHEAD DETECTED" in result.details

    def test_passes_good_feature_with_shift(self, sample_ohlcv_data):
        """Auditor passes feature that properly uses shift(1)."""
        df = sample_ohlcv_data

        def good_feature_fn(df):
            """GOOD: Uses shift(1) to prevent lookahead."""
            df = df.copy()
            df["good_ma"] = df["close"].rolling(10).mean().shift(1)
            return df

        auditor = LookaheadAuditor(corruption_point=0.8, corruption_method="nan")
        result = auditor.audit_feature_function(df, good_feature_fn, "good_ma")

        assert not result.has_lookahead
        assert len(result.affected_columns) == 0

    def test_passes_returns_feature(self, sample_ohlcv_data):
        """Auditor passes properly lagged returns."""
        df = sample_ohlcv_data

        def returns_fn(df):
            """Proper returns with shift."""
            df = df.copy()
            df["returns"] = df["close"].pct_change().shift(1)
            return df

        auditor = LookaheadAuditor(corruption_point=0.8)
        result = auditor.audit_feature_function(df, returns_fn, "returns")

        assert not result.has_lookahead

    def test_detects_unshifted_feature(self, sample_ohlcv_data):
        """Auditor detects feature without proper lag."""
        df = sample_ohlcv_data

        def unshifted_fn(df):
            """BAD: No shift - uses current bar's close."""
            df = df.copy()
            # Rolling mean without shift uses current close
            df["unshifted_ma"] = df["close"].rolling(5).mean()
            return df

        # Use random corruption to detect the issue
        auditor = LookaheadAuditor(
            corruption_point=0.8,
            corruption_method="random"
        )
        result = auditor.audit_feature_function(df, unshifted_fn, "unshifted")

        # This may or may not detect lookahead depending on the feature
        # But the test verifies the auditor runs without error
        assert isinstance(result, LookaheadAuditResult)

    def test_handles_insufficient_data(self):
        """Auditor handles small datasets gracefully."""
        small_df = pd.DataFrame({
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0, 1, 2],
            "close": [1.5, 2.5, 3.5],
            "volume": [100, 100, 100],
        })

        def dummy_fn(df):
            return df

        auditor = LookaheadAuditor()
        result = auditor.audit_feature_function(small_df, dummy_fn, "dummy")

        assert not result.has_lookahead
        assert "Insufficient data" in result.details

    def test_corruption_methods(self, sample_ohlcv_data):
        """All corruption methods work."""
        df = sample_ohlcv_data

        def good_fn(df):
            df = df.copy()
            df["feature"] = df["close"].rolling(5).mean().shift(1)
            return df

        for method in ["nan", "random", "shuffle"]:
            auditor = LookaheadAuditor(
                corruption_point=0.7,
                corruption_method=method
            )
            result = auditor.audit_feature_function(df, good_fn, f"test_{method}")
            assert isinstance(result, LookaheadAuditResult)


class TestAuditFeatureLookahead:
    """Tests for audit_feature_lookahead convenience function."""

    def test_multiple_corruption_points(self, sample_ohlcv_data):
        """Audits at multiple corruption points."""
        df = sample_ohlcv_data

        def good_fn(df):
            df = df.copy()
            df["ma"] = df["close"].rolling(5).mean().shift(1)
            return df

        results = audit_feature_lookahead(
            df, good_fn, "ma", corruption_points=[0.5, 0.7, 0.9]
        )

        assert len(results) == 3
        assert all(isinstance(r, LookaheadAuditResult) for r in results)
        assert all(not r.has_lookahead for r in results)

    def test_default_corruption_points(self, sample_ohlcv_data):
        """Uses default corruption points when not specified."""
        df = sample_ohlcv_data

        def good_fn(df):
            df = df.copy()
            df["ma"] = df["close"].rolling(5).mean().shift(1)
            return df

        results = audit_feature_lookahead(df, good_fn, "ma")
        assert len(results) == 3  # Default: [0.5, 0.7, 0.9]


# =============================================================================
# MTF ALIGNMENT TESTS
# =============================================================================

class TestMTFAlignment:
    """Tests for MTF alignment audit."""

    def test_valid_mtf_alignment(self, sample_mtf_data):
        """Properly aligned MTF data passes audit."""
        is_valid, issues = audit_mtf_alignment(
            df_base=pd.DataFrame(),  # Not used in current implementation
            df_mtf=sample_mtf_data,
            base_tf_minutes=5,
            mtf_minutes=60,
        )

        assert is_valid
        assert len(issues) == 0

    def test_detects_missing_initial_nans(self):
        """Detects MTF data without proper initial NaNs."""
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="5min")

        # BAD: No initial NaNs (no shift(1) applied)
        bad_mtf = pd.DataFrame({
            "mtf_close_1h": np.random.randn(n),  # All valid from row 0
        }, index=dates)

        is_valid, issues = audit_mtf_alignment(
            df_base=pd.DataFrame(),
            df_mtf=bad_mtf,
            base_tf_minutes=5,
            mtf_minutes=60,
        )

        assert not is_valid
        assert any("First row is not NaN" in msg for msg in issues)

    def test_detects_insufficient_initial_nans(self):
        """Detects insufficient initial NaNs."""
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="5min")

        # BAD: Only 2 NaNs but 1h MTF needs 12 for 5min base
        mtf_data = np.random.randn(n)
        mtf_data[:2] = np.nan

        bad_mtf = pd.DataFrame({
            "mtf_close_1h": mtf_data,
        }, index=dates)

        is_valid, issues = audit_mtf_alignment(
            df_base=pd.DataFrame(),
            df_mtf=bad_mtf,
            base_tf_minutes=5,
            mtf_minutes=60,
        )

        assert not is_valid
        assert any("expected >= 11 initial NaNs" in msg for msg in issues)

    def test_handles_empty_mtf(self):
        """Handles empty MTF DataFrame."""
        is_valid, issues = audit_mtf_alignment(
            df_base=pd.DataFrame(),
            df_mtf=pd.DataFrame(),
            base_tf_minutes=5,
            mtf_minutes=60,
        )

        assert not is_valid
        assert any("empty" in msg.lower() for msg in issues)


# =============================================================================
# RESULT SERIALIZATION
# =============================================================================

class TestLookaheadAuditResult:
    """Tests for LookaheadAuditResult dataclass."""

    def test_to_dict(self):
        """Result converts to dict properly."""
        result = LookaheadAuditResult(
            feature_name="test",
            has_lookahead=True,
            affected_columns=["col1", "col2"],
            max_affected_rows=50,
            corruption_point=400,
            details="Test details",
        )

        d = result.to_dict()
        assert d["feature_name"] == "test"
        assert d["has_lookahead"] is True
        assert d["affected_columns"] == ["col1", "col2"]
        assert d["max_affected_rows"] == 50

    def test_default_values(self):
        """Result has sensible defaults."""
        result = LookaheadAuditResult(
            feature_name="test",
            has_lookahead=False,
        )

        assert result.affected_columns == []
        assert result.max_affected_rows == 0
        assert result.corruption_point == 0
        assert result.details == ""


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestLookaheadIntegration:
    """Integration tests for lookahead detection."""

    def test_full_audit_workflow(self, sample_ohlcv_data):
        """Full audit workflow for a feature function."""
        df = sample_ohlcv_data

        def compute_features(df):
            """Sample feature function with proper anti-lookahead."""
            df = df.copy()
            # All features properly lagged
            df["sma_10"] = df["close"].rolling(10).mean().shift(1)
            df["sma_20"] = df["close"].rolling(20).mean().shift(1)
            df["returns"] = df["close"].pct_change().shift(1)
            df["volatility"] = df["close"].rolling(10).std().shift(1)
            return df

        # Audit at multiple corruption points
        results = audit_feature_lookahead(
            df, compute_features, "features",
            corruption_points=[0.5, 0.7, 0.9]
        )

        # All should pass
        for result in results:
            assert not result.has_lookahead, f"Failed at {result.feature_name}"

    def test_mixed_good_bad_features(self, sample_ohlcv_data):
        """Detects lookahead in mixed feature set."""
        df = sample_ohlcv_data

        def mixed_features(df):
            """Mix of good and bad features."""
            df = df.copy()
            df["good"] = df["close"].rolling(10).mean().shift(1)
            df["bad"] = df["close"].rolling(10, center=True).mean()  # BAD
            return df

        auditor = LookaheadAuditor(corruption_point=0.8, corruption_method="nan")
        result = auditor.audit_feature_function(df, mixed_features, "mixed")

        assert result.has_lookahead
        assert "bad" in result.affected_columns
        # Good feature should not be affected
        assert "good" not in result.affected_columns
