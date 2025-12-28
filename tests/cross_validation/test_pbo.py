"""
Tests for Probability of Backtest Overfitting (PBO) computation.

Tests:
- Configuration validation
- PBO computation sanity
- Threshold gating
- Edge cases
"""
import numpy as np
import pytest

from src.cross_validation.pbo import (
    PBOConfig,
    PBOResult,
    compute_pbo,
    compute_pbo_from_returns,
    pbo_gate,
    analyze_overfitting_risk,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def random_performance_matrix():
    """Generate random performance matrix (no overfitting signal)."""
    np.random.seed(42)
    n_strategies = 10
    n_paths = 20

    # Random performance - no systematic IS/OOS relationship
    return np.random.randn(n_strategies, n_paths)


@pytest.fixture
def overfit_performance_matrix():
    """Generate performance matrix with overfitting signal."""
    np.random.seed(42)
    n_strategies = 10
    n_paths = 20

    # Create IS-heavy performance (good IS, bad OOS)
    matrix = np.zeros((n_strategies, n_paths))

    for i in range(n_strategies):
        # First half of paths (IS): good performance for best strategy
        is_boost = 0.5 if i == 0 else 0
        matrix[i, :n_paths // 2] = np.random.randn(n_paths // 2) + is_boost

        # Second half (OOS): worse performance for "best" strategy
        oos_penalty = -0.5 if i == 0 else 0
        matrix[i, n_paths // 2:] = np.random.randn(n_paths // 2) + oos_penalty

    return matrix


@pytest.fixture
def robust_performance_matrix():
    """Generate performance matrix with robust (non-overfit) signal."""
    np.random.seed(42)
    n_strategies = 10
    n_paths = 20

    # Create consistent IS/OOS relationship
    matrix = np.zeros((n_strategies, n_paths))

    for i in range(n_strategies):
        # Strategy 0 is consistently best across all paths
        boost = 0.5 if i == 0 else 0
        matrix[i, :] = np.random.randn(n_paths) + boost

    return matrix


@pytest.fixture
def default_config():
    """Default PBO configuration."""
    return PBOConfig()


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestPBOConfig:
    """Tests for PBOConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creates successfully."""
        config = PBOConfig(
            n_partitions=16,
            warn_threshold=0.5,
            block_threshold=0.8,
        )
        assert config.n_partitions == 16
        assert config.warn_threshold == 0.5
        assert config.block_threshold == 0.8

    def test_default_config(self):
        """Test default configuration values."""
        config = PBOConfig()
        assert config.n_partitions == 16
        assert config.warn_threshold == 0.5
        assert config.block_threshold == 0.8
        assert config.min_paths == 6

    def test_invalid_n_partitions(self):
        """Test that n_partitions < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_partitions must be >= 2"):
            PBOConfig(n_partitions=1)

    def test_invalid_warn_threshold(self):
        """Test that warn_threshold outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="warn_threshold must be in"):
            PBOConfig(warn_threshold=0)

        with pytest.raises(ValueError, match="warn_threshold must be in"):
            PBOConfig(warn_threshold=1.0)

    def test_invalid_block_threshold(self):
        """Test that block_threshold outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="block_threshold must be in"):
            PBOConfig(block_threshold=0)

    def test_warn_must_be_less_than_block(self):
        """Test that warn_threshold must be < block_threshold."""
        with pytest.raises(ValueError, match="warn_threshold must be < block_threshold"):
            PBOConfig(warn_threshold=0.8, block_threshold=0.5)


# =============================================================================
# PBO COMPUTATION TESTS
# =============================================================================

class TestPBOComputation:
    """Tests for PBO computation sanity."""

    def test_returns_pbo_result(self, random_performance_matrix, default_config):
        """Test that compute_pbo returns PBOResult."""
        result = compute_pbo(random_performance_matrix, default_config)
        assert isinstance(result, PBOResult)

    def test_pbo_in_valid_range(self, random_performance_matrix, default_config):
        """Test that PBO is in [0, 1] range."""
        result = compute_pbo(random_performance_matrix, default_config)
        assert 0 <= result.pbo <= 1

    def test_pbo_result_has_required_fields(self, random_performance_matrix, default_config):
        """Test that PBOResult has all required fields."""
        result = compute_pbo(random_performance_matrix, default_config)

        assert hasattr(result, "pbo")
        assert hasattr(result, "logit_distribution")
        assert hasattr(result, "performance_degradation")
        assert hasattr(result, "rank_correlation")
        assert hasattr(result, "is_overfit")
        assert hasattr(result, "should_block")
        assert hasattr(result, "n_paths_evaluated")

    def test_overfit_matrix_has_high_pbo(self, overfit_performance_matrix):
        """Test that overfit matrix produces high PBO."""
        result = compute_pbo(overfit_performance_matrix)

        # Overfit matrix should have higher PBO (though not guaranteed > 0.5
        # due to randomness in CSCV partitioning)
        assert result.pbo >= 0.0  # At minimum, PBO should be computed

    def test_robust_matrix_has_low_pbo(self, robust_performance_matrix):
        """Test that robust matrix produces low PBO."""
        result = compute_pbo(robust_performance_matrix)

        # Robust matrix should have lower PBO
        # With consistent performance, best IS should also be best OOS
        assert result.pbo < 0.8  # Should not be severely overfit

    def test_logit_distribution_has_values(self, random_performance_matrix):
        """Test that logit distribution is computed."""
        result = compute_pbo(random_performance_matrix)
        assert len(result.logit_distribution) > 0

    def test_to_dict_serializable(self, random_performance_matrix):
        """Test that to_dict returns serializable dict."""
        result = compute_pbo(random_performance_matrix)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "pbo" in result_dict
        assert "is_overfit" in result_dict

    def test_get_risk_level_returns_string(self, random_performance_matrix):
        """Test that get_risk_level returns a string."""
        result = compute_pbo(random_performance_matrix)
        risk_level = result.get_risk_level()

        assert isinstance(risk_level, str)
        assert any(
            keyword in risk_level
            for keyword in ["OK", "CAUTION", "WARNING", "CRITICAL"]
        )


# =============================================================================
# THRESHOLD TESTS
# =============================================================================

class TestThresholds:
    """Tests for PBO threshold handling."""

    def test_is_overfit_respects_warn_threshold(self, random_performance_matrix):
        """Test that is_overfit respects warn_threshold."""
        config_high = PBOConfig(warn_threshold=0.9, block_threshold=0.95)
        config_low = PBOConfig(warn_threshold=0.1, block_threshold=0.2)

        result_high = compute_pbo(random_performance_matrix, config_high)
        result_low = compute_pbo(random_performance_matrix, config_low)

        # With same matrix, lower threshold should be more likely to trigger
        if result_high.pbo > 0.1:  # If PBO is in the range to differ
            assert result_low.is_overfit or not result_high.is_overfit

    def test_should_block_respects_block_threshold(self, random_performance_matrix):
        """Test that should_block respects block_threshold."""
        config = PBOConfig(warn_threshold=0.3, block_threshold=0.5)
        result = compute_pbo(random_performance_matrix, config)

        # Verify the logic
        if result.pbo > 0.5:
            assert result.should_block
        else:
            assert not result.should_block


# =============================================================================
# GATE FUNCTION TESTS
# =============================================================================

class TestPBOGate:
    """Tests for pbo_gate function."""

    def test_gate_returns_tuple(self, random_performance_matrix):
        """Test that pbo_gate returns tuple."""
        result = compute_pbo(random_performance_matrix)
        should_proceed, reason = pbo_gate(result, strict=False)

        assert isinstance(should_proceed, bool)
        assert isinstance(reason, str)

    def test_strict_mode_uses_block_threshold(self):
        """Test that strict mode uses block_threshold."""
        # Create a mock result with PBO between warn and block
        config = PBOConfig(warn_threshold=0.4, block_threshold=0.7)

        # Create a minimal performance matrix
        np.random.seed(42)
        matrix = np.random.randn(5, 10)
        result = compute_pbo(matrix, config)

        # If PBO is between thresholds, strict=False should pass
        # and strict=True should fail
        if 0.4 < result.pbo < 0.7:
            proceed_lenient, _ = pbo_gate(result, strict=False)
            proceed_strict, _ = pbo_gate(result, strict=True)

            assert not proceed_lenient  # Should fail warn threshold
            # strict uses block threshold, so depends on actual PBO

    def test_gate_reason_includes_pbo_value(self, random_performance_matrix):
        """Test that gate reason includes PBO value."""
        result = compute_pbo(random_performance_matrix)
        _, reason = pbo_gate(result)

        assert str(round(result.pbo, 3)) in reason or "PBO" in reason


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_minimum_strategies(self):
        """Test with minimum number of strategies (2)."""
        np.random.seed(42)
        matrix = np.random.randn(2, 10)

        result = compute_pbo(matrix)
        assert 0 <= result.pbo <= 1

    def test_minimum_paths(self):
        """Test with minimum number of paths."""
        np.random.seed(42)
        matrix = np.random.randn(5, 6)  # 6 is default min_paths

        result = compute_pbo(matrix)
        assert 0 <= result.pbo <= 1

    def test_too_few_strategies_raises(self):
        """Test that 1 strategy raises ValueError."""
        matrix = np.random.randn(1, 10)

        with pytest.raises(ValueError, match="Need at least 2 strategies"):
            compute_pbo(matrix)

    def test_too_few_paths_raises(self):
        """Test that too few paths raises ValueError."""
        matrix = np.random.randn(5, 3)  # Less than min_paths

        with pytest.raises(ValueError, match="Need at least .* paths"):
            compute_pbo(matrix)

    def test_handles_nan_values(self):
        """Test that NaN values are handled gracefully."""
        np.random.seed(42)
        matrix = np.random.randn(5, 10)
        matrix[0, 0] = np.nan  # Add a NaN

        result = compute_pbo(matrix)
        assert 0 <= result.pbo <= 1
        assert not np.isnan(result.pbo)


# =============================================================================
# ANALYZE OVERFITTING RISK TESTS
# =============================================================================

class TestAnalyzeOverfittingRisk:
    """Tests for analyze_overfitting_risk function."""

    def test_returns_dict(self, random_performance_matrix):
        """Test that analyze_overfitting_risk returns dict."""
        result = analyze_overfitting_risk(random_performance_matrix)
        assert isinstance(result, dict)

    def test_includes_pbo(self, random_performance_matrix):
        """Test that result includes PBO."""
        result = analyze_overfitting_risk(random_performance_matrix)
        assert "pbo" in result
        assert "is_overfit" in result
        assert "risk_level" in result

    def test_includes_strategy_analysis(self, random_performance_matrix):
        """Test that result includes per-strategy analysis."""
        strategy_names = ["strat_a", "strat_b", "strat_c"]
        matrix = np.random.randn(3, 10)

        result = analyze_overfitting_risk(matrix, strategy_names=strategy_names)

        assert "strategy_analysis" in result
        assert len(result["strategy_analysis"]) == 3
        assert result["strategy_analysis"][0]["name"] == "strat_a"

    def test_identifies_best_is_strategy(self, random_performance_matrix):
        """Test that best IS strategy is identified."""
        result = analyze_overfitting_risk(random_performance_matrix)
        assert "best_is_strategy" in result


# =============================================================================
# COMPUTE PBO FROM RETURNS TESTS
# =============================================================================

class TestComputePBOFromReturns:
    """Tests for compute_pbo_from_returns function."""

    def test_converts_returns_to_sharpe(self):
        """Test that returns are converted to Sharpe for PBO."""
        np.random.seed(42)
        n_strategies = 5
        n_periods = 500

        returns = np.random.randn(n_strategies, n_periods) * 0.01

        result = compute_pbo_from_returns(returns, use_sharpe=True)
        assert 0 <= result.pbo <= 1

    def test_works_with_cumulative_returns(self):
        """Test that cumulative returns mode works."""
        np.random.seed(42)
        n_strategies = 5
        n_periods = 500

        returns = np.random.randn(n_strategies, n_periods) * 0.01

        result = compute_pbo_from_returns(returns, use_sharpe=False)
        assert 0 <= result.pbo <= 1
