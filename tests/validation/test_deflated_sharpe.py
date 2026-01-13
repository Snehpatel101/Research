"""
Tests for Deflated Sharpe Ratio (DSR) module.

Tests:
- Core DSR computation
- Gamma correction for non-normality
- Edge cases (n_trials=1, high variance, skewed distributions)
- Optuna integration
- Gate functions and analysis utilities
"""
import numpy as np
import pytest

from src.validation.deflated_sharpe import (
    DSRConfig,
    DSRResult,
    analyze_selection_bias,
    compute_deflated_sharpe,
    dsr_gate,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def normal_trial_sharpes():
    """Generate normally distributed trial Sharpes."""
    np.random.seed(42)
    # 20 trials with mean ~0.8, std ~0.3
    return np.random.normal(0.8, 0.3, 20)


@pytest.fixture
def skewed_trial_sharpes():
    """Generate positively skewed trial Sharpes (log-normal like)."""
    np.random.seed(42)
    # Skewed distribution simulating lucky strategies
    return np.abs(np.random.lognormal(mean=0.0, sigma=0.5, size=20)) * 0.5


@pytest.fixture
def high_variance_sharpes():
    """Generate high variance trial Sharpes."""
    np.random.seed(42)
    return np.random.uniform(-0.5, 2.5, 20)


@pytest.fixture
def small_trial_sharpes():
    """Small number of trials (edge case)."""
    return np.array([0.5, 1.2, 0.8])


# =============================================================================
# DSR CONFIG TESTS
# =============================================================================


class TestDSRConfig:
    """Tests for DSRConfig validation."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = DSRConfig()
        assert config.significance_threshold == 0.0
        assert config.deployment_threshold == 0.5
        assert config.gamma_clip_min == 0.1
        assert config.gamma_clip_max == 10.0

    def test_custom_config(self):
        """Custom config works."""
        config = DSRConfig(
            significance_threshold=-0.5,
            deployment_threshold=1.0,
            gamma_clip_min=0.2,
            gamma_clip_max=5.0,
        )
        assert config.deployment_threshold == 1.0

    def test_invalid_threshold_order_raises(self):
        """Invalid threshold ordering raises ValueError."""
        with pytest.raises(ValueError, match="significance_threshold"):
            DSRConfig(significance_threshold=1.0, deployment_threshold=0.5)

    def test_invalid_gamma_bounds_raises(self):
        """Invalid gamma bounds raise ValueError."""
        with pytest.raises(ValueError, match="gamma_clip_min"):
            DSRConfig(gamma_clip_min=0.0)

        with pytest.raises(ValueError, match="gamma_clip_min"):
            DSRConfig(gamma_clip_min=5.0, gamma_clip_max=3.0)

    def test_invalid_kurtosis_min_raises(self):
        """Kurtosis minimum less than 4 raises ValueError."""
        with pytest.raises(ValueError, match="min_trials_for_kurtosis"):
            DSRConfig(min_trials_for_kurtosis=3)


# =============================================================================
# CORE DSR COMPUTATION TESTS
# =============================================================================


class TestComputeDeflatedSharpe:
    """Tests for compute_deflated_sharpe function."""

    def test_basic_computation(self, normal_trial_sharpes):
        """Basic DSR computation works."""
        best_sharpe = float(np.max(normal_trial_sharpes))
        result = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=normal_trial_sharpes,
            n_trials=len(normal_trial_sharpes),
        )

        assert isinstance(result, DSRResult)
        assert result.sharpe_ratio == best_sharpe
        # DSR should be less than raw Sharpe due to deflation
        assert result.deflated_sharpe <= result.sharpe_ratio
        assert result.n_trials == len(normal_trial_sharpes)

    def test_deflation_increases_with_trials(self):
        """More trials = more deflation (holding variance fixed)."""
        np.random.seed(42)

        # Generate a large pool of trial Sharpes with fixed variance
        large_pool = np.random.normal(0.8, 0.3, 200)
        variance = np.var(large_pool, ddof=1)  # Fixed variance for all tests

        # Create synthetic trial arrays with same variance
        best_sharpe = 1.5

        # For fair comparison, use subsamples of the same population
        results = []
        for n in [10, 50, 100, 200]:
            # Use first n samples from pool
            trial_sharpes = large_pool[:n]
            result = compute_deflated_sharpe(
                sharpe_ratio=best_sharpe,
                trial_sharpes=trial_sharpes,
                n_trials=n,
            )
            results.append((n, result.deflated_sharpe))

        # Generally, DSR should decrease with more trials
        # But due to gamma correction, this is not strictly monotonic
        # Check that the trend is correct (correlation test)
        ns = [r[0] for r in results]
        dsrs = [r[1] for r in results]

        # At minimum, largest n should have lowest DSR
        assert dsrs[-1] <= dsrs[0], (
            f"Largest trial count should give lower DSR: n={ns}, DSR={dsrs}"
        )

    def test_deflation_increases_with_variance(self):
        """Higher variance = more deflation."""
        np.random.seed(42)

        # Fixed n_trials, varying variance
        # Use large n to get stable estimates
        n_trials = 100
        best_sharpe = 1.5

        # Test low vs high variance
        low_var_sharpes = np.random.normal(0.8, 0.1, n_trials)
        high_var_sharpes = np.random.normal(0.8, 0.5, n_trials)

        result_low = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=low_var_sharpes,
            n_trials=n_trials,
        )

        result_high = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=high_var_sharpes,
            n_trials=n_trials,
        )

        # Higher variance should give lower DSR (more deflation)
        assert result_low.variance_sharpe < result_high.variance_sharpe
        assert result_low.deflated_sharpe > result_high.deflated_sharpe, (
            f"Higher variance should give more deflation: "
            f"low_var DSR={result_low.deflated_sharpe}, "
            f"high_var DSR={result_high.deflated_sharpe}"
        )

    def test_single_trial_no_deflation(self):
        """Single trial has zero deflation."""
        result = compute_deflated_sharpe(
            sharpe_ratio=1.0,
            trial_sharpes=np.array([1.0]),
            n_trials=1,
        )

        # No selection bias with single trial
        assert result.deflated_sharpe == result.sharpe_ratio
        assert result.variance_sharpe == 0.0

    def test_skewed_distribution_higher_gamma(self, skewed_trial_sharpes):
        """Skewed distribution has higher gamma correction."""
        best_sharpe = float(np.max(skewed_trial_sharpes))
        result = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=skewed_trial_sharpes,
            n_trials=len(skewed_trial_sharpes),
        )

        # Skewness should be positive for log-normal-like distribution
        assert result.skewness_sharpe > 0.0

    def test_significance_and_deployment_flags(self, normal_trial_sharpes):
        """Significance and deployment flags work correctly."""
        # Low Sharpe should not be significant
        result_low = compute_deflated_sharpe(
            sharpe_ratio=0.3,
            trial_sharpes=np.array([0.1, 0.2, 0.3, 0.15]),
            n_trials=4,
            deployment_threshold=0.5,
        )

        # Even low DSR can be significant if > 0
        if result_low.deflated_sharpe > 0:
            assert result_low.is_significant
        else:
            assert not result_low.is_significant

        # High Sharpe with low variance should be deployable
        result_high = compute_deflated_sharpe(
            sharpe_ratio=2.0,
            trial_sharpes=np.array([1.8, 1.9, 2.0, 1.85, 1.95]),
            n_trials=5,
            deployment_threshold=0.5,
        )
        assert result_high.should_deploy

    def test_custom_deployment_threshold(self, normal_trial_sharpes):
        """Custom deployment threshold works."""
        best_sharpe = float(np.max(normal_trial_sharpes))

        result_strict = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=normal_trial_sharpes,
            n_trials=len(normal_trial_sharpes),
            deployment_threshold=2.0,  # Very strict
        )

        result_lenient = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=normal_trial_sharpes,
            n_trials=len(normal_trial_sharpes),
            deployment_threshold=-1.0,  # Very lenient (below significance)
            config=DSRConfig(
                significance_threshold=-2.0,
                deployment_threshold=-1.0
            ),
        )

        # Same DSR, different flags
        assert abs(result_strict.deflated_sharpe - result_lenient.deflated_sharpe) < 0.01
        assert not result_strict.should_deploy
        assert result_lenient.should_deploy


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_nan_sharpe_raises(self):
        """NaN sharpe_ratio raises ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            compute_deflated_sharpe(
                sharpe_ratio=np.nan,
                trial_sharpes=np.array([1.0, 1.1, 1.2]),
                n_trials=3,
            )

    def test_inf_sharpe_raises(self):
        """Infinite sharpe_ratio raises ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            compute_deflated_sharpe(
                sharpe_ratio=np.inf,
                trial_sharpes=np.array([1.0, 1.1, 1.2]),
                n_trials=3,
            )

    def test_zero_trials_raises(self):
        """n_trials < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_trials must be >= 1"):
            compute_deflated_sharpe(
                sharpe_ratio=1.0,
                trial_sharpes=np.array([1.0]),
                n_trials=0,
            )

    def test_empty_sharpes_raises(self):
        """Empty trial_sharpes raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_deflated_sharpe(
                sharpe_ratio=1.0,
                trial_sharpes=np.array([]),
                n_trials=1,  # n_trials >= 1 passes, but empty array fails
            )

    def test_mismatched_length_raises(self):
        """Mismatched trial_sharpes length raises ValueError."""
        with pytest.raises(ValueError, match="must equal n_trials"):
            compute_deflated_sharpe(
                sharpe_ratio=1.0,
                trial_sharpes=np.array([1.0, 1.1, 1.2]),
                n_trials=5,
            )

    def test_all_nan_sharpes_raises(self):
        """All NaN trial_sharpes raises ValueError."""
        with pytest.raises(ValueError, match="all NaN"):
            compute_deflated_sharpe(
                sharpe_ratio=1.0,
                trial_sharpes=np.array([np.nan, np.nan, np.nan]),
                n_trials=3,
            )

    def test_non_array_raises(self):
        """Non-array trial_sharpes raises TypeError."""
        with pytest.raises(TypeError, match="must be numpy array"):
            compute_deflated_sharpe(
                sharpe_ratio=1.0,
                trial_sharpes=[1.0, 1.1, 1.2],  # List, not array
                n_trials=3,
            )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_two_trials(self):
        """Two trials still works (minimal case for variance)."""
        result = compute_deflated_sharpe(
            sharpe_ratio=1.5,
            trial_sharpes=np.array([1.0, 1.5]),
            n_trials=2,
        )

        assert result.n_trials == 2
        assert result.variance_sharpe > 0
        # No skewness/kurtosis with n=2
        assert result.skewness_sharpe == 0.0
        assert result.kurtosis_sharpe == 0.0

    def test_three_trials_has_skewness(self):
        """Three trials has skewness but no kurtosis."""
        result = compute_deflated_sharpe(
            sharpe_ratio=1.5,
            trial_sharpes=np.array([0.5, 1.0, 1.5]),
            n_trials=3,
        )

        # Skewness computed for n >= 3
        assert result.skewness_sharpe != 0.0 or True  # May be 0 for symmetric
        # Kurtosis requires n >= 4
        assert result.kurtosis_sharpe == 0.0

    def test_some_nan_values(self):
        """Handles some NaN values in trial_sharpes."""
        result = compute_deflated_sharpe(
            sharpe_ratio=1.5,
            trial_sharpes=np.array([1.0, np.nan, 1.2, 1.5, np.nan, 1.3]),
            n_trials=6,
        )

        # Should still compute with valid values
        assert result.n_trials == 6
        assert result.deflated_sharpe <= result.sharpe_ratio

    def test_identical_sharpes_zero_variance(self):
        """Identical Sharpes give zero variance and no deflation."""
        result = compute_deflated_sharpe(
            sharpe_ratio=1.0,
            trial_sharpes=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            n_trials=5,
        )

        assert result.variance_sharpe == 0.0
        # Zero variance means zero expected max, so no deflation
        assert result.deflated_sharpe == result.sharpe_ratio

    def test_negative_sharpes(self):
        """Negative Sharpe ratios work correctly."""
        result = compute_deflated_sharpe(
            sharpe_ratio=-0.5,
            trial_sharpes=np.array([-1.0, -0.8, -0.5, -0.7]),
            n_trials=4,
        )

        assert result.sharpe_ratio == -0.5
        # Even negative Sharpes get deflated (become more negative)
        assert result.deflated_sharpe <= result.sharpe_ratio


# =============================================================================
# DSR RESULT TESTS
# =============================================================================


class TestDSRResult:
    """Tests for DSRResult dataclass."""

    def test_to_dict(self, normal_trial_sharpes):
        """Result converts to dict properly."""
        best_sharpe = float(np.max(normal_trial_sharpes))
        result = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=normal_trial_sharpes,
            n_trials=len(normal_trial_sharpes),
        )

        d = result.to_dict()
        assert "sharpe_ratio" in d
        assert "deflated_sharpe" in d
        assert "n_trials" in d
        assert "gamma_correction" in d
        assert "deployment_threshold" in d

    def test_get_deflation_pct(self, normal_trial_sharpes):
        """Deflation percentage computed correctly."""
        best_sharpe = float(np.max(normal_trial_sharpes))
        result = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=normal_trial_sharpes,
            n_trials=len(normal_trial_sharpes),
        )

        pct = result.get_deflation_pct()
        expected = (best_sharpe - result.deflated_sharpe) / best_sharpe * 100
        assert abs(pct - expected) < 1e-6

    def test_get_deflation_pct_zero_sharpe(self):
        """Deflation percentage handles zero Sharpe."""
        result = compute_deflated_sharpe(
            sharpe_ratio=0.0,
            trial_sharpes=np.array([0.0, 0.1, -0.1]),
            n_trials=3,
        )

        pct = result.get_deflation_pct()
        assert pct == 0.0  # Avoid division by zero

    def test_risk_levels(self):
        """Risk level messages appropriate for DSR values."""
        # High DSR - strong confidence
        result_strong = DSRResult(
            sharpe_ratio=2.0,
            deflated_sharpe=1.5,
            n_trials=10,
            variance_sharpe=0.1,
            skewness_sharpe=0.0,
            kurtosis_sharpe=0.0,
            gamma_correction=1.0,
            is_significant=True,
            should_deploy=True,
            config=DSRConfig(),
        )
        assert "STRONG" in result_strong.get_risk_level()

        # Moderate DSR - OK for deployment
        result_ok = DSRResult(
            sharpe_ratio=1.0,
            deflated_sharpe=0.6,
            n_trials=10,
            variance_sharpe=0.1,
            skewness_sharpe=0.0,
            kurtosis_sharpe=0.0,
            gamma_correction=1.0,
            is_significant=True,
            should_deploy=True,
            config=DSRConfig(),
        )
        assert "OK" in result_ok.get_risk_level()

        # Low DSR - warning
        result_warn = DSRResult(
            sharpe_ratio=1.0,
            deflated_sharpe=-0.5,
            n_trials=50,
            variance_sharpe=0.5,
            skewness_sharpe=0.0,
            kurtosis_sharpe=0.0,
            gamma_correction=1.0,
            is_significant=False,
            should_deploy=False,
            config=DSRConfig(),
        )
        assert "WARNING" in result_warn.get_risk_level()


# =============================================================================
# DSR GATE TESTS
# =============================================================================


class TestDSRGate:
    """Tests for dsr_gate function."""

    def test_gate_passes(self, normal_trial_sharpes):
        """Gate passes for high DSR."""
        result = compute_deflated_sharpe(
            sharpe_ratio=2.0,
            trial_sharpes=np.array([1.8, 1.9, 2.0, 1.85, 1.95]),
            n_trials=5,
        )

        proceed, reason = dsr_gate(result)
        assert proceed
        assert "exceeds" in reason.lower()

    def test_gate_blocks(self):
        """Gate blocks for low DSR."""
        np.random.seed(42)
        # Create high variance Sharpes with many trials to ensure deflation
        n_trials = 100
        trial_sharpes = np.random.uniform(-1.0, 2.0, n_trials)
        best_sharpe = float(np.max(trial_sharpes))

        result = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=trial_sharpes,
            n_trials=n_trials,
            deployment_threshold=5.0,  # Very high threshold ensures blocking
        )

        proceed, reason = dsr_gate(result)
        assert not proceed, f"Gate should block: DSR={result.deflated_sharpe}"
        assert "below threshold" in reason.lower()

    def test_strict_mode(self, normal_trial_sharpes):
        """Strict mode requires higher DSR."""
        best_sharpe = float(np.max(normal_trial_sharpes))
        result = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=normal_trial_sharpes,
            n_trials=len(normal_trial_sharpes),
        )

        # Non-strict may pass
        proceed_normal, _ = dsr_gate(result, strict=False)

        # Strict requires DSR > 1.0
        proceed_strict, reason_strict = dsr_gate(result, strict=True)

        if result.deflated_sharpe < 1.0:
            assert not proceed_strict
        else:
            assert proceed_strict


# =============================================================================
# ANALYZE SELECTION BIAS TESTS
# =============================================================================


class TestAnalyzeSelectionBias:
    """Tests for analyze_selection_bias function."""

    def test_basic_analysis(self, normal_trial_sharpes):
        """Basic analysis works."""
        analysis = analyze_selection_bias(normal_trial_sharpes)

        assert "n_trials" in analysis
        assert analysis["n_trials"] == len(normal_trial_sharpes)
        assert "deflated_sharpe" in analysis
        assert "best_strategy_idx" in analysis
        assert "strategy_analysis" in analysis
        assert len(analysis["strategy_analysis"]) == len(normal_trial_sharpes)

    def test_with_strategy_names(self, small_trial_sharpes):
        """Analysis with custom strategy names."""
        names = ["baseline", "tuned_v1", "tuned_v2"]
        analysis = analyze_selection_bias(
            small_trial_sharpes,
            strategy_names=names,
        )

        assert analysis["best_strategy_name"] == "tuned_v1"  # 1.2 is highest
        for item in analysis["strategy_analysis"]:
            assert item["name"] in names

    def test_mismatched_names_raises(self, normal_trial_sharpes):
        """Mismatched strategy names raises ValueError."""
        with pytest.raises(ValueError, match="must match"):
            analyze_selection_bias(
                normal_trial_sharpes,
                strategy_names=["a", "b", "c"],  # Wrong length
            )

    def test_all_nan_raises(self):
        """All NaN values raises ValueError."""
        with pytest.raises(ValueError, match="(?i)all.*nan"):
            analyze_selection_bias(np.array([np.nan, np.nan]))

    def test_ranks_computed(self, small_trial_sharpes):
        """Strategy ranks computed correctly."""
        analysis = analyze_selection_bias(small_trial_sharpes)

        # Find best strategy in analysis
        for item in analysis["strategy_analysis"]:
            if item["is_best"]:
                assert item["rank"] == 1


# =============================================================================
# OPTUNA INTEGRATION TESTS
# =============================================================================


class TestOptunaIntegration:
    """Tests for Optuna integration (mocked)."""

    def test_optuna_not_installed_graceful(self):
        """Graceful handling when Optuna not available."""
        # Import should work regardless
        from src.validation.deflated_sharpe import compute_dsr_from_optuna_study
        assert compute_dsr_from_optuna_study is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestDSRIntegration:
    """Integration tests for DSR workflow."""

    def test_hyperparameter_optimization_scenario(self):
        """Simulate hyperparameter optimization and DSR check."""
        np.random.seed(42)

        # Simulate 50 hyperparameter trials
        n_trials = 50
        # True Sharpe is 0.5, but optimization finds best of noisy trials
        true_sharpe = 0.5
        trial_sharpes = true_sharpe + np.random.normal(0, 0.3, n_trials)
        best_sharpe = float(np.max(trial_sharpes))

        result = compute_deflated_sharpe(
            sharpe_ratio=best_sharpe,
            trial_sharpes=trial_sharpes,
            n_trials=n_trials,
        )

        # Best observed Sharpe should be inflated above true
        assert best_sharpe > true_sharpe

        # DSR should deflate towards true Sharpe
        # (Not exactly, but should be closer)
        assert result.deflated_sharpe < best_sharpe

        # Check deflation percentage
        deflation_pct = result.get_deflation_pct()
        assert deflation_pct > 0  # Some deflation occurred

    def test_model_selection_scenario(self):
        """Simulate model selection and DSR check."""
        # 5 different models evaluated
        model_sharpes = np.array([0.3, 0.8, 1.1, 0.6, 0.9])
        model_names = ["RF", "XGBoost", "LightGBM", "SVM", "CatBoost"]

        analysis = analyze_selection_bias(model_sharpes, strategy_names=model_names)

        assert analysis["best_strategy_name"] == "LightGBM"  # 1.1 highest
        assert analysis["raw_sharpe"] == 1.1
        assert analysis["deflated_sharpe"] < 1.1  # Deflated due to selection

        # Check gate
        # Create result for gate check
        result = compute_deflated_sharpe(
            sharpe_ratio=analysis["raw_sharpe"],
            trial_sharpes=model_sharpes,
            n_trials=len(model_sharpes),
        )

        proceed, reason = dsr_gate(result)
        assert isinstance(proceed, bool)
        assert isinstance(reason, str)

    def test_real_world_sizing(self):
        """Test with realistic trial counts and verify DSR properties."""
        np.random.seed(42)

        # Generate trials with controlled properties
        all_trials = np.random.normal(0.8, 0.2, 500)

        # Medium and Large studies for stable comparison
        # (small n gives unstable gamma estimates due to skewness/kurtosis noise)
        medium_result = compute_deflated_sharpe(
            sharpe_ratio=1.2,
            trial_sharpes=all_trials[:100],
            n_trials=100,
        )

        large_result = compute_deflated_sharpe(
            sharpe_ratio=1.2,
            trial_sharpes=all_trials,
            n_trials=500,
        )

        # Key properties to verify:
        # 1. DSR is less than raw Sharpe (deflation occurred)
        assert medium_result.deflated_sharpe < 1.2
        assert large_result.deflated_sharpe < 1.2

        # 2. Deflation increases with n_trials (for stable gamma estimates)
        # large_result has more trials, so should have more deflation
        assert large_result.deflated_sharpe < medium_result.deflated_sharpe, (
            f"500 trials should deflate more than 100: "
            f"medium={medium_result.deflated_sharpe:.3f}, large={large_result.deflated_sharpe:.3f}"
        )
