"""
Tests for statistical model comparison tests.

Tests cover:
- Diebold-Mariano test for forecast accuracy comparison
- Paired t-test for cross-validation comparison
- Wilcoxon signed-rank test for non-parametric comparison
- compare_models convenience function
"""

import numpy as np
import pytest

from src.validation.statistical_tests import (
    LossFunction,
    StatisticalTestResult,
    compare_models,
    diebold_mariano_test,
    paired_ttest,
    wilcoxon_test,
)


class TestDieboldMarianoTest:
    """Tests for Diebold-Mariano forecast comparison test."""

    def test_identical_predictions_not_significant(self):
        """Identical predictions should show no significant difference."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        pred = y_true + np.random.randn(100) * 0.1

        result = diebold_mariano_test(y_true, pred, pred)

        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "Diebold-Mariano"
        assert not result.significant
        assert result.p_value > 0.05

    def test_significantly_different_predictions(self):
        """Model with much better predictions should be detected."""
        np.random.seed(42)
        y_true = np.random.randn(200)
        pred1 = y_true + np.random.randn(200) * 2.0  # Bad model
        pred2 = y_true + np.random.randn(200) * 0.1  # Good model

        result = diebold_mariano_test(y_true, pred1, pred2)

        assert result.significant
        assert result.p_value < 0.05
        assert "Model 2 significantly better" in result.conclusion

    def test_model1_better(self):
        """Detect when model 1 is significantly better."""
        np.random.seed(42)
        y_true = np.random.randn(200)
        pred1 = y_true + np.random.randn(200) * 0.1  # Good model
        pred2 = y_true + np.random.randn(200) * 2.0  # Bad model

        result = diebold_mariano_test(y_true, pred1, pred2)

        assert result.significant
        assert "Model 1 significantly better" in result.conclusion

    def test_different_loss_functions(self):
        """Test with different loss functions."""
        np.random.seed(42)
        y_true = np.abs(np.random.randn(100)) + 1  # Positive values for MAPE
        pred1 = y_true + np.random.randn(100) * 0.5
        pred2 = y_true + np.random.randn(100) * 0.5

        # MSE
        result_mse = diebold_mariano_test(y_true, pred1, pred2, loss=LossFunction.MSE)
        assert result_mse.test_name == "Diebold-Mariano"

        # MAE
        result_mae = diebold_mariano_test(y_true, pred1, pred2, loss="mae")
        assert result_mae.test_name == "Diebold-Mariano"

        # MAPE
        result_mape = diebold_mariano_test(y_true, pred1, pred2, loss="mape")
        assert result_mape.test_name == "Diebold-Mariano"

    def test_one_sided_alternatives(self):
        """Test one-sided alternative hypotheses."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        pred1 = y_true + np.random.randn(100) * 1.0
        pred2 = y_true + np.random.randn(100) * 0.2

        # Two-sided
        result_two = diebold_mariano_test(y_true, pred1, pred2, alternative="two-sided")
        assert result_two.alternative == "two-sided"

        # Greater (model 2 better)
        result_greater = diebold_mariano_test(y_true, pred1, pred2, alternative="greater")
        assert result_greater.alternative == "greater"

        # Less (model 1 better)
        result_less = diebold_mariano_test(y_true, pred1, pred2, alternative="less")
        assert result_less.alternative == "less"

    def test_forecast_horizon_adjustment(self):
        """Test with multi-step forecast horizon."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        pred1 = y_true + np.random.randn(100) * 0.5
        pred2 = y_true + np.random.randn(100) * 0.5

        # h=1 (default)
        result_h1 = diebold_mariano_test(y_true, pred1, pred2, h=1)

        # h=5
        result_h5 = diebold_mariano_test(y_true, pred1, pred2, h=5)

        # Both should complete without error
        assert result_h1.test_name == "Diebold-Mariano"
        assert result_h5.test_name == "Diebold-Mariano"

    def test_effect_size_computed(self):
        """Effect size should be computed."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        pred1 = y_true + np.random.randn(100) * 0.5
        pred2 = y_true + np.random.randn(100) * 0.5

        result = diebold_mariano_test(y_true, pred1, pred2)

        assert result.effect_size is not None
        assert isinstance(result.effect_size, float)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        np.random.seed(42)
        y_true = np.random.randn(50)
        pred1 = y_true + np.random.randn(50) * 0.5
        pred2 = y_true + np.random.randn(50) * 0.5

        result = diebold_mariano_test(y_true, pred1, pred2)
        result_dict = result.to_dict()

        assert "test_name" in result_dict
        assert "statistic" in result_dict
        assert "p_value" in result_dict
        assert "significant" in result_dict
        assert "effect_size" in result_dict

    def test_length_mismatch_raises(self):
        """Arrays of different lengths should raise."""
        y_true = np.random.randn(100)
        pred1 = np.random.randn(100)
        pred2 = np.random.randn(50)

        with pytest.raises(ValueError, match="same length"):
            diebold_mariano_test(y_true, pred1, pred2)


class TestPairedTTest:
    """Tests for paired t-test."""

    def test_identical_metrics_not_significant(self):
        """Identical metrics should not be significant."""
        metrics = np.array([0.7, 0.72, 0.68, 0.71, 0.73])

        result = paired_ttest(metrics, metrics)

        assert not result.significant
        # With identical inputs, t-test returns NaN p-value (undefined)
        # The key assertion is that it's not significant
        assert np.isnan(result.p_value) or result.p_value > 0.99

    def test_different_metrics_significant(self):
        """Clearly different metrics should be significant."""
        np.random.seed(42)
        metrics1 = np.array([0.80, 0.82, 0.78, 0.81, 0.79, 0.80, 0.82, 0.78])
        metrics2 = np.array([0.60, 0.62, 0.58, 0.61, 0.59, 0.60, 0.62, 0.58])

        result = paired_ttest(metrics1, metrics2)

        assert result.significant
        assert "Model 1 significantly better" in result.conclusion

    def test_confidence_interval(self):
        """Confidence interval should be computed."""
        np.random.seed(42)
        metrics1 = np.array([0.7, 0.72, 0.68, 0.71, 0.73])
        metrics2 = np.array([0.65, 0.67, 0.63, 0.66, 0.68])

        result = paired_ttest(metrics1, metrics2)

        assert result.confidence_interval is not None
        ci_lower, ci_upper = result.confidence_interval
        assert ci_lower < ci_upper

    def test_effect_size_cohens_d(self):
        """Cohen's d effect size should be computed."""
        np.random.seed(42)
        metrics1 = np.array([0.7, 0.72, 0.68, 0.71, 0.73])
        metrics2 = np.array([0.65, 0.67, 0.63, 0.66, 0.68])

        result = paired_ttest(metrics1, metrics2)

        assert result.effect_size is not None
        # Large effect size expected for clearly different metrics
        assert abs(result.effect_size) > 0.5

    def test_one_sided_alternatives(self):
        """Test one-sided alternatives."""
        metrics1 = np.array([0.8, 0.82, 0.78, 0.81, 0.79])
        metrics2 = np.array([0.7, 0.72, 0.68, 0.71, 0.69])

        result_greater = paired_ttest(metrics1, metrics2, alternative="greater")
        result_less = paired_ttest(metrics1, metrics2, alternative="less")

        # Model 1 is better, so "greater" should be more significant
        assert result_greater.p_value < result_less.p_value

    def test_custom_alpha(self):
        """Test with custom alpha level."""
        metrics1 = np.array([0.7, 0.72, 0.68, 0.71, 0.73])
        metrics2 = np.array([0.69, 0.71, 0.67, 0.70, 0.72])

        result_05 = paired_ttest(metrics1, metrics2, alpha=0.05)
        result_01 = paired_ttest(metrics1, metrics2, alpha=0.01)

        assert result_05.alpha == 0.05
        assert result_01.alpha == 0.01

    def test_minimum_samples_required(self):
        """Need at least 2 samples."""
        metrics1 = np.array([0.7])
        metrics2 = np.array([0.65])

        with pytest.raises(ValueError, match="at least 2"):
            paired_ttest(metrics1, metrics2)

    def test_length_mismatch_raises(self):
        """Different length arrays should raise."""
        metrics1 = np.array([0.7, 0.72, 0.68])
        metrics2 = np.array([0.65, 0.67])

        with pytest.raises(ValueError, match="same length"):
            paired_ttest(metrics1, metrics2)


class TestWilcoxonTest:
    """Tests for Wilcoxon signed-rank test."""

    def test_identical_metrics_not_significant(self):
        """Identical metrics should not be significant."""
        metrics = np.array([0.7, 0.72, 0.68, 0.71, 0.73])

        result = wilcoxon_test(metrics, metrics)

        assert not result.significant

    def test_different_metrics_significant(self):
        """Clearly different metrics should be significant."""
        np.random.seed(42)
        metrics1 = np.array([0.80, 0.82, 0.78, 0.81, 0.79, 0.83, 0.77, 0.80])
        metrics2 = np.array([0.60, 0.62, 0.58, 0.61, 0.59, 0.63, 0.57, 0.60])

        result = wilcoxon_test(metrics1, metrics2)

        assert result.significant
        assert "Model 1 significantly better" in result.conclusion

    def test_effect_size_computed(self):
        """Effect size r should be computed."""
        metrics1 = np.array([0.80, 0.82, 0.78, 0.81, 0.79])
        metrics2 = np.array([0.70, 0.72, 0.68, 0.71, 0.69])

        result = wilcoxon_test(metrics1, metrics2)

        assert result.effect_size is not None

    def test_handles_ties(self):
        """Should handle tied values gracefully."""
        metrics1 = np.array([0.7, 0.7, 0.7, 0.8, 0.8])
        metrics2 = np.array([0.6, 0.6, 0.6, 0.7, 0.7])

        result = wilcoxon_test(metrics1, metrics2)

        assert result.test_name == "Wilcoxon signed-rank"

    def test_insufficient_nonzero_differences(self):
        """Should handle case with no nonzero differences."""
        metrics1 = np.array([0.7, 0.7])
        metrics2 = np.array([0.7, 0.7])

        result = wilcoxon_test(metrics1, metrics2)

        assert not result.significant
        assert "Insufficient" in result.conclusion

    def test_one_sided_alternatives(self):
        """Test one-sided alternatives."""
        metrics1 = np.array([0.8, 0.82, 0.78, 0.81, 0.79, 0.80])
        metrics2 = np.array([0.7, 0.72, 0.68, 0.71, 0.69, 0.70])

        result_greater = wilcoxon_test(metrics1, metrics2, alternative="greater")
        result_less = wilcoxon_test(metrics1, metrics2, alternative="less")

        assert result_greater.alternative == "greater"
        assert result_less.alternative == "less"


class TestCompareModels:
    """Tests for the compare_models convenience function."""

    def test_with_predictions(self):
        """Test comparison using predictions (DM test)."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        pred1 = y_true + np.random.randn(100) * 0.5
        pred2 = y_true + np.random.randn(100) * 0.5

        results = compare_models(y_true=y_true, pred1=pred1, pred2=pred2)

        assert "diebold_mariano" in results
        assert results["diebold_mariano"].test_name == "Diebold-Mariano"

    def test_with_metrics(self):
        """Test comparison using metrics (t-test and Wilcoxon)."""
        metrics1 = np.array([0.7, 0.72, 0.68, 0.71, 0.73])
        metrics2 = np.array([0.65, 0.67, 0.63, 0.66, 0.68])

        results = compare_models(metrics1=metrics1, metrics2=metrics2)

        assert "paired_ttest" in results
        assert "wilcoxon" in results

    def test_with_both(self):
        """Test with both predictions and metrics."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        pred1 = y_true + np.random.randn(100) * 0.5
        pred2 = y_true + np.random.randn(100) * 0.5
        metrics1 = np.array([0.7, 0.72, 0.68, 0.71, 0.73])
        metrics2 = np.array([0.65, 0.67, 0.63, 0.66, 0.68])

        results = compare_models(
            y_true=y_true,
            pred1=pred1,
            pred2=pred2,
            metrics1=metrics1,
            metrics2=metrics2,
        )

        assert "diebold_mariano" in results
        assert "paired_ttest" in results
        assert "wilcoxon" in results

    def test_missing_inputs_raises(self):
        """Should raise if neither predictions nor metrics provided."""
        with pytest.raises(ValueError, match="Provide either"):
            compare_models()

    def test_custom_alpha(self):
        """Test with custom alpha level."""
        metrics1 = np.array([0.7, 0.72, 0.68, 0.71, 0.73])
        metrics2 = np.array([0.65, 0.67, 0.63, 0.66, 0.68])

        results = compare_models(metrics1=metrics1, metrics2=metrics2, alpha=0.01)

        assert results["paired_ttest"].alpha == 0.01
        assert results["wilcoxon"].alpha == 0.01
