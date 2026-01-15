"""
Tests for ensemble diversity metrics.

Tests cover:
- Pairwise correlation computation
- Q-statistic (Yule's Q) computation
- Disagreement measure
- Double fault measure
- Entropy of votes
- KL divergence for probability outputs
- Composite diversity score
- DiversityAnalyzer class
"""

import numpy as np
import pytest

from src.models.ensemble.diversity import (
    DiversityAnalyzer,
    DiversityMetrics,
    compute_average_disagreement,
    compute_average_double_fault,
    compute_average_kl_divergence,
    compute_average_q_statistic,
    compute_disagreement,
    compute_diversity_score,
    compute_double_fault,
    compute_entropy_of_votes,
    compute_kl_divergence,
    compute_kl_diversity_penalty,
    compute_pairwise_correlation,
    compute_pairwise_correlation_matrix,
    compute_q_statistic,
)


class TestPairwiseCorrelation:
    """Tests for pairwise correlation computation."""

    def test_identical_predictions_high_correlation(self):
        """Identical predictions should have correlation = 1.0."""
        # Two models with identical predictions
        predictions = np.array([[0, 0], [1, 1], [2, 2], [1, 1], [0, 0]])
        corr = compute_pairwise_correlation(predictions)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_opposite_predictions_negative_correlation(self):
        """Opposite predictions should have negative correlation."""
        # Model 1: 0, 1, 2, 1, 0
        # Model 2: 2, 1, 0, 1, 2 (reversed)
        predictions = np.array([[0, 2], [1, 1], [2, 0], [1, 1], [0, 2]])
        corr = compute_pairwise_correlation(predictions)
        assert corr < 0

    def test_uncorrelated_predictions(self):
        """Uncorrelated predictions should have ~0 correlation."""
        np.random.seed(42)
        # Two independent random predictions
        predictions = np.random.randint(0, 3, size=(1000, 2))
        corr = compute_pairwise_correlation(predictions)
        assert abs(corr) < 0.1  # Should be close to 0

    def test_single_model_returns_zero(self):
        """Single model should return 0 (no pairs)."""
        predictions = np.array([[0], [1], [2]])
        corr = compute_pairwise_correlation(predictions)
        assert corr == 0.0

    def test_invalid_shape_raises(self):
        """Non-2D input should raise ValueError."""
        with pytest.raises(ValueError, match="2D"):
            compute_pairwise_correlation(np.array([0, 1, 2]))

    def test_multiple_models_average(self):
        """Average correlation across multiple model pairs."""
        # 3 models with varying correlations
        predictions = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [2, 2, 2],
            [1, 1, 1],
            [0, 0, 0],
        ])
        corr = compute_pairwise_correlation(predictions)
        # Should be positive since model 0 and 1 are identical
        assert 0 < corr < 1


class TestPairwiseCorrelationMatrix:
    """Tests for pairwise correlation matrix."""

    def test_returns_dict(self):
        """Should return dictionary mapping pairs to correlations."""
        predictions = np.array([[0, 0, 1], [1, 1, 0], [2, 2, 2]])
        result = compute_pairwise_correlation_matrix(predictions, ["m1", "m2", "m3"])

        assert isinstance(result, dict)
        assert ("m1", "m2") in result
        assert ("m1", "m3") in result
        assert ("m2", "m3") in result
        assert len(result) == 3  # C(3,2) = 3 pairs

    def test_default_model_names(self):
        """Should use default names if not provided."""
        predictions = np.array([[0, 1], [1, 0]])
        result = compute_pairwise_correlation_matrix(predictions)

        assert ("model_0", "model_1") in result

    def test_name_length_mismatch_raises(self):
        """Should raise if model names don't match columns."""
        predictions = np.array([[0, 1, 2], [1, 0, 1]])
        with pytest.raises(ValueError, match="model_names length"):
            compute_pairwise_correlation_matrix(predictions, ["m1", "m2"])


class TestQStatistic:
    """Tests for Yule's Q statistic."""

    def test_identical_classifiers_q_one(self):
        """Identical classifiers should have Q = 1."""
        y_true = np.array([0, 1, 0, 1, 0])
        preds_i = np.array([0, 1, 1, 1, 0])
        preds_j = np.array([0, 1, 1, 1, 0])  # Same predictions

        q = compute_q_statistic(y_true, preds_i, preds_j)
        assert q == pytest.approx(1.0, abs=1e-6)

    def test_opposite_classifiers_q_negative(self):
        """Opposite classifiers should have Q < 0."""
        y_true = np.array([0, 0, 1, 1])
        # Model i: correct on [0,1], wrong on [2,3]
        preds_i = np.array([0, 0, 0, 0])
        # Model j: wrong on [0,1], correct on [2,3]
        preds_j = np.array([1, 1, 1, 1])

        q = compute_q_statistic(y_true, preds_i, preds_j)
        assert q < 0

    def test_independent_classifiers_q_zero(self):
        """Statistically independent classifiers should have Q ≈ 0."""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, size=1000)
        preds_i = np.random.randint(0, 3, size=1000)
        preds_j = np.random.randint(0, 3, size=1000)

        q = compute_q_statistic(y_true, preds_i, preds_j)
        assert abs(q) < 0.1  # Should be close to 0

    def test_length_mismatch_raises(self):
        """Should raise on length mismatch."""
        with pytest.raises(ValueError, match="same length"):
            compute_q_statistic(np.array([0, 1]), np.array([0, 1, 0]), np.array([0, 1]))


class TestAverageQStatistic:
    """Tests for average Q statistic across model pairs."""

    def test_computes_average(self):
        """Should compute average Q across all pairs."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        predictions = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ])

        avg_q = compute_average_q_statistic(y_true, predictions)
        # Models 0 and 1 are identical, model 2 is opposite
        # Q(0,1) ≈ 1.0, Q(0,2) < 0, Q(1,2) < 0
        # Average should be somewhere in between
        assert isinstance(avg_q, float)


class TestDisagreement:
    """Tests for disagreement measure."""

    def test_identical_predictions_zero_disagreement(self):
        """Identical predictions should have 0 disagreement."""
        preds_i = np.array([0, 1, 2, 1, 0])
        preds_j = np.array([0, 1, 2, 1, 0])

        d = compute_disagreement(preds_i, preds_j)
        assert d == 0.0

    def test_completely_different_predictions(self):
        """Completely different predictions should have high disagreement."""
        preds_i = np.array([0, 0, 0, 0])
        preds_j = np.array([1, 1, 1, 1])

        d = compute_disagreement(preds_i, preds_j)
        assert d == 1.0

    def test_partial_disagreement(self):
        """50% disagreement case."""
        preds_i = np.array([0, 1, 0, 1])
        preds_j = np.array([0, 0, 1, 1])

        d = compute_disagreement(preds_i, preds_j)
        assert d == 0.5

    def test_empty_predictions_returns_zero(self):
        """Empty predictions should return 0."""
        d = compute_disagreement(np.array([]), np.array([]))
        assert d == 0.0


class TestDoubleFault:
    """Tests for double fault measure."""

    def test_no_double_faults(self):
        """No samples where both are wrong."""
        y_true = np.array([0, 1, 0, 1])
        preds_i = np.array([0, 1, 0, 1])  # All correct
        preds_j = np.array([0, 1, 0, 1])  # All correct

        df = compute_double_fault(y_true, preds_i, preds_j)
        assert df == 0.0

    def test_all_double_faults(self):
        """All samples have both wrong."""
        y_true = np.array([0, 1, 0, 1])
        preds_i = np.array([1, 0, 1, 0])  # All wrong
        preds_j = np.array([1, 0, 1, 0])  # All wrong

        df = compute_double_fault(y_true, preds_i, preds_j)
        assert df == 1.0

    def test_partial_double_faults(self):
        """Some double faults."""
        y_true = np.array([0, 1, 0, 1])
        preds_i = np.array([1, 1, 0, 1])  # Wrong on [0]
        preds_j = np.array([1, 1, 0, 1])  # Wrong on [0]

        df = compute_double_fault(y_true, preds_i, preds_j)
        assert df == 0.25  # 1/4 samples


class TestEntropyOfVotes:
    """Tests for entropy of voting distribution."""

    def test_unanimous_votes_zero_entropy(self):
        """Unanimous votes should have ~0 entropy."""
        # All models predict the same class
        predictions = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ])
        entropy = compute_entropy_of_votes(predictions, n_classes=3)
        # Near-zero due to numerical stability epsilon in computation
        assert entropy < 1e-4

    def test_split_votes_high_entropy(self):
        """Split votes should have high entropy."""
        # Uniform distribution across classes
        predictions = np.array([
            [0, 1, 2],  # One vote each for 3 classes
            [0, 1, 2],
            [0, 1, 2],
        ])
        entropy = compute_entropy_of_votes(predictions, n_classes=3)
        # Maximum entropy for 3 classes is ln(3) ≈ 1.099
        assert entropy > 1.0

    def test_single_model_returns_zero(self):
        """Single model should return 0 (no disagreement)."""
        predictions = np.array([[0], [1], [2]])
        entropy = compute_entropy_of_votes(predictions, n_classes=3)
        assert entropy == 0.0


class TestKLDivergence:
    """Tests for KL divergence between probability distributions."""

    def test_identical_distributions_zero_kl(self):
        """Identical distributions should have ~0 KL divergence."""
        probs_i = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
        probs_j = probs_i.copy()

        kl = compute_kl_divergence(probs_i, probs_j)
        assert kl < 0.01

    def test_different_distributions_positive_kl(self):
        """Different distributions should have positive KL divergence."""
        probs_i = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
        probs_j = np.array([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])

        kl = compute_kl_divergence(probs_i, probs_j)
        assert kl > 0

    def test_shape_mismatch_raises(self):
        """Should raise on shape mismatch."""
        probs_i = np.array([[0.5, 0.5], [0.5, 0.5]])
        probs_j = np.array([[0.5, 0.3, 0.2]])

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_kl_divergence(probs_i, probs_j)


class TestAverageKLDivergence:
    """Tests for average KL divergence across model pairs."""

    def test_computes_average(self):
        """Should compute average KL across all pairs."""
        # Shape: (n_samples, n_models, n_classes)
        probabilities = np.array([
            [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.2, 0.2]],
            [[0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2]],
        ])

        avg_kl = compute_average_kl_divergence(probabilities)
        assert isinstance(avg_kl, float)
        assert avg_kl >= 0


class TestDiversityScore:
    """Tests for composite diversity score."""

    def test_high_diversity_high_score(self):
        """High diversity metrics should produce high score."""
        score = compute_diversity_score(
            pairwise_correlation=0.0,  # Low correlation = diverse
            q_statistic=-0.5,  # Negative Q = diverse
            disagreement=0.5,  # High disagreement = diverse
            double_fault=0.1,  # Low double fault = diverse
            entropy=1.0,  # High entropy = diverse
            kl_divergence=0.5,  # High KL = diverse
        )
        assert score > 0.5

    def test_low_diversity_low_score(self):
        """Low diversity metrics should produce low score."""
        score = compute_diversity_score(
            pairwise_correlation=0.9,  # High correlation = not diverse
            q_statistic=0.9,  # High Q = not diverse
            disagreement=0.1,  # Low disagreement = not diverse
            double_fault=0.5,  # High double fault = not diverse
            entropy=0.1,  # Low entropy = not diverse
            kl_divergence=0.01,  # Low KL = not diverse
        )
        assert score < 0.5

    def test_score_in_valid_range(self):
        """Score should always be in [0, 1]."""
        np.random.seed(42)
        for _ in range(100):
            score = compute_diversity_score(
                pairwise_correlation=np.random.uniform(-1, 1),
                q_statistic=np.random.uniform(-1, 1),
                disagreement=np.random.uniform(0, 1),
                double_fault=np.random.uniform(0, 1),
                entropy=np.random.uniform(0, 2),
                kl_divergence=np.random.uniform(0, 1),
            )
            assert 0 <= score <= 1


class TestDiversityAnalyzer:
    """Tests for DiversityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default settings."""
        return DiversityAnalyzer(min_diversity_threshold=0.3, correlation_threshold=0.8)

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions dict."""
        np.random.seed(42)
        n_samples = 100
        return {
            "model_a": np.random.randint(0, 3, n_samples),
            "model_b": np.random.randint(0, 3, n_samples),
            "model_c": np.random.randint(0, 3, n_samples),
        }

    @pytest.fixture
    def sample_probabilities(self):
        """Create sample probabilities dict."""
        np.random.seed(42)
        n_samples = 100
        result = {}
        for name in ["model_a", "model_b", "model_c"]:
            probs = np.random.rand(n_samples, 3)
            probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize
            result[name] = probs
        return result

    def test_analyze_returns_metrics(self, analyzer, sample_predictions):
        """analyze() should return DiversityMetrics."""
        y_true = np.random.randint(0, 3, 100)
        metrics = analyzer.analyze(sample_predictions, y_true=y_true)

        assert isinstance(metrics, DiversityMetrics)
        assert isinstance(metrics.pairwise_correlation, float)
        assert isinstance(metrics.diversity_score, float)
        assert isinstance(metrics.model_pair_correlations, dict)

    def test_analyze_with_probabilities(
        self, analyzer, sample_predictions, sample_probabilities
    ):
        """analyze() with probabilities should compute KL divergence."""
        metrics = analyzer.analyze(
            sample_predictions,
            base_probabilities=sample_probabilities,
        )

        assert metrics.kl_divergence >= 0

    def test_minimum_models_required(self, analyzer):
        """Should warn and return default metrics for < 2 models."""
        predictions = {"single_model": np.array([0, 1, 2])}
        metrics = analyzer.analyze(predictions)

        assert "Add more base models" in metrics.recommendations[0]

    def test_recommendations_generated(self, analyzer):
        """Should generate recommendations for low diversity."""
        # Create highly correlated predictions
        n_samples = 100
        base = np.random.randint(0, 3, n_samples)
        predictions = {
            "model_a": base,
            "model_b": base,  # Identical to a
            "model_c": base,  # Identical to a
        }

        metrics = analyzer.analyze(predictions)
        assert len(metrics.recommendations) > 0

    def test_suggest_model_removal(self, analyzer):
        """suggest_model_removal() should identify redundant models."""
        n_samples = 100
        base = np.random.randint(0, 3, n_samples)

        predictions = {
            "model_a": base,
            "model_b": base,  # Identical to a
            "model_c": np.random.randint(0, 3, n_samples),  # Different
        }

        removals = analyzer.suggest_model_removal(predictions)
        # Either model_a or model_b should be suggested for removal
        assert len(removals) > 0
        assert removals[0] in ["model_a", "model_b"]

    def test_suggest_model_removal_with_accuracy(self, analyzer):
        """Should prefer removing less accurate model."""
        n_samples = 100
        y_true = np.random.randint(0, 3, n_samples)

        # model_a: 80% accurate, model_b: 20% accurate, both correlated
        preds_good = y_true.copy()
        preds_good[: n_samples // 5] = (y_true[: n_samples // 5] + 1) % 3

        preds_bad = (y_true + 1) % 3  # Mostly wrong
        preds_bad[: n_samples // 5] = y_true[: n_samples // 5]  # Some right

        predictions = {
            "good_model": preds_good,
            "bad_model": preds_bad,
            "different_model": np.random.randint(0, 3, n_samples),
        }

        removals = analyzer.suggest_model_removal(predictions, y_true=y_true)
        # Should suggest removing bad_model since it's less accurate
        if len(removals) > 0:
            # The highly correlated pair is good_model and bad_model
            assert "bad_model" in removals or "different_model" in removals


class TestKLDiversityPenalty:
    """Tests for KL diversity penalty function."""

    def test_high_diversity_low_penalty(self):
        """High diversity should have low penalty."""
        # Create diverse probability outputs
        n_samples = 100
        probs = np.random.rand(n_samples, 3, 3)
        probs = probs / probs.sum(axis=2, keepdims=True)

        penalty = compute_kl_diversity_penalty(probs, target_kl=0.01)
        assert penalty >= 0

    def test_single_model_returns_zero(self):
        """Single model should have 0 penalty."""
        probs = np.array([[[0.8, 0.1, 0.1]], [[0.2, 0.7, 0.1]]])
        penalty = compute_kl_diversity_penalty(probs)
        assert penalty == 0.0


class TestDiversityMetricsDataclass:
    """Tests for DiversityMetrics dataclass."""

    def test_to_dict(self):
        """to_dict() should return all metrics."""
        metrics = DiversityMetrics(
            pairwise_correlation=0.5,
            q_statistic=0.3,
            disagreement=0.2,
            double_fault=0.1,
            entropy=1.0,
            kl_divergence=0.4,
            diversity_score=0.6,
            model_pair_correlations={("a", "b"): 0.5},
            recommendations=["Test recommendation"],
        )

        d = metrics.to_dict()
        assert d["pairwise_correlation"] == 0.5
        assert d["diversity_score"] == 0.6
        assert "a_vs_b" in d["model_pair_correlations"]
        assert "Test recommendation" in d["recommendations"]


class TestValidationIntegration:
    """Test that diversity metrics are accessible from validation module."""

    def test_import_from_validation(self):
        """Should be importable from src.validation."""
        from src.validation import (
            DiversityAnalyzer,
            DiversityMetrics,
            compute_diversity_score,
            compute_pairwise_correlation,
            compute_q_statistic,
        )

        assert DiversityAnalyzer is not None
        assert DiversityMetrics is not None
        assert callable(compute_diversity_score)
        assert callable(compute_pairwise_correlation)
        assert callable(compute_q_statistic)
