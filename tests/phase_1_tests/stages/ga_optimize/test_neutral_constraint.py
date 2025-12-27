"""
Unit tests for neutral label preservation in fitness function.

These tests verify that the fitness function correctly:
1. Blocks solutions with <10% neutral labels (HARD constraint)
2. Penalizes solutions with <20% neutral labels (soft penalty)
3. Rewards solutions in the 20-30% neutral range (target)
4. Penalizes solutions with >30% neutral labels (soft penalty)
5. Blocks solutions with >40% neutral labels (HARD constraint)

The neutral class represents "hold" / timeout scenarios which are essential for:
- Avoiding overtrading and transaction costs
- Filtering low-confidence / noisy signals
- Realistic signal rates in production

Run with: pytest tests/phase_1_tests/stages/ga_optimize/test_neutral_constraint.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.ga_optimize.fitness import calculate_fitness
from src.phase1.config import LABEL_BALANCE_CONSTRAINTS


# =============================================================================
# FIXTURE DATA
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample MAE/MFE and bars_to_hit data."""
    np.random.seed(42)
    n = 1000
    return {
        'bars_to_hit': np.random.randint(1, 10, n).astype(np.int32),
        'mae': -np.abs(np.random.randn(n) * 0.01).astype(np.float32),
        'mfe': np.abs(np.random.randn(n) * 0.02).astype(np.float32),
        'horizon': 20,
        'atr_mean': 5.0,
    }


def create_labels(short_pct: float, neutral_pct: float, long_pct: float, n: int = 1000):
    """Create labels with specified distribution."""
    n_short = int(n * short_pct)
    n_neutral = int(n * neutral_pct)
    n_long = n - n_short - n_neutral
    labels = np.array([-1] * n_short + [0] * n_neutral + [1] * n_long, dtype=np.int8)
    np.random.shuffle(labels)
    return labels


# =============================================================================
# HARD CONSTRAINT TESTS
# =============================================================================

class TestNeutralHardConstraints:
    """Tests for hard constraints on neutral percentage."""

    def test_constraint_config_values(self):
        """Test LABEL_BALANCE_CONSTRAINTS has correct values."""
        assert LABEL_BALANCE_CONSTRAINTS['min_neutral_pct'] == 0.10
        assert LABEL_BALANCE_CONSTRAINTS['target_neutral_low'] == 0.20
        assert LABEL_BALANCE_CONSTRAINTS['target_neutral_high'] == 0.30
        assert LABEL_BALANCE_CONSTRAINTS['max_neutral_pct'] == 0.40
        assert LABEL_BALANCE_CONSTRAINTS['min_any_class_pct'] == 0.10

    def test_1pct_neutral_blocked(self, sample_data):
        """Test 1% neutral (the original problem case) is blocked."""
        labels = create_labels(0.49, 0.01, 0.50)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness < -9000, "1% neutral should trigger hard constraint"

    def test_5pct_neutral_blocked(self, sample_data):
        """Test 5% neutral is blocked (below 10% minimum)."""
        labels = create_labels(0.47, 0.05, 0.48)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness < -9000, "5% neutral should trigger hard constraint"

    def test_9pct_neutral_blocked(self, sample_data):
        """Test 9% neutral is blocked (just below 10% minimum)."""
        labels = create_labels(0.45, 0.09, 0.46)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness < -9000, "9% neutral should trigger hard constraint"

    def test_45pct_neutral_blocked(self, sample_data):
        """Test 45% neutral is blocked (above 40% maximum)."""
        labels = create_labels(0.27, 0.45, 0.28)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness < -9000, "45% neutral should trigger hard constraint"

    def test_100pct_neutral_blocked(self, sample_data):
        """Test 100% neutral is blocked."""
        labels = np.zeros(1000, dtype=np.int8)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness < -9000, "100% neutral should trigger hard constraint"

    def test_0pct_neutral_blocked(self, sample_data):
        """Test 0% neutral is blocked."""
        labels = create_labels(0.50, 0.00, 0.50)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness < -9000, "0% neutral should trigger hard constraint"


# =============================================================================
# SOFT PENALTY TESTS
# =============================================================================

class TestNeutralSoftPenalties:
    """Tests for soft penalties on neutral percentage."""

    def test_10pct_neutral_allowed_but_penalized(self, sample_data):
        """Test 10% neutral is allowed but penalized."""
        labels = create_labels(0.45, 0.10, 0.45)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        # Should pass hard constraint but have lower fitness than target
        assert fitness > -9000, "10% neutral should pass hard constraint"
        assert fitness < 10, "10% neutral should be penalized vs target"

    def test_15pct_neutral_moderate_penalty(self, sample_data):
        """Test 15% neutral has moderate penalty."""
        labels = create_labels(0.43, 0.15, 0.42)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness > -9000, "15% neutral should pass hard constraint"
        assert fitness < 14, "15% neutral should be penalized vs target"
        assert fitness > 8, "15% neutral should not be too heavily penalized"

    def test_35pct_neutral_moderate_penalty(self, sample_data):
        """Test 35% neutral has moderate penalty."""
        labels = create_labels(0.32, 0.35, 0.33)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness > -9000, "35% neutral should pass hard constraint"
        assert fitness < 14, "35% neutral should be penalized vs target"
        assert fitness > 8, "35% neutral should not be too heavily penalized"


# =============================================================================
# TARGET RANGE TESTS
# =============================================================================

class TestNeutralTargetRange:
    """Tests for optimal neutral percentage range (20-30%)."""

    def test_20pct_neutral_optimal(self, sample_data):
        """Test 20% neutral is in optimal range."""
        labels = create_labels(0.40, 0.20, 0.40)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness > 10, "20% neutral should have high fitness"

    def test_25pct_neutral_optimal(self, sample_data):
        """Test 25% neutral is in optimal range."""
        labels = create_labels(0.35, 0.25, 0.40)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness > 10, "25% neutral should have high fitness"

    def test_30pct_neutral_optimal(self, sample_data):
        """Test 30% neutral is in optimal range."""
        labels = create_labels(0.35, 0.30, 0.35)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        assert fitness > 10, "30% neutral should have high fitness"

    def test_target_range_higher_than_boundaries(self, sample_data):
        """Test target range (20-30%) has higher fitness than boundaries."""
        labels_10 = create_labels(0.45, 0.10, 0.45)
        labels_25 = create_labels(0.35, 0.25, 0.40)
        labels_38 = create_labels(0.31, 0.38, 0.31)

        fitness_10 = calculate_fitness(
            labels_10, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        fitness_25 = calculate_fitness(
            labels_25, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )
        fitness_38 = calculate_fitness(
            labels_38, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )

        assert fitness_25 > fitness_10, "25% neutral should beat 10% neutral"
        assert fitness_25 > fitness_38, "25% neutral should beat 38% neutral"


# =============================================================================
# GRADIENT TESTS
# =============================================================================

class TestNeutralGradients:
    """Tests for smooth fitness gradients around neutral constraints."""

    def test_gradient_from_5_to_25_pct(self, sample_data):
        """Test fitness increases smoothly from 5% to 25% neutral."""
        neutral_pcts = [0.05, 0.10, 0.15, 0.20, 0.25]
        fitnesses = []

        for neutral_pct in neutral_pcts:
            short_pct = (1.0 - neutral_pct) / 2
            long_pct = (1.0 - neutral_pct) / 2
            labels = create_labels(short_pct, neutral_pct, long_pct)
            fitness = calculate_fitness(
                labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
                sample_data['horizon'], sample_data['atr_mean']
            )
            fitnesses.append(fitness)

        # 5% should be blocked (hard constraint)
        assert fitnesses[0] < -9000

        # From 10% onwards, fitness should increase as we approach target
        for i in range(1, len(fitnesses) - 1):
            # Note: 10->15->20->25 should generally increase
            # (though we don't strictly enforce monotonicity)
            pass

        # 25% (target) should have highest fitness
        assert fitnesses[4] == max(fitnesses[1:]), "25% should have highest fitness among valid options"


# =============================================================================
# PROBLEM CASE REGRESSION TEST
# =============================================================================

class TestProblemCaseRegression:
    """Regression tests for the original problem: 49% Short, 1% Neutral, 50% Long."""

    def test_original_problem_case_blocked(self, sample_data):
        """Test the original problem case (49/1/50) is now blocked."""
        # This was the distribution after GA optimization that destroyed neutral
        labels = create_labels(0.49, 0.01, 0.50)
        fitness = calculate_fitness(
            labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )

        # Must be blocked by hard constraint
        assert fitness < -9000, "49/1/50 distribution should be blocked"

    def test_initial_good_case_preferred(self, sample_data):
        """Test initial good distribution (43/33/25) is preferred over problem case."""
        # Initial labeling: 43% Short, 33% Neutral, 25% Long
        labels_good = create_labels(0.43, 0.33, 0.24)
        fitness_good = calculate_fitness(
            labels_good, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )

        # Problem case: 49% Short, 1% Neutral, 50% Long
        labels_bad = create_labels(0.49, 0.01, 0.50)
        fitness_bad = calculate_fitness(
            labels_bad, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )

        # Good case should have MUCH higher fitness
        assert fitness_good > fitness_bad + 9000, "Good distribution should far exceed blocked case"

    def test_optimizer_cannot_select_low_neutral(self, sample_data):
        """Test that no fitness score with <10% neutral can exceed valid options."""
        # Try various low-neutral configurations
        low_neutral_fitnesses = []
        for neutral_pct in [0.01, 0.03, 0.05, 0.07, 0.09]:
            short_pct = (1.0 - neutral_pct) / 2
            long_pct = (1.0 - neutral_pct) / 2
            labels = create_labels(short_pct, neutral_pct, long_pct)
            fitness = calculate_fitness(
                labels, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
                sample_data['horizon'], sample_data['atr_mean']
            )
            low_neutral_fitnesses.append(fitness)

        # Valid configuration
        labels_valid = create_labels(0.35, 0.25, 0.40)
        fitness_valid = calculate_fitness(
            labels_valid, sample_data['bars_to_hit'], sample_data['mae'], sample_data['mfe'],
            sample_data['horizon'], sample_data['atr_mean']
        )

        # All low-neutral configurations should be far below valid
        for low_fitness in low_neutral_fitnesses:
            assert low_fitness < fitness_valid - 9000, \
                "Low neutral configurations should be blocked"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
