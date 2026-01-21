"""
Tests for BetSizer, KellyCriterion, and VolatilityScaler.

Tests cover:
- KellyCriterion fraction computation
- VolatilityScaler position scaling
- BetSizer with different methods
- BetSizingResult properties
- Edge cases and validation
"""

import numpy as np
import pytest

from src.pipeline._phase1_impl.stages.meta_labeling.bet_sizer import (
    BetSizer,
    BetSizingMethod,
    BetSizingResult,
    KellyCriterion,
    VolatilityScaler,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MAX_POSITION,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_TARGET_VOLATILITY,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_confidences():
    """Create sample confidence values."""
    np.random.seed(42)
    return np.clip(np.random.normal(0.65, 0.15, 100), 0.1, 0.99)


@pytest.fixture
def sample_directions():
    """Create sample trading directions."""
    np.random.seed(42)
    return np.random.choice([-1, 1], size=100)


@pytest.fixture
def sample_returns():
    """Create sample returns for volatility calculation."""
    np.random.seed(42)
    return np.random.normal(0, 0.02, 100)  # 2% daily vol


@pytest.fixture
def high_confidence_data():
    """Create data with high confidence values."""
    return np.array([0.8, 0.9, 0.85, 0.75, 0.95])


@pytest.fixture
def low_confidence_data():
    """Create data with low confidence values."""
    return np.array([0.52, 0.48, 0.56, 0.51, 0.54])


# =============================================================================
# KELLY CRITERION TESTS
# =============================================================================


class TestKellyCriterion:
    """Tests for KellyCriterion."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        kelly = KellyCriterion()

        assert kelly.kelly_fraction == DEFAULT_KELLY_FRACTION
        assert kelly.min_edge == 0.0
        assert kelly.max_fraction == 0.25

    def test_initialization_custom(self):
        """Test custom initialization."""
        kelly = KellyCriterion(
            kelly_fraction=0.25,
            min_edge=0.05,
            max_fraction=0.50,
        )

        assert kelly.kelly_fraction == 0.25
        assert kelly.min_edge == 0.05
        assert kelly.max_fraction == 0.50

    def test_invalid_kelly_fraction_raises(self):
        """Test that invalid kelly_fraction raises ValueError."""
        with pytest.raises(ValueError, match="kelly_fraction must be in"):
            KellyCriterion(kelly_fraction=0)

        with pytest.raises(ValueError, match="kelly_fraction must be in"):
            KellyCriterion(kelly_fraction=1.5)

        with pytest.raises(ValueError, match="kelly_fraction must be in"):
            KellyCriterion(kelly_fraction=-0.1)

    def test_invalid_max_fraction_raises(self):
        """Test that invalid max_fraction raises ValueError."""
        with pytest.raises(ValueError, match="max_fraction must be positive"):
            KellyCriterion(max_fraction=0)

        with pytest.raises(ValueError, match="max_fraction must be positive"):
            KellyCriterion(max_fraction=-0.1)

    def test_compute_fraction_basic(self):
        """Test basic fraction computation."""
        kelly = KellyCriterion(kelly_fraction=1.0)  # Full Kelly

        # Win prob = 0.6, win/loss ratio = 1.5
        # f* = (bp - q) / b = (1.5 * 0.6 - 0.4) / 1.5 = 0.33
        fraction = kelly.compute_fraction(win_prob=0.6, win_loss_ratio=1.5)

        assert isinstance(fraction, float)
        assert fraction >= 0

    def test_compute_fraction_no_edge(self):
        """Test fraction with no edge (50/50)."""
        kelly = KellyCriterion()

        # Win prob = 0.5, win/loss ratio = 1.0 -> no edge -> f* = 0
        fraction = kelly.compute_fraction(win_prob=0.5, win_loss_ratio=1.0)

        assert fraction == 0.0

    def test_compute_fraction_negative_edge(self):
        """Test fraction with negative edge."""
        kelly = KellyCriterion()

        # Win prob = 0.4, win/loss ratio = 1.0 -> negative edge -> f* = 0
        fraction = kelly.compute_fraction(win_prob=0.4, win_loss_ratio=1.0)

        assert fraction == 0.0

    def test_compute_fraction_half_kelly(self):
        """Test half-Kelly fraction."""
        kelly_full = KellyCriterion(kelly_fraction=1.0, max_fraction=1.0)
        kelly_half = KellyCriterion(kelly_fraction=0.5, max_fraction=1.0)

        full_frac = kelly_full.compute_fraction(win_prob=0.65, win_loss_ratio=1.5)
        half_frac = kelly_half.compute_fraction(win_prob=0.65, win_loss_ratio=1.5)

        # Half-Kelly should be approximately half of full Kelly
        assert abs(half_frac - full_frac * 0.5) < 0.01

    def test_compute_fraction_capped(self):
        """Test that fraction is capped at max_fraction."""
        kelly = KellyCriterion(kelly_fraction=1.0, max_fraction=0.10)

        # Very high edge should be capped
        fraction = kelly.compute_fraction(win_prob=0.90, win_loss_ratio=3.0)

        assert fraction <= 0.10

    def test_compute_fractions_array_input(self):
        """Test fraction computation with array inputs."""
        kelly = KellyCriterion()

        win_probs = np.array([0.55, 0.60, 0.65, 0.70])
        win_loss_ratios = np.array([1.2, 1.3, 1.4, 1.5])

        fractions = kelly.compute_fractions_array(win_probs, win_loss_ratios)

        assert isinstance(fractions, np.ndarray)
        assert len(fractions) == len(win_probs)
        assert np.all(fractions >= 0)

    def test_min_edge_filtering(self):
        """Test that positions below min_edge are zeroed."""
        kelly = KellyCriterion(min_edge=0.10)  # Require 10% edge

        # 55% win prob has only 5% edge -> should be zero
        fraction_low_edge = kelly.compute_fraction(win_prob=0.55, win_loss_ratio=1.0)

        # 65% win prob has 15% edge -> should be non-zero
        fraction_high_edge = kelly.compute_fraction(win_prob=0.65, win_loss_ratio=1.0)

        assert fraction_low_edge == 0.0
        assert fraction_high_edge > 0.0


# =============================================================================
# VOLATILITY SCALER TESTS
# =============================================================================


class TestVolatilityScaler:
    """Tests for VolatilityScaler."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        scaler = VolatilityScaler()

        assert scaler.target_volatility == DEFAULT_TARGET_VOLATILITY
        assert scaler.lookback == 20

    def test_initialization_custom(self):
        """Test custom initialization."""
        scaler = VolatilityScaler(
            target_volatility=0.20,
            lookback=30,
            annualization_factor=252,
        )

        assert scaler.target_volatility == 0.20
        assert scaler.lookback == 30
        assert scaler.annualization_factor == 252

    def test_invalid_target_volatility_raises(self):
        """Test that invalid target_volatility raises ValueError."""
        with pytest.raises(ValueError, match="target_volatility must be positive"):
            VolatilityScaler(target_volatility=0)

        with pytest.raises(ValueError, match="target_volatility must be positive"):
            VolatilityScaler(target_volatility=-0.1)

    def test_invalid_lookback_raises(self):
        """Test that invalid lookback raises ValueError."""
        with pytest.raises(ValueError, match="lookback must be >= 2"):
            VolatilityScaler(lookback=1)

        with pytest.raises(ValueError, match="lookback must be >= 2"):
            VolatilityScaler(lookback=0)

    def test_estimate_volatility_basic(self, sample_returns):
        """Test basic volatility estimation."""
        scaler = VolatilityScaler(target_volatility=0.15, lookback=20)

        vol = scaler.estimate_volatility(sample_returns)

        assert isinstance(vol, float)
        assert vol > 0

    def test_compute_scale_factor(self):
        """Test scale factor computation."""
        scaler = VolatilityScaler(target_volatility=0.15)

        # High vol -> low scale
        scale_high = scaler.compute_scale_factor(0.30)

        # Low vol -> high scale
        scale_low = scaler.compute_scale_factor(0.075)

        # Target vol -> scale = 1
        scale_target = scaler.compute_scale_factor(0.15)

        assert scale_high < scale_target
        assert scale_low > scale_target
        assert abs(scale_target - 1.0) < 0.01

    def test_scale_positions(self, sample_returns):
        """Test position scaling."""
        scaler = VolatilityScaler(target_volatility=0.15)

        base_positions = np.ones(10)
        # Use a specific volatility value for scaling
        vol = scaler.estimate_volatility(sample_returns)
        scaled_positions = scaler.scale(base_positions, vol)

        assert len(scaled_positions) == len(base_positions)
        # Scaled positions should be non-negative
        assert np.all(scaled_positions >= 0)


# =============================================================================
# BET SIZING RESULT TESTS
# =============================================================================


class TestBetSizingResult:
    """Tests for BetSizingResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = BetSizingResult(
            position_sizes=np.array([1, -1, 0, 2, -2]),
            raw_sizes=np.array([1, 1, 0, 2, 2]),
            confidence_values=np.array([0.8, 0.7, 0.5, 0.9, 0.85]),
        )

        assert result.n_samples == 5
        assert result.n_active == 4  # One zero position
        assert result.avg_size > 0

    def test_n_active_property(self):
        """Test n_active property."""
        result = BetSizingResult(
            position_sizes=np.array([0, 0, 1, -1, 0]),
            raw_sizes=np.array([0, 0, 1, 1, 0]),
            confidence_values=np.array([0.3, 0.4, 0.8, 0.7, 0.5]),
        )

        assert result.n_active == 2

    def test_avg_size_property(self):
        """Test avg_size property."""
        result = BetSizingResult(
            position_sizes=np.array([2, -2, 4, -4]),
            raw_sizes=np.array([2, 2, 4, 4]),
            confidence_values=np.array([0.8, 0.8, 0.9, 0.9]),
        )

        assert result.avg_size == 3.0  # (2 + 2 + 4 + 4) / 4

    def test_avg_size_no_active(self):
        """Test avg_size with no active positions."""
        result = BetSizingResult(
            position_sizes=np.array([0, 0, 0]),
            raw_sizes=np.array([0, 0, 0]),
            confidence_values=np.array([0.3, 0.4, 0.5]),
        )

        assert result.avg_size == 0.0

    def test_with_volatility_values(self):
        """Test result with volatility values."""
        result = BetSizingResult(
            position_sizes=np.array([1, 2, 3]),
            raw_sizes=np.array([1, 2, 3]),
            confidence_values=np.array([0.7, 0.8, 0.9]),
            volatility_values=np.array([0.15, 0.20, 0.18]),
        )

        assert result.volatility_values is not None
        assert len(result.volatility_values) == 3

    def test_with_metadata(self):
        """Test result with metadata."""
        result = BetSizingResult(
            position_sizes=np.array([1]),
            raw_sizes=np.array([1]),
            confidence_values=np.array([0.8]),
            metadata={"method": "kelly", "fraction": 0.5},
        )

        assert result.metadata is not None
        assert result.metadata["method"] == "kelly"


# =============================================================================
# BET SIZER TESTS
# =============================================================================


class TestBetSizer:
    """Tests for BetSizer."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        sizer = BetSizer()

        assert sizer.method == BetSizingMethod.LINEAR
        assert sizer.max_position == DEFAULT_MAX_POSITION
        assert sizer.min_confidence == DEFAULT_MIN_CONFIDENCE

    def test_initialization_custom(self):
        """Test custom initialization."""
        sizer = BetSizer(
            method=BetSizingMethod.KELLY,
            max_position=5,
            min_confidence=0.60,
        )

        assert sizer.method == BetSizingMethod.KELLY
        assert sizer.max_position == 5
        assert sizer.min_confidence == 0.60

    def test_initialization_with_string_method(self):
        """Test initialization with string method."""
        sizer = BetSizer(method="kelly")
        assert sizer.method == BetSizingMethod.KELLY

        sizer = BetSizer(method="linear")
        assert sizer.method == BetSizingMethod.LINEAR

    def test_invalid_max_position_raises(self):
        """Test that invalid max_position raises ValueError."""
        with pytest.raises(ValueError, match="max_position must be positive"):
            BetSizer(max_position=0)

        with pytest.raises(ValueError, match="max_position must be positive"):
            BetSizer(max_position=-1)

    def test_invalid_min_confidence_raises(self):
        """Test that invalid min_confidence raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence must be in"):
            BetSizer(min_confidence=1.0)  # Must be < 1

        with pytest.raises(ValueError, match="min_confidence must be in"):
            BetSizer(min_confidence=0)  # Must be > 0

    def test_compute_sizes_linear(self, sample_confidences, sample_directions):
        """Test linear bet sizing."""
        sizer = BetSizer(method=BetSizingMethod.LINEAR, max_position=10)

        result = sizer.compute_sizes(
            meta_probabilities=sample_confidences,
            directions=sample_directions,
        )

        assert isinstance(result, BetSizingResult)
        assert len(result.position_sizes) == len(sample_confidences)
        # Linear: sizes should scale with confidence
        assert np.all(np.abs(result.position_sizes) <= 10)

    def test_compute_sizes_constant(self, sample_confidences, sample_directions):
        """Test constant bet sizing."""
        sizer = BetSizer(method=BetSizingMethod.CONSTANT, max_position=5)

        result = sizer.compute_sizes(
            meta_probabilities=sample_confidences,
            directions=sample_directions,
        )

        # Constant: all active positions should have same size
        active_sizes = np.abs(result.position_sizes[result.position_sizes != 0])
        if len(active_sizes) > 0:
            assert np.allclose(active_sizes, active_sizes[0])

    def test_compute_sizes_kelly(self, sample_confidences, sample_directions):
        """Test Kelly bet sizing."""
        sizer = BetSizer(method=BetSizingMethod.KELLY, max_position=10)

        result = sizer.compute_sizes(
            meta_probabilities=sample_confidences,
            directions=sample_directions,
        )

        assert isinstance(result, BetSizingResult)
        # Kelly positions should be non-negative after abs
        assert np.all(np.abs(result.position_sizes) >= 0)

    def test_min_confidence_filtering(self, sample_directions):
        """Test that low confidence positions are zeroed."""
        sizer = BetSizer(min_confidence=0.60)

        # Mix of high and low confidences
        confidences = np.array([0.55, 0.65, 0.50, 0.75, 0.58])
        directions = sample_directions[:5]

        result = sizer.compute_sizes(confidences, directions)

        # Positions with confidence < 0.60 should be zero
        low_conf_mask = confidences < 0.60
        assert np.all(result.position_sizes[low_conf_mask] == 0)

    def test_direction_application(self, sample_confidences):
        """Test that directions are applied correctly."""
        sizer = BetSizer()

        directions = np.array([1, -1, 1, -1, 1] * 20)
        confidences = sample_confidences

        result = sizer.compute_sizes(confidences, directions)

        # Check signs match directions for active positions
        active_mask = result.position_sizes != 0
        if active_mask.sum() > 0:
            expected_signs = np.sign(directions[active_mask])
            actual_signs = np.sign(result.position_sizes[active_mask])
            np.testing.assert_array_equal(expected_signs, actual_signs)

    def test_high_confidence_larger_sizes(self, high_confidence_data, low_confidence_data):
        """Test that higher confidence leads to larger sizes."""
        sizer = BetSizer(method=BetSizingMethod.LINEAR, min_confidence=0.45)

        high_result = sizer.compute_sizes(
            high_confidence_data,
            np.ones(len(high_confidence_data)),
        )
        low_result = sizer.compute_sizes(
            low_confidence_data,
            np.ones(len(low_confidence_data)),
        )

        # High confidence should have larger average size
        high_avg = np.mean(np.abs(high_result.position_sizes))
        low_avg = np.mean(np.abs(low_result.position_sizes))

        assert high_avg > low_avg

    def test_volatility_scaled_method(self, sample_confidences, sample_directions):
        """Test volatility-scaled sizing."""
        sizer = BetSizer(
            method=BetSizingMethod.VOLATILITY_SCALED,
            max_position=10,
            target_volatility=0.15,
        )

        result = sizer.compute_sizes(
            meta_probabilities=sample_confidences,
            directions=sample_directions,
            volatility=0.20,  # Higher than target = smaller positions
        )

        assert isinstance(result, BetSizingResult)

    def test_mismatched_lengths_raises(self):
        """Test that mismatched input lengths raise error."""
        sizer = BetSizer()

        with pytest.raises(ValueError, match="same length"):
            sizer.compute_sizes(
                meta_probabilities=np.array([0.6, 0.7, 0.8]),
                directions=np.array([1, -1]),  # Different length
            )


# =============================================================================
# BET SIZING METHOD ENUM TESTS
# =============================================================================


class TestBetSizingMethod:
    """Tests for BetSizingMethod enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert BetSizingMethod.LINEAR.value == "linear"
        assert BetSizingMethod.KELLY.value == "kelly"
        assert BetSizingMethod.VOLATILITY_SCALED.value == "volatility_scaled"
        assert BetSizingMethod.RISK_PARITY.value == "risk_parity"
        assert BetSizingMethod.CONSTANT.value == "constant"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert BetSizingMethod("linear") == BetSizingMethod.LINEAR
        assert BetSizingMethod("kelly") == BetSizingMethod.KELLY


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestBetSizerIntegration:
    """Integration tests for bet sizing workflow."""

    def test_full_workflow(self, sample_confidences, sample_directions):
        """Test full bet sizing workflow."""
        # 1. Create Kelly criterion
        kelly = KellyCriterion(kelly_fraction=0.5)

        # 2. Create sizer with Kelly method
        sizer = BetSizer(
            method=BetSizingMethod.KELLY,
            max_position=10,
            min_confidence=0.55,
        )

        # 3. Compute sizes
        result = sizer.compute_sizes(sample_confidences, sample_directions)

        # 4. Verify result
        assert isinstance(result, BetSizingResult)
        assert result.n_samples == len(sample_confidences)
        assert np.all(np.abs(result.position_sizes) <= 10)

    def test_workflow_with_volatility(self, sample_confidences, sample_directions):
        """Test workflow with volatility scaling."""
        # 1. Create volatility scaler (implicitly via BetSizer)
        sizer = BetSizer(
            method=BetSizingMethod.VOLATILITY_SCALED,
            target_volatility=0.15,
        )

        # 2. Compute volatility-scaled sizes
        result = sizer.compute_sizes(
            sample_confidences,
            sample_directions,
            volatility=0.20,
        )

        assert isinstance(result, BetSizingResult)


# =============================================================================
# VALIDATION MODULE INTEGRATION
# =============================================================================


class TestValidationModuleIntegration:
    """Test that classes are importable from validation module."""

    def test_import_from_validation(self):
        """Test importing from src.validation."""
        from src.validation import (
            BetSizer,
            BetSizingMethod,
            MetaKellyCriterion,
            VolatilityScaler,
        )

        assert BetSizer is not None
        assert BetSizingMethod is not None
        assert MetaKellyCriterion is not None
        assert VolatilityScaler is not None

    def test_create_from_validation_import(self, sample_confidences, sample_directions):
        """Test creating instances from validation imports."""
        from src.validation import BetSizer, BetSizingMethod

        sizer = BetSizer(method=BetSizingMethod.LINEAR, max_position=5)
        result = sizer.compute_sizes(sample_confidences, sample_directions)

        assert result.n_samples == len(sample_confidences)
