"""
Bet Sizing from Meta-Label Confidence.

This module converts meta-model confidence (probability that primary prediction
is correct) into position sizes. Following Lopez de Prado's AFML and Hudson &
Thames implementations, several sizing methods are provided:

1. Linear scaling: position = confidence * max_position
2. Kelly criterion: optimal fraction based on edge and odds
3. Volatility-scaled: adjust for current market volatility
4. Risk-parity: size inversely proportional to expected variance

Engineering Rules Applied:
- Clear contracts with type hints
- Input validation at boundaries
- Fail fast with explicit errors
- Modular design with single responsibility
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# CONSTANTS
# =============================================================================

# Default sizing parameters
DEFAULT_MAX_POSITION = 10
DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_KELLY_FRACTION = 0.5  # Half-Kelly for safety

# Volatility scaling defaults
DEFAULT_TARGET_VOLATILITY = 0.15  # 15% annualized
DEFAULT_VOLATILITY_LOOKBACK = 20  # Bars for volatility estimation


# =============================================================================
# ENUMS
# =============================================================================


class BetSizingMethod(Enum):
    """Available bet sizing methods."""

    LINEAR = "linear"  # Simple linear scaling by confidence
    KELLY = "kelly"  # Kelly criterion for optimal fraction
    VOLATILITY_SCALED = "volatility_scaled"  # Adjust for volatility
    RISK_PARITY = "risk_parity"  # Size inversely to variance
    CONSTANT = "constant"  # Fixed position size


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BetSizingResult:
    """Result of bet sizing computation.

    Attributes:
        position_sizes: Array of position sizes (signed for direction)
        raw_sizes: Array of unsigned sizes before direction
        confidence_values: Array of meta-model confidences used
        volatility_values: Array of volatility values (if applicable)
        metadata: Additional computation metadata
    """

    position_sizes: np.ndarray
    raw_sizes: np.ndarray
    confidence_values: np.ndarray
    volatility_values: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    @property
    def n_samples(self) -> int:
        return len(self.position_sizes)

    @property
    def n_active(self) -> int:
        """Number of non-zero positions."""
        return int((self.position_sizes != 0).sum())

    @property
    def avg_size(self) -> float:
        """Average absolute position size for active positions."""
        active = np.abs(self.position_sizes[self.position_sizes != 0])
        return float(np.mean(active)) if len(active) > 0 else 0.0


# =============================================================================
# KELLY CRITERION
# =============================================================================


class KellyCriterion:
    """
    Kelly Criterion for optimal bet sizing.

    The Kelly criterion calculates the optimal fraction of capital to risk
    based on the probability of winning and the payoff ratio.

    f* = (bp - q) / b

    where:
    - f* = fraction of capital to bet
    - b = odds received on the bet (win/loss ratio)
    - p = probability of winning
    - q = probability of losing (1 - p)

    In practice, fractional Kelly (e.g., half-Kelly) is used for safety.

    Attributes:
        kelly_fraction: Fraction of full Kelly to use (default 0.5)
        min_edge: Minimum edge required to bet (default 0.0)
        max_fraction: Maximum bet fraction cap (default 0.25)

    Example:
        >>> kelly = KellyCriterion(kelly_fraction=0.5)
        >>> fraction = kelly.compute_fraction(win_prob=0.55, win_loss_ratio=1.5)
        >>> print(f"Optimal fraction: {fraction:.2%}")
    """

    def __init__(
        self,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        min_edge: float = 0.0,
        max_fraction: float = 0.25,
    ):
        """
        Initialize KellyCriterion.

        Args:
            kelly_fraction: Fraction of full Kelly (default 0.5 = half-Kelly)
            min_edge: Minimum edge (p - 0.5) to place bet (default 0.0)
            max_fraction: Maximum allowed fraction (default 0.25)
        """
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")

        if max_fraction <= 0:
            raise ValueError(f"max_fraction must be positive, got {max_fraction}")

        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_fraction = max_fraction

    def compute_fraction(
        self,
        win_prob: float,
        win_loss_ratio: float = 1.0,
    ) -> float:
        """
        Compute optimal Kelly fraction.

        Args:
            win_prob: Probability of winning (0 to 1)
            win_loss_ratio: Expected win / expected loss (default 1.0)

        Returns:
            Optimal fraction of capital to risk
        """
        # Validate inputs
        if not 0 <= win_prob <= 1:
            raise ValueError(f"win_prob must be in [0, 1], got {win_prob}")

        if win_loss_ratio <= 0:
            raise ValueError(f"win_loss_ratio must be positive, got {win_loss_ratio}")

        # Check minimum edge
        edge = win_prob - 0.5
        if edge < self.min_edge:
            return 0.0

        # Kelly formula: f* = (bp - q) / b
        # where b = odds (win_loss_ratio), p = win_prob, q = 1 - p
        b = win_loss_ratio
        p = win_prob
        q = 1.0 - p

        kelly = (b * p - q) / b

        # Apply fractional Kelly
        kelly *= self.kelly_fraction

        # Clamp to valid range
        kelly = max(0.0, min(kelly, self.max_fraction))

        return kelly

    def compute_fractions_array(
        self,
        win_probs: np.ndarray,
        win_loss_ratios: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """
        Compute Kelly fractions for arrays of probabilities.

        Args:
            win_probs: Array of win probabilities
            win_loss_ratios: Array or scalar of win/loss ratios

        Returns:
            Array of optimal fractions
        """
        win_probs = np.asarray(win_probs)

        if isinstance(win_loss_ratios, (int, float)):
            win_loss_ratios = np.full_like(win_probs, win_loss_ratios)
        else:
            win_loss_ratios = np.asarray(win_loss_ratios)

        fractions = np.zeros_like(win_probs)

        for i in range(len(win_probs)):
            try:
                fractions[i] = self.compute_fraction(
                    win_prob=float(win_probs[i]),
                    win_loss_ratio=float(win_loss_ratios[i]),
                )
            except ValueError:
                fractions[i] = 0.0

        return fractions


# =============================================================================
# VOLATILITY SCALER
# =============================================================================


class VolatilityScaler:
    """
    Scale position sizes based on volatility.

    Implements volatility targeting: when volatility is high, reduce position
    sizes; when volatility is low, increase sizes. This helps maintain
    consistent risk exposure regardless of market conditions.

    target_size = base_size * (target_vol / current_vol)

    Attributes:
        target_volatility: Target annualized volatility
        lookback: Lookback period for volatility estimation
        min_scale: Minimum scaling factor (floor)
        max_scale: Maximum scaling factor (cap)
        annualization_factor: Factor to annualize volatility

    Example:
        >>> scaler = VolatilityScaler(target_volatility=0.15)
        >>> scaled_sizes = scaler.scale(base_sizes, volatility=0.20)
    """

    def __init__(
        self,
        target_volatility: float = DEFAULT_TARGET_VOLATILITY,
        lookback: int = DEFAULT_VOLATILITY_LOOKBACK,
        min_scale: float = 0.25,
        max_scale: float = 2.0,
        annualization_factor: float = 252.0,
    ):
        """
        Initialize VolatilityScaler.

        Args:
            target_volatility: Target annualized volatility (default 0.15)
            lookback: Bars for volatility estimation (default 20)
            min_scale: Minimum scaling factor (default 0.25)
            max_scale: Maximum scaling factor (default 2.0)
            annualization_factor: Annualization factor (default 252 for daily)
        """
        if target_volatility <= 0:
            raise ValueError(f"target_volatility must be positive, got {target_volatility}")

        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")

        self.target_volatility = target_volatility
        self.lookback = lookback
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.annualization_factor = annualization_factor

    def estimate_volatility(self, returns: np.ndarray) -> float:
        """
        Estimate current volatility from returns.

        Args:
            returns: Array of recent returns

        Returns:
            Annualized volatility estimate
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            logger.warning("Insufficient returns for volatility estimation, using target")
            return self.target_volatility

        # Use rolling std with lookback
        if len(returns) >= self.lookback:
            returns = returns[-self.lookback :]

        vol = np.std(returns, ddof=1) * np.sqrt(self.annualization_factor)
        return float(max(vol, 1e-6))  # Floor to prevent division by zero

    def compute_scale_factor(self, current_volatility: float) -> float:
        """
        Compute scaling factor for given volatility.

        Args:
            current_volatility: Current annualized volatility

        Returns:
            Scaling factor to apply to position sizes
        """
        if current_volatility <= 0:
            return 1.0

        scale = self.target_volatility / current_volatility
        scale = np.clip(scale, self.min_scale, self.max_scale)
        return float(scale)

    def scale(
        self,
        base_sizes: np.ndarray,
        volatility: float | np.ndarray,
    ) -> np.ndarray:
        """
        Scale position sizes by volatility.

        Args:
            base_sizes: Base position sizes
            volatility: Current volatility (scalar or array)

        Returns:
            Volatility-scaled position sizes
        """
        base_sizes = np.asarray(base_sizes)

        if isinstance(volatility, (int, float)):
            scale = self.compute_scale_factor(float(volatility))
            return base_sizes * scale

        # Array of volatilities
        volatility = np.asarray(volatility)
        scales = np.array([self.compute_scale_factor(float(v)) for v in volatility])
        return np.asarray(base_sizes * scales)


# =============================================================================
# BET SIZER
# =============================================================================


class BetSizer:
    """
    Convert meta-model confidence to position sizes.

    This class is the main interface for bet sizing. It combines:
    1. Confidence filtering (minimum threshold)
    2. Size computation (linear, Kelly, etc.)
    3. Optional volatility scaling
    4. Direction assignment (long/short)

    Attributes:
        max_position: Maximum position size
        min_confidence: Minimum confidence to trade
        method: Sizing method (linear, kelly, etc.)
        kelly: KellyCriterion instance (if using Kelly method)
        volatility_scaler: VolatilityScaler instance (if using vol scaling)

    Example:
        >>> sizer = BetSizer(max_position=10, min_confidence=0.55)
        >>> result = sizer.compute_sizes(
        ...     meta_probabilities=proba,
        ...     directions=sides,
        ...     volatility=vol
        ... )
        >>> positions = result.position_sizes
    """

    def __init__(
        self,
        max_position: int = DEFAULT_MAX_POSITION,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        method: str | BetSizingMethod = BetSizingMethod.LINEAR,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        target_volatility: float | None = None,
        risk_per_trade: float = 0.02,
        discrete_sizes: bool = True,
    ):
        """
        Initialize BetSizer.

        Args:
            max_position: Maximum position size (default 10)
            min_confidence: Minimum confidence to trade (default 0.55)
            method: Sizing method ('linear', 'kelly', etc.)
            kelly_fraction: Fraction of full Kelly (default 0.5)
            target_volatility: Target volatility for scaling (None = no scaling)
            risk_per_trade: Risk per trade for risk-parity method
            discrete_sizes: Round to discrete integers (default True)
        """
        if max_position <= 0:
            raise ValueError(f"max_position must be positive, got {max_position}")

        if not 0 < min_confidence < 1:
            raise ValueError(f"min_confidence must be in (0, 1), got {min_confidence}")

        self.max_position = max_position
        self.min_confidence = min_confidence
        self.risk_per_trade = risk_per_trade
        self.discrete_sizes = discrete_sizes

        # Convert method to enum
        if isinstance(method, str):
            try:
                method = BetSizingMethod(method.lower())
            except ValueError:
                valid = [m.value for m in BetSizingMethod]
                raise ValueError(f"Invalid method '{method}'. Valid: {valid}")
        self.method = method

        # Initialize Kelly criterion if needed
        self.kelly: KellyCriterion | None = None
        if self.method == BetSizingMethod.KELLY:
            self.kelly = KellyCriterion(kelly_fraction=kelly_fraction)

        # Initialize volatility scaler if target provided
        self.volatility_scaler: VolatilityScaler | None = None
        if target_volatility is not None:
            self.volatility_scaler = VolatilityScaler(target_volatility=target_volatility)

    def compute_sizes(
        self,
        meta_probabilities: np.ndarray,
        directions: np.ndarray,
        volatility: float | np.ndarray | None = None,
        win_loss_ratio: float | np.ndarray = 1.0,
    ) -> BetSizingResult:
        """
        Compute position sizes from meta-model probabilities.

        Args:
            meta_probabilities: Meta-model confidence (P(primary is correct))
            directions: Trading directions from primary model {-1, 1}
            volatility: Current volatility (optional, for scaling)
            win_loss_ratio: Expected win/loss ratio (for Kelly method)

        Returns:
            BetSizingResult with position sizes and metadata
        """
        # Validate inputs
        meta_probs = np.asarray(meta_probabilities).ravel()
        directions = np.asarray(directions).ravel()

        if len(meta_probs) != len(directions):
            raise ValueError(
                f"meta_probabilities and directions must have same length, "
                f"got {len(meta_probs)} and {len(directions)}"
            )

        n = len(meta_probs)
        logger.info(f"Computing bet sizes for {n} samples, method={self.method.value}")

        # Initialize output arrays
        raw_sizes = np.zeros(n, dtype=np.float32)
        position_sizes = np.zeros(n, dtype=np.float32)

        # Apply confidence filter
        above_threshold = meta_probs >= self.min_confidence
        n_above = above_threshold.sum()
        logger.info(f"  {n_above} samples above confidence threshold {self.min_confidence:.2%}")

        # Compute raw sizes based on method
        if self.method == BetSizingMethod.CONSTANT:
            raw_sizes[above_threshold] = self.max_position

        elif self.method == BetSizingMethod.LINEAR:
            # Linear scaling: size = confidence * max_position
            # Rescale confidence to [0, 1] above threshold
            conf_rescaled = (meta_probs - self.min_confidence) / (1.0 - self.min_confidence)
            conf_rescaled = np.clip(conf_rescaled, 0, 1)
            raw_sizes = np.asarray(conf_rescaled * self.max_position, dtype=np.float32)
            raw_sizes[~above_threshold] = 0.0

        elif self.method == BetSizingMethod.KELLY:
            # Kelly criterion sizing
            if self.kelly is None:
                raise RuntimeError("Kelly criterion not initialized")
            fractions = self.kelly.compute_fractions_array(
                win_probs=meta_probs,
                win_loss_ratios=win_loss_ratio,
            )
            raw_sizes = np.asarray(fractions * self.max_position, dtype=np.float32)
            raw_sizes[~above_threshold] = 0.0

        elif self.method == BetSizingMethod.RISK_PARITY:
            # Size inversely proportional to expected variance
            if volatility is None:
                logger.warning("Risk-parity method requires volatility, falling back to linear")
                raw_sizes = self._linear_sizing(meta_probs, above_threshold)
            else:
                vol_arr = np.atleast_1d(volatility)
                if len(vol_arr) == 1:
                    vol_arr = np.full(n, vol_arr[0])
                # Size = risk_budget / volatility
                vol_arr = np.maximum(vol_arr, 1e-6)  # Floor
                raw_sizes = (self.risk_per_trade / vol_arr) * meta_probs * self.max_position
                raw_sizes[~above_threshold] = 0.0

        elif self.method == BetSizingMethod.VOLATILITY_SCALED:
            # Linear sizing with volatility scaling
            raw_sizes = self._linear_sizing(meta_probs, above_threshold)
            if volatility is not None and self.volatility_scaler is not None:
                raw_sizes = self.volatility_scaler.scale(raw_sizes, volatility)

        # Apply direction
        position_sizes = raw_sizes * np.sign(directions)

        # Handle zero directions (shouldn't happen but be defensive)
        zero_dir_mask = directions == 0
        if zero_dir_mask.any():
            position_sizes[zero_dir_mask] = 0.0

        # Optional volatility scaling (if method didn't already do it)
        vol_values = None
        if volatility is not None:
            vol_values = np.atleast_1d(volatility)
            if len(vol_values) == 1:
                vol_values = np.full(n, vol_values[0])

            if (
                self.volatility_scaler is not None
                and self.method != BetSizingMethod.VOLATILITY_SCALED
            ):
                position_sizes = self.volatility_scaler.scale(position_sizes, vol_values)

        # Clip to max position
        position_sizes = np.clip(position_sizes, -self.max_position, self.max_position)

        # Discretize if requested
        if self.discrete_sizes:
            position_sizes = np.round(position_sizes).astype(int)
            raw_sizes = np.round(raw_sizes).astype(int)

        # Build result
        result = BetSizingResult(
            position_sizes=position_sizes,
            raw_sizes=np.abs(raw_sizes),
            confidence_values=meta_probs,
            volatility_values=vol_values,
            metadata={
                "method": self.method.value,
                "max_position": self.max_position,
                "min_confidence": self.min_confidence,
                "n_above_threshold": int(n_above),
                "n_active_positions": int((position_sizes != 0).sum()),
            },
        )

        self._log_summary(result)
        return result

    def _linear_sizing(
        self,
        meta_probs: np.ndarray,
        above_threshold: np.ndarray,
    ) -> np.ndarray:
        """Compute linear sizing."""
        conf_rescaled = (meta_probs - self.min_confidence) / (1.0 - self.min_confidence)
        conf_rescaled = np.clip(conf_rescaled, 0, 1)
        sizes = conf_rescaled * self.max_position
        sizes[~above_threshold] = 0.0
        return sizes

    def _log_summary(self, result: BetSizingResult) -> None:
        """Log sizing summary."""
        meta = result.metadata or {}
        logger.info("Bet sizing summary:")
        logger.info(f"  Method: {meta.get('method', 'unknown')}")
        logger.info(f"  Active positions: {result.n_active} / {result.n_samples}")
        logger.info(f"  Avg position size: {result.avg_size:.2f}")

        if result.n_active > 0:
            active = result.position_sizes[result.position_sizes != 0]
            n_long = (active > 0).sum()
            n_short = (active < 0).sum()
            logger.info(f"  Long positions: {n_long}, Short positions: {n_short}")

    def get_position_size(
        self,
        meta_probability: float,
        direction: int,
        volatility: float | None = None,
    ) -> int:
        """
        Get position size for a single prediction.

        Convenience method for real-time sizing.

        Args:
            meta_probability: Meta-model confidence
            direction: Trading direction (-1 or 1)
            volatility: Current volatility (optional)

        Returns:
            Position size (signed integer)
        """
        result = self.compute_sizes(
            meta_probabilities=np.array([meta_probability]),
            directions=np.array([direction]),
            volatility=volatility,
        )
        return int(result.position_sizes[0])
