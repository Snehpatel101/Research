"""
Position sizing algorithms for risk management.

This module provides various position sizing methods:
- Kelly Criterion: Optimal growth-maximizing position size
- Fixed Fractional: Risk a fixed percentage per trade
- Volatility-Targeted: Scale position by inverse volatility
- Equal Weight: Simple equal allocation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class PositionSizingMethod(Enum):
    """Available position sizing methods."""

    KELLY = "kelly"
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_TARGETED = "volatility_targeted"
    EQUAL_WEIGHT = "equal_weight"
    FIXED_CONTRACTS = "fixed_contracts"
    BET_SIZING = "bet_sizing"  # Phase 4G: Variable sizing from model confidence


class BasePositionSizer(ABC):
    """Abstract base class for position sizing algorithms."""

    @abstractmethod
    def calculate_position_size(
        self,
        account_equity: float,
        **kwargs: Any,
    ) -> int:
        """
        Calculate position size in contracts.

        Args:
            account_equity: Current account equity in dollars
            **kwargs: Algorithm-specific parameters

        Returns:
            Number of contracts to trade (integer)
        """
        pass

    def _round_to_contracts(self, fractional_contracts: float) -> int:
        """Round fractional contracts to integer, minimum 0."""
        return max(0, int(np.floor(fractional_contracts)))


@dataclass
class KellyCriterion(BasePositionSizer):
    """
    Kelly Criterion position sizing.

    The Kelly formula maximizes long-term growth rate:
    f* = (p * b - q) / b

    Where:
        p = win probability
        b = win/loss ratio (avg_win / avg_loss)
        q = 1 - p = loss probability
        f* = fraction of capital to risk

    In practice, we use fractional Kelly (typically 0.25-0.5) to reduce
    volatility and drawdowns while maintaining most of the growth rate.

    Attributes:
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)
        max_position_pct: Maximum position as % of equity (risk limit)
        min_position_pct: Minimum position as % of equity (avoid tiny positions)
        point_value: Dollar value per point for the contract
    """

    kelly_fraction: float = 0.25
    max_position_pct: float = 0.10
    min_position_pct: float = 0.01
    point_value: float = 5.0  # MES: $5 per point

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate the Kelly fraction f*.

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade (dollars or points)
            avg_loss: Average losing trade (dollars or points, positive value)

        Returns:
            Optimal fraction of capital to risk (can be negative)
        """
        if avg_loss <= 0:
            return 0.0

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss  # Win/loss ratio

        # Kelly formula: f* = (p * b - q) / b
        # Equivalent to: f* = p - q/b
        kelly_f = (p * b - q) / b

        return kelly_f

    def calculate_position_size(
        self,
        account_equity: float,
        win_rate: float = 0.5,
        avg_win: float = 100.0,
        avg_loss: float = 100.0,
        current_price: float = 5000.0,
        **kwargs: Any,
    ) -> int:
        """
        Calculate position size using Kelly criterion.

        Args:
            account_equity: Current account equity
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade in dollars
            avg_loss: Average losing trade in dollars (positive)
            current_price: Current price of the instrument

        Returns:
            Number of contracts to trade
        """
        # Calculate full Kelly fraction
        full_kelly = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)

        # Apply fractional Kelly
        adjusted_kelly = full_kelly * self.kelly_fraction

        # Clamp to bounds
        adjusted_kelly = np.clip(adjusted_kelly, 0, self.max_position_pct)

        # If Kelly suggests not trading (negative edge), return 0
        if adjusted_kelly <= 0:
            return 0

        # Calculate dollar amount to risk
        risk_amount = account_equity * adjusted_kelly

        # Convert to contracts
        # Contract notional = price * point_value
        contract_notional = current_price * self.point_value
        if contract_notional <= 0:
            return 0

        fractional_contracts = risk_amount / contract_notional

        # Apply minimum position threshold
        min_contracts = (account_equity * self.min_position_pct) / contract_notional
        if fractional_contracts < min_contracts:
            return 0

        return self._round_to_contracts(fractional_contracts)

    def calculate_optimal_f(
        self,
        returns: np.ndarray,
        method: str = "empirical",
    ) -> float:
        """
        Calculate optimal f from historical returns.

        Args:
            returns: Array of trade returns (positive and negative)
            method: Calculation method ('empirical' or 'parametric')

        Returns:
            Optimal fraction to risk
        """
        if len(returns) == 0:
            return 0.0

        if method == "empirical":
            # Empirical Kelly from returns
            wins = returns[returns > 0]
            losses = returns[returns < 0]

            if len(wins) == 0 or len(losses) == 0:
                return 0.0

            win_rate = len(wins) / len(returns)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())

            return self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)

        elif method == "parametric":
            # Parametric Kelly assuming normal returns
            mean_return = returns.mean()
            var_return = returns.var()

            if var_return <= 0:
                return 0.0

            # For normal returns: f* = mu / sigma^2
            return float(mean_return / var_return)

        return 0.0


@dataclass
class FixedFractional(BasePositionSizer):
    """
    Fixed fractional position sizing.

    Risk a fixed percentage of equity per trade based on the
    distance to the stop loss.

    Position Size = (Equity * Risk%) / (Stop Distance * Point Value)

    Attributes:
        risk_per_trade: Fraction of equity to risk per trade (e.g., 0.02 = 2%)
        max_contracts: Maximum contracts regardless of equity
        min_contracts: Minimum contracts to trade (0 means skip small positions)
        point_value: Dollar value per point
    """

    risk_per_trade: float = 0.02
    max_contracts: int = 100
    min_contracts: int = 1
    point_value: float = 5.0

    def calculate_position_size(
        self,
        account_equity: float,
        stop_distance: float = 10.0,
        **kwargs: Any,
    ) -> int:
        """
        Calculate position size based on stop distance.

        Args:
            account_equity: Current account equity
            stop_distance: Distance to stop loss in points

        Returns:
            Number of contracts to trade
        """
        if stop_distance <= 0:
            return 0

        # Dollar risk per trade
        risk_amount = account_equity * self.risk_per_trade

        # Risk per contract = stop distance * point value
        risk_per_contract = stop_distance * self.point_value

        # Calculate contracts
        fractional_contracts = risk_amount / risk_per_contract

        # Apply limits
        contracts = self._round_to_contracts(fractional_contracts)
        contracts = np.clip(contracts, self.min_contracts, self.max_contracts)

        # If below minimum, don't trade
        if fractional_contracts < self.min_contracts:
            return 0

        return int(contracts)


@dataclass
class VolatilityTargeted(BasePositionSizer):
    """
    Volatility-targeted position sizing.

    Scale position size inversely with volatility to maintain
    consistent risk across different market conditions.

    Position Size = Target Vol / Current Vol * Base Position

    Attributes:
        target_volatility: Target annualized volatility (e.g., 0.10 = 10%)
        base_position_pct: Base position as % of equity at target vol
        max_position_pct: Maximum position as % of equity
        min_position_pct: Minimum position as % of equity
        point_value: Dollar value per point
        trading_days: Trading days per year (for annualization)
    """

    target_volatility: float = 0.10
    base_position_pct: float = 0.05
    max_position_pct: float = 0.15
    min_position_pct: float = 0.01
    point_value: float = 5.0
    trading_days: int = 252

    def calculate_position_size(
        self,
        account_equity: float,
        current_volatility: float = 0.15,
        current_price: float = 5000.0,
        **kwargs: Any,
    ) -> int:
        """
        Calculate volatility-adjusted position size.

        Args:
            account_equity: Current account equity
            current_volatility: Current annualized volatility
            current_price: Current price of the instrument

        Returns:
            Number of contracts to trade
        """
        if current_volatility <= 0:
            current_volatility = self.target_volatility

        # Scale factor: target_vol / current_vol
        vol_scalar = self.target_volatility / current_volatility

        # Base position adjusted by volatility
        adjusted_pct = self.base_position_pct * vol_scalar

        # Clamp to limits
        adjusted_pct = np.clip(adjusted_pct, self.min_position_pct, self.max_position_pct)

        # Convert to dollar amount
        position_value = account_equity * adjusted_pct

        # Convert to contracts
        contract_notional = current_price * self.point_value
        if contract_notional <= 0:
            return 0

        fractional_contracts = position_value / contract_notional

        return self._round_to_contracts(fractional_contracts)

    def calculate_realized_volatility(
        self,
        returns: np.ndarray,
        window: int = 20,
    ) -> float:
        """
        Calculate realized volatility from returns.

        Args:
            returns: Array of returns
            window: Lookback window

        Returns:
            Annualized volatility
        """
        if len(returns) < window:
            return self.target_volatility

        # Rolling standard deviation
        recent_returns = returns[-window:]
        daily_vol = np.std(recent_returns)

        # Annualize
        annualized_vol = daily_vol * np.sqrt(self.trading_days)

        return float(annualized_vol)


@dataclass
class EqualWeight(BasePositionSizer):
    """
    Equal weight position sizing.

    Allocate a fixed percentage of equity to each position.

    Attributes:
        weight_pct: Fraction of equity per position (e.g., 0.05 = 5%)
        max_contracts: Maximum contracts per position
        point_value: Dollar value per point
    """

    weight_pct: float = 0.05
    max_contracts: int = 50
    point_value: float = 5.0

    def calculate_position_size(
        self,
        account_equity: float,
        current_price: float = 5000.0,
        **kwargs: Any,
    ) -> int:
        """
        Calculate equal-weighted position size.

        Args:
            account_equity: Current account equity
            current_price: Current price

        Returns:
            Number of contracts to trade
        """
        # Allocate fixed percentage
        position_value = account_equity * self.weight_pct

        # Convert to contracts
        contract_notional = current_price * self.point_value
        if contract_notional <= 0:
            return 0

        fractional_contracts = position_value / contract_notional
        contracts = self._round_to_contracts(fractional_contracts)

        return min(contracts, self.max_contracts)


@dataclass
class FixedContracts(BasePositionSizer):
    """
    Fixed number of contracts regardless of equity.

    Simple position sizing for testing or small accounts.

    Attributes:
        contracts: Fixed number of contracts per trade
    """

    contracts: int = 1

    def calculate_position_size(
        self,
        account_equity: float,
        **kwargs: Any,
    ) -> int:
        """Return fixed number of contracts."""
        return self.contracts


@dataclass
class BetSizingPositioner(BasePositionSizer):
    """
    Variable position sizing based on model confidence (Phase 4G).

    Connects meta-labeling bet sizing to position sizing.
    Uses model prediction probability to determine position size.

    Attributes:
        strategy: Bet sizing strategy ("binary", "proportional", "kelly", etc.)
        threshold: Minimum probability to trade
        max_size: Maximum position size (fraction of equity)
        min_size: Minimum position size if trading
        kelly_fraction: Fraction of Kelly to use (for Kelly strategies)
        point_value: Dollar value per point
    """

    strategy: str = "binary"
    threshold: float = 0.5
    max_size: float = 1.0
    min_size: float = 0.0
    kelly_fraction: float = 0.5
    point_value: float = 5.0

    def calculate_position_size(
        self,
        account_equity: float,
        probability: float = 0.5,
        current_price: float = 5000.0,
        **kwargs: Any,
    ) -> int:
        """
        Calculate position size based on model confidence.

        Args:
            account_equity: Current account equity
            probability: Model confidence/probability (0-1)
            current_price: Current instrument price
            **kwargs: Additional parameters

        Returns:
            Number of contracts to trade
        """
        try:
            from src.models.training.meta_labeling.bet_sizing import compute_bet_sizes

            # Compute fractional position size
            size_fraction = compute_bet_sizes(
                probabilities=np.array([probability]),
                strategy=self.strategy,
                threshold=self.threshold,
                max_size=self.max_size,
                min_size=self.min_size,
                kelly_fraction=self.kelly_fraction,
            )[0]

            if size_fraction == 0:
                return 0

            # Convert fraction to contracts
            dollar_amount = account_equity * size_fraction
            contract_notional = current_price * self.point_value

            if contract_notional <= 0:
                return 0

            fractional_contracts = dollar_amount / contract_notional
            return self._round_to_contracts(fractional_contracts)

        except ImportError:
            # Fallback to binary if bet sizing module unavailable
            if probability > self.threshold:
                dollar_amount = account_equity * self.max_size
                contract_notional = current_price * self.point_value
                if contract_notional > 0:
                    return self._round_to_contracts(dollar_amount / contract_notional)
            return 0


@dataclass
class PositionSizerConfig:
    """Configuration for position sizing."""

    method: PositionSizingMethod = PositionSizingMethod.FIXED_FRACTIONAL
    risk_per_trade: float = 0.02
    kelly_fraction: float = 0.25
    target_volatility: float = 0.10
    max_position_pct: float = 0.15
    min_position_pct: float = 0.01
    fixed_contracts: int = 1
    point_value: float = 5.0
    max_contracts: int = 100
    # Phase 4G: Bet sizing parameters
    bet_sizing_strategy: str = "binary"
    bet_sizing_threshold: float = 0.5
    bet_sizing_max_size: float = 1.0


def create_position_sizer(
    method: str | PositionSizingMethod = "fixed_fractional",
    **kwargs: Any,
) -> BasePositionSizer:
    """
    Factory function to create a position sizer.

    Args:
        method: Position sizing method name or enum
        **kwargs: Method-specific parameters

    Returns:
        Configured position sizer instance
    """
    if isinstance(method, str):
        method = PositionSizingMethod(method.lower())

    point_value = kwargs.get("point_value", 5.0)

    if method == PositionSizingMethod.KELLY:
        return KellyCriterion(
            kelly_fraction=kwargs.get("kelly_fraction", 0.25),
            max_position_pct=kwargs.get("max_position_pct", 0.10),
            min_position_pct=kwargs.get("min_position_pct", 0.01),
            point_value=point_value,
        )

    elif method == PositionSizingMethod.FIXED_FRACTIONAL:
        return FixedFractional(
            risk_per_trade=kwargs.get("risk_per_trade", 0.02),
            max_contracts=kwargs.get("max_contracts", 100),
            min_contracts=kwargs.get("min_contracts", 1),
            point_value=point_value,
        )

    elif method == PositionSizingMethod.VOLATILITY_TARGETED:
        return VolatilityTargeted(
            target_volatility=kwargs.get("target_volatility", 0.10),
            base_position_pct=kwargs.get("base_position_pct", 0.05),
            max_position_pct=kwargs.get("max_position_pct", 0.15),
            min_position_pct=kwargs.get("min_position_pct", 0.01),
            point_value=point_value,
        )

    elif method == PositionSizingMethod.EQUAL_WEIGHT:
        return EqualWeight(
            weight_pct=kwargs.get("weight_pct", 0.05),
            max_contracts=kwargs.get("max_contracts", 50),
            point_value=point_value,
        )

    elif method == PositionSizingMethod.FIXED_CONTRACTS:
        return FixedContracts(
            contracts=kwargs.get("contracts", 1),
        )

    elif method == PositionSizingMethod.BET_SIZING:
        return BetSizingPositioner(
            strategy=kwargs.get("strategy", "binary"),
            threshold=kwargs.get("threshold", 0.5),
            max_size=kwargs.get("max_size", 1.0),
            min_size=kwargs.get("min_size", 0.0),
            kelly_fraction=kwargs.get("kelly_fraction", 0.5),
            point_value=point_value,
        )

    else:
        return FixedFractional(
            risk_per_trade=0.02,
            point_value=point_value,
        )


__all__ = [
    "PositionSizingMethod",
    "BasePositionSizer",
    "KellyCriterion",
    "FixedFractional",
    "VolatilityTargeted",
    "EqualWeight",
    "FixedContracts",
    "BetSizingPositioner",  # Phase 4G
    "PositionSizerConfig",
    "create_position_sizer",
]
