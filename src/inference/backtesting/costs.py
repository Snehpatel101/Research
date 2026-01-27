"""
Transaction costs and slippage models for backtesting.

This module provides realistic cost modeling for futures trading including:
- Commission costs (per-contract or percentage)
- Slippage models (fixed, linear, square-root impact)
- Market impact estimation

Known limitation: Overnight financing / carry / swap costs are not modeled.
For intraday futures strategies (the primary use case), this is negligible.
For multi-day holding periods, users should account for roll costs and
margin interest externally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SlippageModel(Enum):
    """Slippage model types."""

    FIXED = "fixed"
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    VOLATILITY_SCALED = "volatility_scaled"


@dataclass
class TransactionCosts:
    """
    Transaction cost configuration for futures trading.

    Attributes:
        commission_per_contract: Round-trip commission per contract (dollars)
        slippage_ticks: Expected slippage in ticks (one-way)
        tick_value: Dollar value per tick (e.g., 1.25 for MES, 0.10 for MGC)
        tick_size: Minimum price increment (e.g., 0.25 for MES, 0.10 for MGC)
        exchange_fee: Exchange fee per contract (round-trip)
        nfa_fee: NFA regulatory fee per contract (round-trip)
    """

    commission_per_contract: float = 2.50
    slippage_ticks: float = 1.0
    tick_value: float = 1.25
    tick_size: float = 0.25
    exchange_fee: float = 0.52
    nfa_fee: float = 0.02

    @property
    def total_fixed_cost_per_contract(self) -> float:
        """Total fixed costs per contract (round-trip)."""
        return self.commission_per_contract + self.exchange_fee + self.nfa_fee

    @property
    def slippage_cost_per_contract(self) -> float:
        """Expected slippage cost per contract (round-trip = 2x one-way)."""
        return 2 * self.slippage_ticks * self.tick_value

    @property
    def total_cost_per_contract(self) -> float:
        """Total expected cost per contract (round-trip)."""
        return self.total_fixed_cost_per_contract + self.slippage_cost_per_contract

    def calculate_entry_cost(self, contracts: int, entry_price: float) -> float:
        """
        Calculate cost of entering a position.

        Args:
            contracts: Number of contracts
            entry_price: Entry price

        Returns:
            Total entry cost in dollars
        """
        fixed_cost = self.commission_per_contract / 2 + self.exchange_fee / 2 + self.nfa_fee / 2
        slippage_cost = self.slippage_ticks * self.tick_value
        return contracts * (fixed_cost + slippage_cost)

    def calculate_exit_cost(self, contracts: int, exit_price: float) -> float:
        """
        Calculate cost of exiting a position.

        Args:
            contracts: Number of contracts
            exit_price: Exit price

        Returns:
            Total exit cost in dollars
        """
        return self.calculate_entry_cost(contracts, exit_price)

    def calculate_round_trip_cost(self, contracts: int) -> float:
        """
        Calculate total round-trip cost for a trade.

        Args:
            contracts: Number of contracts

        Returns:
            Total round-trip cost in dollars
        """
        return contracts * self.total_cost_per_contract

    @classmethod
    def for_mes(cls) -> TransactionCosts:
        """Create transaction costs for Micro E-mini S&P 500 (MES)."""
        return cls(
            commission_per_contract=2.50,
            slippage_ticks=1.0,
            tick_value=1.25,
            tick_size=0.25,
            exchange_fee=0.52,
            nfa_fee=0.02,
        )

    @classmethod
    def for_mgc(cls) -> TransactionCosts:
        """Create transaction costs for Micro Gold (MGC)."""
        return cls(
            commission_per_contract=2.50,
            slippage_ticks=1.0,
            tick_value=1.00,
            tick_size=0.10,
            exchange_fee=0.52,
            nfa_fee=0.02,
        )

    @classmethod
    def for_mnq(cls) -> TransactionCosts:
        """Create transaction costs for Micro E-mini Nasdaq-100 (MNQ)."""
        return cls(
            commission_per_contract=2.50,
            slippage_ticks=1.0,
            tick_value=0.50,
            tick_size=0.25,
            exchange_fee=0.52,
            nfa_fee=0.02,
        )


class BaseSlippageModel(ABC):
    """Abstract base class for slippage models."""

    @abstractmethod
    def estimate_slippage(
        self,
        order_size: int,
        price: float,
        volatility: float | None = None,
        spread: float | None = None,
        volume: float | None = None,
    ) -> float:
        """
        Estimate slippage for an order.

        Args:
            order_size: Number of contracts
            price: Current price
            volatility: Current volatility (optional)
            spread: Current bid-ask spread (optional)
            volume: Current volume (optional)

        Returns:
            Estimated slippage in price units
        """
        pass


@dataclass
class FixedSlippage(BaseSlippageModel):
    """
    Fixed slippage model - constant slippage per trade.

    Simple model assuming fixed number of ticks slippage regardless
    of order size or market conditions.
    """

    ticks: float = 1.0
    tick_size: float = 0.25

    def estimate_slippage(
        self,
        order_size: int,
        price: float,
        volatility: float | None = None,
        spread: float | None = None,
        volume: float | None = None,
    ) -> float:
        """Return fixed slippage in price units."""
        return self.ticks * self.tick_size


@dataclass
class LinearSlippage(BaseSlippageModel):
    """
    Linear slippage model - slippage scales linearly with order size.

    Slippage = base_ticks + (order_size * size_factor) ticks

    More realistic for larger orders that move the market.
    """

    base_ticks: float = 0.5
    size_factor: float = 0.1
    tick_size: float = 0.25
    max_slippage_ticks: float = 10.0

    def estimate_slippage(
        self,
        order_size: int,
        price: float,
        volatility: float | None = None,
        spread: float | None = None,
        volume: float | None = None,
    ) -> float:
        """Return linear slippage scaled by order size."""
        slippage_ticks = self.base_ticks + order_size * self.size_factor
        slippage_ticks = min(slippage_ticks, self.max_slippage_ticks)
        return slippage_ticks * self.tick_size


@dataclass
class SquareRootSlippage(BaseSlippageModel):
    """
    Square root market impact model.

    Based on the empirical finding that market impact scales with
    the square root of order size (Kyle, 1985; Almgren, 2005).

    Slippage = impact_coefficient * sqrt(order_size / typical_volume) * volatility * price

    This is the most realistic model for institutional-scale orders.
    """

    impact_coefficient: float = 0.1
    typical_volume: float = 1000.0
    tick_size: float = 0.25

    def estimate_slippage(
        self,
        order_size: int,
        price: float,
        volatility: float | None = None,
        spread: float | None = None,
        volume: float | None = None,
    ) -> float:
        """
        Estimate slippage using square root impact model.

        Args:
            order_size: Number of contracts
            price: Current price
            volatility: Annualized volatility (default: 0.15 = 15%)
            spread: Bid-ask spread (not used in this model)
            volume: Current volume (default: typical_volume)

        Returns:
            Estimated slippage in price units
        """
        vol = volatility if volatility is not None else 0.15
        vol_used = volume if volume is not None else self.typical_volume

        # Prevent division by zero
        if vol_used <= 0:
            vol_used = self.typical_volume

        # Square root impact
        participation_rate = order_size / vol_used
        impact = self.impact_coefficient * np.sqrt(participation_rate) * vol * price

        # Round to tick size
        return float(np.ceil(impact / self.tick_size) * self.tick_size)


@dataclass
class VolatilityScaledSlippage(BaseSlippageModel):
    """
    Volatility-scaled slippage model.

    Slippage scales with market volatility - more slippage in
    volatile conditions, less in calm markets.

    Slippage = base_ticks * (1 + volatility_multiplier * volatility / base_volatility)
    """

    base_ticks: float = 1.0
    base_volatility: float = 0.15
    volatility_multiplier: float = 2.0
    tick_size: float = 0.25
    min_slippage_ticks: float = 0.5
    max_slippage_ticks: float = 5.0

    def estimate_slippage(
        self,
        order_size: int,
        price: float,
        volatility: float | None = None,
        spread: float | None = None,
        volume: float | None = None,
    ) -> float:
        """
        Estimate volatility-scaled slippage.

        Args:
            order_size: Number of contracts
            price: Current price
            volatility: Current volatility (default: base_volatility)
            spread: Bid-ask spread (not used)
            volume: Volume (not used)

        Returns:
            Estimated slippage in price units
        """
        vol = volatility if volatility is not None else self.base_volatility

        # Scale by volatility ratio
        vol_ratio = vol / self.base_volatility
        scaled_ticks = self.base_ticks * (1 + self.volatility_multiplier * (vol_ratio - 1))

        # Clamp to bounds
        scaled_ticks = np.clip(scaled_ticks, self.min_slippage_ticks, self.max_slippage_ticks)

        return float(scaled_ticks * self.tick_size)


@dataclass
class CostCalculator:
    """
    Unified cost calculator combining transaction costs and slippage.

    This is the main interface for calculating total trading costs
    in the backtester.
    """

    transaction_costs: TransactionCosts = field(default_factory=TransactionCosts.for_mes)
    # Phase 12A-2: Use VolatilityScaledSlippage as default for more realistic costs
    slippage_model: BaseSlippageModel = field(default_factory=VolatilityScaledSlippage)

    def calculate_trade_cost(
        self,
        contracts: int,
        entry_price: float,
        exit_price: float,
        entry_volatility: float | None = None,
        exit_volatility: float | None = None,
    ) -> dict[str, float]:
        """
        Calculate total cost for a complete trade (round-trip).

        Args:
            contracts: Number of contracts traded
            entry_price: Entry price
            exit_price: Exit price
            entry_volatility: Volatility at entry
            exit_volatility: Volatility at exit

        Returns:
            Dict with cost breakdown:
                - commission: Total commission costs
                - entry_slippage: Entry slippage in dollars
                - exit_slippage: Exit slippage in dollars
                - total_slippage: Total slippage in dollars
                - total_cost: Total cost in dollars
                - cost_per_contract: Average cost per contract
        """
        # Commission costs (round-trip)
        commission = contracts * self.transaction_costs.total_fixed_cost_per_contract

        # Slippage at entry
        entry_slip_price = self.slippage_model.estimate_slippage(
            contracts, entry_price, volatility=entry_volatility
        )
        entry_slippage = (
            contracts
            * (entry_slip_price / self.transaction_costs.tick_size)
            * self.transaction_costs.tick_value
        )

        # Slippage at exit
        exit_slip_price = self.slippage_model.estimate_slippage(
            contracts, exit_price, volatility=exit_volatility
        )
        exit_slippage = (
            contracts
            * (exit_slip_price / self.transaction_costs.tick_size)
            * self.transaction_costs.tick_value
        )

        total_slippage = entry_slippage + exit_slippage
        total_cost = commission + total_slippage

        return {
            "commission": commission,
            "entry_slippage": entry_slippage,
            "exit_slippage": exit_slippage,
            "total_slippage": total_slippage,
            "total_cost": total_cost,
            "cost_per_contract": total_cost / contracts if contracts > 0 else 0.0,
        }

    def calculate_pnl(
        self,
        contracts: int,
        entry_price: float,
        exit_price: float,
        direction: int,
        point_value: float = 5.0,
        include_costs: bool = True,
        entry_volatility: float | None = None,
        exit_volatility: float | None = None,
    ) -> dict[str, float]:
        """
        Calculate P&L for a trade including costs.

        Args:
            contracts: Number of contracts
            entry_price: Entry price
            exit_price: Exit price
            direction: 1 for long, -1 for short
            point_value: Dollar value per point (e.g., 5 for MES, 10 for MGC)
            include_costs: Whether to include transaction costs
            entry_volatility: Volatility at entry
            exit_volatility: Volatility at exit

        Returns:
            Dict with P&L breakdown:
                - gross_pnl: P&L before costs
                - costs: Total costs
                - net_pnl: P&L after costs
                - return_pct: Return as percentage of entry value
        """
        # Gross P&L (before costs)
        price_change = exit_price - entry_price
        gross_pnl = direction * contracts * price_change * point_value

        # Costs
        if include_costs:
            cost_breakdown = self.calculate_trade_cost(
                contracts, entry_price, exit_price, entry_volatility, exit_volatility
            )
            total_costs = cost_breakdown["total_cost"]
        else:
            total_costs = 0.0

        # Net P&L
        net_pnl = gross_pnl - total_costs

        # Return percentage (relative to entry notional)
        entry_notional = contracts * entry_price * point_value
        return_pct = (net_pnl / entry_notional * 100) if entry_notional > 0 else 0.0

        return {
            "gross_pnl": gross_pnl,
            "costs": total_costs,
            "net_pnl": net_pnl,
            "return_pct": return_pct,
        }


def create_cost_calculator(
    symbol: str = "MES",
    slippage_model: str = "fixed",
    **kwargs: Any,
) -> CostCalculator:
    """
    Factory function to create cost calculator for a symbol.

    Args:
        symbol: Trading symbol (MES, MGC, MNQ)
        slippage_model: Type of slippage model (fixed, linear, square_root, volatility_scaled)
        **kwargs: Additional parameters for slippage model

    Returns:
        Configured CostCalculator
    """
    # Get transaction costs for symbol
    symbol_upper = symbol.upper()
    if symbol_upper == "MES":
        tx_costs = TransactionCosts.for_mes()
    elif symbol_upper == "MGC":
        tx_costs = TransactionCosts.for_mgc()
    elif symbol_upper == "MNQ":
        tx_costs = TransactionCosts.for_mnq()
    else:
        tx_costs = TransactionCosts.for_mes()

    # Create slippage model
    tick_size = tx_costs.tick_size
    slippage_model_lower = slippage_model.lower()

    slip_model: BaseSlippageModel
    if slippage_model_lower == "fixed":
        slip_model = FixedSlippage(
            ticks=kwargs.get("slippage_ticks", 1.0),
            tick_size=tick_size,
        )
    elif slippage_model_lower == "linear":
        slip_model = LinearSlippage(
            base_ticks=kwargs.get("base_ticks", 0.5),
            size_factor=kwargs.get("size_factor", 0.1),
            tick_size=tick_size,
        )
    elif slippage_model_lower == "square_root":
        slip_model = SquareRootSlippage(
            impact_coefficient=kwargs.get("impact_coefficient", 0.1),
            typical_volume=kwargs.get("typical_volume", 1000.0),
            tick_size=tick_size,
        )
    elif slippage_model_lower == "volatility_scaled":
        slip_model = VolatilityScaledSlippage(
            base_ticks=kwargs.get("base_ticks", 1.0),
            base_volatility=kwargs.get("base_volatility", 0.15),
            volatility_multiplier=kwargs.get("volatility_multiplier", 2.0),
            tick_size=tick_size,
        )
    else:
        slip_model = FixedSlippage(ticks=1.0, tick_size=tick_size)

    return CostCalculator(transaction_costs=tx_costs, slippage_model=slip_model)


__all__ = [
    "TransactionCosts",
    "SlippageModel",
    "BaseSlippageModel",
    "FixedSlippage",
    "LinearSlippage",
    "SquareRootSlippage",
    "VolatilityScaledSlippage",
    "CostCalculator",
    "create_cost_calculator",
]
