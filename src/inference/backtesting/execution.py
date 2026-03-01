"""
Execution models with realistic constraints.

Phase 12A-3: Market Hours Filtering

Includes:
- Market hours filtering (CME calendar)
- Adverse selection bias
- Volume-relative position limits
"""

from __future__ import annotations

from datetime import datetime, time
from typing import TYPE_CHECKING

import pytz

if TYPE_CHECKING:
    pass


# Per-contract liquid session times (Eastern Time)
# Equity micros: NYSE cash session 9:30-16:00 ET
# Gold micros: COMEX primary session 8:20-13:30 ET (London/NY overlap)
CONTRACT_SESSION_TIMES: dict[str, tuple[time, time]] = {
    "MES": (time(9, 30), time(16, 0)),
    "ES": (time(9, 30), time(16, 0)),
    "MNQ": (time(9, 30), time(16, 0)),
    "NQ": (time(9, 30), time(16, 0)),
    "MGC": (time(8, 20), time(13, 30)),
    "GC": (time(8, 20), time(13, 30)),
}
DEFAULT_SESSION = (time(9, 30), time(16, 0))

# Legacy aliases for backward compatibility
NY_SESSION_START = time(9, 30)  # 9:30 AM ET
NY_SESSION_END = time(16, 0)  # 4:00 PM ET

# Eastern timezone
ET_TZ = pytz.timezone("US/Eastern")


class MarketHoursFilter:
    """
    Execution model with realistic trading constraints.

    Filters trades outside liquid market hours and applies
    adverse selection adjustments to fill prices.
    """

    def __init__(
        self,
        contract: str = "MES",
        enable_market_hours_filter: bool = True,
        enable_adverse_selection: bool = True,
        base_volatility: float = 0.15,
        adverse_base_ticks: float = 0.5,
        adverse_vol_scale: float = 0.5,
        max_participation: float = 0.01,
    ) -> None:
        """
        Initialize execution model.

        Args:
            contract: Trading contract (MES, MGC, MNQ)
            enable_market_hours_filter: Filter trades outside NY session
            enable_adverse_selection: Apply adverse selection to fills
            base_volatility: Baseline annualized volatility for adverse selection scaling
            adverse_base_ticks: Base adverse selection in ticks
            adverse_vol_scale: Volatility scaling coefficient for adverse selection
            max_participation: Max fraction of volume for position sizing (1% default)
        """
        self.contract = contract
        self.enable_market_hours_filter = enable_market_hours_filter
        self.enable_adverse_selection = enable_adverse_selection
        self.base_volatility = base_volatility
        self.adverse_base_ticks = adverse_base_ticks
        self.adverse_vol_scale = adverse_vol_scale
        self.max_participation = max_participation
        self._calendar: CMECalendar | None = None

        # Resolve per-contract session times
        session = CONTRACT_SESSION_TIMES.get(contract.upper(), DEFAULT_SESSION)
        self._session_start = session[0]
        self._session_end = session[1]

    @property
    def calendar(self) -> CMECalendar:
        """Lazy-load CME calendar."""
        if self._calendar is None:
            from src.data.pipeline.stages.sessions import CMECalendar

            self._calendar = CMECalendar()
        return self._calendar

    def is_tradeable_time(self, timestamp: datetime) -> bool:
        """
        Check if timestamp is during liquid trading hours.

        Filters:
        - CME holidays
        - Weekend days
        - Outside NY session (9:30 AM - 4:00 PM ET)

        Args:
            timestamp: UTC or timezone-aware datetime

        Returns:
            True if timestamp is tradeable
        """
        if not self.enable_market_hours_filter:
            return True

        # Convert to Eastern Time if needed
        if timestamp.tzinfo is None:
            # Assume UTC if no timezone
            timestamp = pytz.UTC.localize(timestamp)

        et_time = timestamp.astimezone(ET_TZ)

        # Check weekend
        if et_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check holidays
        if self.calendar.is_holiday(et_time):
            return False

        # Check if within liquid session hours (per-contract)
        current_time = et_time.time()
        return self._session_start <= current_time < self._session_end

    def apply_adverse_selection(
        self,
        signal_price: float,
        signal_direction: int,
        realized_volatility: float,
    ) -> float:
        """
        Adjust fill price for adverse selection.

        When a model predicts a move, the market is often already moving
        in that direction, leading to worse fills than the signal price.

        Args:
            signal_price: Price when signal was generated
            signal_direction: +1 for long, -1 for short
            realized_volatility: Recent realized volatility (annualized)

        Returns:
            Adjusted fill price accounting for adverse selection
        """
        if not self.enable_adverse_selection:
            return signal_price

        # Adverse selection in ticks, scaled by volatility
        # Higher volatility = more adverse selection
        vol_ratio = realized_volatility / self.base_volatility if self.base_volatility > 0 else 1.0
        adverse_ticks = self.adverse_base_ticks + self.adverse_vol_scale * vol_ratio

        # Get tick value for contract
        tick_value = self._get_tick_value()

        # Apply adverse selection in direction of trade
        # Long = we pay more, Short = we receive less
        if signal_direction > 0:  # Long
            return signal_price + adverse_ticks * tick_value
        else:  # Short
            return signal_price - adverse_ticks * tick_value

    def calculate_max_position_size(
        self,
        avg_volume: float,
        max_participation: float | None = None,
    ) -> int:
        """
        Limit position size to fraction of market volume.

        Prevents excessive market impact by limiting participation rate.

        Args:
            avg_volume: Rolling average volume (contracts)
            max_participation: Max fraction of volume (overrides constructor default)

        Returns:
            Maximum contracts tradeable without excessive market impact
        """
        participation = (
            max_participation if max_participation is not None else self.max_participation
        )
        max_contracts = int(avg_volume * participation)
        return max(1, max_contracts)  # At least 1 contract

    def _get_tick_value(self) -> float:
        """Get tick value for the contract."""
        tick_values = {
            "MES": 0.25,  # Micro E-mini S&P 500
            "MGC": 0.10,  # Micro Gold
            "MNQ": 0.25,  # Micro E-mini Nasdaq-100
            "ES": 0.25,  # E-mini S&P 500
            "NQ": 0.25,  # E-mini Nasdaq-100
            "GC": 0.10,  # Gold
        }
        return tick_values.get(self.contract.upper(), 0.25)


class CMECalendar:
    """
    Fallback CME calendar for holiday checking.

    Uses the full calendar from sessions module when available,
    provides basic weekend checking otherwise.
    """

    def __init__(self) -> None:
        """Initialize calendar."""
        self._full_calendar: object | None = None
        try:
            from src.data.pipeline.stages.sessions import CMECalendar as FullCalendar

            self._full_calendar = FullCalendar()
        except ImportError:
            pass

    def is_holiday(self, timestamp: datetime) -> bool:
        """Check if timestamp is a CME holiday."""
        if self._full_calendar is not None:
            return self._full_calendar.is_holiday(timestamp)

        # Fallback: basic weekend check (holidays not available)
        return timestamp.weekday() >= 5


def create_market_hours_filter(
    contract: str = "MES",
    enable_market_hours_filter: bool = True,
    enable_adverse_selection: bool = True,
) -> MarketHoursFilter:
    """
    Factory function to create market hours filter.

    Args:
        contract: Trading contract
        enable_market_hours_filter: Filter trades outside market hours
        enable_adverse_selection: Apply adverse selection to fills

    Returns:
        Configured MarketHoursFilter instance
    """
    return MarketHoursFilter(
        contract=contract,
        enable_market_hours_filter=enable_market_hours_filter,
        enable_adverse_selection=enable_adverse_selection,
    )


__all__ = [
    "MarketHoursFilter",
    "CMECalendar",
    "create_market_hours_filter",
    "CONTRACT_SESSION_TIMES",
    "DEFAULT_SESSION",
    "NY_SESSION_START",
    "NY_SESSION_END",
]
