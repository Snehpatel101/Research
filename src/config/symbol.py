"""
Symbol Configuration - Contract specifications for traded instruments.

This module provides SymbolConfig, the canonical source of truth for
per-symbol contract specifications (tick value, tick size, point value, etc.).

Usage:
    from src.config.symbol import SymbolConfig

    # Get config for a known symbol (raises ValueError if unknown)
    mes = SymbolConfig.from_symbol("MES")

    # Use preset class methods
    mgc = SymbolConfig.for_mgc()

    # Explicit fallback to MES defaults for unknown symbols
    cfg = SymbolConfig.from_symbol_or_default("ZB")

    # Fully custom symbol (no lookup needed)
    custom = SymbolConfig(
        symbol="ZB",
        tick_value=31.25,
        tick_size=1 / 32,
        point_value=1000.0,
        exchange="CBOT",
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from src.config.base import BaseConfig


@dataclass
class SymbolConfig(BaseConfig):
    """
    Contract specifications for a traded instrument.

    This is the CANONICAL source for per-symbol contract specs. Use this
    instead of hardcoded dicts in:
    - src/inference/backtesting/backtest.py (BacktestConfig.for_mes/for_mgc)
    - src/inference/backtesting/costs.py (TransactionCosts.for_mes/for_mgc)
    - src/inference/backtesting/execution.py (_get_tick_value)
    - src/data/pipeline/config/barriers_config.py (TICK_VALUES)

    Attributes:
        symbol: Instrument symbol (e.g., "MES", "MGC")
        tick_value: Dollar value per tick
        tick_size: Minimum price increment
        point_value: Dollar value per point (full contract value factor)
        exchange: Exchange name (optional)
        contract_size: Contract multiplier (optional)
    """

    symbol: str = "MES"
    tick_value: float = 1.25
    tick_size: float = 0.25
    point_value: float = 5.0
    exchange: str = ""
    contract_size: float = 1.0

    def validate(self) -> list[str]:
        """Validate symbol configuration."""
        issues = super().validate()

        if not self.symbol:
            issues.append("symbol is required")

        if self.tick_value <= 0:
            issues.append(f"tick_value must be positive, got {self.tick_value}")

        if self.tick_size <= 0:
            issues.append(f"tick_size must be positive, got {self.tick_size}")

        if self.point_value <= 0:
            issues.append(f"point_value must be positive, got {self.point_value}")

        if self.contract_size <= 0:
            issues.append(f"contract_size must be positive, got {self.contract_size}")

        return issues

    # -------------------------------------------------------------------------
    # Presets for known symbols
    # -------------------------------------------------------------------------

    @classmethod
    def for_mes(cls) -> SymbolConfig:
        """Micro E-mini S&P 500."""
        return cls(
            symbol="MES",
            tick_value=1.25,
            tick_size=0.25,
            point_value=5.0,
            exchange="CME",
        )

    @classmethod
    def for_mgc(cls) -> SymbolConfig:
        """Micro Gold."""
        return cls(
            symbol="MGC",
            tick_value=1.00,
            tick_size=0.10,
            point_value=10.0,
            exchange="COMEX",
        )

    @classmethod
    def for_mnq(cls) -> SymbolConfig:
        """Micro E-mini Nasdaq-100."""
        return cls(
            symbol="MNQ",
            tick_value=0.50,
            tick_size=0.25,
            point_value=2.0,
            exchange="CME",
        )

    # -------------------------------------------------------------------------
    # Factory method
    # -------------------------------------------------------------------------

    # Registry of known symbol presets (populated after class definition)
    _PRESETS: ClassVar[dict[str, Callable[[], SymbolConfig]]] = {}  # noqa: RUF012

    @classmethod
    def supported_symbols(cls) -> list[str]:
        """Return sorted list of supported symbol names."""
        return sorted(cls._PRESETS.keys())

    @classmethod
    def from_symbol(cls, symbol: str) -> SymbolConfig:
        """
        Get SymbolConfig for a known symbol.

        Raises ValueError for unknown symbols to prevent silent fallback to
        incorrect contract specifications.

        Args:
            symbol: Instrument symbol (case-insensitive)

        Returns:
            SymbolConfig with appropriate contract specifications

        Raises:
            ValueError: If symbol is not in the supported presets
        """
        key = symbol.upper()
        if key in cls._PRESETS:
            return cls._PRESETS[key]()
        supported = ", ".join(cls.supported_symbols())
        raise ValueError(
            f"Unknown symbol '{symbol}'. Supported symbols: {supported}. "
            f"If you need a custom symbol, use SymbolConfig(...) directly or "
            f"SymbolConfig.from_symbol_or_default('{symbol}') for explicit "
            f"fallback to MES defaults."
        )

    @classmethod
    def from_symbol_or_default(cls, symbol: str) -> SymbolConfig:
        """
        Get SymbolConfig for a known symbol, or MES defaults for unknown ones.

        Use this method only when you explicitly want fallback behavior.
        Prefer ``from_symbol()`` to catch misconfiguration early.

        Args:
            symbol: Instrument symbol (case-insensitive)

        Returns:
            SymbolConfig with appropriate contract specifications, or
            MES-default SymbolConfig with the given symbol name for
            unknown symbols.
        """
        key = symbol.upper()
        if key in cls._PRESETS:
            return cls._PRESETS[key]()
        # Unknown symbol — return defaults with the given symbol name
        return cls(symbol=key)


# Populate the presets registry after class definition
SymbolConfig._PRESETS = {
    "MES": SymbolConfig.for_mes,
    "MGC": SymbolConfig.for_mgc,
    "MNQ": SymbolConfig.for_mnq,
}


__all__ = [
    "SymbolConfig",
]
