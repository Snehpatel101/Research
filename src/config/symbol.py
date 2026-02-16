"""
Symbol Configuration - Contract specifications for traded instruments.

This module provides SymbolConfig, the canonical source of truth for
per-symbol contract specifications (tick value, tick size, point value, etc.).

Usage:
    from src.config.symbol import SymbolConfig

    # Get config for a known symbol
    mes = SymbolConfig.from_symbol("MES")

    # Use preset class methods
    mgc = SymbolConfig.for_mgc()

    # Custom symbol
    custom = SymbolConfig(
        symbol="ZB",
        tick_value=31.25,
        tick_size=1 / 32,
        point_value=1000.0,
        exchange="CBOT",
    )
"""

from __future__ import annotations

from dataclasses import dataclass

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

    @classmethod
    def from_symbol(cls, symbol: str) -> SymbolConfig:
        """
        Get SymbolConfig for a known symbol, or a default for unknown ones.

        Args:
            symbol: Instrument symbol (case-insensitive)

        Returns:
            SymbolConfig with appropriate contract specifications
        """
        presets = {
            "MES": cls.for_mes,
            "MGC": cls.for_mgc,
            "MNQ": cls.for_mnq,
        }
        key = symbol.upper()
        if key in presets:
            return presets[key]()
        # Unknown symbol — return defaults with the given symbol name
        return cls(symbol=key)


__all__ = [
    "SymbolConfig",
]
