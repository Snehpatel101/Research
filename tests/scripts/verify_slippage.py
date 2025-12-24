#!/usr/bin/env python3
"""
Verification script for slippage modeling.

Demonstrates the new slippage configuration and cost calculation functions.

Usage:
    python scripts/verify_slippage.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.config import (
    SLIPPAGE_TICKS,
    TRANSACTION_COSTS,
    TICK_VALUES,
    get_slippage_ticks,
    get_total_trade_cost,
)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def main():
    """Demonstrate slippage configuration and calculations."""

    print_section("Slippage Configuration")
    print("\nSLIPPAGE_TICKS (per-fill slippage in ticks):")
    for symbol, regimes in SLIPPAGE_TICKS.items():
        print(f"\n{symbol}:")
        for regime, slippage in regimes.items():
            print(f"  {regime:12s}: {slippage:.2f} ticks")

    print("\n\nTRANSACTION_COSTS (commission, round-trip in ticks):")
    for symbol, cost in TRANSACTION_COSTS.items():
        print(f"  {symbol}: {cost:.2f} ticks")

    print_section("Cost Breakdown by Symbol and Regime")

    symbols = ['MES', 'MGC']
    regimes = ['low_vol', 'high_vol']

    for symbol in symbols:
        print(f"\n{symbol} ({TICK_VALUES[symbol]:.2f} $/tick):")
        print(f"  Commission (round-trip):    {TRANSACTION_COSTS[symbol]:.2f} ticks")

        for regime in regimes:
            slippage_per_fill = get_slippage_ticks(symbol, regime)
            total_slippage = 2 * slippage_per_fill
            total_cost = get_total_trade_cost(symbol, regime, include_slippage=True)

            print(f"\n  {regime.upper()} REGIME:")
            print(f"    Slippage per fill:        {slippage_per_fill:.2f} ticks")
            print(f"    Slippage round-trip:      {total_slippage:.2f} ticks")
            print(f"    TOTAL (comm + slip):      {total_cost:.2f} ticks")
            print(f"    TOTAL in dollars:         ${total_cost * TICK_VALUES[symbol]:.2f}")

    print_section("Regime Cost Comparisons")

    for symbol in symbols:
        low_vol_cost = get_total_trade_cost(symbol, 'low_vol')
        high_vol_cost = get_total_trade_cost(symbol, 'high_vol')
        cost_increase_ticks = high_vol_cost - low_vol_cost
        cost_increase_pct = (cost_increase_ticks / low_vol_cost) * 100

        print(f"\n{symbol}:")
        print(f"  Low vol cost:     {low_vol_cost:.2f} ticks (${low_vol_cost * TICK_VALUES[symbol]:.2f})")
        print(f"  High vol cost:    {high_vol_cost:.2f} ticks (${high_vol_cost * TICK_VALUES[symbol]:.2f})")
        print(f"  Cost increase:    {cost_increase_ticks:.2f} ticks ({cost_increase_pct:.1f}%)")

    print_section("Minimum Profit Thresholds")
    print("\nMinimum profit needed to break even (round-trip):")

    for symbol in symbols:
        print(f"\n{symbol}:")
        for regime in regimes:
            cost_ticks = get_total_trade_cost(symbol, regime)
            cost_dollars = cost_ticks * TICK_VALUES[symbol]
            print(f"  {regime:12s}: {cost_ticks:.2f} ticks (${cost_dollars:.2f})")

    print_section("Practical Example: 100 Round-Trip Trades")
    print("\nTotal transaction costs for 100 round-trip trades:")

    num_trades = 100
    for symbol in symbols:
        print(f"\n{symbol}:")
        for regime in regimes:
            cost_per_trade = get_total_trade_cost(symbol, regime)
            total_cost_ticks = cost_per_trade * num_trades
            total_cost_dollars = total_cost_ticks * TICK_VALUES[symbol]

            print(f"  {regime:12s}: {total_cost_ticks:.0f} ticks (${total_cost_dollars:.2f})")

    print_section("API Usage Examples")

    print("""
# Get slippage for a specific regime
from config import get_slippage_ticks
slippage = get_slippage_ticks('MES', 'high_vol')  # 1.0 ticks per fill

# Get total round-trip cost (commission + slippage)
from config import get_total_trade_cost
cost = get_total_trade_cost('MES', 'low_vol')  # 1.5 ticks

# Get commission only (no slippage)
commission = get_total_trade_cost('MES', include_slippage=False)  # 0.5 ticks

# Calculate fitness with slippage
from src.phase1.stages.ga_optimize.fitness import calculate_fitness
fitness = calculate_fitness(
    labels, bars_to_hit, mae, mfe, horizon, atr_mean,
    symbol='MES',
    regime='high_vol',      # Use high volatility slippage
    include_slippage=True   # Include slippage in cost
)
""")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
