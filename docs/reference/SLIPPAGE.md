# Slippage Modeling

## Overview
Slippage is modeled alongside commission in the GA fitness function. Costs are defined per symbol and volatility regime and used by `get_total_trade_cost()`.

## Configuration

Defined in `src/phase1/config/barriers_config.py`:

```python
SLIPPAGE_TICKS = {
    "MES": {"low_vol": 0.5, "high_vol": 1.0},
    "MGC": {"low_vol": 0.75, "high_vol": 1.5},
}

TRANSACTION_COSTS = {
    "MES": 0.5,
    "MGC": 0.3,
}
```

Total round-trip cost (commission + slippage) is computed by:

```python
from src.phase1.config import get_total_trade_cost

cost = get_total_trade_cost("MES", regime="low_vol")
```

## GA Fitness Integration

GA fitness uses `get_total_trade_cost()` from:
- `src/phase1/stages/ga_optimize/fitness.py`

## References

- Config: `src/phase1/config/barriers_config.py`
- Fitness: `src/phase1/stages/ga_optimize/fitness.py`
- Tests: `tests/phase_1_tests/config/test_slippage.py`
- Script: `tests/scripts/verify_slippage.py`
