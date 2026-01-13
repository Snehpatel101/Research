# Trading Simulator - Quick Start Guide

## 5-Minute Setup

### Step 1: Configure Simulation (Cell 1.2)

```python
# Recommended settings for liquid futures (MGC, MES)
POSITION_SIZING_METHOD = "kelly_half"  # Conservative Kelly
SLIPPAGE_BPS = 5.0                     # 5 bps (realistic for liquid futures)
INITIAL_CAPITAL = 100000               # $100k starting capital
MAX_POSITION_PCT = 0.25                # Never risk more than 25%

# Optional: Risk management
USE_STOP_LOSS = True
STOP_LOSS_PCT = 0.02  # 2% stop loss
```

### Step 2: Run Pipeline and Training (Cells 3.x - 4.3)

Run the standard notebook workflow up to Cell 4.3.

### Step 3: Run Simulation (Cell 4.5)

```python
run_trading_sim = True
show_equity_curve = True
show_drawdown_chart = True
show_trade_distribution = True
show_kelly_evolution = True
```

Execute Cell 4.5 to see results.

## Position Sizing Methods

| Method | Use Case | Risk Level |
|--------|----------|------------|
| `fixed` | Baseline comparison | Low (conservative) |
| `kelly` | Maximum growth rate | High (aggressive) |
| `kelly_half` | **Recommended for live** | Medium (balanced) |
| `volatility_scaled` | Risk-adjusted sizing | Medium (adaptive) |

## Interpreting Results

### Key Metrics

**Net Sharpe Ratio** (after costs):
- `< 1.0`: Barely profitable, not worth trading
- `1.0 - 1.5`: Decent, but monitor closely
- `1.5 - 2.0`: Good, worth trading
- `> 2.0`: Excellent, rare in practice

**Max Drawdown**:
- `< 10%`: Low risk
- `10-20%`: Moderate risk
- `20-30%`: High risk
- `> 30%`: Too risky for most traders

**Profit Factor**:
- `< 1.2`: Weak edge
- `1.2 - 1.5`: Decent edge
- `1.5 - 2.0`: Strong edge
- `> 2.0`: Very strong edge

**Costs as % of Gross P&L**:
- `< 10%`: Efficient (low turnover)
- `10-20%`: Moderate (acceptable)
- `20-30%`: High (over-trading warning)
- `> 30%`: Too high (reduce frequency)

### Red Flags

🚩 **Net Sharpe < 1.0** → Strategy barely profitable after costs
🚩 **Max Drawdown > 30%** → Too risky, reduce position sizes
🚩 **Profit Factor < 1.2** → Weak edge, need better signals
🚩 **Costs > 20% of Gross** → Over-trading, reduce frequency

## Common Adjustments

### If Sharpe is Low

1. **Reduce turnover** (trade less frequently)
2. **Increase slippage estimate** (be more conservative)
3. **Use half-Kelly** instead of full Kelly
4. **Add stop loss** to limit losses

### If Drawdown is High

1. **Reduce MAX_POSITION_PCT** (e.g., 0.15 instead of 0.25)
2. **Use kelly_half** or fixed sizing
3. **Enable USE_STOP_LOSS** with 2% threshold
4. **Filter trades** (only high-confidence predictions)

### If Costs are High

1. **Increase slippage threshold** (only trade when edge > costs)
2. **Reduce position size** (smaller trades = lower impact)
3. **Trade less frequently** (longer horizons)
4. **Use limit orders** (reduce slippage in practice)

## Research Workflow

### Compare Position Sizing Methods

```python
# Run 4 simulations with different methods
for method in ["fixed", "kelly", "kelly_half", "volatility_scaled"]:
    POSITION_SIZING_METHOD = method
    # Run Cell 4.5
    # Record Net Sharpe, Max DD, Costs
```

### Cost Sensitivity Analysis

```python
# Test different slippage assumptions
for slippage in [2.0, 5.0, 10.0, 20.0]:
    SLIPPAGE_BPS = slippage
    # Run Cell 4.5
    # Record how costs affect profitability
```

### Risk Management Impact

```python
# Test with/without stops
USE_STOP_LOSS = False  # Baseline
# Run Cell 4.5 → Record metrics

USE_STOP_LOSS = True   # With stops
# Run Cell 4.5 → Compare metrics
```

## Advanced Configuration

### For High-Frequency Strategies

```python
SLIPPAGE_BPS = 10.0           # Higher slippage (more trades)
COMMISSION_PER_SHARE = 0.001  # Higher commission (HFT rates)
MAX_POSITION_PCT = 0.10       # Lower position size (more trades)
```

### For Swing Trading

```python
SLIPPAGE_BPS = 2.0            # Lower slippage (longer holds)
MAX_POSITION_PCT = 0.50       # Higher position size (fewer trades)
USE_TAKE_PROFIT = True
TAKE_PROFIT_PCT = 0.10        # 10% take profit
```

### Conservative Risk Management

```python
POSITION_SIZING_METHOD = "kelly_half"
MAX_POSITION_PCT = 0.15       # Max 15% per trade
USE_STOP_LOSS = True
STOP_LOSS_PCT = 0.01          # Tight 1% stop
USE_TAKE_PROFIT = True
TAKE_PROFIT_PCT = 0.03        # Quick 3% profit
```

## Kelly Criterion Explained

**What it does:**
Calculates optimal position size to maximize long-term growth rate.

**Formula:**
```
kelly_fraction = (win_rate * avg_win/avg_loss - (1 - win_rate)) / (avg_win/avg_loss)
```

**Example:**
- Win rate: 60%
- Avg win: $150
- Avg loss: $100
- b = 150/100 = 1.5
- kelly = (0.6 × 1.5 - 0.4) / 1.5 = 0.333 = **33.3% of capital**

**Why half-Kelly?**
- Full Kelly can be too aggressive
- Estimation errors cause over-leveraging
- Half-Kelly reduces volatility
- Practitioners prefer 0.25-0.5x Kelly

## Troubleshooting

### "No trades executed"

**Cause:** Model only predicts "Hold" (class 1)
**Fix:** Check model predictions in Cell 4.3

### "Kelly fraction is 0"

**Cause:** Not enough trades for calculation (< 30 bars)
**Fix:** Run longer backtest or reduce warmup period

### "Costs exceed gross P&L"

**Cause:** Over-trading with high costs
**Fix:** Reduce trading frequency or check slippage assumptions

### "Equity curve is flat"

**Cause:** No profitable trades
**Fix:** Check model performance, may need retraining

## Output Storage

Results are stored in `CONFIG.trading_sim_results`:

```python
# Access results programmatically
for model_name, result in config.trading_sim_results.items():
    print(f"{model_name}:")
    print(f"  Net Sharpe: {result['sharpe']:.3f}")
    print(f"  Max DD: {result['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {result['win_rate']*100:.2f}%")
    print(f"  Net P&L: ${result['net_pnl']:.2f}")
```

## Next Steps

1. **Compare models** with realistic costs
2. **Sensitivity analysis** on slippage assumptions
3. **Optimize position sizing** method for your risk tolerance
4. **Export results** for portfolio-level analysis

## References

- Thorp (2008): Kelly criterion for portfolio management
- Almgren-Chriss (2000): Market impact models
- 2025 RL + Kelly: Adaptive position sizing research

For detailed documentation, see: `TRADING_SIMULATOR.md`
