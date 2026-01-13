# Trading Simulator - Implementation Notes

## What Was Added

Successfully implemented a realistic trading simulator in `ML_Pipeline.ipynb` with 4 new components:

### 1. Cell 1.2: Trading Simulation Configuration

**Location:** After Cell 1.1 (Master Configuration Panel)

**Purpose:** Configure all trading simulation parameters

**Parameters:**
- Transaction costs (commission, fixed fees, slippage)
- Position sizing method (fixed, kelly, kelly_half, volatility_scaled)
- Risk management (stop loss, take profit)
- Initial capital

**Key Settings:**
```python
POSITION_SIZING_METHOD = "kelly"  # Adaptive position sizing
SLIPPAGE_BPS = 5.0                # 5 basis points (realistic for liquid futures)
INITIAL_CAPITAL = 100000          # $100k starting capital
MAX_POSITION_PCT = 0.25           # Max 25% per position
```

### 2. Cell 2.3: Kelly Criterion Calculator

**Location:** Added to Checkpoint Utilities cell

**Purpose:** Calculate optimal position sizes using Kelly criterion

**Features:**
- Expanding window recalculation (100-trade lookback)
- 30-bar warmup period
- Maximum position limiter
- Adaptive to changing market conditions

**Function Signature:**
```python
def calculate_kelly_fraction(
    predictions,  # Model predictions (0=Short, 1=Hold, 2=Long)
    actuals,      # Actual outcomes
    prices,       # Asset prices
    max_position_pct=0.25  # Maximum position cap
) -> np.ndarray:  # Returns Kelly fractions for each bar
```

### 3. Cell 2.1: NotebookConfig Updates

**Location:** Modified existing NotebookConfig dataclass

**Purpose:** Store simulation configuration and results

**New Fields:**
```python
@dataclass
class NotebookConfig:
    # ... existing fields ...

    # Trading simulation
    trading_sim_results: Dict[str, Any] = field(default_factory=dict)
    position_sizing_method: str = "kelly"
    initial_capital: float = 100000.0
```

### 4. Cell 4.5: Realistic Trading Simulation

**Location:** After Cell 4.4 (Trading Performance Metrics)

**Purpose:** Full backtesting engine with realistic costs and position sizing

**Key Functions:**

**a) Market Impact-Adjusted Slippage:**
```python
def calculate_slippage_with_impact(price, shares, typical_volume=10000, base_slippage_bps=5.0):
    """Square-root market impact model."""
    impact_multiplier = (shares / typical_volume) ** 0.5
    effective_slippage_bps = base_slippage_bps * max(1.0, impact_multiplier)
    return price * effective_slippage_bps / 10000
```

**b) Transaction Costs:**
```python
def calculate_transaction_costs(shares, entry_price, exit_price):
    """Round-trip costs = commission + slippage (entry + exit)."""
    commission = 2 * (shares * COMMISSION_PER_SHARE + FIXED_FEE_PER_TRADE)
    slippage_cost = shares * (entry_slippage + exit_slippage)
    return commission + slippage_cost
```

**c) Trade Simulation:**
```python
def simulate_trades(predictions, actuals, prices, initial_capital=100000):
    """
    Simulate realistic trading with:
    - Transaction costs (commission + slippage)
    - Kelly criterion position sizing
    - Risk management (stop loss / take profit)
    - Comprehensive metrics
    """
```

**Output Metrics:**
- Total Return (%)
- Net Sharpe Ratio (annualized, after costs)
- Maximum Drawdown (%)
- Number of Trades
- Win Rate (%)
- Profit Factor (gross win / gross loss)
- Average Win / Loss ($)
- Gross P&L
- Total Transaction Costs
- Net P&L
- Costs as % of Gross P&L

**Visualizations:**
1. Equity curve (capital over time)
2. Drawdown chart (% from peak)
3. Trade P&L distribution (histogram)
4. Kelly fraction evolution (adaptive sizing)

## How It Works

### Workflow

```
User Configures Simulation (Cell 1.2)
         ↓
User Runs Data Pipeline + Training (Cells 3.x - 4.3)
         ↓
Simulator Loads Test Predictions (Cell 4.5)
         ↓
For Each Model:
  1. Calculate Kelly Fractions (adaptive, expanding window)
  2. Simulate Trades:
     - Open position when signal != Hold
     - Apply slippage at entry (market impact model)
     - Check stop loss / take profit every bar
     - Close position on signal change or risk limit
     - Apply slippage at exit
     - Deduct commission + slippage costs
  3. Calculate Performance Metrics:
     - Net Sharpe (after costs)
     - Max Drawdown (from running peak)
     - Win Rate, Profit Factor
     - Cost Analysis
  4. Generate Visualizations:
     - Equity curve
     - Drawdown chart
     - Trade distribution
     - Kelly evolution
  5. Store Results in CONFIG.trading_sim_results
         ↓
User Reviews Results and Adjusts Configuration
```

### Position Sizing Logic

**Fixed:**
```python
shares = FIXED_POSITION_SIZE  # e.g., 100
```

**Kelly (Full):**
```python
# Calculate based on last 100 trades
win_rate = wins / total_trades
avg_win_loss_ratio = avg_win / avg_loss
kelly = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
kelly = clip(kelly, 0, MAX_POSITION_PCT)
shares = int(capital * kelly / price)
```

**Half-Kelly (Recommended):**
```python
kelly_half = kelly * 0.5
shares = int(capital * kelly_half / price)
```

**Volatility-Scaled:**
```python
target_risk = 0.02  # 2% risk per trade
volatility = std(recent_returns)
shares = int(capital * target_risk / volatility / price)
```

### Transaction Cost Model

**Commission:**
```
Round-trip commission = 2 × (shares × commission_per_share + fixed_fee_per_trade)

Default: 2 × (shares × $0.0005 + $1.00)
```

**Slippage (Market Impact):**
```
Entry slippage = price × (base_slippage_bps / 10000) × (shares / typical_volume)^0.5
Exit slippage = price × (base_slippage_bps / 10000) × (shares / typical_volume)^0.5

Total slippage cost = shares × (entry_slippage + exit_slippage)

Default base: 5 bps (0.05%)
```

**Total Cost:**
```
Total round-trip cost = Commission + Slippage cost
```

### Risk Management Logic

**Stop Loss:**
```python
if USE_STOP_LOSS:
    pnl_pct = (current_price - entry_price) / entry_price
    if pnl_pct < -STOP_LOSS_PCT:
        # Close position with slippage
        exit_triggered = True
```

**Take Profit:**
```python
if USE_TAKE_PROFIT:
    pnl_pct = (current_price - entry_price) / entry_price
    if pnl_pct > TAKE_PROFIT_PCT:
        # Close position with slippage
        exit_triggered = True
```

## Key Design Decisions

### 1. Expanding Window Kelly Calculation

**Why:** Adaptive to changing market conditions

**Implementation:**
- 30-bar warmup (minimal position before enough data)
- 100-trade lookback window
- Recalculates every bar
- Clips to maximum position limit

**Rationale:**
- Static Kelly assumes stationary win rate (unrealistic)
- Expanding window adapts to regime changes
- 100-trade lookback balances responsiveness and stability

### 2. Square-Root Market Impact

**Why:** Empirically validated in literature

**Formula:**
```
impact_multiplier = (trade_size / typical_volume)^0.5
```

**Rationale:**
- Linear impact too conservative for small trades
- Square-root captures non-linear price impact
- Aligns with Almgren-Chriss (2000) research

### 3. Half-Kelly as Default Recommendation

**Why:** More conservative than full Kelly

**Rationale:**
- Full Kelly can be too aggressive
- Estimation errors in win rate cause over-leveraging
- Half-Kelly reduces volatility by ~50%
- Practitioners prefer 0.25-0.5x Kelly for live trading

### 4. Annualized Sharpe Calculation

**Formula:**
```python
periods_per_year = 252 * (1440 / bar_minutes)  # 252 trading days
sharpe = (mean_return / std_return) * sqrt(periods_per_year)
```

**Example (5-min bars):**
- 288 bars/day (24h futures)
- 72,576 bars/year
- Annualization factor = sqrt(72,576) = 269.4

## Testing Recommendations

### Before Live Trading

1. **Sensitivity Analysis on Slippage:**
   ```python
   for slippage in [2, 5, 10, 20]:
       SLIPPAGE_BPS = slippage
       # Run simulation
       # Check if still profitable
   ```

2. **Position Sizing Comparison:**
   ```python
   for method in ["fixed", "kelly", "kelly_half", "volatility_scaled"]:
       POSITION_SIZING_METHOD = method
       # Run simulation
       # Compare Sharpe and Max DD
   ```

3. **Risk Management Impact:**
   ```python
   # Baseline (no risk management)
   USE_STOP_LOSS = False
   USE_TAKE_PROFIT = False
   # Run → Record metrics

   # With stop loss
   USE_STOP_LOSS = True
   STOP_LOSS_PCT = 0.02
   # Run → Compare metrics
   ```

4. **Capital Scaling:**
   ```python
   for capital in [10000, 50000, 100000, 500000]:
       INITIAL_CAPITAL = capital
       # Run simulation
       # Check if results scale linearly
   ```

### Red Flags to Monitor

🚩 **Net Sharpe < 1.0:** Strategy barely profitable after costs
🚩 **Max Drawdown > 30%:** Too risky for most traders
🚩 **Costs > 20% of Gross P&L:** Over-trading (reduce frequency)
🚩 **Win Rate < 45%:** Need better signals or higher avg win/loss ratio
🚩 **Profit Factor < 1.2:** Weak edge, not worth trading

## Known Limitations

### 1. Simplified Price Model (Demo)

**Current:**
```python
# Synthetic prices for demo
returns = np.random.randn(len(y_pred)) * 0.01
prices = base_price * np.exp(np.cumsum(returns))
```

**TODO:**
```python
# Load actual test set prices from OHLCV data
test_data = load_test_data(symbol, horizon)
prices = test_data['close'].values
```

### 2. No Partial Fills

**Current:** Assumes all orders fill at calculated price

**Future Enhancement:**
- Model partial fills based on volume
- Adjust position size if fill < requested

### 3. No Latency

**Current:** Assumes instant execution

**Future Enhancement:**
- Add latency (1-100ms)
- Model price movement during execution

### 4. Constant Slippage Base

**Current:** Base slippage is constant (5 bps)

**Future Enhancement:**
- Time-of-day effects (higher at open/close)
- Session-dependent slippage (Asia, Europe, US)

## Accessing Results

### Programmatic Access

```python
# After running Cell 4.5
for model_name, result in config.trading_sim_results.items():
    print(f"\n{model_name}:")
    print(f"  Net Sharpe: {result['sharpe']:.3f}")
    print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"  Total Return: {result['total_return']*100:.2f}%")
    print(f"  Win Rate: {result['win_rate']*100:.2f}%")
    print(f"  Profit Factor: {result['profit_factor']:.2f}")
    print(f"  Net P&L: ${result['net_pnl']:.2f}")
    print(f"  Total Costs: ${result['total_costs']:.2f}")
```

### Export to DataFrame

```python
import pandas as pd

# Summary table
summary = []
for model_name, result in config.trading_sim_results.items():
    summary.append({
        'model': model_name,
        'net_sharpe': result['sharpe'],
        'max_drawdown': result['max_drawdown'],
        'total_return': result['total_return'],
        'win_rate': result['win_rate'],
        'profit_factor': result['profit_factor'],
        'net_pnl': result['net_pnl'],
    })

df = pd.DataFrame(summary)
df.to_csv('trading_sim_results.csv', index=False)
print(df)
```

### Individual Trade Analysis

```python
# Get trades for a specific model
model_name = 'CatBoost_H20'
trades_df = config.trading_sim_results[model_name]['trades']

# Analyze winning trades
winning_trades = trades_df[trades_df['net_pnl'] > 0]
print(f"Avg win: ${winning_trades['net_pnl'].mean():.2f}")
print(f"Max win: ${winning_trades['net_pnl'].max():.2f}")

# Analyze losing trades
losing_trades = trades_df[trades_df['net_pnl'] <= 0]
print(f"Avg loss: ${abs(losing_trades['net_pnl'].mean()):.2f}")
print(f"Max loss: ${abs(losing_trades['net_pnl'].min()):.2f}")

# Exit reasons
print("\nExit reasons:")
print(trades_df['exit_reason'].value_counts())
```

## References

### Research Papers

1. **Kelly Criterion:**
   - Thorp (2008): "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
   - 2025 RL + Kelly: Adaptive position sizing with reinforcement learning

2. **Market Impact:**
   - Almgren & Chriss (2000): "Optimal Execution of Portfolio Transactions"
   - Barra (2010): "Market Impact Model"

3. **Slippage Studies:**
   - 2025 research on micro-futures liquidity (MGC, MES)
   - Large-caps: 2-5 bps per round-trip
   - Mid-caps: 10-50 bps
   - Small-caps: 50-200 bps

## Support

For questions or issues:
1. Check `TRADING_SIMULATOR.md` (comprehensive guide)
2. Check `docs/TRADING_SIMULATOR_QUICKSTART.md` (5-min quick start)
3. Review this implementation notes file

## Changelog

**2026-01-12:**
- Initial implementation
- 4 components added to ML_Pipeline.ipynb
- Kelly criterion calculator with expanding window
- Market impact-adjusted slippage model
- 4 position sizing methods
- Risk management (stop loss / take profit)
- Comprehensive metrics and visualizations
- Documentation created (TRADING_SIMULATOR.md, QUICKSTART.md)
