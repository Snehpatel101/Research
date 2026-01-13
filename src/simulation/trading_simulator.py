import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Default constants from notebook
COMMISSION_PER_SHARE = 0.0005
FIXED_FEE_PER_TRADE = 1.0
SLIPPAGE_BPS = 5.0

def calculate_slippage_with_impact(
    price: float, 
    shares: int, 
    typical_volume: int = 10000, 
    base_slippage_bps: float = 5.0
) -> float:
    """Calculate market impact-adjusted slippage."""
    impact_multiplier = (shares / typical_volume) ** 0.5
    effective_slippage_bps = base_slippage_bps * max(1.0, impact_multiplier)
    return price * effective_slippage_bps / 10000

def calculate_transaction_costs(
    shares: int, 
    entry_price: float, 
    exit_price: float,
    commission_per_share: float = COMMISSION_PER_SHARE,
    fixed_fee_per_trade: float = FIXED_FEE_PER_TRADE,
    base_slippage_bps: float = SLIPPAGE_BPS
) -> float:
    """Calculate total round-trip transaction costs."""
    # Commission
    commission = 2 * (shares * commission_per_share + fixed_fee_per_trade)  # Round-trip
    
    # Slippage (entry + exit)
    entry_slippage = calculate_slippage_with_impact(entry_price, shares, base_slippage_bps=base_slippage_bps)
    exit_slippage = calculate_slippage_with_impact(exit_price, shares, base_slippage_bps=base_slippage_bps)
    slippage_cost = shares * (entry_slippage + exit_slippage)
    
    return commission + slippage_cost

def calculate_kelly_fraction(
    predictions: np.ndarray, 
    actuals: np.ndarray, 
    prices: np.ndarray, 
    max_position_pct: float = 0.25
) -> np.ndarray:
    """
    Calculate Kelly criterion position sizing based on historical performance.
    
    Args:
        predictions: Model predictions (0=Short, 1=Hold, 2=Long)
        actuals: Actual outcomes
        prices: Asset prices
        max_position_pct: Maximum position size as fraction of capital
    
    Returns:
        kelly_fractions: Array of position sizes for each trade
    """
    # Filter to actual trades (exclude Hold)
    trades_mask = predictions != 1
    trade_preds = predictions[trades_mask]
    trade_actuals = actuals[trades_mask]
    trade_prices = prices[trades_mask]
    
    if len(trade_preds) < 2:
        return np.full(len(predictions), 0.01)  # Minimal position
    
    # Calculate wins/losses
    correct = (trade_preds == trade_actuals)
    
    if len(correct) < 2:
        return np.full(len(predictions), 0.01)
    
    # Calculate returns for wins and losses
    returns = np.diff(trade_prices) / trade_prices[:-1]
    
    # We need to align returns with the trades. 
    # Note: The notebook implementation logic for returns alignment was:
    # returns = np.diff(trade_prices) / trade_prices[:-1]
    # win_returns = returns[correct[:-1]]
    # This implies 'correct' is aligned with 'trade_prices' (which it is),
    # but 'returns' is length N-1. 'correct[:-1]' aligns with 'returns'.
    
    win_returns = returns[correct[:-1]]
    loss_returns = returns[~correct[:-1]]
    
    if len(win_returns) == 0 or len(loss_returns) == 0:
        return np.full(len(predictions), 0.01)  # Minimal position
    
    win_rate = len(win_returns) / len(returns)
    avg_win = np.abs(np.mean(win_returns))
    avg_loss = np.abs(np.mean(loss_returns))
    
    # Kelly formula: f = (p*b - q) / b
    # where p = win_rate, q = 1-p, b = avg_win/avg_loss
    b = avg_win / avg_loss if avg_loss > 0 else 1.0
    
    # Initial global Kelly (not strictly used for the dynamic part, but good for reference)
    # kelly = (win_rate * b - (1 - win_rate)) / b
    # kelly = np.clip(kelly, 0, max_position_pct)
    
    # Expanding window: recalculate Kelly as more data arrives
    kelly_fractions = np.zeros(len(predictions))
    
    # Optimization: Pre-calculate indices to map back to original time
    # But strictly following notebook logic for fidelity:
    
    for i in range(len(predictions)):
        if i < 30:  # Minimum warmup period
            kelly_fractions[i] = 0.01
        else:
            # Recalculate based on last 100 trades
            lookback = min(i, 100)
            recent_preds = predictions[max(0, i-lookback):i]
            recent_actuals = actuals[max(0, i-lookback):i]
            recent_trades_mask = recent_preds != 1
            
            if np.sum(recent_trades_mask) > 10:  # Need at least 10 trades
                recent_correct = (recent_preds[recent_trades_mask] == recent_actuals[recent_trades_mask])
                recent_win_rate = np.mean(recent_correct)
                
                # NOTE: Notebook used global 'b' here. We keep that behavior for now.
                recent_kelly = (recent_win_rate * b - (1 - recent_win_rate)) / b
                kelly_fractions[i] = np.clip(recent_kelly, 0, max_position_pct)
            else:
                kelly_fractions[i] = 0.01
    
    return kelly_fractions

def simulate_trades(
    predictions: np.ndarray, 
    actuals: np.ndarray, 
    prices: np.ndarray, 
    initial_capital: float = 100000.0,
    position_sizing_method: str = "kelly",
    max_position_pct: float = 0.25,
    fixed_position_size: int = 100,
    use_stop_loss: bool = False,
    stop_loss_pct: float = 0.02,
    use_take_profit: bool = False,
    take_profit_pct: float = 0.06,
    commission_per_share: float = COMMISSION_PER_SHARE,
    fixed_fee_per_trade: float = FIXED_FEE_PER_TRADE,
    slippage_bps: float = SLIPPAGE_BPS
) -> Dict[str, Any]:
    """Simulate realistic trading with costs and position sizing."""
    capital = initial_capital
    equity_curve = [capital]
    trades = []
    
    # Calculate Kelly fractions
    kelly_fractions = None
    if position_sizing_method in ["kelly", "kelly_half"]:
        kelly_fractions = calculate_kelly_fraction(predictions, actuals, prices, max_position_pct)
        if position_sizing_method == "kelly_half":
            kelly_fractions *= 0.5  # Half-Kelly
            
    current_position = None
    peak_equity = capital
    
    for i in range(len(predictions)):
        pred = predictions[i]
        price = prices[i]
        
        # Check for stop loss / take profit
        if current_position is not None:
            entry_price = current_position['entry_price']
            pnl_pct = (price - entry_price) / entry_price
            
            exit_triggered = False
            exit_reason = None
            
            if use_stop_loss and pnl_pct < -stop_loss_pct:
                exit_triggered = True
                exit_reason = 'stop_loss'
            elif use_take_profit and pnl_pct > take_profit_pct:
                exit_triggered = True
                exit_reason = 'take_profit'
            
            if exit_triggered:
                # Close position
                shares = current_position['shares']
                gross_pnl = shares * (price - entry_price)
                costs = calculate_transaction_costs(
                    shares, entry_price, price, 
                    commission_per_share, fixed_fee_per_trade, slippage_bps
                )
                net_pnl = gross_pnl - costs
                
                capital += net_pnl
                equity_curve.append(capital)
                
                trades.append({
                    'entry_idx': current_position['entry_idx'],
                    'exit_idx': i,
                    'shares': shares,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'gross_pnl': gross_pnl,
                    'costs': costs,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason,
                })
                
                current_position = None
                peak_equity = max(peak_equity, capital)
                continue
        
        # Trading logic
        if pred == 1:  # Hold
            equity_curve.append(capital)
            continue
        
        # Close existing position if direction changes
        if current_position is not None:
            shares = current_position['shares']
            entry_price = current_position['entry_price']
            gross_pnl = shares * (price - entry_price)
            costs = calculate_transaction_costs(
                shares, entry_price, price,
                commission_per_share, fixed_fee_per_trade, slippage_bps
            )
            net_pnl = gross_pnl - costs
            
            capital += net_pnl
            
            trades.append({
                'entry_idx': current_position['entry_idx'],
                'exit_idx': i,
                'shares': shares,
                'entry_price': entry_price,
                'exit_price': price,
                'gross_pnl': gross_pnl,
                'costs': costs,
                'net_pnl': net_pnl,
                'exit_reason': 'signal_change',
            })
            
            current_position = None
        
        # Open new position
        shares = 0
        if position_sizing_method == "fixed":
            shares = fixed_position_size
        elif position_sizing_method in ["kelly", "kelly_half"]:
            position_value = capital * kelly_fractions[i]
            shares = int(position_value / price)
        elif position_sizing_method == "volatility_scaled":
            # Use inverse volatility scaling
            lookback = min(i, 20)
            if lookback > 1:
                recent_returns = np.diff(prices[max(0, i-lookback):i+1]) / prices[max(0, i-lookback):i]
                volatility = np.std(recent_returns)
                target_risk = 0.02  # 2% risk per trade
                position_value = capital * target_risk / (volatility if volatility > 0 else 0.01)
                shares = int(position_value / price)
            else:
                shares = fixed_position_size
        else:
            shares = fixed_position_size
        
        # Cap position size
        max_shares = int((capital * max_position_pct) / price)
        shares = min(shares, max_shares)
        shares = max(1, shares)  # At least 1 share
        
        # Apply slippage to entry
        entry_slippage = calculate_slippage_with_impact(price, shares, base_slippage_bps=slippage_bps)
        effective_entry_price = price * (1 + entry_slippage / price)
        
        current_position = {
            'entry_idx': i,
            'shares': shares,
            'entry_price': effective_entry_price,
            'signal': pred,
        }
        
        equity_curve.append(capital)
        peak_equity = max(peak_equity, capital)
    
    # Close any open position at end
    if current_position is not None:
        shares = current_position['shares']
        entry_price = current_position['entry_price']
        price = prices[-1]
        gross_pnl = shares * (price - entry_price)
        costs = calculate_transaction_costs(
            shares, entry_price, price,
            commission_per_share, fixed_fee_per_trade, slippage_bps
        )
        net_pnl = gross_pnl - costs
        
        capital += net_pnl
        equity_curve.append(capital)
        
        trades.append({
            'entry_idx': current_position['entry_idx'],
            'exit_idx': len(prices) - 1,
            'shares': shares,
            'entry_price': entry_price,
            'exit_price': price,
            'gross_pnl': gross_pnl,
            'costs': costs,
            'net_pnl': net_pnl,
            'exit_reason': 'end_of_data',
        })
    
    return {
        'final_capital': capital,
        'equity_curve': np.array(equity_curve),
        'trades': trades,
        'kelly_fractions': kelly_fractions,
    }

def plot_simulation_results(
    model_name: str,
    sim_result: Dict[str, Any],
    trades_df: pd.DataFrame,
    initial_capital: float,
    max_position_pct: float,
    show_equity_curve: bool = True,
    show_drawdown_chart: bool = True,
    show_trade_distribution: bool = True,
    show_kelly_evolution: bool = True
) -> None:
    """Visualize trading simulation results."""
    
    equity = sim_result['equity_curve']
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    
    n_plots = sum([show_equity_curve, show_drawdown_chart, show_trade_distribution, show_kelly_evolution])
    if n_plots == 0:
        return
        
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    if show_equity_curve:
        axes[plot_idx].plot(sim_result['equity_curve'], linewidth=2)
        axes[plot_idx].axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
        axes[plot_idx].set_title(f"{model_name} - Equity Curve", fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel("Trade Number")
        axes[plot_idx].set_ylabel("Capital ($)")
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    if show_drawdown_chart:
        axes[plot_idx].fill_between(range(len(drawdown)), drawdown * 100, 0, alpha=0.3, color='red')
        axes[plot_idx].plot(drawdown * 100, color='red', linewidth=2)
        axes[plot_idx].set_title(f"{model_name} - Drawdown", fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel("Trade Number")
        axes[plot_idx].set_ylabel("Drawdown (%)")
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    if show_trade_distribution:
        axes[plot_idx].hist(trades_df['net_pnl'], bins=30, edgecolor='black', alpha=0.7)
        axes[plot_idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[plot_idx].set_title(f"{model_name} - Trade P&L Distribution", fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel("Net P&L ($)")
        axes[plot_idx].set_ylabel("Frequency")
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    if show_kelly_evolution and sim_result['kelly_fractions'] is not None:
        axes[plot_idx].plot(sim_result['kelly_fractions'] * 100, linewidth=2)
        axes[plot_idx].axhline(y=max_position_pct * 100, color='red', linestyle='--', alpha=0.5, label='Max Position')
        axes[plot_idx].set_title(f"{model_name} - Kelly Fraction Evolution", fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel("Bar Index")
        axes[plot_idx].set_ylabel("Kelly Fraction (%)")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()
