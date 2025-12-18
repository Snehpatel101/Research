"""
Baseline Backtest Strategy
Simple label-following strategy to verify labels have predictive signal
NOT meant to be profitable - just a sanity check
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineBacktest:
    """
    Simple baseline strategy: trade in direction of label when quality > threshold.
    Uses shifted labels to prevent lookahead bias.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        quality_threshold: float = 0.5,
        commission: float = 0.0001,  # 0.01% per trade
        initial_capital: float = 100000.0
    ):
        self.df = df.copy()
        self.horizon = horizon
        self.quality_threshold = quality_threshold
        self.commission = commission
        self.initial_capital = initial_capital

        self.label_col = f'label_h{horizon}'
        self.quality_col = f'quality_h{horizon}'

        self.trades = []
        self.equity_curve = []
        self.metrics = {}

    def prepare_data(self):
        """Prepare data with shifted labels to prevent lookahead."""
        logger.info(f"Preparing data for backtest (horizon={self.horizon}, quality_threshold={self.quality_threshold})")

        # Shift labels to prevent lookahead
        # At time t, we can only use labels computed at t-1
        self.df['signal'] = self.df[self.label_col].shift(1)
        self.df['signal_quality'] = self.df[self.quality_col].shift(1)

        # Filter by quality threshold
        self.df['trade_signal'] = 0
        mask = self.df['signal_quality'] >= self.quality_threshold
        self.df.loc[mask, 'trade_signal'] = self.df.loc[mask, 'signal']

        # Remove NaN rows
        self.df = self.df.dropna(subset=['signal', 'signal_quality', 'trade_signal'])

        logger.info(f"  Total bars: {len(self.df):,}")
        logger.info(f"  Bars with quality >= {self.quality_threshold}: {mask.sum():,} ({mask.sum()/len(self.df)*100:.1f}%)")

        signal_counts = self.df['trade_signal'].value_counts().sort_index()
        logger.info(f"  Trade signals: Long={signal_counts.get(1, 0):,}, Short={signal_counts.get(-1, 0):,}, Neutral={signal_counts.get(0, 0):,}")

    def run_backtest(self):
        """Execute the backtest."""
        logger.info("\nRunning backtest...")

        position = 0  # Current position: -1 (short), 0 (flat), 1 (long)
        capital = self.initial_capital
        entry_price = 0.0
        entry_time = None
        entry_idx = None
        trade_count = 0

        for bar_num, (idx, row) in enumerate(self.df.iterrows()):
            signal = row['trade_signal']
            current_price = row['close']
            current_time = row['datetime']

            # Close existing position if signal changes
            if position != 0 and signal != position:
                # Exit trade
                exit_price = current_price
                if position == 1:  # Long exit
                    pnl = exit_price - entry_price
                else:  # Short exit
                    pnl = entry_price - exit_price

                # Apply commission
                pnl_pct = (pnl / entry_price) - 2 * self.commission
                capital *= (1 + pnl_pct)

                # Record trade
                trade = {
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'duration_bars': bar_num - entry_idx if entry_idx is not None else 0,
                    'capital': capital
                }
                self.trades.append(trade)
                trade_count += 1

                position = 0

            # Enter new position if signal is not neutral
            if position == 0 and signal != 0:
                position = int(signal)
                entry_price = current_price
                entry_time = current_time
                entry_idx = bar_num

            # Record equity
            current_equity = capital
            if position != 0:
                # Mark-to-market unrealized PnL
                if position == 1:
                    unrealized_pnl_pct = (current_price - entry_price) / entry_price
                else:
                    unrealized_pnl_pct = (entry_price - current_price) / entry_price
                current_equity *= (1 + unrealized_pnl_pct)

            self.equity_curve.append({
                'datetime': current_time,
                'equity': current_equity,
                'position': position
            })

        # Close any open position at the end
        if position != 0:
            exit_price = self.df.iloc[-1]['close']
            if position == 1:
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            pnl_pct = (pnl / entry_price) - 2 * self.commission
            capital *= (1 + pnl_pct)

            trade = {
                'entry_time': entry_time,
                'exit_time': self.df.iloc[-1]['datetime'],
                'direction': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'duration_bars': len(self.df) - entry_idx - 1 if entry_idx is not None else 0,
                'capital': capital
            }
            self.trades.append(trade)

        logger.info(f"  Completed {len(self.trades)} trades")

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        logger.info("\nCalculating performance metrics...")

        if len(self.trades) == 0:
            logger.warning("  No trades executed!")
            return {
                'total_trades': 0,
                'status': 'No trades executed'
            }

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # PnL metrics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        total_return_pct = (trades_df.iloc[-1]['capital'] - self.initial_capital) / self.initial_capital * 100

        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Sharpe ratio (approximate, assuming daily returns)
        returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Max drawdown
        equity_series = equity_df['equity'].values
        running_max = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100

        # Average trade duration
        avg_duration = trades_df['duration_bars'].mean()

        self.metrics = {
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'win_rate': float(win_rate * 100),
            'total_return_pct': float(total_return_pct),
            'total_pnl': float(total_pnl),
            'avg_pnl': float(avg_pnl),
            'gross_profit': float(gross_profit),
            'gross_loss': float(gross_loss),
            'profit_factor': float(profit_factor) if np.isfinite(profit_factor) else 999.9,
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown_pct': float(max_drawdown),
            'avg_trade_duration_bars': float(avg_duration),
            'initial_capital': float(self.initial_capital),
            'final_capital': float(trades_df.iloc[-1]['capital']),
            'commission_per_trade': float(self.commission * 100)
        }

        # Log metrics
        logger.info(f"\n  {'='*50}")
        logger.info(f"  BACKTEST RESULTS")
        logger.info(f"  {'='*50}")
        logger.info(f"  Total Trades:        {self.metrics['total_trades']:,}")
        logger.info(f"  Win Rate:            {self.metrics['win_rate']:.2f}%")
        logger.info(f"  Total Return:        {self.metrics['total_return_pct']:.2f}%")
        logger.info(f"  Profit Factor:       {self.metrics['profit_factor']:.2f}")
        logger.info(f"  Sharpe Ratio:        {self.metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown:        {self.metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"  Avg Trade Duration:  {self.metrics['avg_trade_duration_bars']:.1f} bars")
        logger.info(f"  {'='*50}")

        return self.metrics

    def plot_equity_curve(self, output_path: Path):
        """Generate equity curve plot."""
        logger.info(f"\nGenerating equity curve plot: {output_path}")

        equity_df = pd.DataFrame(self.equity_curve)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Equity curve
        ax1.plot(equity_df['datetime'], equity_df['equity'], linewidth=1.5, color='blue', label='Equity')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', linewidth=1, label='Initial Capital')
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.set_title(f'Baseline Backtest - Horizon {self.horizon} (Quality >= {self.quality_threshold})', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(equity_df['datetime'].min(), equity_df['datetime'].max())

        # Position indicator
        ax2.fill_between(equity_df['datetime'], 0, equity_df['position'],
                         where=(equity_df['position'] > 0), color='green', alpha=0.3, label='Long')
        ax2.fill_between(equity_df['datetime'], 0, equity_df['position'],
                         where=(equity_df['position'] < 0), color='red', alpha=0.3, label='Short')
        ax2.set_ylabel('Position', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Short', 'Flat', 'Long'])
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(equity_df['datetime'].min(), equity_df['datetime'].max())

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved plot to {output_path}")


def run_baseline_backtest(
    data_path: Path,
    split_indices_path: Path,
    output_dir: Path,
    horizon: int = 5,
    quality_threshold: float = 0.5,
    commission: float = 0.0001
) -> Dict:
    """
    Run baseline backtest on test set.

    Args:
        data_path: Path to combined labeled data
        split_indices_path: Path to test indices (.npy)
        output_dir: Directory for outputs
        horizon: Label horizon to use
        quality_threshold: Minimum quality score to trade
        commission: Commission per trade (as fraction)

    Returns:
        Dictionary with backtest results
    """
    logger.info("="*70)
    logger.info("BASELINE BACKTEST")
    logger.info("="*70)

    # Load data
    logger.info(f"\nLoading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Load test indices
    logger.info(f"Loading test indices from {split_indices_path}")
    test_idx = np.load(split_indices_path)
    test_df = df.iloc[test_idx].copy()

    logger.info(f"Test set: {len(test_df):,} samples")
    logger.info(f"Date range: {test_df['datetime'].min()} to {test_df['datetime'].max()}")

    # Create and run backtest
    backtest = BaselineBacktest(
        df=test_df,
        horizon=horizon,
        quality_threshold=quality_threshold,
        commission=commission
    )

    backtest.prepare_data()
    backtest.run_backtest()
    metrics = backtest.calculate_metrics()

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"baseline_equity_curve_h{horizon}.png"
    backtest.plot_equity_curve(plot_path)

    # Save results
    results = {
        'horizon': horizon,
        'quality_threshold': quality_threshold,
        'commission': commission,
        'test_samples': int(len(test_df)),
        'test_date_start': str(test_df['datetime'].min()),
        'test_date_end': str(test_df['datetime'].max()),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    results_path = output_dir / f"baseline_backtest_h{horizon}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Plot saved to {plot_path}")

    logger.info("\n" + "="*70)
    logger.info("BASELINE BACKTEST COMPLETE")
    logger.info("="*70)

    return results


def main():
    """Run baseline backtest on default configuration."""
    import sys
    sys.path.insert(0, '/home/user/Research/src')

    from config import FINAL_DATA_DIR, SPLITS_DIR, RESULTS_DIR

    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"
    test_indices_path = SPLITS_DIR / "test_indices.npy"
    output_dir = RESULTS_DIR / "baseline_backtest"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    if not test_indices_path.exists():
        logger.error(f"Test indices not found: {test_indices_path}")
        logger.info("Run Stage 7 (splits) first!")
        return

    # Run backtest for all horizons
    for horizon in [1, 5, 20]:
        logger.info(f"\n{'='*70}")
        logger.info(f"HORIZON {horizon}")
        logger.info(f"{'='*70}")

        results = run_baseline_backtest(
            data_path=data_path,
            split_indices_path=test_indices_path,
            output_dir=output_dir,
            horizon=horizon,
            quality_threshold=0.5,
            commission=0.0001
        )


if __name__ == "__main__":
    main()
