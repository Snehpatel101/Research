"""
Equity curve management and visualization.

This module provides the EquityCurve class for tracking, analyzing,
and visualizing portfolio equity over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    PerformanceMetrics,
    calculate_all_metrics,
    calculate_max_drawdown,
)


@dataclass
class Trade:
    """
    Represents a single completed trade.

    Attributes:
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        direction: 1 for long, -1 for short
        contracts: Number of contracts
        entry_price: Entry price
        exit_price: Exit price
        gross_pnl: P&L before costs
        costs: Total transaction costs
        net_pnl: P&L after costs
        return_pct: Return as percentage
        label: Actual label at entry (for analysis)
        prediction: Model prediction
        confidence: Model confidence (0-1)
    """

    entry_time: datetime
    exit_time: datetime
    direction: int
    contracts: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    label: int | None = None
    prediction: int | None = None
    confidence: float | None = None
    initial_risk_1r: float = 0.0
    r_multiple: float = 0.0
    stop_loss_price: float | None = None

    @property
    def is_winner(self) -> bool:
        """Check if trade is profitable after costs."""
        return self.net_pnl > 0

    @property
    def holding_period(self) -> int:
        """Calculate holding period in bars/periods."""
        if hasattr(self.exit_time, "timestamp") and hasattr(self.entry_time, "timestamp"):
            return int((self.exit_time - self.entry_time).total_seconds())
        return 0

    def calculate_r_multiple(self, point_value: float = 5.0) -> None:
        """
        Calculate R-multiple from stop loss.

        R-multiple = Net P&L / Initial Risk (1R)
        where 1R = distance to stop loss in dollars

        Args:
            point_value: Dollar value per point (default 5.0 for MES)
        """
        if self.stop_loss_price is None or self.stop_loss_price == 0:
            self.r_multiple = 0.0
            self.initial_risk_1r = 0.0
            return

        # Calculate initial risk per contract
        risk_per_contract = abs(self.entry_price - self.stop_loss_price) * point_value
        self.initial_risk_1r = risk_per_contract * self.contracts

        # Calculate R-multiple
        if self.initial_risk_1r > 0:
            self.r_multiple = self.net_pnl / self.initial_risk_1r
        else:
            self.r_multiple = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "direction": self.direction,
            "contracts": self.contracts,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "gross_pnl": self.gross_pnl,
            "costs": self.costs,
            "net_pnl": self.net_pnl,
            "return_pct": self.return_pct,
            "label": self.label,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "is_winner": self.is_winner,
            "initial_risk_1r": self.initial_risk_1r,
            "r_multiple": self.r_multiple,
            "stop_loss_price": self.stop_loss_price,
        }


@dataclass
class EquityCurve:
    """
    Manages equity curve data with analysis and visualization.

    This class tracks portfolio equity over time, maintains trade history,
    and provides methods for performance analysis and visualization.

    Attributes:
        initial_equity: Starting portfolio value
        timestamps: List of timestamps for each equity point
        equity_values: List of equity values at each timestamp
        trades: List of completed trades
        daily_returns: Computed daily returns
    """

    initial_equity: float = 100000.0
    timestamps: list[datetime] = field(default_factory=list)
    equity_values: list[float] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    _cached_metrics: PerformanceMetrics | None = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize with starting equity if empty."""
        if len(self.equity_values) == 0:
            self.equity_values = [self.initial_equity]
            self.timestamps = [datetime.now()]

    def add_equity_point(self, timestamp: datetime, equity: float) -> None:
        """
        Add a new equity point.

        Args:
            timestamp: Timestamp for the equity point
            equity: Current equity value
        """
        self.timestamps.append(timestamp)
        self.equity_values.append(equity)
        self._cached_metrics = None  # Invalidate cache

    def add_trade(self, trade: Trade) -> None:
        """
        Record a completed trade.

        Args:
            trade: Completed trade to record
        """
        self.trades.append(trade)
        self._cached_metrics = None

    @property
    def current_equity(self) -> float:
        """Get current equity value."""
        return self.equity_values[-1] if self.equity_values else self.initial_equity

    @property
    def total_return(self) -> float:
        """Calculate total return from initial equity."""
        if not self.equity_values or self.initial_equity <= 0:
            return 0.0
        return (self.equity_values[-1] / self.initial_equity) - 1

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L."""
        return self.current_equity - self.initial_equity

    @property
    def returns(self) -> np.ndarray:
        """Calculate period returns from equity curve."""
        if len(self.equity_values) < 2:
            return np.array([])
        equity = np.array(self.equity_values)
        result: np.ndarray = np.diff(equity) / equity[:-1]
        return result

    @property
    def trade_returns(self) -> np.ndarray:
        """Get array of trade returns (net P&L)."""
        return np.array([t.net_pnl for t in self.trades])

    def get_drawdowns(self) -> np.ndarray:
        """Calculate drawdown series."""
        equity = np.array(self.equity_values)
        if len(equity) == 0:
            return np.array([])
        running_max = np.maximum.accumulate(equity)
        result: np.ndarray = (equity - running_max) / running_max
        return result

    def get_metrics(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> PerformanceMetrics:
        """
        Get comprehensive performance metrics.

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year

        Returns:
            PerformanceMetrics dataclass
        """
        if self._cached_metrics is not None:
            return self._cached_metrics

        trade_returns = self.trade_returns
        equity = np.array(self.equity_values)

        metrics = calculate_all_metrics(
            trade_returns,
            equity,
            risk_free_rate,
            periods_per_year,
        )

        # Add date info if available
        if self.timestamps:
            metrics.start_date = str(self.timestamps[0])
            metrics.end_date = str(self.timestamps[-1])

        self._cached_metrics = metrics
        return metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to DataFrame."""
        return pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "equity": self.equity_values,
                "drawdown": self.get_drawdowns(),
            }
        )

    def trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trade history to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        metrics = self.get_metrics()
        return {
            "initial_equity": self.initial_equity,
            "final_equity": self.current_equity,
            "total_return_pct": self.total_return * 100,
            "total_pnl": self.total_pnl,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown_pct": metrics.max_drawdown * 100,
            "total_trades": metrics.total_trades,
            "win_rate_pct": metrics.win_rate * 100,
            "profit_factor": metrics.profit_factor,
            "expectancy": metrics.expectancy,
        }

    def plot(
        self,
        figsize: tuple[int, int] = (14, 10),
        show_drawdowns: bool = True,
        show_trades: bool = True,
        save_path: Path | str | None = None,
    ) -> Any:
        """
        Plot equity curve with optional drawdowns and trade markers.

        Args:
            figsize: Figure size
            show_drawdowns: Whether to show drawdown panel
            show_trades: Whether to mark trades on the chart
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
        except ImportError as err:
            raise ImportError("matplotlib is required for plotting") from err

        n_panels = 2 if show_drawdowns else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)

        if n_panels == 1:
            axes = [axes]

        # Convert to pandas for easier plotting
        df = self.to_dataframe()

        # Panel 1: Equity curve
        ax1 = axes[0]
        ax1.plot(df["timestamp"], df["equity"], "b-", linewidth=1.5, label="Equity")
        ax1.axhline(
            y=self.initial_equity,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Initial: ${self.initial_equity:,.0f}",
        )

        # Mark trades if requested
        if show_trades and self.trades:
            wins = [t for t in self.trades if t.is_winner]
            losses = [t for t in self.trades if not t.is_winner]

            if wins:
                win_times = [t.exit_time for t in wins]
                # Find equity at exit times
                win_equities = [self._get_equity_at_time(t) for t in win_times]
                ax1.scatter(
                    win_times,
                    win_equities,
                    c="green",
                    marker="^",
                    s=50,
                    alpha=0.7,
                    label=f"Wins ({len(wins)})",
                )

            if losses:
                loss_times = [t.exit_time for t in losses]
                loss_equities = [self._get_equity_at_time(t) for t in loss_times]
                ax1.scatter(
                    loss_times,
                    loss_equities,
                    c="red",
                    marker="v",
                    s=50,
                    alpha=0.7,
                    label=f"Losses ({len(losses)})",
                )

        ax1.set_ylabel("Equity ($)")
        ax1.set_title("Equity Curve")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Format y-axis with dollar signs
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Panel 2: Drawdowns
        if show_drawdowns:
            ax2 = axes[1]
            ax2.fill_between(df["timestamp"], df["drawdown"] * 100, 0, color="red", alpha=0.3)
            ax2.plot(df["timestamp"], df["drawdown"] * 100, "r-", linewidth=1)

            # Mark max drawdown
            max_dd, peak_idx, trough_idx = calculate_max_drawdown(np.array(self.equity_values))
            if trough_idx < len(df):
                ax2.axhline(
                    y=max_dd * 100,
                    color="darkred",
                    linestyle="--",
                    label=f"Max DD: {max_dd*100:.1f}%",
                )

            ax2.set_ylabel("Drawdown (%)")
            ax2.set_xlabel("Date")
            ax2.set_title("Drawdown")
            ax2.legend(loc="lower left")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(min(df["drawdown"] * 100) * 1.1, 5)

        # Format x-axis dates
        for ax in axes:
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def _get_equity_at_time(self, timestamp: datetime) -> float:
        """Get equity value at a specific timestamp."""
        if not self.timestamps:
            return self.initial_equity

        # Find closest timestamp
        for i, ts in enumerate(self.timestamps):
            if ts >= timestamp:
                return self.equity_values[i]
        return self.equity_values[-1]

    def plot_returns_distribution(
        self,
        figsize: tuple[int, int] = (12, 6),
        bins: int = 50,
        save_path: Path | str | None = None,
    ) -> Any:
        """
        Plot distribution of returns.

        Args:
            figsize: Figure size
            bins: Number of histogram bins
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError("matplotlib is required for plotting") from err

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Period returns histogram
        ax1 = axes[0]
        returns = self.returns * 100  # Convert to percentage
        if len(returns) > 0:
            ax1.hist(returns, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
            ax1.axvline(
                x=np.mean(returns),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(returns):.2f}%",
            )
            ax1.axvline(x=0, color="black", linestyle="-", alpha=0.5)
        ax1.set_xlabel("Return (%)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Period Returns Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Trade returns histogram
        ax2 = axes[1]
        trade_returns = self.trade_returns
        if len(trade_returns) > 0:
            ax2.hist(trade_returns, bins=bins, color="green", alpha=0.7, edgecolor="black")
            ax2.axvline(
                x=np.mean(trade_returns),
                color="red",
                linestyle="--",
                label=f"Mean: ${np.mean(trade_returns):.2f}",
            )
            ax2.axvline(x=0, color="black", linestyle="-", alpha=0.5)
        ax2.set_xlabel("P&L ($)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Trade P&L Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_monthly_returns(
        self,
        figsize: tuple[int, int] = (14, 8),
        save_path: Path | str | None = None,
    ) -> Any:
        """
        Plot monthly returns heatmap.

        Args:
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError("matplotlib is required for plotting") from err

        df = self.to_dataframe()
        if len(df) < 2:
            return None

        # Calculate returns
        df["returns"] = df["equity"].pct_change()
        df["year"] = pd.to_datetime(df["timestamp"]).dt.year
        df["month"] = pd.to_datetime(df["timestamp"]).dt.month

        # Aggregate monthly returns
        monthly = df.groupby(["year", "month"])["returns"].sum().unstack()

        if monthly.empty:
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
        norm = mcolors.TwoSlopeNorm(vmin=monthly.values.min(), vcenter=0, vmax=monthly.values.max())

        im = ax.imshow(monthly.values * 100, cmap=cmap, norm=norm, aspect="auto")

        # Labels
        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_labels)
        ax.set_yticks(range(len(monthly.index)))
        ax.set_yticklabels(monthly.index)

        # Add values
        for i in range(len(monthly.index)):
            for j in range(len(monthly.columns)):
                val = monthly.values[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > 0.05 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val*100:.1f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                    )

        ax.set_title("Monthly Returns (%)")
        plt.colorbar(im, label="Return (%)")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def save(self, path: Path | str) -> None:
        """
        Save equity curve to parquet file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_parquet(path)

        # Also save trades if present
        if self.trades:
            trades_path = path.with_suffix(".trades.parquet")
            self.trades_to_dataframe().to_parquet(trades_path)

    @classmethod
    def load(cls, path: Path | str) -> EquityCurve:
        """
        Load equity curve from parquet file.

        Args:
            path: Path to load from

        Returns:
            EquityCurve instance
        """
        path = Path(path)
        df = pd.read_parquet(path)

        curve = cls(
            initial_equity=df["equity"].iloc[0],
            timestamps=df["timestamp"].tolist(),
            equity_values=df["equity"].tolist(),
        )

        # Load trades if available
        trades_path = path.with_suffix(".trades.parquet")
        if trades_path.exists():
            trades_df = pd.read_parquet(trades_path)
            for _, row in trades_df.iterrows():
                trade = Trade(
                    entry_time=row["entry_time"],
                    exit_time=row["exit_time"],
                    direction=row["direction"],
                    contracts=row["contracts"],
                    entry_price=row["entry_price"],
                    exit_price=row["exit_price"],
                    gross_pnl=row["gross_pnl"],
                    costs=row["costs"],
                    net_pnl=row["net_pnl"],
                    return_pct=row["return_pct"],
                    label=row.get("label"),
                    prediction=row.get("prediction"),
                    confidence=row.get("confidence"),
                )
                curve.trades.append(trade)

        return curve


def create_equity_curve_from_trades(
    trades: list[Trade],
    initial_equity: float = 100000.0,
) -> EquityCurve:
    """
    Create an equity curve from a list of trades.

    Args:
        trades: List of Trade objects
        initial_equity: Starting equity

    Returns:
        EquityCurve instance
    """
    curve = EquityCurve(initial_equity=initial_equity)

    current_equity = initial_equity
    for trade in sorted(trades, key=lambda t: t.exit_time):
        current_equity += trade.net_pnl
        curve.add_equity_point(trade.exit_time, current_equity)
        curve.add_trade(trade)

    return curve


__all__ = [
    "Trade",
    "EquityCurve",
    "create_equity_curve_from_trades",
]
