"""
Post-Training Financial Report Generator.

This module generates comprehensive financial reports after model training,
including trade simulation, performance metrics, and visualizations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.backtesting.equity_curve import EquityCurve, Trade

from .charts import generate_all_charts

logger = logging.getLogger(__name__)


@dataclass
class FinancialReportConfig:
    """Configuration for financial report generation."""

    initial_equity: float = 100000.0
    commission_per_trade: float = 2.50  # Per side
    slippage_ticks: float = 1.0
    tick_value: float = 1.25  # For MES
    contracts_per_trade: int = 1
    risk_free_rate: float = 0.0
    periods_per_year: int = 252


@dataclass
class FinancialReport:
    """
    Comprehensive financial evaluation report.

    Contains all metrics, trade statistics, and chart paths
    generated from post-training evaluation.
    """

    # Identification
    run_id: str
    model_name: str
    horizon: int
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration
    config: dict[str, Any] = field(default_factory=dict)

    # Performance Metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # By Direction
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    long_pnl: float = 0.0
    short_pnl: float = 0.0

    # Classification Metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Chart Paths
    charts: dict[str, str] = field(default_factory=dict)

    # Data Paths
    equity_curve_path: str | None = None
    trades_path: str | None = None

    # Report Output Paths (set after generation)
    html_path: Path | None = field(default=None, repr=False)
    json_path: Path | None = field(default=None, repr=False)
    charts_dir: Path | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)

    def save_json(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved financial report JSON to {path}")


def simulate_trades(
    predictions: np.ndarray,
    prices: np.ndarray,
    timestamps: pd.DatetimeIndex | np.ndarray | None = None,
    labels: np.ndarray | None = None,
    config: FinancialReportConfig | None = None,
) -> EquityCurve:
    """
    Simulate trades from model predictions.

    This function converts classification predictions (-1, 0, 1) into
    simulated trades with realistic P&L including transaction costs.

    Args:
        predictions: Model predictions (-1=short, 0=neutral, 1=long)
        prices: Close prices corresponding to predictions
        timestamps: Optional timestamps for each prediction
        labels: Optional true labels for analysis
        config: Report configuration

    Returns:
        EquityCurve with simulated trades
    """
    if config is None:
        config = FinancialReportConfig()

    equity_curve = EquityCurve(initial_equity=config.initial_equity)

    if timestamps is None:
        timestamps = pd.date_range(start="2020-01-01", periods=len(predictions), freq="5min")

    current_equity = config.initial_equity
    position = 0  # Current position: -1, 0, or 1
    entry_price = 0.0
    entry_time: datetime | None = None
    entry_label: int | None = None
    entry_pred: int | None = None

    for i in range(len(predictions)):
        pred = int(predictions[i])
        price = float(prices[i])
        ts = pd.Timestamp(timestamps[i]).to_pydatetime()  # type: ignore[union-attr]

        # Record equity point
        equity_curve.add_equity_point(ts, current_equity)

        # Skip neutral predictions (no trade)
        if pred == 0:
            continue

        # If we have a position and prediction changes, close it
        if position != 0 and pred != position:
            # Close existing position
            exit_price = price

            # Calculate P&L
            if position == 1:  # Long
                gross_pnl = (
                    (exit_price - entry_price)
                    * config.contracts_per_trade
                    * (1 / config.tick_value)
                )
            else:  # Short
                gross_pnl = (
                    (entry_price - exit_price)
                    * config.contracts_per_trade
                    * (1 / config.tick_value)
                )

            # Scale by tick value
            gross_pnl *= config.tick_value

            # Transaction costs (round trip)
            costs = (config.commission_per_trade * 2) + (
                config.slippage_ticks * config.tick_value * 2
            )
            net_pnl = gross_pnl - costs

            # Calculate return
            return_pct = net_pnl / current_equity if current_equity > 0 else 0

            # Create trade record
            trade = Trade(
                entry_time=entry_time,  # type: ignore[arg-type]
                exit_time=ts,
                direction=position,
                contracts=config.contracts_per_trade,
                entry_price=entry_price,
                exit_price=exit_price,
                gross_pnl=gross_pnl,
                costs=costs,
                net_pnl=net_pnl,
                return_pct=return_pct,
                label=entry_label,
                prediction=entry_pred,
            )
            equity_curve.add_trade(trade)

            # Update equity
            current_equity += net_pnl
            position = 0

        # Open new position if not already in one
        if position == 0 and pred != 0:
            position = pred
            entry_price = price
            entry_time = ts
            entry_pred = pred
            entry_label = int(labels[i]) if labels is not None else None

    # Close any remaining position at the end
    if position != 0 and len(prices) > 0:
        exit_price = float(prices[-1])
        ts = pd.Timestamp(timestamps[-1]).to_pydatetime()  # type: ignore[union-attr]

        if position == 1:
            gross_pnl = (
                (exit_price - entry_price) * config.contracts_per_trade * (1 / config.tick_value)
            )
        else:
            gross_pnl = (
                (entry_price - exit_price) * config.contracts_per_trade * (1 / config.tick_value)
            )

        gross_pnl *= config.tick_value
        costs = (config.commission_per_trade * 2) + (config.slippage_ticks * config.tick_value * 2)
        net_pnl = gross_pnl - costs
        return_pct = net_pnl / current_equity if current_equity > 0 else 0

        trade = Trade(
            entry_time=entry_time,  # type: ignore[arg-type]
            exit_time=ts,
            direction=position,
            contracts=config.contracts_per_trade,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            costs=costs,
            net_pnl=net_pnl,
            return_pct=return_pct,
            label=entry_label,
            prediction=entry_pred,
        )
        equity_curve.add_trade(trade)
        current_equity += net_pnl

    # Final equity point
    if len(timestamps) > 0:
        equity_curve.add_equity_point(pd.Timestamp(timestamps[-1]).to_pydatetime(), current_equity)  # type: ignore[union-attr]

    return equity_curve


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of classification metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def calculate_streak_stats(trades: list[Trade]) -> tuple[int, int]:
    """Calculate max consecutive wins and losses."""
    if not trades:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for trade in trades:
        if trade.is_winner:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    return max_wins, max_losses


def generate_financial_report(
    model_name: str,
    horizon: int,
    predictions: np.ndarray,
    y_true: np.ndarray,
    prices: np.ndarray,
    timestamps: pd.DatetimeIndex | np.ndarray | None = None,
    output_dir: Path | str = Path("./reports"),
    run_id: str | None = None,
    config: FinancialReportConfig | None = None,
    feature_names: list[str] | None = None,
    feature_importance: np.ndarray | None = None,
) -> FinancialReport:
    """
    Generate comprehensive financial report after model training.

    This is the main entry point for post-training evaluation. It:
    1. Simulates trades from predictions
    2. Calculates all performance metrics
    3. Generates visualization charts
    4. Produces HTML and JSON reports

    Args:
        model_name: Name of the model
        horizon: Prediction horizon
        predictions: Model predictions (-1, 0, 1)
        y_true: True labels
        prices: Close prices
        timestamps: Optional timestamps
        output_dir: Directory for output files
        run_id: Optional run identifier
        config: Report configuration
        feature_names: Optional feature names for importance chart
        feature_importance: Optional feature importance values

    Returns:
        FinancialReport with all metrics and chart paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = FinancialReportConfig()

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("GENERATING FINANCIAL REPORT")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}, Horizon: {horizon}")
    logger.info(f"Samples: {len(predictions)}")
    logger.info(f"Output: {output_dir}")

    # 1. Simulate trades
    logger.info("\nSimulating trades from predictions...")
    equity_curve = simulate_trades(
        predictions=predictions,
        prices=prices,
        timestamps=timestamps,
        labels=y_true,
        config=config,
    )
    logger.info(f"Generated {len(equity_curve.trades)} trades")

    # 2. Get performance metrics
    logger.info("\nCalculating performance metrics...")
    metrics = equity_curve.get_metrics(
        risk_free_rate=config.risk_free_rate,
        periods_per_year=config.periods_per_year,
    )

    # 3. Calculate classification metrics
    logger.info("Calculating classification metrics...")
    class_metrics = calculate_classification_metrics(y_true, predictions)

    # 4. Calculate streak stats
    max_wins, max_losses = calculate_streak_stats(equity_curve.trades)

    # 5. Calculate direction-specific stats
    trades_df = equity_curve.trades_to_dataframe() if equity_curve.trades else pd.DataFrame()
    long_trades = trades_df[trades_df["direction"] == 1] if len(trades_df) > 0 else pd.DataFrame()
    short_trades = trades_df[trades_df["direction"] == -1] if len(trades_df) > 0 else pd.DataFrame()

    long_wins = long_trades[long_trades["is_winner"]] if len(long_trades) > 0 else pd.DataFrame()
    short_wins = (
        short_trades[short_trades["is_winner"]] if len(short_trades) > 0 else pd.DataFrame()
    )

    # Compute winning/losing trade counts and averages from trades_df
    winning_trades_df = trades_df[trades_df["is_winner"]] if len(trades_df) > 0 else pd.DataFrame()
    losing_trades_df = trades_df[~trades_df["is_winner"]] if len(trades_df) > 0 else pd.DataFrame()
    n_winning = len(winning_trades_df)
    n_losing = len(losing_trades_df)
    avg_win = float(winning_trades_df["net_pnl"].mean()) if n_winning > 0 else 0.0
    avg_loss = float(losing_trades_df["net_pnl"].mean()) if n_losing > 0 else 0.0

    # 6. Generate charts
    logger.info("\nGenerating charts...")
    charts_dir = output_dir / "charts"
    charts = generate_all_charts(
        equity_curve=equity_curve,
        y_true=y_true,
        y_pred=predictions,
        output_dir=charts_dir,
        feature_names=feature_names,
        feature_importance=feature_importance,
    )

    # 7. Save equity curve and trades data
    equity_path = output_dir / "equity_curve.csv"
    equity_curve.to_dataframe().to_csv(equity_path, index=False)

    trades_path = output_dir / "trades.csv"
    if equity_curve.trades:
        trades_df.to_csv(trades_path, index=False)

    # 8. Create report object
    report = FinancialReport(
        run_id=run_id,
        model_name=model_name,
        horizon=horizon,
        config=asdict(config),
        # Performance metrics
        sharpe_ratio=metrics.sharpe_ratio,
        sortino_ratio=metrics.sortino_ratio,
        calmar_ratio=metrics.calmar_ratio,
        max_drawdown=metrics.max_drawdown,
        max_drawdown_duration=metrics.max_drawdown_duration,
        total_return=metrics.total_return,
        annual_return=metrics.cagr,  # Use CAGR for annual return
        volatility=metrics.std_return,  # Use std_return for volatility
        var_95=metrics.var_95,
        cvar_95=metrics.cvar_95,
        # Trade statistics (computed from trades_df above)
        total_trades=metrics.total_trades,
        winning_trades=n_winning,
        losing_trades=n_losing,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
        expectancy=metrics.expectancy,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,
        # By direction
        long_trades=len(long_trades),
        short_trades=len(short_trades),
        long_win_rate=len(long_wins) / len(long_trades) if len(long_trades) > 0 else 0,
        short_win_rate=len(short_wins) / len(short_trades) if len(short_trades) > 0 else 0,
        long_pnl=float(long_trades["net_pnl"].sum()) if len(long_trades) > 0 else 0,
        short_pnl=float(short_trades["net_pnl"].sum()) if len(short_trades) > 0 else 0,
        # Classification
        accuracy=class_metrics["accuracy"],
        precision=class_metrics["precision"],
        recall=class_metrics["recall"],
        f1_score=class_metrics["f1_score"],
        # Paths
        charts={k: str(v) for k, v in charts.items()},
        equity_curve_path=str(equity_path),
        trades_path=str(trades_path) if equity_curve.trades else None,
    )

    # 9. Save JSON report
    json_path = output_dir / "financial_report.json"
    report.save_json(json_path)

    # 10. Generate HTML report
    logger.info("\nGenerating HTML report...")
    html_path = output_dir / "financial_report.html"
    render_html_report(report, charts, html_path)

    # Set output paths on report for convenience
    report.html_path = html_path
    report.json_path = json_path
    report.charts_dir = charts_dir

    logger.info("=" * 70)
    logger.info("FINANCIAL REPORT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {report.max_drawdown * 100:.1f}%")
    logger.info(f"Win Rate: {report.win_rate * 100:.1f}%")
    logger.info(f"Total Trades: {report.total_trades}")
    logger.info(f"Report saved to: {html_path}")

    return report


def render_html_report(
    report: FinancialReport,
    charts: dict[str, Path],
    output_path: Path,
) -> None:
    """
    Render financial report as HTML.

    Args:
        report: FinancialReport object
        charts: Dictionary of chart name to path
        output_path: Path for HTML file
    """
    import base64

    def embed_image(path: Path) -> str:
        """Embed image as base64 in HTML."""
        if not Path(path).exists():
            return ""
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{data}"

    # HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Report - {report.model_name}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header .meta {{ opacity: 0.8; font-size: 0.9em; }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            font-size: 1.3em;
            color: #2c5282;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #f7fafc;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c5282;
        }}
        .metric .label {{
            font-size: 0.85em;
            color: #718096;
            margin-top: 5px;
        }}
        .metric.positive .value {{ color: #38a169; }}
        .metric.negative .value {{ color: #e53e3e; }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f7fafc;
            font-weight: 600;
            color: #4a5568;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 768px) {{
            .two-col {{ grid-template-columns: 1fr; }}
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #718096;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Financial Performance Report</h1>
            <div class="meta">
                <strong>Model:</strong> {report.model_name} |
                <strong>Horizon:</strong> {report.horizon} bars |
                <strong>Run ID:</strong> {report.run_id} |
                <strong>Generated:</strong> {report.generated_at}
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="card">
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric {'positive' if report.sharpe_ratio > 1 else 'negative' if report.sharpe_ratio < 0 else ''}">
                    <div class="value">{report.sharpe_ratio:.2f}</div>
                    <div class="label">Sharpe Ratio</div>
                </div>
                <div class="metric {'positive' if report.total_return > 0 else 'negative'}">
                    <div class="value">{report.total_return * 100:.1f}%</div>
                    <div class="label">Total Return</div>
                </div>
                <div class="metric negative">
                    <div class="value">{report.max_drawdown * 100:.1f}%</div>
                    <div class="label">Max Drawdown</div>
                </div>
                <div class="metric {'positive' if report.win_rate > 0.5 else 'negative'}">
                    <div class="value">{report.win_rate * 100:.1f}%</div>
                    <div class="label">Win Rate</div>
                </div>
                <div class="metric {'positive' if report.profit_factor > 1 else 'negative'}">
                    <div class="value">{report.profit_factor:.2f}</div>
                    <div class="label">Profit Factor</div>
                </div>
                <div class="metric">
                    <div class="value">{report.total_trades}</div>
                    <div class="label">Total Trades</div>
                </div>
            </div>
        </div>

        <!-- Equity Curve -->
        {"<div class='card'><h2>Equity Curve</h2><div class='chart-container'><img src='" + embed_image(charts.get('equity_curve', Path(''))) + "' alt='Equity Curve'></div></div>" if 'equity_curve' in charts else ''}

        <!-- Risk Metrics -->
        <div class="card">
            <h2>Risk-Adjusted Performance</h2>
            <div class="two-col">
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Sharpe Ratio</td><td>{report.sharpe_ratio:.3f}</td></tr>
                    <tr><td>Sortino Ratio</td><td>{report.sortino_ratio:.3f}</td></tr>
                    <tr><td>Calmar Ratio</td><td>{report.calmar_ratio:.3f}</td></tr>
                    <tr><td>Annual Return</td><td>{report.annual_return * 100:.2f}%</td></tr>
                    <tr><td>Volatility</td><td>{report.volatility * 100:.2f}%</td></tr>
                </table>
                <table>
                    <tr><th>Risk Metric</th><th>Value</th></tr>
                    <tr><td>Max Drawdown</td><td>{report.max_drawdown * 100:.2f}%</td></tr>
                    <tr><td>Max DD Duration</td><td>{report.max_drawdown_duration} bars</td></tr>
                    <tr><td>VaR (95%)</td><td>{report.var_95 * 100:.2f}%</td></tr>
                    <tr><td>CVaR (95%)</td><td>{report.cvar_95 * 100:.2f}%</td></tr>
                </table>
            </div>
        </div>

        <!-- Trade Statistics -->
        <div class="card">
            <h2>Trade Statistics</h2>
            <div class="two-col">
                <table>
                    <tr><th>Statistic</th><th>Value</th></tr>
                    <tr><td>Total Trades</td><td>{report.total_trades}</td></tr>
                    <tr><td>Winning Trades</td><td>{report.winning_trades}</td></tr>
                    <tr><td>Losing Trades</td><td>{report.losing_trades}</td></tr>
                    <tr><td>Win Rate</td><td>{report.win_rate * 100:.1f}%</td></tr>
                    <tr><td>Profit Factor</td><td>{report.profit_factor:.2f}</td></tr>
                    <tr><td>Expectancy</td><td>${report.expectancy:.2f}</td></tr>
                </table>
                <table>
                    <tr><th>Statistic</th><th>Value</th></tr>
                    <tr><td>Average Win</td><td>${report.avg_win:.2f}</td></tr>
                    <tr><td>Average Loss</td><td>${report.avg_loss:.2f}</td></tr>
                    <tr><td>Max Consecutive Wins</td><td>{report.max_consecutive_wins}</td></tr>
                    <tr><td>Max Consecutive Losses</td><td>{report.max_consecutive_losses}</td></tr>
                    <tr><td>Long Trades</td><td>{report.long_trades} ({report.long_win_rate * 100:.1f}% win)</td></tr>
                    <tr><td>Short Trades</td><td>{report.short_trades} ({report.short_win_rate * 100:.1f}% win)</td></tr>
                </table>
            </div>
        </div>

        <!-- Trade Analysis Chart -->
        {"<div class='card'><h2>Trade Analysis</h2><div class='chart-container'><img src='" + embed_image(charts.get('trade_analysis', Path(''))) + "' alt='Trade Analysis'></div></div>" if 'trade_analysis' in charts else ''}

        <!-- Monthly Returns -->
        {"<div class='card'><h2>Monthly Returns</h2><div class='chart-container'><img src='" + embed_image(charts.get('monthly_returns', Path(''))) + "' alt='Monthly Returns'></div></div>" if 'monthly_returns' in charts else ''}

        <!-- Classification Metrics -->
        <div class="card">
            <h2>Classification Performance</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="value">{report.accuracy * 100:.1f}%</div>
                    <div class="label">Accuracy</div>
                </div>
                <div class="metric">
                    <div class="value">{report.precision * 100:.1f}%</div>
                    <div class="label">Precision</div>
                </div>
                <div class="metric">
                    <div class="value">{report.recall * 100:.1f}%</div>
                    <div class="label">Recall</div>
                </div>
                <div class="metric">
                    <div class="value">{report.f1_score * 100:.1f}%</div>
                    <div class="label">F1 Score</div>
                </div>
            </div>
        </div>

        <!-- Confusion Matrix -->
        {"<div class='card'><h2>Confusion Matrix</h2><div class='chart-container'><img src='" + embed_image(charts.get('confusion_matrix', Path(''))) + "' alt='Confusion Matrix'></div></div>" if 'confusion_matrix' in charts else ''}

        <!-- Rolling Sharpe -->
        {"<div class='card'><h2>Rolling Sharpe Ratio</h2><div class='chart-container'><img src='" + embed_image(charts.get('rolling_sharpe', Path(''))) + "' alt='Rolling Sharpe'></div></div>" if 'rolling_sharpe' in charts else ''}

        <!-- Returns Distribution -->
        {"<div class='card'><h2>Returns Distribution</h2><div class='chart-container'><img src='" + embed_image(charts.get('returns_distribution', Path(''))) + "' alt='Returns Distribution'></div></div>" if 'returns_distribution' in charts else ''}

        <!-- Feature Importance -->
        {"<div class='card'><h2>Feature Importance</h2><div class='chart-container'><img src='" + embed_image(charts.get('feature_importance', Path(''))) + "' alt='Feature Importance'></div></div>" if 'feature_importance' in charts else ''}

        <div class="footer">
            Generated by ML Pipeline Financial Report Generator | {report.generated_at}
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Saved HTML report to {output_path}")
