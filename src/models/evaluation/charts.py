"""
Financial Report Chart Generation.

This module provides chart generation functions for the post-training
financial report, including classification and trading performance visuals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.inference.backtesting.equity_curve import EquityCurve

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    figsize: tuple[int, int] = (8, 6),
    save_path: Path | str | None = None,
    title: str = "Confusion Matrix",
) -> Any:
    """
    Plot confusion matrix for classification results.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (default: ["Short", "Neutral", "Long"])
        figsize: Figure size
        save_path: Path to save figure
        title: Chart title

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    if labels is None:
        labels = ["Short (-1)", "Neutral (0)", "Long (+1)"]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize for percentages
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)  # type: ignore[attr-defined]
    ax.figure.colorbar(im, ax=ax, label="Proportion")

    # Labels and ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels[: cm.shape[1]],
        yticklabels=labels[: cm.shape[0]],
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_normalized.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=10,
            )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")

    return fig


def plot_rolling_sharpe(
    returns: np.ndarray | pd.Series,
    window: int = 30,
    periods_per_year: int = 252,
    figsize: tuple[int, int] = (12, 5),
    save_path: Path | str | None = None,
    title: str = "Rolling Sharpe Ratio",
) -> Any:
    """
    Plot rolling Sharpe ratio over time.

    Args:
        returns: Array of period returns
        window: Rolling window size
        periods_per_year: Number of periods per year for annualization
        figsize: Figure size
        save_path: Path to save figure
        title: Chart title

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns

    # Calculate rolling Sharpe
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    # Annualize
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot rolling Sharpe
    ax.plot(
        rolling_sharpe.values,
        color="steelblue",
        linewidth=1.5,
        label=f"{window}-period Rolling Sharpe",
    )

    # Add reference lines
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="Sharpe = 1")
    ax.axhline(y=2, color="green", linestyle=":", alpha=0.5, label="Sharpe = 2")
    ax.axhline(y=-1, color="red", linestyle="--", alpha=0.5)

    # Fill positive/negative regions
    ax.fill_between(
        range(len(rolling_sharpe)),
        rolling_sharpe.values,
        0,
        where=rolling_sharpe.values > 0,
        color="green",
        alpha=0.1,
    )
    ax.fill_between(
        range(len(rolling_sharpe)),
        rolling_sharpe.values,
        0,
        where=rolling_sharpe.values < 0,
        color="red",
        alpha=0.1,
    )

    ax.set_xlabel("Period")
    ax.set_ylabel("Sharpe Ratio (Annualized)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved rolling Sharpe chart to {save_path}")

    return fig


def plot_trade_analysis(
    equity_curve: EquityCurve,
    figsize: tuple[int, int] = (14, 10),
    save_path: Path | str | None = None,
) -> Any:
    """
    Plot comprehensive trade analysis charts.

    Includes:
    - P&L by trade direction (long vs short)
    - Trade holding period distribution
    - Cumulative P&L over trades
    - Win/Loss streak analysis

    Args:
        equity_curve: EquityCurve object with trade history
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if not equity_curve.trades:
        logger.warning("No trades to analyze")
        return None

    trades_df = equity_curve.trades_to_dataframe()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. P&L by Direction
    ax1 = axes[0, 0]
    long_pnl = trades_df[trades_df["direction"] == 1]["net_pnl"]
    short_pnl = trades_df[trades_df["direction"] == -1]["net_pnl"]

    positions = [1, 2]
    bp = ax1.boxplot(
        [
            long_pnl.values if len(long_pnl) > 0 else [0],
            short_pnl.values if len(short_pnl) > 0 else [0],
        ],
        positions=positions,
        widths=0.6,
        patch_artist=True,
    )

    colors = ["green", "red"]
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.set_xticklabels(["Long", "Short"])
    ax1.set_ylabel("P&L ($)")
    ax1.set_title("P&L by Trade Direction")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add stats text
    long_mean = long_pnl.mean() if len(long_pnl) > 0 else 0
    short_mean = short_pnl.mean() if len(short_pnl) > 0 else 0
    ax1.text(
        0.02,
        0.98,
        f"Long: n={len(long_pnl)}, avg=${long_mean:.2f}\nShort: n={len(short_pnl)}, avg=${short_mean:.2f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # 2. Cumulative P&L
    ax2 = axes[0, 1]
    cumulative_pnl = trades_df["net_pnl"].cumsum()
    ax2.plot(range(len(cumulative_pnl)), cumulative_pnl.values, color="steelblue", linewidth=1.5)
    ax2.fill_between(
        range(len(cumulative_pnl)),
        cumulative_pnl.values,
        0,
        where=cumulative_pnl.values > 0,
        color="green",
        alpha=0.2,
    )
    ax2.fill_between(
        range(len(cumulative_pnl)),
        cumulative_pnl.values,
        0,
        where=cumulative_pnl.values < 0,
        color="red",
        alpha=0.2,
    )
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Trade Number")
    ax2.set_ylabel("Cumulative P&L ($)")
    ax2.set_title("Cumulative P&L Over Trades")
    ax2.grid(True, alpha=0.3)

    # 3. Win/Loss Distribution
    ax3 = axes[1, 0]
    wins = trades_df[trades_df["is_winner"]]["net_pnl"]
    losses = trades_df[~trades_df["is_winner"]]["net_pnl"]

    ax3.hist(
        wins, bins=20, color="green", alpha=0.6, label=f"Wins (n={len(wins)})", edgecolor="black"
    )
    ax3.hist(
        losses,
        bins=20,
        color="red",
        alpha=0.6,
        label=f"Losses (n={len(losses)})",
        edgecolor="black",
    )
    ax3.axvline(x=0, color="black", linestyle="-", alpha=0.5)

    if len(wins) > 0:
        ax3.axvline(
            x=wins.mean(), color="darkgreen", linestyle="--", label=f"Avg Win: ${wins.mean():.2f}"
        )
    if len(losses) > 0:
        ax3.axvline(
            x=losses.mean(),
            color="darkred",
            linestyle="--",
            label=f"Avg Loss: ${losses.mean():.2f}",
        )

    ax3.set_xlabel("P&L ($)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Win/Loss Distribution")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Win/Loss Streaks
    ax4 = axes[1, 1]
    is_winner = trades_df["is_winner"].values

    # Calculate streaks
    streaks = []
    current_streak = 1
    current_type = is_winner[0] if len(is_winner) > 0 else True

    for i in range(1, len(is_winner)):
        if is_winner[i] == current_type:
            current_streak += 1
        else:
            streaks.append((current_type, current_streak))
            current_streak = 1
            current_type = is_winner[i]
    if len(is_winner) > 0:
        streaks.append((current_type, current_streak))

    win_streaks = [s[1] for s in streaks if s[0]]
    loss_streaks = [s[1] for s in streaks if not s[0]]

    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0

    # Bar chart of streak lengths
    streak_range = range(1, max(max_win_streak, max_loss_streak, 5) + 1)
    win_counts = [win_streaks.count(i) for i in streak_range]
    loss_counts = [loss_streaks.count(i) for i in streak_range]

    x = np.arange(len(streak_range))
    width = 0.35

    ax4.bar(x - width / 2, win_counts, width, label="Win Streaks", color="green", alpha=0.7)
    ax4.bar(x + width / 2, loss_counts, width, label="Loss Streaks", color="red", alpha=0.7)

    ax4.set_xlabel("Streak Length")
    ax4.set_ylabel("Frequency")
    ax4.set_title(f"Win/Loss Streaks (Max Win: {max_win_streak}, Max Loss: {max_loss_streak})")
    ax4.set_xticks(x)
    ax4.set_xticklabels(streak_range)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved trade analysis chart to {save_path}")

    return fig


def plot_feature_importance(
    feature_names: list[str],
    importance_values: np.ndarray,
    top_n: int = 20,
    figsize: tuple[int, int] = (10, 8),
    save_path: Path | str | None = None,
    title: str = "Feature Importance",
) -> Any:
    """
    Plot feature importance bar chart.

    Args:
        feature_names: Names of features
        importance_values: Importance scores
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Path to save figure
        title: Chart title

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    # Sort and get top N
    indices = np.argsort(importance_values)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_values = importance_values[indices]

    fig, ax = plt.subplots(figsize=figsize)

    # Horizontal bar chart
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_values, color="steelblue", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved feature importance chart to {save_path}")

    return fig


def generate_all_charts(
    equity_curve: EquityCurve,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    feature_names: list[str] | None = None,
    feature_importance: np.ndarray | None = None,
) -> dict[str, Path]:
    """
    Generate all financial report charts.

    Args:
        equity_curve: EquityCurve object with trade history
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save charts
        feature_names: Optional feature names for importance chart
        feature_importance: Optional feature importance values

    Returns:
        Dictionary mapping chart names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = {}

    # 1. Equity Curve with Drawdown
    try:
        equity_path = output_dir / "equity_curve.png"
        equity_curve.plot(save_path=equity_path)
        charts["equity_curve"] = equity_path
        logger.info("Generated equity curve chart")
    except Exception as e:
        logger.error(f"Failed to generate equity curve: {e}")

    # 2. Returns Distribution
    try:
        returns_path = output_dir / "returns_distribution.png"
        equity_curve.plot_returns_distribution(save_path=returns_path)
        charts["returns_distribution"] = returns_path
        logger.info("Generated returns distribution chart")
    except Exception as e:
        logger.error(f"Failed to generate returns distribution: {e}")

    # 3. Monthly Returns Heatmap
    try:
        monthly_path = output_dir / "monthly_returns.png"
        equity_curve.plot_monthly_returns(save_path=monthly_path)
        charts["monthly_returns"] = monthly_path
        logger.info("Generated monthly returns heatmap")
    except Exception as e:
        logger.error(f"Failed to generate monthly returns: {e}")

    # 4. Confusion Matrix
    try:
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        charts["confusion_matrix"] = cm_path
        logger.info("Generated confusion matrix")
    except Exception as e:
        logger.error(f"Failed to generate confusion matrix: {e}")

    # 5. Rolling Sharpe Ratio
    try:
        if len(equity_curve.returns) > 30:
            sharpe_path = output_dir / "rolling_sharpe.png"
            plot_rolling_sharpe(equity_curve.returns, save_path=sharpe_path)
            charts["rolling_sharpe"] = sharpe_path
            logger.info("Generated rolling Sharpe chart")
    except Exception as e:
        logger.error(f"Failed to generate rolling Sharpe: {e}")

    # 6. Trade Analysis
    try:
        if equity_curve.trades:
            trade_path = output_dir / "trade_analysis.png"
            plot_trade_analysis(equity_curve, save_path=trade_path)
            charts["trade_analysis"] = trade_path
            logger.info("Generated trade analysis chart")
    except Exception as e:
        logger.error(f"Failed to generate trade analysis: {e}")

    # 7. Feature Importance (if provided)
    if feature_names is not None and feature_importance is not None:
        try:
            fi_path = output_dir / "feature_importance.png"
            plot_feature_importance(feature_names, feature_importance, save_path=fi_path)
            charts["feature_importance"] = fi_path
            logger.info("Generated feature importance chart")
        except Exception as e:
            logger.error(f"Failed to generate feature importance: {e}")

    logger.info(f"Generated {len(charts)} charts in {output_dir}")
    return charts
