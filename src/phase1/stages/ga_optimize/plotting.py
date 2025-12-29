"""
Plotting functions for GA optimization results.

Contains:
    - plot_convergence: Plot GA convergence curve and label distribution
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def plot_convergence(results: dict, output_path: Path) -> None:
    """
    Plot GA convergence curve and label distribution.

    Parameters:
    -----------
    results : Dict with 'convergence' and 'validation' keys
    output_path : Path to save the plot
    """
    convergence = results["convergence"]

    gens = [c["gen"] for c in convergence]
    avg_fits = [c["avg"] for c in convergence]
    max_fits = [c["max"] for c in convergence]
    min_fits = [c["min"] for c in convergence]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Convergence plot
    ax1.plot(gens, max_fits, "g-", label="Best", linewidth=2)
    ax1.plot(gens, avg_fits, "b-", label="Average", linewidth=2)
    ax1.plot(gens, min_fits, "r-", label="Worst", linewidth=1, alpha=0.5)
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Fitness", fontsize=12)
    ax1.set_title(
        f'GA Convergence - Horizon {results["horizon"]}', fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Label distribution (if validation data available)
    if "validation" in results:
        val = results["validation"]
        labels = ["Long", "Short", "Neutral"]
        sizes = [val["pct_long"], val["pct_short"], val["pct_neutral"]]
        colors = ["#2ecc71", "#e74c3c", "#95a5a6"]

        ax2.bar(labels, sizes, color=colors, edgecolor="black", linewidth=1.2)
        ax2.axhline(y=40, color="orange", linestyle="--", label="40% min signal", alpha=0.7)
        ax2.set_ylabel("Percentage (%)", fontsize=12)
        ax2.set_title(
            f'Label Distribution (Full Data)\nSignal Rate: {val["signal_rate"]*100:.1f}%',
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_ylim(0, 60)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # Add percentage labels on bars
        for i, (label, pct) in enumerate(zip(labels, sizes, strict=False)):
            ax2.text(i, pct + 1, f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved convergence plot to {output_path}")
