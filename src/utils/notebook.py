"""Notebook utilities for Jupyter/Google Colab environments."""
from __future__ import annotations

import logging
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


def setup_notebook(
    log_level: str = "INFO",
    suppress_warnings: bool = True,
    seed: int = 42,
    max_display_rows: int = 100,
    max_display_cols: int = 50,
) -> dict[str, Any]:
    """Initialize notebook environment with optimal settings."""
    import pandas as pd

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress common warnings
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        warnings.filterwarnings("ignore", message=".*pynvml.*")

    # Set random seeds
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Configure pandas display
    pd.set_option("display.max_rows", max_display_rows)
    pd.set_option("display.max_columns", max_display_cols)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    # Configure matplotlib
    try:
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 100
    except Exception:
        pass

    # Get environment info
    from src.models.device import get_best_gpu, get_environment_info, get_mixed_precision_config

    env_info = get_environment_info()
    gpu_info = get_best_gpu()

    result = {
        **env_info,
        "gpu_available": gpu_info is not None,
        "gpu_name": gpu_info.name if gpu_info else None,
        "gpu_memory_gb": gpu_info.total_memory_gb if gpu_info else 0,
        "seed": seed,
    }

    if gpu_info:
        mp_config = get_mixed_precision_config(gpu_info)
        result["mixed_precision"] = mp_config

    # Print summary
    print("=" * 60)
    print("Notebook Environment Setup Complete")
    print("=" * 60)
    print(f"  Platform: {result['platform']}")
    print(f"  Python: {result['python_version']}")
    print(f"  Colab: {result['is_colab']}")
    print(f"  Kaggle: {result['is_kaggle']}")
    if result["gpu_available"]:
        print(f"  GPU: {result['gpu_name']} ({result['gpu_memory_gb']:.1f} GB)")
        print(f"  Mixed Precision: {result.get('mixed_precision', {}).get('dtype', 'N/A')}")
    else:
        print("  GPU: Not available (using CPU)")
    print(f"  Random Seed: {seed}")
    print("=" * 60)

    return result


def install_dependencies(
    packages: list[str] | None = None,
    quiet: bool = True,
) -> dict[str, bool]:
    """Install required packages in Colab/Kaggle."""
    import subprocess

    if packages is None:
        packages = [
            "xgboost>=1.7.0",
            "lightgbm>=3.3.0",
            "catboost>=1.1.0",
            "optuna>=3.0.0",
            "scikit-learn>=1.2.0",
            "torch>=2.0.0",
            "tqdm>=4.64.0",
        ]

    results = {}
    quiet_flag = "-q" if quiet else ""

    for package in packages:
        package_name = package.split(">=")[0].split("==")[0]
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", quiet_flag, package],
                stdout=subprocess.DEVNULL if quiet else None,
                stderr=subprocess.DEVNULL if quiet else None,
            )
            results[package_name] = True
            if not quiet:
                print(f"Installed {package_name}")
        except subprocess.CalledProcessError:
            results[package_name] = False
            print(f"Failed to install {package_name}")

    return results


def mount_drive(mount_path: str = "/content/drive") -> bool:
    """Mount Google Drive in Colab."""
    from src.models.device import is_colab

    if not is_colab():
        print("Not running in Colab - Drive mounting skipped")
        return False

    try:
        from google.colab import drive
        drive.mount(mount_path)
        print(f"Google Drive mounted at {mount_path}")
        return True
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
        return False


def download_sample_data(
    output_dir: str = "data/sample",
    symbols: list[str] | None = None,
) -> dict[str, str]:
    """Generate synthetic OHLCV data for testing."""
    import pandas as pd

    if symbols is None:
        symbols = ["SAMPLE"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    for symbol in symbols:
        # Generate synthetic OHLCV data
        n_bars = 50000  # About 6 months of 5-min data

        # Generate base price with trend and noise
        np.random.seed(hash(symbol) % (2**31))
        base_price = 4000  # Starting price
        returns = np.random.normal(0.0001, 0.001, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        volatility = prices * 0.001
        high = prices + np.abs(np.random.normal(0, 1, n_bars)) * volatility
        low = prices - np.abs(np.random.normal(0, 1, n_bars)) * volatility
        open_prices = prices + np.random.normal(0, 0.5, n_bars) * volatility

        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_prices, prices))
        low = np.minimum(low, np.minimum(open_prices, prices))

        # Generate volume with intraday pattern
        hours = np.arange(n_bars) % 78  # 78 bars per day (6.5 hours)
        volume_pattern = 1 + 0.5 * np.sin(np.pi * hours / 78)
        volume = (1000 + np.random.exponential(500, n_bars)) * volume_pattern

        # Create datetime index
        start_date = pd.Timestamp("2023-01-01 09:30:00")
        dates = pd.date_range(start=start_date, periods=n_bars, freq="5min")

        # Create DataFrame
        df = pd.DataFrame({
            "datetime": dates,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume.astype(int),
            "symbol": symbol,
        })

        # Save
        file_path = output_path / f"{symbol}_5m.parquet"
        df.to_parquet(file_path, index=False)
        paths[symbol] = str(file_path)

        print(f"Generated {len(df):,} bars for {symbol} -> {file_path}")

    return paths


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================


def display_metrics(
    results: dict[str, Any],
    title: str = "Model Results",
    show_confusion: bool = True,
) -> None:
    """Pretty-print model results as formatted table."""

    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

    # Basic info
    if "model_name" in results:
        print(f"  Model: {results['model_name']}")
    if "horizon" in results:
        print(f"  Horizon: {results['horizon']}")
    if "run_id" in results:
        print(f"  Run ID: {results['run_id']}")

    print("-" * 60)

    # Training metrics
    if "training_metrics" in results:
        tm = results["training_metrics"]
        print("  Training Metrics:")
        print(f"    Train Loss: {tm.get('train_loss', 'N/A'):.4f}")
        print(f"    Val Loss: {tm.get('val_loss', 'N/A'):.4f}")
        print(f"    Train F1: {tm.get('train_f1', 'N/A'):.4f}")
        print(f"    Val F1: {tm.get('val_f1', 'N/A'):.4f}")
        print(f"    Epochs: {tm.get('epochs_trained', 'N/A')}")
        print(f"    Training Time: {tm.get('training_time_seconds', 0):.1f}s")

    print("-" * 60)

    # Evaluation metrics
    if "evaluation_metrics" in results:
        em = results["evaluation_metrics"]
        print("  Evaluation Metrics:")
        print(f"    Accuracy: {em.get('accuracy', 'N/A'):.4f}")
        print(f"    Macro F1: {em.get('macro_f1', 'N/A'):.4f}")
        print(f"    Weighted F1: {em.get('weighted_f1', 'N/A'):.4f}")
        print(f"    Precision: {em.get('precision', 'N/A'):.4f}")
        print(f"    Recall: {em.get('recall', 'N/A'):.4f}")

        # Per-class F1
        if "per_class_f1" in em:
            print("  Per-Class F1:")
            for cls, f1 in em["per_class_f1"].items():
                print(f"    {cls}: {f1:.4f}")

        # Trading metrics
        if "trading" in em:
            tm = em["trading"]
            print("  Trading Stats:")
            print(f"    Long Signals: {tm.get('long_signals', 'N/A')}")
            print(f"    Short Signals: {tm.get('short_signals', 'N/A')}")
            print(f"    Position Win Rate: {tm.get('position_win_rate', 0):.2%}")

        # Confusion matrix
        if show_confusion and "confusion_matrix" in em:
            print("  Confusion Matrix:")
            cm = np.array(em["confusion_matrix"])
            labels = ["Short", "Neutral", "Long"]
            print("           " + "  ".join(f"{l:>8}" for l in labels))
            for i, row in enumerate(cm):
                print(f"    {labels[i]:>7}" + "  ".join(f"{v:>8}" for v in row))

    print("=" * 60 + "\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    save_path: str | None = None,
) -> None:
    """Plot confusion matrix with matplotlib."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    if labels is None:
        labels = ["Short", "Neutral", "Long"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


def plot_training_history(
    history: dict[str, list[float]],
    metrics: list[str] | None = None,
    title: str = "Training History",
    figsize: tuple[int, int] = (12, 4),
    save_path: str | None = None,
) -> None:
    """Plot training loss/accuracy curves."""
    import matplotlib.pyplot as plt

    if not history:
        print("No training history available")
        return

    if metrics is None:
        metrics = list(history.keys())

    # Group metrics by type
    loss_metrics = [m for m in metrics if "loss" in m.lower()]
    acc_metrics = [m for m in metrics if "acc" in m.lower() or "f1" in m.lower()]
    other_metrics = [m for m in metrics if m not in loss_metrics and m not in acc_metrics]

    n_plots = sum([len(loss_metrics) > 0, len(acc_metrics) > 0, len(other_metrics) > 0])
    if n_plots == 0:
        print("No plottable metrics found")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot losses
    if loss_metrics:
        ax = axes[plot_idx]
        for metric in loss_metrics:
            if metric in history:
                ax.plot(history[metric], label=metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot accuracy/F1
    if acc_metrics:
        ax = axes[plot_idx]
        for metric in acc_metrics:
            if metric in history:
                ax.plot(history[metric], label=metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Accuracy/F1 Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot other metrics
    if other_metrics:
        ax = axes[plot_idx]
        for metric in other_metrics:
            if metric in history:
                ax.plot(history[metric], label=metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Other Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training history to {save_path}")

    plt.show()


def plot_model_comparison(
    results: dict[str, dict[str, Any]],
    metric: str = "macro_f1",
    title: str = "Model Comparison",
    figsize: tuple[int, int] = (10, 6),
    save_path: str | None = None,
) -> None:
    """Compare multiple models on a given metric."""
    import matplotlib.pyplot as plt

    models = []
    values = []

    for model_name, result in results.items():
        if "evaluation_metrics" in result:
            value = result["evaluation_metrics"].get(metric)
            if value is not None:
                models.append(model_name)
                values.append(value)

    if not models:
        print(f"No models found with metric '{metric}'")
        return

    # Sort by value
    sorted_pairs = sorted(zip(models, values, strict=False), key=lambda x: x[1], reverse=True)
    models, values = zip(*sorted_pairs, strict=False)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax.barh(models, values, color=colors)

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            value + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}", va="center", fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved model comparison to {save_path}")

    plt.show()


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================


def get_sample_config(
    model_name: str,
    horizon: int = 20,
    quick_mode: bool = False,
) -> dict[str, Any]:
    """Get sample training configuration for a model."""
    base_config = {
        "model_type": model_name,
        "horizon": horizon,
    }

    # Quick mode settings for fast iteration
    if quick_mode:
        model_overrides = {
            "xgboost": {"n_estimators": 50, "max_depth": 4},
            "lightgbm": {"n_estimators": 50, "max_depth": 4},
            "catboost": {"iterations": 50, "depth": 4},
            "lstm": {"hidden_size": 64, "num_layers": 1, "epochs": 5},
            "gru": {"hidden_size": 64, "num_layers": 1, "epochs": 5},
            "tcn": {"num_channels": [32, 32], "epochs": 5},
            "random_forest": {"n_estimators": 50, "max_depth": 8},
            "logistic": {"max_iter": 100},
            "svm": {"max_iter": 100},
        }

        if model_name in model_overrides:
            base_config["model_config"] = model_overrides[model_name]

    return base_config


def create_progress_callback(total_epochs: int, description: str = "Training") -> Callable:
    """Create a tqdm progress callback for training."""
    from tqdm.auto import tqdm

    pbar = tqdm(total=total_epochs, desc=description, unit="epoch")

    def callback(epoch: int, metrics: dict[str, float]) -> None:
        pbar.update(1)
        postfix = {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))}
        pbar.set_postfix(postfix)
        if epoch >= total_epochs - 1:
            pbar.close()

    return callback


__all__ = [
    "setup_notebook",
    "install_dependencies",
    "mount_drive",
    "download_sample_data",
    "display_metrics",
    "plot_confusion_matrix",
    "plot_training_history",
    "plot_model_comparison",
    "get_sample_config",
    "create_progress_callback",
]
