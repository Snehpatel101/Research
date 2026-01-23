"""
Learning rate finder for neural networks.

Implements the learning rate range test (Smith 2017) to find optimal
learning rates for neural network training. The algorithm trains the
model with exponentially increasing learning rate while tracking loss.
The optimal LR is where loss decreases fastest (steepest gradient).

Usage:
    >>> from src.models.neural.lr_finder import LRFinder
    >>> model = create_model(...)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    >>> criterion = nn.CrossEntropyLoss()
    >>> finder = LRFinder(model, optimizer, criterion, device="cuda")
    >>> results = finder.find(train_loader)
    >>> suggested_lr = results["suggested_lr"]
    >>> finder.plot()

References:
    - Smith, L.N. (2017). Cyclical Learning Rates for Training Neural Networks
    - fastai's lr_find implementation
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as _DataLoader

# Type alias for DataLoader with Any type parameter to satisfy mypy strict mode
DataLoader = _DataLoader[Any]

logger = logging.getLogger(__name__)


@dataclass
class LRFinderResult:
    """Results from learning rate range test."""

    lrs: list[float]
    losses: list[float]
    suggested_lr: float
    method: str
    min_lr_tested: float
    max_lr_tested: float
    num_iterations: int
    best_loss: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lrs": self.lrs,
            "losses": self.losses,
            "suggested_lr": self.suggested_lr,
            "method": self.method,
            "min_lr_tested": self.min_lr_tested,
            "max_lr_tested": self.max_lr_tested,
            "num_iterations": self.num_iterations,
            "best_loss": self.best_loss,
            "metadata": self.metadata,
        }


class LRFinder:
    """
    Learning rate range test (Smith 2017).

    Trains model with exponentially increasing LR, tracks loss.
    Optimal LR is where loss decreases fastest (steepest gradient).

    Example:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
        >>> criterion = nn.CrossEntropyLoss()
        >>> finder = LRFinder(model, optimizer, criterion, device="cuda")
        >>> results = finder.find(train_loader, start_lr=1e-7, end_lr=10)
        >>> print(f"Suggested LR: {results.suggested_lr:.2e}")
        >>> finder.plot()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str | torch.device = "cuda",
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Initialize LRFinder.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer (initial LR will be overwritten)
            criterion: Loss function
            device: Device to train on
            use_amp: Whether to use automatic mixed precision
            amp_dtype: AMP dtype (float16 or bfloat16)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_amp = use_amp and self.device.type == "cuda"
        self.amp_dtype = amp_dtype

        # Store original state for restoration
        self._model_state: dict[str, Any] | None = None
        self._optimizer_state: dict[str, Any] | None = None

        # Results storage
        self._result: LRFinderResult | None = None

    def _save_state(self) -> None:
        """Save model and optimizer state for restoration."""
        self._model_state = copy.deepcopy(self.model.state_dict())
        self._optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def _restore_state(self) -> None:
        """Restore model and optimizer to original state."""
        if self._model_state is not None:
            self.model.load_state_dict(self._model_state)
        if self._optimizer_state is not None:
            self.optimizer.load_state_dict(self._optimizer_state)

    def find(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_threshold: float = 5.0,
    ) -> LRFinderResult:
        """
        Run LR range test.

        Trains model with exponentially increasing learning rate from
        start_lr to end_lr over num_iter iterations. Tracks smoothed
        loss at each LR to identify optimal training rate.

        Args:
            train_loader: Training DataLoader
            start_lr: Starting learning rate (very small)
            end_lr: Ending learning rate (typically 1-10)
            num_iter: Number of iterations to run
            smooth_f: Smoothing factor for loss (0 = no smoothing)
            diverge_threshold: Stop if loss exceeds best_loss * threshold

        Returns:
            LRFinderResult with learning rates, losses, and suggested LR
        """
        # Save state for restoration
        self._save_state()

        # Move model to device
        self.model.to(self.device)
        self.model.train()

        # Calculate LR multiplier for exponential growth
        # lr(i) = start_lr * mult^i where mult = (end_lr/start_lr)^(1/num_iter)
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)

        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr

        # Initialize gradient scaler for mixed precision (only for float16)
        scaler = None
        if self.use_amp and self.amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")

        # Tracking variables
        lrs: list[float] = []
        losses: list[float] = []
        best_loss = float("inf")
        smoothed_loss = 0.0
        current_lr = start_lr

        logger.info(
            f"Starting LR finder: start_lr={start_lr:.2e}, end_lr={end_lr:.2e}, "
            f"num_iter={num_iter}"
        )

        # Create iterator from dataloader
        data_iter = iter(train_loader)

        for iteration in range(num_iter):
            # Get next batch (cycle if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Unpack batch (handles both weighted and unweighted)
            if len(batch) == 3:
                X_batch, y_batch, _ = batch  # Ignore weights for LR finding
            else:
                X_batch, y_batch = batch

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass with optional AMP
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
            else:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

            # Check for divergence
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Loss became NaN/Inf at LR={current_lr:.2e}, stopping")
                break

            loss_val = loss.item()

            # Smooth the loss
            if iteration == 0:
                smoothed_loss = loss_val
            else:
                smoothed_loss = smooth_f * loss_val + (1 - smooth_f) * smoothed_loss

            # Track best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > best_loss * diverge_threshold:
                logger.info(f"Loss diverged at LR={current_lr:.2e}, stopping")
                break

            # Record
            lrs.append(current_lr)
            losses.append(smoothed_loss)

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update learning rate
            current_lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            # Log progress
            if (iteration + 1) % 20 == 0:
                logger.debug(
                    f"  iter {iteration + 1}/{num_iter}: lr={current_lr:.2e}, loss={smoothed_loss:.4f}"
                )

        # Restore original state
        self._restore_state()

        if len(lrs) < 10:
            logger.warning("LR finder completed with too few iterations for reliable suggestion")
            suggested_lr = start_lr
            method = "fallback"
        else:
            # Suggest optimal LR
            suggested_lr, method = self._suggest_lr(lrs, losses)

        # Store results
        self._result = LRFinderResult(
            lrs=lrs,
            losses=losses,
            suggested_lr=suggested_lr,
            method=method,
            min_lr_tested=start_lr,
            max_lr_tested=current_lr,
            num_iterations=len(lrs),
            best_loss=best_loss,
        )

        logger.info(
            f"LR finder complete: suggested_lr={suggested_lr:.2e} (method={method}), "
            f"best_loss={best_loss:.4f}"
        )

        return self._result

    def _suggest_lr(
        self,
        lrs: list[float],
        losses: list[float],
        method: str = "steepest",
    ) -> tuple[float, str]:
        """
        Suggest optimal learning rate.

        Methods:
        - "steepest": LR where loss gradient is steepest (most negative)
        - "minimum": LR at minimum loss / 10
        - "valley": Find the LR just before loss starts increasing sharply

        Args:
            lrs: List of learning rates tested
            losses: List of smoothed losses at each LR
            method: Selection method

        Returns:
            Tuple of (suggested_lr, method_used)
        """
        lrs_arr = np.array(lrs)
        losses_arr = np.array(losses)

        if method == "steepest":
            # Find where the gradient is most negative
            # Use log scale for LR for better gradient estimation
            log_lrs = np.log10(lrs_arr)

            # Compute gradient (skip first and last few points for stability)
            skip = max(3, len(losses) // 20)
            if len(losses) <= 2 * skip:
                return self._suggest_lr(lrs, losses, method="minimum")

            gradients = np.gradient(losses_arr[skip:-skip], log_lrs[skip:-skip])

            # Find most negative gradient
            min_grad_idx = np.argmin(gradients) + skip
            suggested_lr = float(lrs_arr[min_grad_idx])

            # Safety: use LR before the steepest point (typically more stable)
            safe_idx = int(max(0, int(min_grad_idx) - 2))
            suggested_lr = float(lrs_arr[safe_idx])

            return suggested_lr, "steepest"

        elif method == "minimum":
            # Find minimum loss, then use LR at 1/10 of that point
            min_loss_idx = np.argmin(losses_arr)
            suggested_lr = float(lrs_arr[min_loss_idx]) / 10.0

            # Ensure suggested LR is within tested range
            suggested_lr = max(suggested_lr, float(lrs_arr[0]))

            return suggested_lr, "minimum"

        elif method == "valley":
            # Find the valley before loss starts increasing
            # Look for local minimum in smoothed loss
            from scipy.ndimage import gaussian_filter1d

            # Extra smoothing
            smooth_losses = gaussian_filter1d(losses_arr, sigma=2)

            # Find valley (local minimum before divergence)
            for i in range(len(smooth_losses) - 1, 0, -1):
                if smooth_losses[i - 1] < smooth_losses[i]:
                    # Found valley
                    suggested_lr = float(lrs_arr[i - 1])
                    return suggested_lr, "valley"

            # Fallback to minimum
            return self._suggest_lr(lrs, losses, method="minimum")

        else:
            raise ValueError(f"Unknown method: {method}")

    def plot(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        show_suggestion: bool = True,
        log_lr: bool = True,
        save_path: str | None = None,
    ) -> None:
        """
        Plot loss vs learning rate.

        Args:
            skip_start: Number of initial points to skip (often noisy)
            skip_end: Number of final points to skip (often diverged)
            show_suggestion: Whether to show suggested LR line
            log_lr: Whether to use log scale for LR axis
            save_path: Path to save figure (if provided)
        """
        if self._result is None:
            raise RuntimeError("Must run find() before plot()")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return

        lrs = self._result.lrs
        losses = self._result.losses

        # Skip noisy start/end
        if len(lrs) <= skip_start + skip_end:
            skip_start = 0
            skip_end = 0

        plot_lrs = lrs[skip_start : len(lrs) - skip_end if skip_end > 0 else None]
        plot_losses = losses[skip_start : len(losses) - skip_end if skip_end > 0 else None]

        fig, ax = plt.subplots(figsize=(10, 6))

        if log_lr:
            ax.semilogx(plot_lrs, plot_losses, "b-", linewidth=2)
        else:
            ax.plot(plot_lrs, plot_losses, "b-", linewidth=2)

        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Learning Rate Finder", fontsize=14)
        ax.grid(True, alpha=0.3)

        if show_suggestion:
            ax.axvline(
                x=self._result.suggested_lr,
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"Suggested LR: {self._result.suggested_lr:.2e}",
            )
            ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved LR finder plot to {save_path}")
        else:
            plt.show()

        plt.close()

    @property
    def result(self) -> LRFinderResult | None:
        """Get the latest result."""
        return self._result


def find_lr_for_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module | None = None,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 100,
    device: str = "cuda",
    use_amp: bool = True,
) -> float:
    """
    Convenience function to find optimal learning rate.

    Creates optimizer and LR finder, runs the test, returns suggested LR.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        criterion: Loss function (defaults to CrossEntropyLoss)
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iter: Number of iterations
        device: Device to train on
        use_amp: Whether to use automatic mixed precision

    Returns:
        Suggested learning rate
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Create fresh optimizer with start_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    finder = LRFinder(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    result = finder.find(
        train_loader=train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
    )

    return result.suggested_lr


__all__ = [
    "LRFinder",
    "LRFinderResult",
    "find_lr_for_model",
]
