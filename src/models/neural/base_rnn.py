"""
Base RNN class for LSTM and GRU models.

Provides shared training infrastructure:
- PyTorch training loop with early stopping
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)
- Gradient clipping and learning rate scheduling
- Sequence handling utilities

Supports any NVIDIA GPU (GTX 10xx, RTX 20xx/30xx/40xx, Tesla T4/V100/A100).
"""

from __future__ import annotations

import logging
import math
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import TensorDataset

from src.core.reproducibility import set_all_seeds

from ..base import BaseModel, PredictionResult, TrainingMetrics
from ..common import map_classes_to_labels, map_labels_to_classes
from ..device import get_amp_dtype, get_best_gpu, get_mixed_precision_config
from .checkpointing import CheckpointConfig, CheckpointManager
from .numerical_stability import NumericalValidator, validate_training_inputs
from .oom_recovery import OOMConfig, OOMRecoveryManager

# Type alias for DataLoader with Any type parameter
DataLoader = _DataLoader[Any]

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingState:
    """Tracks early stopping state during training."""

    best_loss: float = float("inf")
    best_epoch: int = 0
    patience_counter: int = 0
    best_state_dict: dict[str, Any] | None = None

    def check(
        self, val_loss: float, epoch: int, model: nn.Module, patience: int, min_delta: float
    ) -> bool:
        """Check if training should stop. Returns True if should stop."""
        if val_loss < self.best_loss - min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience


# Checkpoints written before the architecture version was recorded (and every
# checkpoint written by the removed iTransformer save() override) carry no
# "arch_version" key; they were produced by version-1.0 networks.
_LEGACY_ARCH_VERSION = "1.0"


class WarmupCosineSchedule:
    """Linear-warmup + cosine-decay LR multiplier driven by fractional epochs.

    The multiplier is a function of *epoch position* (completed epochs plus the
    fraction of the current epoch), not of the raw optimizer-step count. That
    keeps the schedule correct when OOM recovery shrinks the batch size mid
    training (steps per epoch changes) or an epoch is retried: the fit loop
    reports epoch boundaries via :meth:`end_epoch` / :meth:`restart_epoch`.

    Instances are callable objects (not lambdas), so ``LambdaLR.state_dict()``
    persists the clock alongside the step counter.
    """

    def __init__(self, max_epochs: int, warmup_epochs: int, steps_per_epoch: int) -> None:
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = max(int(steps_per_epoch), 1)
        self.completed_epochs = 0
        self.epoch_start_step = 0

    def epoch_position(self, step: int) -> float:
        return self.completed_epochs + (step - self.epoch_start_step) / self.steps_per_epoch

    def __call__(self, step: int) -> float:
        position = self.epoch_position(step)
        if position < self.warmup_epochs:
            return position / self.warmup_epochs
        decay_epochs = max(self.max_epochs - self.warmup_epochs, 1)
        progress = min((position - self.warmup_epochs) / decay_epochs, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def end_epoch(self, completed_epochs: int, step: int) -> None:
        """Record a successfully completed epoch (``step`` = scheduler step count)."""
        self.completed_epochs = completed_epochs
        self.epoch_start_step = step

    def restart_epoch(self, steps_per_epoch: int, step: int) -> None:
        """Retry the current epoch with a new loader length (after OOM recovery)."""
        self.steps_per_epoch = max(int(steps_per_epoch), 1)
        self.epoch_start_step = step


def _check_cuda_available() -> bool:
    """Check if CUDA is available for PyTorch."""
    return torch.cuda.is_available()


def _get_device(use_cuda: bool) -> torch.device:
    """Get the appropriate device."""
    if use_cuda and _check_cuda_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RNNNetwork(nn.Module):
    """
    Base RNN neural network architecture.

    Architecture:
        Input (batch, seq_len, features)
        -> RNN layers (LSTM/GRU)
        -> Take last hidden state
        -> LayerNorm + Dropout
        -> Linear -> hidden_size
        -> ReLU + Dropout
        -> Linear -> 3 classes
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # RNN layer placeholder - subclasses set this
        self.rnn: nn.RNNBase | None = None

        # Output dimension from RNN
        rnn_output_size = hidden_size * self.num_directions

        # Post-RNN layers
        self.layer_norm = nn.LayerNorm(rnn_output_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_output_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    @abstractmethod
    def _init_rnn(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> nn.Module:
        """Initialize the RNN layer (LSTM or GRU). Implemented by subclasses."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Output logits, shape (batch, n_classes)
        """
        # RNN forward pass
        # output: (batch, seq_len, hidden_size * num_directions)
        # hidden: tuple for LSTM, tensor for GRU
        if self.rnn is None:
            raise RuntimeError("RNN layer not initialized")
        output, hidden = self.rnn(x)

        # Take last timestep output
        # For bidirectional, concatenate both directions' last hidden states
        if self.bidirectional:
            # Forward direction: last timestep
            forward_out = output[:, -1, : self.hidden_size]
            # Backward direction: first timestep (contains full backward pass)
            backward_out = output[:, 0, self.hidden_size :]
            last_output = torch.cat([forward_out, backward_out], dim=1)
        else:
            last_output = output[:, -1, :]

        # Classification head
        x = self.layer_norm(last_output)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class BaseRNNModel(BaseModel):
    """
    Base class for RNN-based models (LSTM, GRU).

    Provides shared training infrastructure with:
    - GPU training with CUDA (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection:
      - bfloat16 for Ampere+ (RTX 30xx/40xx, A100, H100)
      - float16 for Volta/Turing (RTX 20xx, GTX 16xx, T4, V100)
      - float32 for older GPUs or CPU
    - AdamW optimizer with cosine annealing
    - Gradient clipping
    - Early stopping on validation loss

    Note on Bidirectional Mode:
        When bidirectional=True, the backward RNN pass sees 'future' positions
        within each sequence window. While not technically lookahead bias (data
        is within the observed window), this can capture non-causal patterns
        that may not generalize to real-time inference.
    """

    # Version of the network definition. A subclass bumps its own value whenever
    # its state_dict layout OR its forward semantics change; load() refuses a
    # checkpoint whose recorded version differs instead of silently loading
    # weights into a network that computes something else.
    ARCH_VERSION = "1.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: nn.Module = None  # type: ignore[assignment]
        self._n_features: int | None = None
        # Training window length; persisted so seq_len-dependent networks
        # (iTransformer, N-BEATS, channel-independent PatchTST) rebuild on load.
        self._seq_len: int | None = None
        self._bidirectional_warning_logged: bool = False

        # Device setup with "auto" detection support
        device_config = self._config.get("device", "auto")
        if device_config == "auto" or device_config == "cuda":
            self._device = _get_device(use_cuda=True)
        else:
            self._device = torch.device(device_config)

        # Mixed precision setup based on GPU capabilities
        self._gpu_info = get_best_gpu() if self._device.type == "cuda" else None
        self._mp_config = get_mixed_precision_config(self._gpu_info)
        self._use_amp = self._config.get("mixed_precision", self._mp_config["enabled"])
        self._amp_dtype = get_amp_dtype(self._gpu_info)

        # Gradient scaler needed for float16 but not bfloat16
        self._use_grad_scaler = self._mp_config.get("grad_scaler", False) and self._use_amp

        if self._device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            dtype_str = str(self._amp_dtype).replace("torch.", "")
            logger.info(f"Using GPU: {gpu_name}, AMP dtype: {dtype_str}")
        else:
            logger.info("Using CPU for training")

    @property
    def model_family(self) -> str:
        return "neural"

    def _unwrapped_model(self) -> nn.Module:
        """The eager network, looking through a ``torch.compile`` wrapper.

        ``torch.compile`` returns an ``OptimizedModule`` that fails
        ``isinstance(m, <Network class>)`` checks; interpretability accessors
        must go through this helper to reach the real network.
        """
        return getattr(self._model, "_orig_mod", self._model)

    @property
    def requires_scaling(self) -> bool:
        return True

    @property
    def requires_sequences(self) -> bool:
        return True

    @property
    def is_production_safe(self) -> bool:
        """
        Check if this model configuration is safe for production trading.

        A model is considered production-safe when it uses only causal patterns
        that will be available during real-time inference (i.e., only past data).

        Returns:
            True if bidirectional=False (causal model), False otherwise.
        """
        return not self._config.get("bidirectional", False)

    def _log_bidirectional_warning(self) -> None:
        """Log a warning about bidirectional mode implications (only once)."""
        if self._bidirectional_warning_logged:
            return

        if self._config.get("bidirectional", False):
            logger.warning(
                "BIDIRECTIONAL RNN ENABLED: The backward pass sees 'future' positions "
                "within each sequence window (not calendar future, but later indices in "
                "the current input sequence). While not technically lookahead bias "
                "(data is within the observed window), this can capture non-causal patterns "
                "that may not generalize to real-time inference.\n"
                "Recommendations:\n"
                "  - For production trading models: Set bidirectional=False\n"
                "  - For research/analysis: Acceptable if you understand the implications\n"
                "  - For real-time inference: Predictions use incomplete backward context"
            )
            self._bidirectional_warning_logged = True

    @abstractmethod
    def _create_network(self, input_size: int) -> nn.Module:
        """Create the neural network. Implemented by subclasses."""
        pass

    @abstractmethod
    def _get_model_type(self) -> str:
        """Return model type string (lstm/gru). Implemented by subclasses."""
        pass

    def _on_training_start(self, train_config: dict[str, Any], seq_len: int) -> dict[str, Any]:
        """
        Hook called at the start of training, after model is created.

        Subclasses can override this to add model-specific logging or setup.
        For example, TCN uses this to log receptive field information.

        Args:
            train_config: Training configuration dictionary
            seq_len: Sequence length of training data

        Returns:
            Dict of additional metadata to include in TrainingMetrics
        """
        return {}

    def get_default_config(self) -> dict[str, Any]:
        """Return default hyperparameters."""
        return {
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": False,
            "sequence_length": 60,
            "batch_size": 512,
            "max_epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "early_stopping_patience": 7,
            "min_delta": 0.0001,
            "warmup_epochs": 5,
            "label_smoothing": 0.0,  # CrossEntropyLoss label smoothing (0 = off)
            "device": "auto",  # Auto-detect GPU/CPU
            "mixed_precision": True,  # Use GPU-appropriate precision
            # DataLoader workers/pinning auto-tuned in _create_dataloader for CUDA
            # None = auto-detect (4 for CUDA, 0 for CPU)
            "num_workers": None,
            "pin_memory": None,  # None = auto (pin only on CUDA)
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> TrainingMetrics:
        """Train the RNN model with early stopping."""
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")

        # Validate inputs for NaN/Inf before training
        validate_training_inputs(X_train, y_train, X_val, y_val, sample_weights)

        start_time = time.time()

        # Merge config
        train_config = self._config.copy()
        if config:
            train_config.update(config)

        # Set seeds for reproducibility
        random_seed = train_config.get("random_seed", 42)
        deterministic = train_config.get("deterministic_mode", False)
        set_all_seeds(random_seed, deterministic=deterministic)

        # Extract dimensions (handle both 3D and 4D inputs)
        if X_train.ndim == 4:
            n_samples, n_timeframes, seq_len, n_features_per_tf = X_train.shape
            # Forward methods flatten 4D to 3D: (batch, seq, n_tf * feat)
            n_features = n_timeframes * n_features_per_tf
        else:
            n_samples, seq_len, n_features = X_train.shape
        self._n_features = n_features
        self._seq_len = seq_len

        # Use dynamically detected AMP dtype
        amp_dtype = self._amp_dtype

        # Create network (compiled on CUDA, with eager fallback)
        self._build_network(n_features, train_config, X_train, amp_dtype)

        # Log bidirectional warning if applicable (only logged once per model)
        self._log_bidirectional_warning()

        # Call training start hook for subclass-specific setup/logging
        extra_metadata = self._on_training_start(train_config, seq_len)

        # Prepare data
        train_loader, val_loader = self._create_fit_loaders(
            X_train, y_train, sample_weights, X_val, y_val, train_config
        )

        # Setup training components
        optimizer = self._create_optimizer(train_config)
        scheduler, schedule = self._create_scheduler(optimizer, train_config, len(train_loader))
        criterion = self._create_criterion(y_train, train_config)

        # Mixed precision scaler (only needed for float16, not bfloat16)
        scaler = (
            torch.amp.GradScaler("cuda")
            if self._use_grad_scaler and self._device.type == "cuda"
            else None
        )

        # Training state
        early_stopping = EarlyStoppingState()
        history = self._new_history()

        # Numerical stability validator for training loop
        self._numerical_validator = NumericalValidator(
            raise_on_error=train_config.get("nan_check_raise_error", True),
            log_warnings=True,
        )

        # Checkpoint manager for periodic saving and best model tracking
        checkpoint_interval = train_config.get("checkpoint_interval", 50)
        checkpoint_dir = train_config.get("checkpoint_dir", None)
        self._checkpoint_manager: CheckpointManager | None = None
        if checkpoint_dir:
            ckpt_config = CheckpointConfig(
                checkpoint_dir=Path(checkpoint_dir),
                interval_epochs=checkpoint_interval,
                keep_n_best=train_config.get("keep_n_checkpoints", 3),
                save_optimizer=True,
                save_scheduler=True,
                metric_name="val_loss",
                metric_mode="min",
            )
            self._checkpoint_manager = CheckpointManager(ckpt_config)

        # OOM recovery manager for graceful batch size reduction on CUDA OOM
        oom_config = OOMConfig(
            enabled=train_config.get("oom_recovery_enabled", True),
            max_retries=train_config.get("oom_max_retries", 6),
            batch_reduction_factor=train_config.get("oom_batch_reduction_factor", 0.5),
            min_batch_size=train_config.get("oom_min_batch_size", 2),
        )
        self._oom_manager = OOMRecoveryManager(oom_config)
        current_batch_size = train_config.get("batch_size", 512)
        initial_batch_size = current_batch_size  # Save for dtype-fallback reset

        max_epochs = train_config.get("max_epochs", 100)
        patience = train_config.get("early_stopping_patience", 7)
        min_delta = train_config.get("min_delta", 0.0001)
        grad_clip = train_config.get("gradient_clip", 1.0)

        logger.info(
            f"Training {self._get_model_type().upper()}: "
            f"epochs={max_epochs}, batch_size={current_batch_size}, "
            f"hidden={train_config.get('hidden_size')}, layers={train_config.get('num_layers')}, "
            f"mixed_precision={'on' if self._use_amp else 'off'}"
        )

        # Training loop with OOM recovery
        epoch = 0
        while epoch < max_epochs:
            try:
                # Training phase (now returns gradient norm as third value)
                train_loss, train_acc, avg_grad_norm = self._train_epoch(
                    train_loader, optimizer, criterion, scheduler, scaler, amp_dtype, grad_clip
                )
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["gradient_norms"].append(avg_grad_norm)

                # Validation phase
                val_loss, val_acc = self._validate_epoch(val_loader, criterion, amp_dtype)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # Mark OOM recovery as successful if we had any OOM events
                if self._oom_manager.total_oom_count > 0:
                    self._oom_manager.mark_success()

                # Logging (include gradient norm for debugging)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{max_epochs} - "
                        f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                        f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, "
                        f"grad_norm: {avg_grad_norm:.4f}"
                    )

                # Advance the LR clock to the epoch boundary
                schedule.end_epoch(completed_epochs=epoch + 1, step=scheduler.last_epoch)

                # Early stopping check
                if early_stopping.check(val_loss, epoch, self._model, patience, min_delta):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                # Periodic checkpoint save
                if self._checkpoint_manager is not None:
                    epoch_metrics = {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "grad_norm": avg_grad_norm,
                    }
                    self._checkpoint_manager.maybe_save_checkpoint(
                        self._model,
                        optimizer,
                        scheduler,
                        epoch,
                        epoch_metrics,
                        model_config=train_config,
                    )

                # Increment epoch only on successful completion
                epoch += 1

            except RuntimeError as e:
                # Handle CUDA OOM errors with automatic batch size reduction
                if not self._oom_manager.is_oom_error(e):
                    raise

                error_msg = str(e).lower()
                is_cublas_error = "cublas" in error_msg or "cudnn" in error_msg

                # cuBLAS/cuDNN internal errors are often bf16 dtype issues,
                # not memory issues. Fall back to fp32 FIRST, restoring the
                # original batch size (the error was dtype, not memory), and
                # restart training from scratch: fresh network, optimizer,
                # schedule, history and early-stopping state, so no state of
                # the aborted mixed-precision run leaks into the fp32 run.
                if is_cublas_error and self._use_amp and amp_dtype != torch.float32:
                    logger.warning(
                        f"cuBLAS/cuDNN error detected with {amp_dtype} — "
                        "disabling mixed precision and restarting training in fp32"
                    )
                    from src.models.device import release_gpu_memory

                    self._model = None  # type: ignore[assignment]
                    release_gpu_memory()
                    self._use_amp = False
                    amp_dtype = torch.float32
                    scaler = None
                    # Reset OOM state — this is a dtype fix, not memory
                    self._oom_manager.reset()
                    current_batch_size = initial_batch_size
                    train_config["batch_size"] = current_batch_size

                    set_all_seeds(random_seed, deterministic=deterministic)
                    self._build_network(n_features, train_config, X_train, amp_dtype)
                    train_loader, val_loader = self._create_fit_loaders(
                        X_train, y_train, sample_weights, X_val, y_val, train_config
                    )
                    optimizer = self._create_optimizer(train_config)
                    scheduler, schedule = self._create_scheduler(
                        optimizer, train_config, len(train_loader)
                    )
                    early_stopping = EarlyStoppingState()
                    history = self._new_history()
                    epoch = 0
                    continue

                new_batch_size = self._oom_manager.handle_oom(current_batch_size)
                if new_batch_size is None:
                    # Recovery failed - re-raise the error
                    raise RuntimeError(
                        f"OOM recovery failed after {self._oom_manager.config.max_retries} retries. "
                        f"Final batch size: {current_batch_size}. Consider reducing model size or sequence length."
                    ) from e

                # Rebuild ONLY the data loaders with the reduced batch size.
                # Optimizer, scheduler (LR position) and early-stopping state are
                # kept; the schedule clock rewinds to the start of the retried
                # epoch and adopts the new steps-per-epoch.
                current_batch_size = new_batch_size
                train_config["batch_size"] = current_batch_size
                logger.info(f"Recreating data loaders with batch_size={current_batch_size}")
                train_loader, val_loader = self._create_fit_loaders(
                    X_train, y_train, sample_weights, X_val, y_val, train_config
                )
                schedule.restart_epoch(steps_per_epoch=len(train_loader), step=scheduler.last_epoch)
                for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs, strict=True):
                    group["lr"] = base_lr * schedule(scheduler.last_epoch)

                # Don't increment epoch - retry the same epoch with smaller batch
                continue

        # Restore best model
        if early_stopping.best_state_dict is not None:
            self._model.load_state_dict(early_stopping.best_state_dict)

        # Inference always runs the eager network: fit() -> predict() then
        # computes exactly what a load()ed checkpoint computes, and a lazy
        # recompilation for eval-mode shapes can never fail at predict time.
        self._model = self._unwrapped_model()

        training_time = time.time() - start_time
        epochs_trained = len(history["train_loss"])

        # Compute final metrics. The training loader shuffles, so train metrics
        # are computed on a fresh, order-preserving loader aligned with y_train.
        eval_config = {**train_config, "batch_size": train_config.get("batch_size", 512) * 2}
        train_eval_loader = self._create_dataloader(
            X_train, y_train, None, eval_config, shuffle=False
        )
        train_metrics = self._compute_final_metrics(train_eval_loader, amp_dtype, y_train)
        val_metrics = self._compute_final_metrics(val_loader, amp_dtype, y_val)

        self._is_fitted = True

        logger.info(
            f"Training complete: epochs={epochs_trained}, "
            f"best_epoch={early_stopping.best_epoch + 1}, "
            f"val_f1={val_metrics['f1']:.4f}, time={training_time:.1f}s"
        )

        # Build metadata with base info + any extra from subclass hook
        metadata = {
            "model_type": self._get_model_type(),
            "n_features": n_features,
            "n_train_samples": n_samples,
            "n_val_samples": len(X_val),
            "device": str(self._device),
            "mixed_precision": self._use_amp,
        }
        metadata.update(extra_metadata)

        return TrainingMetrics(
            train_loss=history["train_loss"][-1],
            val_loss=early_stopping.best_loss,
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"],
            train_f1=train_metrics["f1"],
            val_f1=val_metrics["f1"],
            epochs_trained=epochs_trained,
            training_time_seconds=training_time,
            early_stopped=epochs_trained < max_epochs,
            best_epoch=early_stopping.best_epoch,
            history=history,
            metadata=metadata,
        )

    @staticmethod
    def _new_history() -> dict[str, list[float]]:
        return {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "gradient_norms": [],  # Track gradient norms for debugging
        }

    def _create_fit_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weights: np.ndarray | None,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_config: dict[str, Any],
    ) -> tuple[DataLoader, DataLoader]:
        """Shuffled training loader + ordered validation loader."""
        train_loader = self._create_dataloader(
            X_train, y_train, sample_weights, train_config, shuffle=True
        )
        # Use 2x batch size for validation (no gradients stored, so memory allows it)
        val_config = {**train_config, "batch_size": train_config.get("batch_size", 512) * 2}
        val_loader = self._create_dataloader(X_val, y_val, None, val_config, shuffle=False)
        return train_loader, val_loader

    def _create_criterion(
        self, y_train: np.ndarray, train_config: dict[str, Any]
    ) -> nn.CrossEntropyLoss:
        """Cross-entropy with optional inverse-frequency class weights and label smoothing."""
        label_smoothing = float(train_config.get("label_smoothing", 0.0))
        # Class weights for imbalanced datasets (common in trading: neutral >> long/short)
        if not train_config.get("use_class_weights", True):
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Same label -> class-index mapping the loss sees ({-1,0,1} or binary {0,1})
        y_class = self._convert_labels_to_class(y_train).astype(np.int64)
        class_counts = np.bincount(y_class, minlength=self._n_classes)
        # Handle edge case of zero counts (shouldn't happen in practice)
        class_counts = np.maximum(class_counts, 1)
        # Inverse frequency weighting: rarer classes get higher weights
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        logger.debug(f"Class weights: {class_weights}")
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self._device)
        return nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)

    def _build_network(
        self,
        n_features: int,
        train_config: dict[str, Any],
        X_train: np.ndarray,
        amp_dtype: torch.dtype,
    ) -> None:
        """Create the network on the target device; compile it on CUDA only.

        CPU compilation needs a system C++ compiler and brings little, so the
        CPU path is always eager. ``torch_compile: False`` disables it on CUDA.
        """
        self._model = self._create_network(n_features).to(self._device)
        if (
            self._device.type == "cuda"
            and hasattr(torch, "compile")
            and train_config.get("torch_compile", True)
        ):
            batch_size = int(train_config.get("batch_size", 512))
            self._model = self._compile_with_fallback(self._model, X_train[:batch_size], amp_dtype)

    def _compile_with_fallback(
        self, eager: nn.Module, X_sample: np.ndarray, amp_dtype: torch.dtype
    ) -> nn.Module:
        """Compile ``eager`` and prove the compiled graphs run, else stay eager.

        ``torch.compile`` is lazy: graph capture and Inductor code generation
        happen on the first call, so wrapping only the ``compile()`` call in a
        try/except never catches real failures. This runs a guarded warm-up —
        one train-mode forward+backward and one eval-mode forward on a real
        batch under the training autocast — and returns the eager module if
        any of it raises. Parameters, buffers (BatchNorm running stats) and
        gradients are restored afterwards, so the warm-up leaves no trace.
        """
        try:
            compiled = torch.compile(eager, mode="max-autotune")
        except Exception as exc:  # noqa: BLE001 - any compile failure -> eager
            logger.warning(f"torch.compile unavailable ({exc!r}); training eagerly")
            return eager

        snapshot = {k: v.detach().clone() for k, v in eager.state_dict().items()}
        sample = torch.from_numpy(np.ascontiguousarray(X_sample)).float().to(self._device)
        was_training = eager.training
        try:
            compiled.train()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
                logits = compiled(sample)
            logits.float().sum().backward()
            compiled.eval()
            with (
                torch.no_grad(),
                torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp),
            ):
                compiled(sample)
        except Exception as exc:  # noqa: BLE001 - any first-call failure -> eager
            logger.warning(
                f"torch.compile failed on its first forward/backward ({exc!r}); "
                "falling back to eager execution"
            )
            return eager
        finally:
            eager.load_state_dict(snapshot)
            for param in eager.parameters():
                param.grad = None
            eager.train(was_training)

        logger.info("torch.compile(mode='max-autotune') applied")
        return compiled

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions with class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        if self._model is None:
            raise RuntimeError("Model is not fitted")

        self._model.eval()
        amp_dtype = self._amp_dtype

        # Zero-copy tensor on CPU; each batch moves to GPU individually
        X_tensor = torch.from_numpy(np.ascontiguousarray(X).astype(np.float32))

        all_probs = []
        batch_size = self._config.get("batch_size", 512)

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size].to(self._device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
                    logits = self._model(batch)
                    probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())

        probabilities = np.concatenate(all_probs, axis=0)
        class_predictions_int = np.argmax(probabilities, axis=1)
        class_predictions = self._convert_labels_from_class(class_predictions_int)
        confidence = np.max(probabilities, axis=1)

        return PredictionResult(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={"model": self._get_model_type()},
        )

    def save(self, path: Path) -> None:
        """Save model to disk."""
        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "config": self._config,
                "n_features": self._n_features,
                "n_classes": self._n_classes,
                "seq_len": self._seq_len,  # seq_len-dependent networks rebuild from it
                "arch_version": self.ARCH_VERSION,
            },
            path / "model.pt",
        )

        logger.info(f"Saved {self._get_model_type().upper()} model to {path}")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)
        model_path = path / "model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(
            model_path, map_location=self._device, weights_only=False
        )  # nosec: loads state_dict + config metadata from trusted internal checkpoints

        # Refuse checkpoints of a different network definition: weights of an
        # older architecture may still fit the state_dict yet compute something
        # else (e.g. a different pooling or padding), which must fail loudly.
        saved_version = str(checkpoint.get("arch_version") or _LEGACY_ARCH_VERSION)
        if saved_version != self.ARCH_VERSION:
            upgraded = self._upgrade_checkpoint(checkpoint, saved_version)
            if upgraded is not None:
                logger.info(
                    f"Converted {self._get_model_type().upper()} checkpoint {model_path} "
                    f"from architecture version {saved_version} to {self.ARCH_VERSION}"
                )
                checkpoint, saved_version = upgraded, self.ARCH_VERSION
        if saved_version != self.ARCH_VERSION:
            raise ValueError(
                f"{self._get_model_type().upper()} checkpoint {model_path} was saved by "
                f"architecture version {saved_version}, but this code builds version "
                f"{self.ARCH_VERSION}. The network definition changed; retrain the model."
            )

        self._config = checkpoint["config"]
        self._n_features = checkpoint["n_features"]
        self._n_classes = checkpoint["n_classes"]
        self._seq_len = checkpoint.get("seq_len")

        # Recreate and load model
        self._model = self._create_network(self._n_features)
        state_dict = checkpoint["model_state_dict"]
        # Strip _orig_mod. prefix added by torch.compile (if present)
        cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        self._model.load_state_dict(cleaned)
        self._model = self._model.to(self._device)
        self._model.eval()

        self._is_fitted = True
        logger.info(f"Loaded {self._get_model_type().upper()} model from {path}")

    def _upgrade_checkpoint(
        self, checkpoint: dict[str, Any], saved_version: str
    ) -> dict[str, Any] | None:
        """
        Deterministically convert a checkpoint of an older architecture version.

        Returns the checkpoint rewritten for ``ARCH_VERSION``, or None when no
        exact conversion exists (``load`` then refuses the checkpoint). Only a
        subclass whose version change is a pure re-parameterization of the same
        function overrides this.
        """
        return None

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None,
        config: dict[str, Any],
        shuffle: bool,
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        X_tensor = torch.from_numpy(np.ascontiguousarray(X))
        y_tensor = torch.from_numpy(
            np.ascontiguousarray(self._convert_labels_to_class(y).astype(np.int64))
        )

        if sample_weights is not None:
            weights_tensor = torch.from_numpy(np.ascontiguousarray(sample_weights))
            dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)

        # CUDA-optimized DataLoader settings
        use_cuda = self._device.type == "cuda"
        # Auto-detect: 4 workers for CUDA (overlaps data loading with GPU compute), 0 for CPU
        num_workers_cfg = config.get("num_workers")
        num_workers = (4 if use_cuda else 0) if num_workers_cfg is None else num_workers_cfg
        pin_memory_cfg = config.get("pin_memory")
        pin_memory = use_cuda if pin_memory_cfg is None else bool(pin_memory_cfg)
        persistent_workers = num_workers > 0

        # Seed each DataLoader worker uniquely for reproducibility
        worker_init_fn = None
        if num_workers > 0:
            from src.core.reproducibility import get_worker_init_fn

            worker_init_fn = get_worker_init_fn(seed=config.get("random_seed", 42))

        return DataLoader(
            dataset,
            batch_size=config.get("batch_size", 512),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )

    def _create_optimizer(self, config: dict[str, Any]) -> torch.optim.Optimizer:
        """Create AdamW optimizer."""
        return torch.optim.AdamW(
            self._model.parameters(),
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 0.0001),
        )

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: dict[str, Any],
        steps_per_epoch: int,
    ) -> tuple[torch.optim.lr_scheduler.LambdaLR, WarmupCosineSchedule]:
        """Create the per-step warmup + cosine scheduler and its epoch clock."""
        schedule = WarmupCosineSchedule(
            max_epochs=config.get("max_epochs", 100),
            warmup_epochs=config.get("warmup_epochs", 5),
            steps_per_epoch=steps_per_epoch,
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), schedule

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler | None,
        amp_dtype: torch.dtype,
        grad_clip: float,
    ) -> tuple[float, float, float]:
        """Run one training epoch.

        Returns:
            Tuple of (avg_loss, accuracy, avg_gradient_norm)
        """
        self._model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_norms: list[float] = []

        non_blocking = self._device.type == "cuda"

        for batch in loader:
            if len(batch) == 3:
                X_batch, y_batch, weights = batch
                X_batch = X_batch.to(self._device, non_blocking=non_blocking)
                y_batch = y_batch.to(self._device, non_blocking=non_blocking)
                weights = weights.to(self._device, non_blocking=non_blocking)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self._device, non_blocking=non_blocking)
                y_batch = y_batch.to(self._device, non_blocking=non_blocking)
                weights = None

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
                logits = self._model(X_batch)

                # NaN/Inf check on forward pass output
                if hasattr(self, "_numerical_validator"):
                    self._numerical_validator.check_tensor(logits, "logits", raise_on_error=True)

                if weights is not None:
                    # Use reduction='none' to get per-sample losses for weighting
                    # Preserve class_weights if active (criterion already has them)
                    criterion_unreduced = nn.CrossEntropyLoss(
                        weight=criterion.weight,
                        reduction="none",
                        label_smoothing=criterion.label_smoothing,
                    )
                    per_sample_loss = criterion_unreduced(logits, y_batch)
                    loss = (per_sample_loss * weights).mean()
                else:
                    loss = criterion(logits, y_batch)

                # NaN/Inf check on loss
                if hasattr(self, "_numerical_validator"):
                    self._numerical_validator.check_loss(loss, raise_on_error=True)

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Capture gradient norm before clipping for logging
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Capture gradient norm before clipping for logging
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
                optimizer.step()

            # Track gradient norm (convert to float for storage)
            if hasattr(pre_clip_norm, "item"):
                batch_grad_norm = pre_clip_norm.item()
            else:
                batch_grad_norm = float(pre_clip_norm)
            gradient_norms.append(batch_grad_norm)

            scheduler.step()

            # Track metrics
            total_loss += loss.item() * len(y_batch)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)

        if total == 0:
            return 0.0, 0.0, 0.0
        avg_grad_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
        return total_loss / total, correct / total, avg_grad_norm

    def _validate_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        amp_dtype: torch.dtype,
    ) -> tuple[float, float]:
        """Run one validation epoch."""
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        non_blocking = self._device.type == "cuda"

        with torch.no_grad():
            for batch in loader:
                X_batch, y_batch = batch[0], batch[1]
                X_batch = X_batch.to(self._device, non_blocking=non_blocking)
                y_batch = y_batch.to(self._device, non_blocking=non_blocking)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
                    logits = self._model(X_batch)
                    loss = criterion(logits, y_batch)

                total_loss += loss.item() * len(y_batch)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += len(y_batch)

        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total

    def _compute_final_metrics(
        self, loader: DataLoader, amp_dtype: torch.dtype, y_true: np.ndarray
    ) -> dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        from sklearn.metrics import accuracy_score, f1_score

        self._model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(self._device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
                    preds = torch.argmax(self._model(X_batch), dim=1)
                all_preds.append(preds.cpu().numpy())
        y_pred = self._convert_labels_from_class(np.concatenate(all_preds))
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def _convert_labels_to_class(self, labels: np.ndarray) -> np.ndarray:
        return map_labels_to_classes(labels, self._n_classes)

    def _convert_labels_from_class(self, labels: np.ndarray) -> np.ndarray:
        return map_classes_to_labels(labels, self._n_classes)


__all__ = [
    "BaseRNNModel",
    "RNNNetwork",
    "EarlyStoppingState",
]
