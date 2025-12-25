"""
TCN Model - Temporal Convolutional Network for 3-class prediction.

GPU-accelerated TCN with:
- Dilated causal convolutions for long-range dependencies
- Weight normalization for stable training
- Residual connections with 1x1 convolutions
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)

Supports any NVIDIA GPU (GTX 10xx, RTX 20xx/30xx/40xx, Tesla T4/V100/A100).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from ..base import TrainingMetrics
from ..registry import register
from .base_rnn import BaseRNNModel, EarlyStoppingState

logger = logging.getLogger(__name__)


# =============================================================================
# TCN NETWORK COMPONENTS
# =============================================================================


class CausalConv1d(nn.Module):
    """Causal convolution that only looks at past data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                dilation=dilation,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Temporal block with two causal convolutions and residual connection."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if n_inputs != n_outputs:
            self.downsample = weight_norm(nn.Conv1d(n_inputs, n_outputs, 1))
        else:
            self.downsample = None
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCNNetwork(nn.Module):
    """
    Temporal Convolutional Network for sequence classification.

    Architecture:
        Input (batch, seq_len, features)
        -> Transpose to (batch, features, seq_len)
        -> TemporalBlock 1 (dilation=1)
        -> TemporalBlock 2 (dilation=2)
        -> TemporalBlock 3 (dilation=4)
        -> TemporalBlock 4 (dilation=8)
        -> Global average pooling
        -> Linear -> 3 classes
    """

    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        kernel_size: int,
        dropout: float,
        dilation_base: int = 2,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self._kernel_size = kernel_size
        self._dilation_base = dilation_base

        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = dilation_base ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, seq, feat) -> (batch, feat, seq)
        x = self.network(x)
        x = x.mean(dim=2)  # Global average pooling
        return self.fc(x)

    @property
    def receptive_field(self) -> int:
        """Calculate effective receptive field."""
        rf = 1
        for i in range(len(self.num_channels)):
            dilation = self._dilation_base ** i
            rf += 2 * (self._kernel_size - 1) * dilation
        return rf


# =============================================================================
# TCN MODEL
# =============================================================================


@register(
    name="tcn",
    family="neural",
    description="Temporal Convolutional Network with dilated causal convolutions",
    aliases=["temporal_convolutional_network"],
)
class TCNModel(BaseRNNModel):
    """
    Temporal Convolutional Network classifier with GPU support.

    Inherits training infrastructure from BaseRNNModel:
    - GPU training with CUDA (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection:
      - bfloat16 for Ampere+ (RTX 30xx/40xx, A100, H100)
      - float16 for Volta/Turing (RTX 20xx, GTX 16xx, T4, V100)
      - float32 for older GPUs or CPU
    - AdamW optimizer with cosine annealing
    - Gradient clipping and early stopping

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("tcn", config={"num_channels": [64, 64, 64, 64]})
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
    """

    def get_default_config(self) -> Dict[str, Any]:
        """Return default TCN hyperparameters."""
        defaults = super().get_default_config()
        defaults.update({
            "num_channels": [64, 64, 64, 64],
            "kernel_size": 3,
            "dropout": 0.2,
            "dilation_base": 2,
            "sequence_length": 120,  # Longer than LSTM
        })
        return defaults

    def _create_network(self, input_size: int) -> nn.Module:
        """Create the TCN network."""
        return TCNNetwork(
            input_size=input_size,
            num_channels=self._config.get("num_channels", [64, 64, 64, 64]),
            kernel_size=self._config.get("kernel_size", 3),
            dropout=self._config.get("dropout", 0.2),
            dilation_base=self._config.get("dilation_base", 2),
            n_classes=self._n_classes,
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "tcn"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingMetrics:
        """
        Train the TCN with receptive field logging.

        Overrides parent to add TCN-specific receptive field checks.
        """
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        start_time = time.time()

        # Merge config
        train_config = self._config.copy()
        if config:
            train_config.update(config)

        # Extract dimensions
        n_samples, seq_len, n_features = X_train.shape
        self._n_features = n_features

        # Create network
        self._model = self._create_network(n_features)
        self._model = self._model.to(self._device)

        # Log receptive field
        rf = self._model.receptive_field
        logger.info(f"TCN receptive field: {rf} timesteps (seq_len={seq_len})")
        if rf < seq_len:
            logger.warning(
                f"Receptive field ({rf}) < sequence length ({seq_len}). "
                f"Consider adding more layers or increasing kernel_size."
            )

        # Prepare data
        train_loader = self._create_dataloader(
            X_train, y_train, sample_weights, train_config, shuffle=True
        )
        val_loader = self._create_dataloader(X_val, y_val, None, train_config, shuffle=False)

        # Setup training components
        optimizer = self._create_optimizer(train_config)
        scheduler = self._create_scheduler(optimizer, train_config, len(train_loader))
        criterion = nn.CrossEntropyLoss()

        # Mixed precision scaler (only needed for float16, not bfloat16)
        scaler = (
            torch.amp.GradScaler("cuda")
            if self._use_grad_scaler and self._device.type == "cuda"
            else None
        )
        # Use dynamically detected AMP dtype
        amp_dtype = self._amp_dtype

        early_stopping = EarlyStoppingState()
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
        }

        max_epochs = train_config.get("max_epochs", 100)
        patience = train_config.get("early_stopping_patience", 15)
        min_delta = train_config.get("min_delta", 0.0001)
        grad_clip = train_config.get("gradient_clip", 1.0)

        logger.info(
            f"Training TCN: epochs={max_epochs}, batch_size={train_config.get('batch_size')}, "
            f"channels={train_config.get('num_channels')}, kernel={train_config.get('kernel_size')}, "
            f"mixed_precision={'on' if self._use_amp else 'off'}"
        )

        for epoch in range(max_epochs):
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion, scheduler, scaler, amp_dtype, grad_clip
            )
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            val_loss, val_acc = self._validate_epoch(val_loader, criterion, amp_dtype)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{max_epochs} - "
                    f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                    f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"
                )

            if early_stopping.check(val_loss, epoch, self._model, patience, min_delta):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if early_stopping.best_state_dict is not None:
            self._model.load_state_dict(early_stopping.best_state_dict)

        training_time = time.time() - start_time
        epochs_trained = len(history["train_loss"])

        train_metrics = self._compute_final_metrics(train_loader, amp_dtype, y_train)
        val_metrics = self._compute_final_metrics(val_loader, amp_dtype, y_val)

        self._is_fitted = True

        logger.info(
            f"Training complete: epochs={epochs_trained}, "
            f"best_epoch={early_stopping.best_epoch + 1}, "
            f"val_f1={val_metrics['f1']:.4f}, time={training_time:.1f}s"
        )

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
            metadata={
                "model_type": "tcn",
                "n_features": n_features,
                "n_train_samples": n_samples,
                "n_val_samples": len(X_val),
                "device": str(self._device),
                "mixed_precision": self._use_amp,
                "receptive_field": self._model.receptive_field,
            },
        )


__all__ = ["TCNModel", "TCNNetwork", "TemporalBlock", "CausalConv1d"]
