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
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from ..registry import register
from .base_rnn import BaseRNNModel

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

    def _on_training_start(self, train_config: Dict[str, Any], seq_len: int) -> Dict[str, Any]:
        """
        Log TCN-specific receptive field information at training start.

        Overrides parent hook to add receptive field logging and validation.

        Args:
            train_config: Training configuration dictionary
            seq_len: Sequence length of training data

        Returns:
            Dict with receptive_field metadata for TrainingMetrics
        """
        rf = self._model.receptive_field
        logger.info(f"TCN receptive field: {rf} timesteps (seq_len={seq_len})")

        if rf < seq_len:
            logger.warning(
                f"Receptive field ({rf}) < sequence length ({seq_len}). "
                f"Consider adding more layers or increasing kernel_size."
            )

        return {"receptive_field": rf}


__all__ = ["TCNModel", "TCNNetwork", "TemporalBlock", "CausalConv1d"]
