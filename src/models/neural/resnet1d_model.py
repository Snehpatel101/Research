"""
ResNet1D Model for Time Series Classification.

GPU-accelerated 1D ResNet with:
- Residual blocks with skip connections
- Configurable depth and width (channels)
- Optional bottleneck blocks for larger models
- Progressive downsampling for hierarchical features
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)

References:
    Adapted from Wang et al. "Time Series Classification
                from Scratch with Deep Neural Networks" (2017)

Supports any NVIDIA GPU (GTX 10xx, RTX 20xx/30xx/40xx, Tesla T4/V100/A100).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..base import PredictionResult
from ..registry import register
from .base_rnn import BaseRNNModel

logger = logging.getLogger(__name__)


# =============================================================================
# RESNET1D COMPONENTS
# =============================================================================


class ResidualBlock1D(nn.Module):
    """
    1D Residual block with two convolutional layers.

    Architecture:
        Input -> Conv1D -> BatchNorm -> ReLU -> Conv1D -> BatchNorm
             |                                               |
             +----------- Shortcut (optional 1x1) -----------+
             -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut: nn.Module
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, channels, seq_len)

        Returns:
            Output tensor, shape (batch, out_channels, seq_len//stride)
        """
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Handle sequence length mismatch from even kernel sizes
        # (padding = kernel_size // 2 doesn't preserve length for even kernels)
        if out.shape[2] != identity.shape[2]:
            min_len = min(out.shape[2], identity.shape[2])
            out = out[:, :, :min_len]
            identity = identity[:, :, :min_len]

        out = out + identity
        result: torch.Tensor = self.relu(out)

        return result


class ResidualBlock1DBottleneck(nn.Module):
    """
    1D Residual bottleneck block (similar to ResNet-50+).

    Architecture:
        Input -> Conv1D 1x1 -> BatchNorm -> ReLU
              -> Conv1D 3x3 -> BatchNorm -> ReLU
              -> Conv1D 1x1 -> BatchNorm
             |                                |
             +------- Shortcut (1x1) ---------+
             -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        # 1x1 reduce
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)

        # 3x3 conv
        self.conv2 = nn.Conv1d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)

        # 1x1 expand
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Shortcut
        self.shortcut: nn.Module
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Handle sequence length mismatch from even kernel sizes
        if out.shape[2] != identity.shape[2]:
            min_len = min(out.shape[2], identity.shape[2])
            out = out[:, :, :min_len]
            identity = identity[:, :, :min_len]

        out = out + identity
        result: torch.Tensor = self.relu(out)

        return result


class ResNet1DNetwork(nn.Module):
    """
    ResNet-style 1D CNN for sequence classification.

    Architecture:
        Input (batch, seq_len, features)
        -> Transpose to (batch, features, seq_len)
        -> Conv1D stem (7x7)
        -> ResidualBlock layers (configurable depth)
        -> Global Average Pooling
        -> Dropout
        -> Linear -> n_classes
    """

    def __init__(
        self,
        input_size: int,
        n_blocks: tuple[int, ...] | list[int] = (2, 2, 2, 2),
        channels: tuple[int, ...] | list[int] = (64, 128, 256, 512),
        kernel_size: int = 3,
        stem_kernel_size: int = 7,
        use_bottleneck: bool = False,
        dropout: float = 0.0,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.n_blocks = n_blocks
        self.channels = channels

        # Stem layer
        stem_padding = stem_kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_size,
                channels[0],
                stem_kernel_size,
                stride=1,
                padding=stem_padding,
                bias=False,
            ),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
        )

        # Build residual stages
        self.stages = nn.ModuleList()
        in_channels = channels[0]

        for i, (n_block, out_channels) in enumerate(zip(n_blocks, channels, strict=False)):
            stride = 2 if i > 0 else 1  # Downsample after first stage

            blocks: list[nn.Module] = []
            for j in range(n_block):
                block_stride = stride if j == 0 else 1
                block_in_channels = in_channels if j == 0 else out_channels

                if use_bottleneck:
                    bottleneck_ch = out_channels // 4
                    block: nn.Module = ResidualBlock1DBottleneck(
                        in_channels=block_in_channels,
                        bottleneck_channels=bottleneck_ch,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        dropout=dropout,
                    )
                else:
                    block = ResidualBlock1D(
                        in_channels=block_in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        dropout=dropout,
                    )
                blocks.append(block)

            self.stages.append(nn.Sequential(*blocks))
            in_channels = out_channels

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(channels[-1], n_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Output logits, shape (batch, n_classes)
        """
        # Transpose: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Stem
        x = self.stem(x)

        # Residual stages
        for stage in self.stages:
            x = stage(x)

        # Global average pooling and classification
        x = self.avgpool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        x = self.dropout(x)
        logits: torch.Tensor = self.fc(x)

        return logits


# =============================================================================
# RESNET1D MODEL
# =============================================================================


@register(
    name="resnet1d",
    family="neural",
    description="ResNet1D: 1D ResNet with residual blocks for time series",
    aliases=["resnet_1d", "resnet"],
)
class ResNet1DModel(BaseRNNModel):
    """
    ResNet1D classifier with GPU support.

    Inherits training infrastructure from BaseRNNModel:
    - GPU training with CUDA (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection:
      - bfloat16 for Ampere+ (RTX 30xx/40xx, A100, H100)
      - float16 for Volta/Turing (RTX 20xx, GTX 16xx, T4, V100)
      - float32 for older GPUs or CPU
    - AdamW optimizer with cosine annealing
    - Gradient clipping and early stopping

    Features:
    - Residual connections enable very deep networks
    - Configurable depth and width (channels)
    - Optional bottleneck blocks for larger models
    - Progressive downsampling for hierarchical features

    Note on Causality:
        ResNet1D uses standard (non-causal) convolutions where each position
        can see neighboring positions in both directions. This is non-causal.

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("resnet1d", config={
        ...     "n_blocks": (2, 2, 2, 2),
        ...     "channels": (64, 128, 256, 512),
        ... })
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        logger.debug(f"Initialized ResNet1DModel with config: {self._config}")

    @property
    def is_production_safe(self) -> bool:
        """
        Check if this model configuration is safe for production trading.

        ResNet1D uses standard convolutions that see both past and future
        within the window, so it is non-causal.

        Returns:
            False - ResNet1D is not inherently causal.
        """
        return False

    def _log_bidirectional_warning(self) -> None:
        """Log a warning about non-causal convolutions (only once)."""
        if self._bidirectional_warning_logged:
            return

        logger.warning(
            "RESNET1D NON-CAUSAL CONVOLUTIONS: ResNet1D uses standard convolutions "
            "where each position can see neighboring positions in both directions. "
            "This is non-causal.\n"
            "Implications:\n"
            "  - Each position uses information from later timesteps in the window\n"
            "  - Patterns learned may not be available during real-time inference\n"
            "Recommendations:\n"
            "  - For production trading: Use TCN which has causal convolutions\n"
            "  - For research/pattern analysis: ResNet1D is acceptable"
        )
        self._bidirectional_warning_logged = True

    def get_default_config(self) -> dict[str, Any]:
        """Return default ResNet1D hyperparameters."""
        defaults = super().get_default_config()
        defaults.update(
            {
                # Architecture
                "n_blocks": (2, 2, 2, 2),  # ResNet-18 style
                "channels": (64, 128, 256, 512),
                "kernel_size": 3,
                "stem_kernel_size": 7,
                "use_bottleneck": False,
                "dropout": 0.0,
                # Training
                "sequence_length": 60,
                "batch_size": 64,
                "max_epochs": 100,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip": 1.0,
                "early_stopping_patience": 15,
                "warmup_epochs": 5,
            }
        )
        return defaults

    def _create_network(self, input_size: int) -> nn.Module:
        """Create the ResNet1D network."""
        n_blocks = self._config.get("n_blocks", (2, 2, 2, 2))
        channels = self._config.get("channels", (64, 128, 256, 512))

        # Ensure lists become tuples
        if isinstance(n_blocks, list):
            n_blocks = tuple(n_blocks)
        if isinstance(channels, list):
            channels = tuple(channels)

        return ResNet1DNetwork(
            input_size=input_size,
            n_blocks=n_blocks,
            channels=channels,
            kernel_size=self._config.get("kernel_size", 3),
            stem_kernel_size=self._config.get("stem_kernel_size", 7),
            use_bottleneck=self._config.get("use_bottleneck", False),
            dropout=self._config.get("dropout", 0.0),
            n_classes=self._n_classes,
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "resnet1d"

    def _on_training_start(self, train_config: dict[str, Any], seq_len: int) -> dict[str, Any]:
        """
        Log ResNet1D-specific information at training start.

        Args:
            train_config: Training configuration dictionary
            seq_len: Sequence length of training data

        Returns:
            Dict with ResNet1D metadata for TrainingMetrics
        """
        n_blocks = train_config.get("n_blocks", (2, 2, 2, 2))
        channels = train_config.get("channels", (64, 128, 256, 512))
        use_bottleneck = train_config.get("use_bottleneck", False)

        total_blocks = sum(n_blocks)
        arch_name = "ResNet1D-Bottleneck" if use_bottleneck else "ResNet1D"

        logger.info(
            f"{arch_name} architecture: {n_blocks} blocks ({total_blocks} total), "
            f"channels={channels}, seq_len={seq_len}"
        )

        return {
            "n_blocks": n_blocks,
            "channels": channels,
            "total_blocks": total_blocks,
            "use_bottleneck": use_bottleneck,
        }

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions with class probabilities.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            PredictionResult with predictions, probabilities, and metadata
        """
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        if self._model is None:
            raise RuntimeError("Model is not fitted")

        self._model.eval()
        amp_dtype = self._amp_dtype

        # Zero-copy tensor on CPU; each batch moves to GPU individually
        X_tensor = torch.from_numpy(np.ascontiguousarray(X).astype(np.float32))

        all_probs = []
        batch_size = self._config.get("batch_size", 64)

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
            metadata={
                "model": "resnet1d",
                "n_blocks": self._config.get("n_blocks"),
                "channels": self._config.get("channels"),
            },
        )

    def get_feature_maps(self, X: np.ndarray, stage_idx: int = -1) -> np.ndarray | None:
        """
        Extract feature maps from a specific stage for visualization.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            stage_idx: Index of stage to extract features from (-1 for last)

        Returns:
            Feature maps, shape (n_samples, channels, seq_len//stride),
            or None if model is not fitted
        """
        if not self._is_fitted:
            return None

        self._validate_input_shape(X, "X")

        resnet_network = self._model
        if not isinstance(resnet_network, ResNet1DNetwork):
            return None

        resnet_network.eval()
        X_tensor = torch.from_numpy(
            np.ascontiguousarray(X).astype(np.float32)
        ).to(self._device)

        with torch.no_grad():
            # Transpose
            x = X_tensor.transpose(1, 2)

            # Stem
            x = resnet_network.stem(x)

            # Pass through stages up to stage_idx
            n_stages = len(resnet_network.stages)
            target_idx = stage_idx if stage_idx >= 0 else n_stages + stage_idx

            for i, stage in enumerate(resnet_network.stages):
                x = stage(x)
                if i == target_idx:
                    break

            result: np.ndarray = x.cpu().numpy()
            return result


__all__ = [
    "ResNet1DModel",
    "ResNet1DNetwork",
    "ResidualBlock1D",
    "ResidualBlock1DBottleneck",
]
