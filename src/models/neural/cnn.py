"""
CNN Models for Time Series - InceptionTime and ResNet1D.

GPU-accelerated CNN architectures with:
- InceptionTime: Inception-based CNN with multi-scale convolutions
- ResNet1D: 1D ResNet with residual blocks and skip connections
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)

References:
    InceptionTime: Fawaz et al. "InceptionTime: Finding AlexNet for
                   Time Series Classification" (2020)
    ResNet1D: Adapted from Wang et al. "Time Series Classification
              from Scratch with Deep Neural Networks" (2017)

Supports any NVIDIA GPU (GTX 10xx, RTX 20xx/30xx/40xx, Tesla T4/V100/A100).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import PredictionOutput
from ..registry import register
from .base_rnn import BaseRNNModel

logger = logging.getLogger(__name__)


# =============================================================================
# INCEPTION TIME COMPONENTS
# =============================================================================


class InceptionModule(nn.Module):
    """
    Inception module for time series.

    Applies multiple parallel convolutions with different kernel sizes
    and concatenates the outputs. Includes a bottleneck layer to reduce
    computational cost.

    Architecture:
        Input -> Bottleneck (optional)
              -> Conv1D k=10 (long-range)
              -> Conv1D k=20 (medium-range)
              -> Conv1D k=40 (short-range)
              -> MaxPool + Conv1D k=1 (local)
        -> Concatenate -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (10, 20, 40),
        bottleneck_channels: int = 32,
        use_bottleneck: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes

        # Bottleneck layer
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in_channels = bottleneck_channels
        else:
            self.bottleneck = None
            conv_in_channels = in_channels

        # Parallel convolutions with different kernel sizes
        self.convolutions = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            conv = nn.Conv1d(
                conv_in_channels,
                n_filters,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
            self.convolutions.append(conv)

        # MaxPool branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        # Output processing
        total_filters = n_filters * (len(kernel_sizes) + 1)  # +1 for maxpool branch
        self.batch_norm = nn.BatchNorm1d(total_filters)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, channels, seq_len)

        Returns:
            Output tensor, shape (batch, total_filters, seq_len)
        """
        seq_len = x.size(2)

        # Bottleneck
        if self.use_bottleneck:
            x_bottleneck = self.bottleneck(x)
        else:
            x_bottleneck = x

        # Parallel convolutions
        conv_outputs = []
        for conv in self.convolutions:
            out = conv(x_bottleneck)
            # Ensure output matches input sequence length
            if out.size(2) != seq_len:
                out = out[:, :, :seq_len]
            conv_outputs.append(out)

        # MaxPool branch (from original input, not bottleneck)
        maxpool_out = self.maxpool(x)
        maxpool_out = self.maxpool_conv(maxpool_out)
        # Ensure output matches input sequence length
        if maxpool_out.size(2) != seq_len:
            maxpool_out = maxpool_out[:, :, :seq_len]
        conv_outputs.append(maxpool_out)

        # Concatenate all branches
        out = torch.cat(conv_outputs, dim=1)

        # Batch norm and activation
        out = self.batch_norm(out)
        out = self.activation(out)

        return out


class InceptionBlock(nn.Module):
    """
    Inception block with residual connection.

    Stacks multiple inception modules and adds a residual connection
    from input to output.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (10, 20, 40),
        bottleneck_channels: int = 32,
        n_modules: int = 3,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.n_modules = n_modules

        # Calculate output channels from one inception module
        self.out_channels = n_filters * (len(kernel_sizes) + 1)

        # Build inception modules
        self.modules_list = nn.ModuleList()
        for i in range(n_modules):
            input_ch = in_channels if i == 0 else self.out_channels
            module = InceptionModule(
                in_channels=input_ch,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_bottleneck=True,
            )
            self.modules_list.append(module)

        # Residual connection with 1x1 conv if dimensions don't match
        if use_residual:
            if in_channels != self.out_channels:
                self.residual_conv = nn.Conv1d(
                    in_channels, self.out_channels, kernel_size=1, bias=False
                )
                self.residual_bn = nn.BatchNorm1d(self.out_channels)
            else:
                self.residual_conv = None
                self.residual_bn = None

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, in_channels, seq_len)

        Returns:
            Output tensor, shape (batch, out_channels, seq_len)
        """
        # Store input for residual
        residual = x

        # Pass through inception modules
        out = x
        for module in self.modules_list:
            out = module(out)

        # Residual connection
        if self.use_residual:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
                residual = self.residual_bn(residual)
            out = out + residual
            out = self.activation(out)

        return out


class InceptionTimeNetwork(nn.Module):
    """
    InceptionTime network for sequence classification.

    Architecture:
        Input (batch, seq_len, features)
        -> Transpose to (batch, features, seq_len)
        -> InceptionBlock 1
        -> InceptionBlock 2
        -> ...
        -> Global Average Pooling
        -> Linear -> n_classes
    """

    def __init__(
        self,
        input_size: int,
        n_blocks: int = 6,
        n_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (10, 20, 40),
        bottleneck_channels: int = 32,
        n_modules_per_block: int = 3,
        use_residual: bool = True,
        dropout: float = 0.0,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.n_blocks = n_blocks
        self.n_filters = n_filters

        # Build inception blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            in_channels = input_size if i == 0 else n_filters * (len(kernel_sizes) + 1)
            block = InceptionBlock(
                in_channels=in_channels,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                n_modules=n_modules_per_block,
                use_residual=use_residual,
            )
            self.blocks.append(block)

        # Classification head
        final_channels = n_filters * (len(kernel_sizes) + 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(final_channels, n_classes)

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

        # Pass through inception blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=2)  # (batch, channels)

        # Classification
        x = self.dropout(x)
        logits = self.fc(x)

        return logits


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

        out = out + identity
        out = self.relu(out)

        return out


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

        out = out + identity
        out = self.relu(out)

        return out


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
        n_blocks: list[int] = (2, 2, 2, 2),
        channels: list[int] = (64, 128, 256, 512),
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

        for i, (n_block, out_channels) in enumerate(zip(n_blocks, channels)):
            stride = 2 if i > 0 else 1  # Downsample after first stage

            blocks = []
            for j in range(n_block):
                block_stride = stride if j == 0 else 1
                block_in_channels = in_channels if j == 0 else out_channels

                if use_bottleneck:
                    bottleneck_ch = out_channels // 4
                    block = ResidualBlock1DBottleneck(
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
        logits = self.fc(x)

        return logits


# =============================================================================
# INCEPTION TIME MODEL
# =============================================================================


@register(
    name="inceptiontime",
    family="neural",
    description="InceptionTime: Inception-based CNN for time series classification",
    aliases=["inception_time", "inception"],
)
class InceptionTimeModel(BaseRNNModel):
    """
    InceptionTime classifier with GPU support.

    Inherits training infrastructure from BaseRNNModel:
    - GPU training with CUDA (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection:
      - bfloat16 for Ampere+ (RTX 30xx/40xx, A100, H100)
      - float16 for Volta/Turing (RTX 20xx, GTX 16xx, T4, V100)
      - float32 for older GPUs or CPU
    - AdamW optimizer with cosine annealing
    - Gradient clipping and early stopping

    Features:
    - Multi-scale convolutions capture patterns at different temporal scales
    - Bottleneck layers reduce computational cost
    - Residual connections enable deeper networks
    - Global average pooling for translation invariance

    Note on Causality:
        InceptionTime uses standard (non-causal) convolutions where each position
        can see neighboring positions in both directions. This is non-causal but
        provides stronger pattern recognition within the observation window.

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("inceptiontime", config={
        ...     "n_blocks": 6,
        ...     "n_filters": 32,
        ...     "kernel_sizes": (10, 20, 40),
        ... })
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        logger.debug(f"Initialized InceptionTimeModel with config: {self._config}")

    @property
    def is_production_safe(self) -> bool:
        """
        Check if this model configuration is safe for production trading.

        InceptionTime uses standard convolutions that see both past and future
        within the window, so it is non-causal.

        Returns:
            False - InceptionTime is not inherently causal.
        """
        return False

    def _log_bidirectional_warning(self) -> None:
        """Log a warning about non-causal convolutions (only once)."""
        if self._bidirectional_warning_logged:
            return

        logger.warning(
            "INCEPTIONTIME NON-CAUSAL CONVOLUTIONS: InceptionTime uses standard "
            "convolutions where each position can see neighboring positions in both "
            "directions. This provides stronger pattern recognition but is non-causal.\n"
            "Implications:\n"
            "  - Each position uses information from later timesteps in the window\n"
            "  - Patterns learned may not be available during real-time inference\n"
            "Recommendations:\n"
            "  - For production trading: Use TCN which has causal convolutions\n"
            "  - For research/pattern analysis: InceptionTime is acceptable"
        )
        self._bidirectional_warning_logged = True

    def get_default_config(self) -> dict[str, Any]:
        """Return default InceptionTime hyperparameters."""
        defaults = super().get_default_config()
        defaults.update(
            {
                # Architecture
                "n_blocks": 6,
                "n_filters": 32,
                "kernel_sizes": (10, 20, 40),
                "bottleneck_channels": 32,
                "n_modules_per_block": 3,
                "use_residual": True,
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
        """Create the InceptionTime network."""
        kernel_sizes = self._config.get("kernel_sizes", (10, 20, 40))
        # Ensure kernel_sizes is a tuple
        if isinstance(kernel_sizes, list):
            kernel_sizes = tuple(kernel_sizes)

        return InceptionTimeNetwork(
            input_size=input_size,
            n_blocks=self._config.get("n_blocks", 6),
            n_filters=self._config.get("n_filters", 32),
            kernel_sizes=kernel_sizes,
            bottleneck_channels=self._config.get("bottleneck_channels", 32),
            n_modules_per_block=self._config.get("n_modules_per_block", 3),
            use_residual=self._config.get("use_residual", True),
            dropout=self._config.get("dropout", 0.0),
            n_classes=self._n_classes,
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "inceptiontime"

    def _on_training_start(self, train_config: dict[str, Any], seq_len: int) -> dict[str, Any]:
        """
        Log InceptionTime-specific information at training start.

        Args:
            train_config: Training configuration dictionary
            seq_len: Sequence length of training data

        Returns:
            Dict with InceptionTime metadata for TrainingMetrics
        """
        n_blocks = train_config.get("n_blocks", 6)
        n_filters = train_config.get("n_filters", 32)
        kernel_sizes = train_config.get("kernel_sizes", (10, 20, 40))

        logger.info(
            f"InceptionTime architecture: {n_blocks} blocks, "
            f"filters={n_filters}, kernels={kernel_sizes}, seq_len={seq_len}"
        )

        return {
            "n_blocks": n_blocks,
            "n_filters": n_filters,
            "kernel_sizes": kernel_sizes,
        }

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """
        Generate predictions with class probabilities.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            PredictionOutput with predictions, probabilities, and metadata
        """
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        self._model.eval()
        amp_dtype = self._amp_dtype

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        all_probs = []
        batch_size = self._config.get("batch_size", 64)

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size]

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
                    logits = self._model(batch)
                    probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())

        probabilities = np.concatenate(all_probs, axis=0)
        class_predictions_int = np.argmax(probabilities, axis=1)
        class_predictions = self._convert_labels_from_class(class_predictions_int)
        confidence = np.max(probabilities, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={
                "model": "inceptiontime",
                "n_blocks": self._config.get("n_blocks"),
                "n_filters": self._config.get("n_filters"),
            },
        )


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

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """
        Generate predictions with class probabilities.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            PredictionOutput with predictions, probabilities, and metadata
        """
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        self._model.eval()
        amp_dtype = self._amp_dtype

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        all_probs = []
        batch_size = self._config.get("batch_size", 64)

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size]

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
                    logits = self._model(batch)
                    probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())

        probabilities = np.concatenate(all_probs, axis=0)
        class_predictions_int = np.argmax(probabilities, axis=1)
        class_predictions = self._convert_labels_from_class(class_predictions_int)
        confidence = np.max(probabilities, axis=1)

        return PredictionOutput(
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

        self._model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            # Transpose
            x = X_tensor.transpose(1, 2)

            # Stem
            x = self._model.stem(x)

            # Pass through stages up to stage_idx
            n_stages = len(self._model.stages)
            target_idx = stage_idx if stage_idx >= 0 else n_stages + stage_idx

            for i, stage in enumerate(self._model.stages):
                x = stage(x)
                if i == target_idx:
                    break

            return x.cpu().numpy()


__all__ = [
    # InceptionTime
    "InceptionTimeModel",
    "InceptionTimeNetwork",
    "InceptionBlock",
    "InceptionModule",
    # ResNet1D
    "ResNet1DModel",
    "ResNet1DNetwork",
    "ResidualBlock1D",
    "ResidualBlock1DBottleneck",
]
