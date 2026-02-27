"""
InceptionTime Model for Time Series Classification.

GPU-accelerated InceptionTime with:
- Inception modules with multi-scale convolutions
- Bottleneck layers for computational efficiency
- Residual connections for deeper networks
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)

References:
    Fawaz et al. "InceptionTime: Finding AlexNet for
                 Time Series Classification" (2020)

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
        self.bottleneck: nn.Conv1d | None
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

        self.activation: nn.ReLU | nn.GELU
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
        if self.use_bottleneck and self.bottleneck is not None:
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
        result: torch.Tensor = self.activation(out)

        return result


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
        self.residual_conv: nn.Conv1d | None = None
        self.residual_bn: nn.BatchNorm1d | None = None
        if use_residual and in_channels != self.out_channels:
            self.residual_conv = nn.Conv1d(
                in_channels, self.out_channels, kernel_size=1, bias=False
            )
            self.residual_bn = nn.BatchNorm1d(self.out_channels)

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
            if self.residual_conv is not None and self.residual_bn is not None:
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
        logits: torch.Tensor = self.fc(x)

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
                "model": "inceptiontime",
                "n_blocks": self._config.get("n_blocks"),
                "n_filters": self._config.get("n_filters"),
            },
        )


__all__ = [
    "InceptionTimeModel",
    "InceptionTimeNetwork",
    "InceptionBlock",
    "InceptionModule",
]
