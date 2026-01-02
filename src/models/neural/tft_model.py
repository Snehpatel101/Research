"""
TFT Model - Temporal Fusion Transformer for 3-class prediction.

GPU-accelerated TFT with:
- Variable Selection Networks for feature importance
- LSTM encoder for temporal dependencies
- Interpretable Multi-Head Attention
- Gated Residual Networks (GRN) for information flow control
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)

Reference: Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" (2021)

Supports any NVIDIA GPU (GTX 10xx, RTX 20xx/30xx/40xx, Tesla T4/V100/A100).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..base import PredictionOutput
from ..registry import register
from .base_rnn import BaseRNNModel

logger = logging.getLogger(__name__)


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) for controlling information flow.

    GLU(x) = x * sigmoid(gate(x))

    This allows the network to suppress or amplify information
    based on learned gating weights.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GLU activation."""
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from TFT.

    GRN applies non-linear processing with gated skip connections:
        eta_1 = ELU(W_1 * x + b_1)
        eta_2 = W_2 * eta_1 + b_2
        GRN(x) = LayerNorm(x + GLU(eta_2))

    With optional context: GRN(x, c) uses concatenation or addition
    of context c to the primary input.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        # Skip connection projection if input != output dims
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None

        # Context projection if context is provided
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_proj = None

        # Main transformation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply GRN transformation.

        Args:
            x: Input tensor, shape (..., input_dim)
            context: Optional context tensor, shape (..., context_dim)

        Returns:
            Output tensor, shape (..., output_dim)
        """
        # Skip connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x

        # First layer
        hidden = self.fc1(x)

        # Add context if provided
        if context is not None and self.context_proj is not None:
            hidden = hidden + self.context_proj(context)

        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # GLU and residual
        gated = self.glu(hidden)
        output = self.layer_norm(skip + gated)

        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) for feature importance.

    VSN applies GRN to each feature independently, then uses a softmax
    to learn variable weights. This provides interpretable feature importance.

    Output = sum_i (softmax(weights)[i] * GRN(x_i))
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.input_dim = input_dim

        # GRN for each feature
        self.feature_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    context_dim=context_dim,
                )
                for _ in range(n_features)
            ]
        )

        # Variable weights GRN
        # Takes flattened features as input
        self.weight_grn = GatedResidualNetwork(
            input_dim=hidden_dim * n_features,
            hidden_dim=hidden_dim,
            output_dim=n_features,
            dropout=dropout,
            context_dim=context_dim,
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply variable selection.

        Args:
            x: Input tensor, shape (batch, seq_len, n_features, input_dim)
               or (batch, n_features, input_dim)
            context: Optional context tensor

        Returns:
            Tuple of:
            - Selected output, shape (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            - Variable weights, shape (batch, seq_len, n_features) or (batch, n_features)
        """
        has_time = x.dim() == 4

        if has_time:
            batch_size, seq_len, n_features, input_dim = x.shape
            # Reshape for processing: (batch * seq_len, n_features, input_dim)
            x = x.view(batch_size * seq_len, n_features, input_dim)
            if context is not None:
                context = context.view(batch_size * seq_len, -1)

        # Apply GRN to each feature
        transformed = []
        for i, grn in enumerate(self.feature_grns):
            xi = x[:, i, :]  # (batch, input_dim)
            transformed.append(grn(xi, context))

        # Stack: (batch, n_features, hidden_dim)
        transformed = torch.stack(transformed, dim=1)

        # Compute variable weights
        # Flatten for weight computation
        flat_transformed = transformed.view(transformed.shape[0], -1)
        weights = self.weight_grn(flat_transformed, context)  # (batch, n_features)
        weights = self.softmax(weights)

        # Weighted sum
        # (batch, n_features, hidden_dim) * (batch, n_features, 1) -> sum
        output = (transformed * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)

        if has_time:
            output = output.view(batch_size, seq_len, -1)
            weights = weights.view(batch_size, seq_len, n_features)

        return output, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention from TFT.

    Unlike standard multi-head attention, this version uses shared
    attention weights across heads, enabling interpretation of which
    timesteps are most important.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Store attention weights for interpretability
        self._attention_weights: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply interpretable multi-head attention.

        Args:
            query: Query tensor, shape (batch, seq_len_q, d_model)
            key: Key tensor, shape (batch, seq_len_k, d_model)
            value: Value tensor, shape (batch, seq_len_v, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor, shape (batch, seq_len_q, d_model)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        # Project to multi-head representation
        q = self.q_proj(query).view(batch_size, seq_len_q, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention weights (interpretable - averaged across heads)
        attention = torch.softmax(scores, dim=-1)
        self._attention_weights = attention.mean(dim=1).detach()  # (batch, seq_q, seq_k)

        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, v)

        # Reshape and project
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len_q, self.d_model)
        output = self.out_proj(context)

        return output

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return the last computed attention weights."""
        return self._attention_weights


class TFTNetwork(nn.Module):
    """
    Temporal Fusion Transformer network for classification.

    Architecture:
        Input (batch, seq_len, features)
        -> Variable Selection Network (learn feature importance)
        -> LSTM Encoder (capture temporal dependencies)
        -> Gated Skip Connection
        -> Interpretable Multi-Head Attention
        -> Feed-Forward with GRN
        -> Global pooling
        -> Classification head -> 3 classes

    Key components:
        1. Variable Selection: Learns which features are most relevant
        2. LSTM Encoder: Captures local temporal patterns
        3. Multi-Head Attention: Captures long-range dependencies
        4. Gated Residual Networks: Control information flow
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_heads: int,
        lstm_layers: int,
        attention_layers: int,
        d_ff: int,
        dropout: float,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.lstm_layers = lstm_layers
        self.attention_layers = attention_layers

        # Initial embedding: project each feature to d_model
        self.input_embedding = nn.Linear(1, d_model)

        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            input_dim=d_model,
            n_features=input_size,
            hidden_dim=d_model,
            dropout=dropout,
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Post-LSTM gated residual
        self.lstm_grn = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
        )

        # Attention layers
        self.attention_layers_list = nn.ModuleList()
        self.attention_grns = nn.ModuleList()
        for _ in range(attention_layers):
            self.attention_layers_list.append(
                InterpretableMultiHeadAttention(d_model, n_heads, dropout)
            )
            self.attention_grns.append(
                GatedResidualNetwork(
                    input_dim=d_model,
                    hidden_dim=d_ff,
                    output_dim=d_model,
                    dropout=dropout,
                )
            )

        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_out = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_model // 2, n_classes)

        # Store variable weights for interpretability
        self._variable_weights: torch.Tensor | None = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TFT.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Output logits, shape (batch, n_classes)
        """
        batch_size, seq_len, n_features = x.shape

        # Embed each feature independently
        # (batch, seq_len, features) -> (batch, seq_len, features, d_model)
        x = x.unsqueeze(-1)  # (batch, seq_len, features, 1)
        x = self.input_embedding(x)  # (batch, seq_len, features, d_model)

        # Variable Selection: learn feature importance
        # (batch, seq_len, features, d_model) -> (batch, seq_len, d_model)
        x, var_weights = self.vsn(x)
        self._variable_weights = var_weights.detach()

        # LSTM encoder
        lstm_out, _ = self.lstm(x)

        # Gated residual connection
        x = self.lstm_grn(lstm_out)

        # Multi-head attention layers
        for attn_layer, grn in zip(self.attention_layers_list, self.attention_grns):
            # Self-attention
            attn_out = attn_layer(x, x, x)
            # Gated residual
            x = grn(attn_out) + x

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification head
        x = self.layer_norm(x)
        x = self.dropout_out(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x

    def get_variable_weights(self) -> torch.Tensor | None:
        """Return the last computed variable selection weights."""
        return self._variable_weights

    def get_attention_weights(self) -> list[torch.Tensor | None]:
        """Return attention weights from all attention layers."""
        return [layer.get_attention_weights() for layer in self.attention_layers_list]


@register(
    name="tft",
    family="neural",
    description="Temporal Fusion Transformer: interpretable multi-horizon forecaster",
    aliases=["temporal_fusion_transformer"],
)
class TFTModel(BaseRNNModel):
    """
    Temporal Fusion Transformer classifier with GPU support.

    TFT combines LSTM for local patterns with attention for long-range
    dependencies. It's highly interpretable through:
    - Variable Selection Networks showing feature importance
    - Interpretable attention weights showing temporal focus

    Inherits training infrastructure from BaseRNNModel:
    - GPU training with CUDA (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection
    - AdamW optimizer with cosine annealing
    - Gradient clipping and early stopping

    Key Features:
    - Variable Selection Networks for automatic feature importance
    - LSTM encoder for local temporal dependencies
    - Interpretable Multi-Head Attention for long-range patterns
    - Gated Residual Networks for controlled information flow

    Note on Causality:
        TFT uses bidirectional LSTM and non-causal attention by default.
        For production trading, consider using causal variants.

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("tft", config={
        ...     "d_model": 256,
        ...     "n_heads": 4,
        ...     "lstm_layers": 2,
        ...     "attention_layers": 1,
        ... })
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
        >>> # Get feature importance
        >>> importance = model.get_feature_importance()
    """

    _noncausal_warning_logged: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._noncausal_warning_logged = False
        logger.debug(f"Initialized TFTModel with config: {self._config}")

    @property
    def is_production_safe(self) -> bool:
        """
        TFT uses bidirectional attention, so not production-safe.

        Returns:
            False - TFT is not production-safe for trading.
        """
        return False

    def _log_bidirectional_warning(self) -> None:
        """Log warning about non-causal components (only once)."""
        if self._noncausal_warning_logged:
            return

        logger.warning(
            "TFT NON-CAUSAL COMPONENTS: TFT uses attention that can attend to "
            "all positions in the sequence, including future positions.\n"
            "Implications:\n"
            "  - Attention can focus on future timesteps within the window\n"
            "  - Patterns may not generalize to real-time inference\n"
            "Recommendations:\n"
            "  - For production trading: Use LSTM (bidirectional=False) or TCN\n"
            "  - For research/pattern analysis: TFT provides excellent interpretability"
        )
        self._noncausal_warning_logged = True

    def get_default_config(self) -> dict[str, Any]:
        """Return default TFT hyperparameters."""
        defaults = super().get_default_config()
        defaults.update(
            {
                # Architecture
                "d_model": 256,
                "n_heads": 4,  # Fewer heads for interpretability
                "lstm_layers": 2,
                "attention_layers": 1,  # Single attention layer often sufficient
                "d_ff": 512,
                "dropout": 0.1,
                # Training
                "sequence_length": 60,
                "batch_size": 128,
                "max_epochs": 50,
                "learning_rate": 0.0001,
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
                "early_stopping_patience": 10,
                "warmup_epochs": 3,
            }
        )
        return defaults

    def _create_network(self, input_size: int) -> nn.Module:
        """Create the TFT network."""
        return TFTNetwork(
            input_size=input_size,
            d_model=self._config.get("d_model", 256),
            n_heads=self._config.get("n_heads", 4),
            lstm_layers=self._config.get("lstm_layers", 2),
            attention_layers=self._config.get("attention_layers", 1),
            d_ff=self._config.get("d_ff", 512),
            dropout=self._config.get("dropout", 0.1),
            n_classes=self._n_classes,
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "tft"

    def _on_training_start(self, train_config: dict[str, Any], seq_len: int) -> dict[str, Any]:
        """
        Log TFT-specific information at training start.

        Args:
            train_config: Training configuration
            seq_len: Sequence length of training data

        Returns:
            Dict with metadata for TrainingMetrics
        """
        n_features = self._n_features

        logger.info(
            f"TFT: seq_len={seq_len}, n_features={n_features}, "
            f"d_model={train_config.get('d_model', 256)}, "
            f"lstm_layers={train_config.get('lstm_layers', 2)}, "
            f"attention_layers={train_config.get('attention_layers', 1)}"
        )

        return {
            "lstm_layers": train_config.get("lstm_layers", 2),
            "attention_layers": train_config.get("attention_layers", 1),
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
        batch_size = self._config.get("batch_size", 128)

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
                "model": "tft",
                "d_model": self._config.get("d_model"),
                "n_heads": self._config.get("n_heads"),
                "lstm_layers": self._config.get("lstm_layers"),
                "attention_layers": self._config.get("attention_layers"),
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Return feature importance from Variable Selection Network.

        TFT's VSN learns explicit feature importance weights, making this
        highly interpretable.

        Returns:
            Dict mapping feature indices to importance scores,
            or None if model is not fitted
        """
        if not self._is_fitted:
            return None

        # Get variable weights from last forward pass
        var_weights = self._model.get_variable_weights()

        if var_weights is None:
            # Need to run a forward pass first
            logger.warning(
                "No variable weights available. Run predict() first to compute "
                "feature importance."
            )
            return None

        # Average across batch and sequence
        # var_weights shape: (batch, seq_len, n_features) or (batch, n_features)
        if var_weights.dim() == 3:
            importance = var_weights.mean(dim=[0, 1]).cpu().numpy()
        else:
            importance = var_weights.mean(dim=0).cpu().numpy()

        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def get_temporal_attention(self, X: np.ndarray, sample_idx: int = 0) -> list[np.ndarray] | None:
        """
        Extract temporal attention weights for interpretability.

        Shows which timesteps the model focuses on for predictions.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            sample_idx: Index of sample to extract attention for

        Returns:
            List of attention matrices from each attention layer,
            each with shape (seq_len, seq_len), or None if not fitted
        """
        if not self._is_fitted:
            return None

        self._validate_input_shape(X, "X")

        if sample_idx >= len(X):
            logger.warning(f"sample_idx {sample_idx} >= n_samples {len(X)}, using idx 0")
            sample_idx = 0

        self._model.eval()
        X_tensor = torch.tensor(X[sample_idx : sample_idx + 1], dtype=torch.float32).to(
            self._device
        )

        with torch.no_grad():
            # Forward pass to populate attention weights
            _ = self._model(X_tensor)

            # Extract attention weights
            attention_weights = self._model.get_attention_weights()

            return [
                attn[0].cpu().numpy() if attn is not None else None for attn in attention_weights
            ]

    def get_variable_selection_weights(self, X: np.ndarray) -> np.ndarray | None:
        """
        Get variable selection weights for a batch of inputs.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            Variable weights, shape (n_samples, seq_len, n_features) or
            (n_samples, n_features), or None if not fitted
        """
        if not self._is_fitted:
            return None

        self._validate_input_shape(X, "X")

        self._model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            # Forward pass to compute variable weights
            _ = self._model(X_tensor)
            var_weights = self._model.get_variable_weights()

            if var_weights is not None:
                return var_weights.cpu().numpy()
            return None


__all__ = [
    "TFTModel",
    "TFTNetwork",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "InterpretableMultiHeadAttention",
]
