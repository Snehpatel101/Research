"""
iTransformer Model - Inverted Transformer for 3-class prediction.

GPU-accelerated iTransformer with:
- Inverted attention: attention over features (channels) instead of time
- Each feature becomes a token with temporal embedding
- Cross-feature attention captures feature correlations
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective
for Time Series Forecasting" (ICLR 2024)

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


class TemporalEmbedding(nn.Module):
    """
    Temporal embedding layer for iTransformer.

    Embeds the temporal dimension of each feature (channel) into a fixed-size
    representation. Uses a 1D convolution followed by linear projection.

    This converts input from (batch, seq_len, features) to (batch, features, d_model)
    where each feature becomes a token with temporal information embedded.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Temporal projection: embed seq_len timesteps into d_model
        self.temporal_proj = nn.Linear(seq_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create temporal embeddings for each feature.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Feature tokens with temporal embedding, shape (batch, features, d_model)
        """
        # Transpose to (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Project temporal dimension to d_model: (batch, features, d_model)
        x = self.temporal_proj(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x


class FeaturePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for feature tokens.

    In iTransformer, positions correspond to different features (channels),
    not timesteps. This captures the ordering/relationship between features.
    """

    def __init__(
        self,
        d_model: int,
        max_features: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable feature position embeddings
        self.pe = nn.Parameter(torch.zeros(1, max_features, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to feature tokens.

        Args:
            x: Feature tokens, shape (batch, n_features, d_model)

        Returns:
            Position-encoded tokens, shape (batch, n_features, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class iTransformerNetwork(nn.Module):
    """
    iTransformer network architecture for sequence classification.

    Architecture:
        Input (batch, seq_len, features)
        -> Temporal embedding: (batch, features, d_model)
           (Each feature becomes a token with embedded temporal info)
        -> Feature positional encoding
        -> Transformer encoder (attention over features)
        -> Global average pooling over features
        -> LayerNorm + Dropout
        -> Linear classifier -> 3 classes

    Key insight:
        Standard transformers apply attention over time positions. iTransformer
        inverts this - applying attention over features. This allows the model
        to learn cross-feature correlations effectively, which is often more
        important in multivariate time series than long-range temporal patterns.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        activation: str = "gelu",
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Temporal embedding: project each feature's temporal sequence to d_model
        self.temporal_embed = TemporalEmbedding(seq_len, d_model, dropout)

        # Feature positional encoding
        self.feature_pos = FeaturePositionalEncoding(
            d_model, max_features=input_size, dropout=dropout
        )

        # Transformer encoder (attention over features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-LN architecture
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Classification head
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_cls = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.gelu = nn.GELU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, n_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through iTransformer.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Output logits, shape (batch, n_classes)
        """
        # Temporal embedding: (batch, seq_len, features) -> (batch, features, d_model)
        x = self.temporal_embed(x)

        # Add feature positional encoding
        x = self.feature_pos(x)

        # Transformer encoder with attention over features
        x = self.transformer_encoder(x)  # (batch, features, d_model)

        # Global average pooling over features
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification head
        x = self.layer_norm(x)
        x = self.dropout_cls(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

    def get_feature_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights between features.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Attention weights, shape (n_layers, batch, n_heads, n_features, n_features)
        """
        # Get temporal embeddings
        x = self.temporal_embed(x)
        x = self.feature_pos(x)

        attention_weights = []

        # Iterate through encoder layers to capture attention
        for layer in self.transformer_encoder.layers:
            attn_output, attn_weights = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            attention_weights.append(attn_weights.detach())

            # Complete the layer forward pass
            x = layer.norm1(x + layer.dropout1(attn_output))
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff_output))

        return torch.stack(attention_weights, dim=0)


@register(
    name="itransformer",
    family="neural",
    description="iTransformer: Inverted Transformer with attention over features",
    aliases=["i_transformer", "inverted_transformer"],
)
class iTransformerModel(BaseRNNModel):
    """
    iTransformer classifier with GPU support.

    iTransformer inverts the attention mechanism of standard transformers:
    instead of attending over time positions, it attends over features.
    This allows the model to learn cross-feature correlations effectively.

    Inherits training infrastructure from BaseRNNModel:
    - GPU training with CUDA (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection
    - AdamW optimizer with cosine annealing
    - Gradient clipping and early stopping

    Key Features:
    - Inverted attention: attends over features, not time
    - Temporal embedding: projects each feature's time series to d_model
    - Effective for multivariate time series with many correlated features
    - Typically needs fewer layers than standard transformers

    Note on Sequence Length:
        iTransformer is sensitive to sequence length since it's used in the
        temporal embedding. Changing seq_len at inference requires re-training.

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("itransformer", config={
        ...     "d_model": 256,
        ...     "n_heads": 8,
        ...     "n_layers": 2,
        ...     "sequence_length": 60,
        ... })
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    _seq_len_set: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._seq_len: int | None = None
        logger.debug(f"Initialized iTransformerModel with config: {self._config}")

    @property
    def is_production_safe(self) -> bool:
        """
        iTransformer processes all timesteps jointly in temporal embedding.

        Since the temporal embedding uses the full sequence, this model
        is not strictly causal. However, it does not have attention over
        time (only over features), so it's somewhat different from standard
        transformers.

        Returns:
            False - for consistency with other transformer models.
        """
        return False

    def _log_bidirectional_warning(self) -> None:
        """iTransformer doesn't have traditional bidirectional concerns."""
        pass  # Attention is over features, not time

    def get_default_config(self) -> dict[str, Any]:
        """Return default iTransformer hyperparameters."""
        defaults = super().get_default_config()
        defaults.update(
            {
                # Architecture
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 2,  # Fewer layers often sufficient
                "d_ff": 512,
                "dropout": 0.1,
                "activation": "gelu",
                # Training
                "sequence_length": 60,  # Must be fixed at training
                "batch_size": 256,
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
        """Create the iTransformer network."""
        # Get sequence length from training data or config
        seq_len = self._seq_len or self._config.get("sequence_length", 60)

        return iTransformerNetwork(
            input_size=input_size,
            seq_len=seq_len,
            d_model=self._config.get("d_model", 256),
            n_heads=self._config.get("n_heads", 8),
            n_layers=self._config.get("n_layers", 2),
            d_ff=self._config.get("d_ff", 512),
            dropout=self._config.get("dropout", 0.1),
            activation=self._config.get("activation", "gelu"),
            n_classes=self._n_classes,
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "itransformer"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Train the iTransformer model.

        Overrides parent to capture and store sequence length from training data.

        Args:
            X_train: Training features, shape (n_samples, seq_len, n_features)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_weights: Optional sample weights
            config: Optional config overrides

        Returns:
            TrainingMetrics with training results
        """
        # Store sequence length from training data
        self._seq_len = X_train.shape[1]

        return super().fit(X_train, y_train, X_val, y_val, sample_weights, config)

    def _on_training_start(self, train_config: dict[str, Any], seq_len: int) -> dict[str, Any]:
        """
        Log iTransformer-specific information at training start.

        Args:
            train_config: Training configuration
            seq_len: Sequence length of training data

        Returns:
            Dict with metadata for TrainingMetrics
        """
        n_features = self._n_features

        logger.info(
            f"iTransformer: seq_len={seq_len}, n_features={n_features}, "
            f"d_model={train_config.get('d_model', 256)}, "
            f"n_layers={train_config.get('n_layers', 2)}"
        )

        if n_features > 256:
            logger.warning(
                f"Large number of features ({n_features}). iTransformer attention "
                f"complexity is O(n_features^2). Consider feature selection."
            )

        return {
            "seq_len_embedded": seq_len,
            "n_feature_tokens": n_features,
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

        # Validate sequence length matches training
        if X.shape[1] != self._seq_len:
            raise ValueError(
                f"Input sequence length ({X.shape[1]}) does not match "
                f"training sequence length ({self._seq_len}). "
                f"iTransformer requires fixed sequence length."
            )

        self._model.eval()
        amp_dtype = self._amp_dtype

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        all_probs = []
        batch_size = self._config.get("batch_size", 256)

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
                "model": "itransformer",
                "d_model": self._config.get("d_model"),
                "n_heads": self._config.get("n_heads"),
                "n_layers": self._config.get("n_layers"),
                "seq_len": self._seq_len,
            },
        )

    def save(self, path) -> None:
        """Save model with sequence length metadata."""
        self._validate_fitted()
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "config": self._config,
                "n_features": self._n_features,
                "n_classes": self._n_classes,
                "seq_len": self._seq_len,  # Store sequence length
            },
            path / "model.pt",
        )

        logger.info(f"Saved iTransformer model to {path}")

    def load(self, path) -> None:
        """Load model with sequence length metadata."""
        from pathlib import Path

        path = Path(path)
        model_path = path / "model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)

        self._config = checkpoint["config"]
        self._n_features = checkpoint["n_features"]
        self._n_classes = checkpoint["n_classes"]
        self._seq_len = checkpoint["seq_len"]  # Restore sequence length

        # Recreate and load model
        self._model = self._create_network(self._n_features)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(self._device)
        self._model.eval()

        self._is_fitted = True
        logger.info(f"Loaded iTransformer model from {path} (seq_len={self._seq_len})")

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Return feature importance based on temporal embedding weights.

        For iTransformer, we analyze how much each feature contributes
        to the final representation by examining the temporal embedding.

        Returns:
            Dict mapping feature indices to importance scores,
            or None if model is not fitted
        """
        if not self._is_fitted:
            return None

        # Get temporal projection weights: (d_model, seq_len)
        weights = self._model.temporal_embed.temporal_proj.weight.detach().cpu().numpy()

        # L2 norm across d_model dimension gives temporal importance
        # Then average across features (each feature uses same projection)
        temporal_importance = np.linalg.norm(weights, axis=0)

        # For feature importance, we use the feature position embeddings
        # Get position embeddings: (1, max_features, d_model)
        pos_embed = self._model.feature_pos.pe.detach().cpu().numpy()[0]

        # Only use embeddings for actual features
        pos_embed = pos_embed[: self._n_features]

        # L2 norm of each feature's position embedding
        feature_importance = np.linalg.norm(pos_embed, axis=1)

        # Normalize
        feature_importance = feature_importance / feature_importance.sum()

        return {f"feature_{i}": float(imp) for i, imp in enumerate(feature_importance)}

    def get_feature_attention_matrix(self, X: np.ndarray, sample_idx: int = 0) -> np.ndarray | None:
        """
        Extract feature-to-feature attention weights.

        Unlike standard transformers, iTransformer attention shows
        how features attend to each other, revealing correlations.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            sample_idx: Index of sample to extract attention for

        Returns:
            Attention matrix, shape (n_layers, n_heads, n_features, n_features)
            or None if model is not fitted
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
            attention = self._model.get_feature_attention(X_tensor)
            # Shape: (n_layers, 1, n_heads, n_features, n_features)
            return attention[:, 0, :, :, :].cpu().numpy()


__all__ = [
    "iTransformerModel",
    "iTransformerNetwork",
    "TemporalEmbedding",
    "FeaturePositionalEncoding",
]
