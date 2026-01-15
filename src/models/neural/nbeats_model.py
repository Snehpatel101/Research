"""
N-BEATS Model - Neural Basis Expansion Analysis for Time Series.

GPU-accelerated N-BEATS with:
- Stack-based architecture with generic/trend/seasonality blocks
- Interpretable basis expansion (trend: polynomial, seasonality: Fourier)
- Backcast/forecast mechanism with residual learning
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)

Reference:
    Oreshkin et al. "N-BEATS: Neural basis expansion analysis for
    interpretable time series forecasting" (ICLR 2020)

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


# =============================================================================
# N-BEATS BUILDING BLOCKS
# =============================================================================


class NBEATSBlock(nn.Module):
    """
    Base N-BEATS block with fully connected layers.

    Each block produces:
    - backcast: reconstruction of the input (used for residual)
    - forecast: contribution to the final prediction

    Architecture:
        Input -> FC stack -> theta_backcast, theta_forecast
        -> basis expansion -> backcast, forecast
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_size: int,
        n_layers: int,
        basis_function: nn.Module,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size

        # Build FC stack
        layers = []
        for i in range(n_layers):
            in_features = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.fc_stack = nn.Sequential(*layers)

        # Theta layers for backcast and forecast
        self.theta_backcast = nn.Linear(hidden_size, theta_size)
        self.theta_forecast = nn.Linear(hidden_size, theta_size)

        # Basis function (identity, trend, or seasonality)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, input_size)

        Returns:
            backcast: Reconstruction, shape (batch, input_size)
            forecast: Forecast contribution, shape (batch, forecast_size)
        """
        # FC stack
        h = self.fc_stack(x)

        # Compute theta coefficients
        theta_b = self.theta_backcast(h)
        theta_f = self.theta_forecast(h)

        # Apply basis function
        backcast, forecast = self.basis_function(theta_b, theta_f)

        return backcast, forecast


class GenericBasis(nn.Module):
    """
    Generic basis function using learnable linear projections.

    For the generic stack, we use simple linear layers without
    explicit structure (polynomial/Fourier).
    """

    def __init__(self, theta_size: int, backcast_size: int, forecast_size: int) -> None:
        super().__init__()
        self.backcast_fc = nn.Linear(theta_size, backcast_size, bias=False)
        self.forecast_fc = nn.Linear(theta_size, forecast_size, bias=False)

    def forward(
        self, theta_backcast: torch.Tensor, theta_forecast: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backcast_fc(theta_backcast), self.forecast_fc(theta_forecast)


class TrendBasis(nn.Module):
    """
    Trend basis function using polynomial expansion.

    Uses Legendre-like polynomial basis:
        T(t) = sum(theta_i * t^i) for i in [0, degree]

    This captures monotonic trends, linear/quadratic patterns.
    """

    def __init__(
        self,
        degree: int,
        backcast_size: int,
        forecast_size: int,
    ) -> None:
        super().__init__()
        self.degree = degree
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

        # Precompute polynomial basis matrices
        backcast_basis = self._create_polynomial_basis(backcast_size, degree)
        forecast_basis = self._create_polynomial_basis(forecast_size, degree)

        # Register as buffers (not trainable, but saved with model)
        self.register_buffer("backcast_basis", backcast_basis)
        self.register_buffer("forecast_basis", forecast_basis)

    def _create_polynomial_basis(self, size: int, degree: int) -> torch.Tensor:
        """Create polynomial basis matrix: (size, degree+1)."""
        t = torch.linspace(0, 1, size).unsqueeze(1)  # (size, 1)
        powers = torch.arange(degree + 1).float()  # (degree+1,)
        basis = t**powers  # (size, degree+1)
        return basis.T  # (degree+1, size)

    def forward(
        self, theta_backcast: torch.Tensor, theta_forecast: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # theta: (batch, degree+1)
        # basis: (degree+1, size)
        # output: (batch, size)
        backcast = torch.matmul(theta_backcast, self.backcast_basis)
        forecast = torch.matmul(theta_forecast, self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """
    Seasonality basis function using Fourier expansion.

    Uses Fourier basis:
        S(t) = sum(a_i * cos(2*pi*i*t) + b_i * sin(2*pi*i*t))

    This captures periodic patterns at multiple frequencies.
    """

    def __init__(
        self,
        n_harmonics: int,
        backcast_size: int,
        forecast_size: int,
    ) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

        # Precompute Fourier basis matrices
        backcast_basis = self._create_fourier_basis(backcast_size, n_harmonics)
        forecast_basis = self._create_fourier_basis(forecast_size, n_harmonics)

        self.register_buffer("backcast_basis", backcast_basis)
        self.register_buffer("forecast_basis", forecast_basis)

    def _create_fourier_basis(self, size: int, n_harmonics: int) -> torch.Tensor:
        """Create Fourier basis matrix: (2*n_harmonics, size)."""
        t = torch.linspace(0, 1, size)  # (size,)
        basis_functions = []

        for i in range(1, n_harmonics + 1):
            # Cosine component
            basis_functions.append(torch.cos(2 * math.pi * i * t))
            # Sine component
            basis_functions.append(torch.sin(2 * math.pi * i * t))

        basis = torch.stack(basis_functions, dim=0)  # (2*n_harmonics, size)
        return basis

    def forward(
        self, theta_backcast: torch.Tensor, theta_forecast: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # theta: (batch, 2*n_harmonics)
        # basis: (2*n_harmonics, size)
        # output: (batch, size)
        backcast = torch.matmul(theta_backcast, self.backcast_basis)
        forecast = torch.matmul(theta_forecast, self.forecast_basis)
        return backcast, forecast


class NBEATSStack(nn.Module):
    """
    N-BEATS stack: sequence of blocks with residual connections.

    The stack applies blocks sequentially, where each block:
    1. Receives the residual from the previous block
    2. Produces backcast (subtracted for next block) and forecast (accumulated)
    """

    def __init__(
        self,
        n_blocks: int,
        input_size: int,
        theta_size: int,
        hidden_size: int,
        n_layers: int,
        basis_function: nn.Module,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                NBEATSBlock(
                    input_size=input_size,
                    theta_size=theta_size,
                    hidden_size=hidden_size,
                    n_layers=n_layers,
                    basis_function=basis_function,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the stack.

        Args:
            x: Input tensor, shape (batch, input_size)

        Returns:
            residual: Final residual after all blocks, shape (batch, input_size)
            forecast: Sum of all block forecasts, shape (batch, forecast_size)
        """
        residual = x
        forecast = None

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast

            if forecast is None:
                forecast = block_forecast
            else:
                forecast = forecast + block_forecast

        return residual, forecast


# =============================================================================
# N-BEATS NETWORK
# =============================================================================


class NBEATSNetwork(nn.Module):
    """
    N-BEATS network for sequence classification.

    Adapts N-BEATS (originally for forecasting) to classification by:
    1. Processing input sequence through interpretable stacks
    2. Using the forecast output as feature representation
    3. Applying classification head

    Architecture:
        Input (batch, seq_len, features)
        -> Flatten to (batch, seq_len * features) or pool features
        -> Generic Stack -> Trend Stack -> Seasonality Stack
        -> Accumulate forecasts
        -> Classification head -> n_classes
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        n_stacks: int = 3,
        n_blocks_per_stack: int = 3,
        hidden_size: int = 256,
        n_layers: int = 4,
        theta_size: int = 32,
        dropout: float = 0.1,
        stack_types: list[str] | None = None,
        n_harmonics: int = 4,
        polynomial_degree: int = 3,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.n_stacks = n_stacks

        # Default stack types: generic, trend, seasonality
        if stack_types is None:
            stack_types = ["generic", "trend", "seasonality"][:n_stacks]
        elif len(stack_types) < n_stacks:
            # Pad with generic stacks
            stack_types = stack_types + ["generic"] * (n_stacks - len(stack_types))

        self.stack_types = stack_types[:n_stacks]

        # Input dimension after feature pooling
        self.input_dim = seq_len

        # Feature aggregation: pool across features
        self.feature_pool = nn.Sequential(
            nn.Linear(input_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
        )

        # Build stacks
        self.stacks = nn.ModuleList()
        for stack_type in self.stack_types:
            # Compute theta_size based on stack type
            stack_theta_size = self._get_theta_size(
                stack_type=stack_type,
                generic_theta_size=theta_size,
                n_harmonics=n_harmonics,
                polynomial_degree=polynomial_degree,
            )

            basis_function = self._create_basis(
                stack_type=stack_type,
                theta_size=stack_theta_size,
                input_dim=self.input_dim,
                n_harmonics=n_harmonics,
                polynomial_degree=polynomial_degree,
            )

            stack = NBEATSStack(
                n_blocks=n_blocks_per_stack,
                input_size=self.input_dim,
                theta_size=stack_theta_size,
                hidden_size=hidden_size,
                n_layers=n_layers,
                basis_function=basis_function,
                dropout=dropout,
            )
            self.stacks.append(stack)

        # Classification head
        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.input_dim, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, n_classes)

        # Store stack outputs for interpretability
        self._stack_forecasts: list[torch.Tensor] | None = None

    def _get_theta_size(
        self,
        stack_type: str,
        generic_theta_size: int,
        n_harmonics: int,
        polynomial_degree: int,
    ) -> int:
        """
        Get the theta size for a given stack type.

        Each basis function has specific requirements:
        - Generic: uses configurable theta_size
        - Trend: needs polynomial_degree + 1 coefficients
        - Seasonality: needs 2 * n_harmonics coefficients (sin + cos)
        """
        if stack_type == "trend":
            return polynomial_degree + 1
        elif stack_type == "seasonality":
            return 2 * n_harmonics
        else:  # generic
            return generic_theta_size

    def _create_basis(
        self,
        stack_type: str,
        theta_size: int,
        input_dim: int,
        n_harmonics: int,
        polynomial_degree: int,
    ) -> nn.Module:
        """Create appropriate basis function for stack type."""
        if stack_type == "trend":
            return TrendBasis(
                degree=polynomial_degree,
                backcast_size=input_dim,
                forecast_size=input_dim,
            )
        elif stack_type == "seasonality":
            actual_theta_size = 2 * n_harmonics
            return SeasonalityBasis(
                n_harmonics=n_harmonics,
                backcast_size=input_dim,
                forecast_size=input_dim,
            )
        else:  # generic
            return GenericBasis(
                theta_size=theta_size,
                backcast_size=input_dim,
                forecast_size=input_dim,
            )

    def forward(
        self, x: torch.Tensor, return_components: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, features)
            return_components: If True, return stack-wise forecasts

        Returns:
            Output logits, shape (batch, n_classes)
            Optionally: list of stack forecasts for interpretability
        """
        batch_size = x.size(0)

        # Pool features: (batch, seq_len, features) -> (batch, seq_len)
        x_pooled = self.feature_pool(x).squeeze(-1)

        # Process through stacks with residual connections
        residual = x_pooled
        total_forecast = torch.zeros_like(x_pooled)
        stack_forecasts = []

        for stack in self.stacks:
            residual, forecast = stack(residual)
            total_forecast = total_forecast + forecast
            stack_forecasts.append(forecast.detach())

        self._stack_forecasts = stack_forecasts

        # Classification head
        out = self.layer_norm(total_forecast)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)

        if return_components:
            return logits, stack_forecasts
        return logits


# =============================================================================
# N-BEATS MODEL
# =============================================================================


@register(
    name="nbeats",
    family="neural",
    description="N-BEATS: Neural Basis Expansion Analysis for interpretable time series",
    aliases=["n_beats", "neural_basis_expansion"],
)
class NBEATSModel(BaseRNNModel):
    """
    N-BEATS classifier with GPU support.

    Inherits training infrastructure from BaseRNNModel:
    - GPU training with CUDA (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection:
      - bfloat16 for Ampere+ (RTX 30xx/40xx, A100, H100)
      - float16 for Volta/Turing (RTX 20xx, GTX 16xx, T4, V100)
      - float32 for older GPUs or CPU
    - AdamW optimizer with cosine annealing
    - Gradient clipping and early stopping

    Features:
    - Interpretable stack-based architecture (trend, seasonality, generic)
    - Polynomial and Fourier basis expansions
    - Residual learning through backcast mechanism
    - Stack-wise decomposition for explainability

    Note on Causality:
        N-BEATS processes the entire input sequence with fully connected layers,
        which is non-causal (each position sees all positions). For strict
        causality in production trading, consider TCN or LSTM with bidirectional=False.

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("nbeats", config={
        ...     "n_stacks": 3,
        ...     "n_blocks_per_stack": 3,
        ...     "hidden_size": 256,
        ... })
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._seq_len: int | None = None
        logger.debug(f"Initialized NBEATSModel with config: {self._config}")

    @property
    def is_production_safe(self) -> bool:
        """
        Check if this model configuration is safe for production trading.

        N-BEATS uses fully connected layers that process the entire sequence
        without causal masking, so it is non-causal.

        Returns:
            False - N-BEATS is not inherently causal.
        """
        return False

    def _log_bidirectional_warning(self) -> None:
        """Log a warning about non-causal processing (only once)."""
        if self._bidirectional_warning_logged:
            return

        logger.warning(
            "N-BEATS NON-CAUSAL PROCESSING: N-BEATS uses fully connected layers that "
            "process the entire input sequence without causal masking. Each position "
            "has access to all other positions in the window.\n"
            "Implications:\n"
            "  - Predictions use information from later timesteps in the window\n"
            "  - Patterns learned may not be available during real-time inference\n"
            "Recommendations:\n"
            "  - For production trading: Use TCN or LSTM (bidirectional=False)\n"
            "  - For research/pattern analysis: N-BEATS is acceptable"
        )
        self._bidirectional_warning_logged = True

    def get_default_config(self) -> dict[str, Any]:
        """Return default N-BEATS hyperparameters."""
        defaults = super().get_default_config()
        defaults.update(
            {
                # Architecture
                "n_stacks": 3,
                "n_blocks_per_stack": 3,
                "hidden_size": 256,
                "n_layers": 4,
                "theta_size": 32,
                "dropout": 0.1,
                "stack_types": ["generic", "trend", "seasonality"],
                "n_harmonics": 4,
                "polynomial_degree": 3,
                # Training
                "sequence_length": 60,
                "batch_size": 128,
                "max_epochs": 100,
                "learning_rate": 0.0005,
                "weight_decay": 0.0001,
                "gradient_clip": 1.0,
                "early_stopping_patience": 15,
                "warmup_epochs": 5,
            }
        )
        return defaults

    def _create_network(self, input_size: int) -> nn.Module:
        """Create the N-BEATS network."""
        return NBEATSNetwork(
            input_size=input_size,
            seq_len=self._seq_len,
            n_stacks=self._config.get("n_stacks", 3),
            n_blocks_per_stack=self._config.get("n_blocks_per_stack", 3),
            hidden_size=self._config.get("hidden_size", 256),
            n_layers=self._config.get("n_layers", 4),
            theta_size=self._config.get("theta_size", 32),
            dropout=self._config.get("dropout", 0.1),
            stack_types=self._config.get("stack_types"),
            n_harmonics=self._config.get("n_harmonics", 4),
            polynomial_degree=self._config.get("polynomial_degree", 3),
            n_classes=self._n_classes,
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "nbeats"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Train the N-BEATS model with early stopping."""
        # Store sequence length for network creation
        self._seq_len = X_train.shape[1]
        return super().fit(X_train, y_train, X_val, y_val, sample_weights, config)

    def _on_training_start(self, train_config: dict[str, Any], seq_len: int) -> dict[str, Any]:
        """
        Log N-BEATS-specific information at training start.

        Args:
            train_config: Training configuration dictionary
            seq_len: Sequence length of training data

        Returns:
            Dict with N-BEATS metadata for TrainingMetrics
        """
        stack_types = train_config.get("stack_types", ["generic", "trend", "seasonality"])
        n_stacks = train_config.get("n_stacks", 3)
        n_blocks = train_config.get("n_blocks_per_stack", 3)

        logger.info(
            f"N-BEATS architecture: {n_stacks} stacks x {n_blocks} blocks, "
            f"types={stack_types[:n_stacks]}, seq_len={seq_len}"
        )

        return {
            "stack_types": stack_types[:n_stacks],
            "n_stacks": n_stacks,
            "n_blocks_per_stack": n_blocks,
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
                "model": "nbeats",
                "n_stacks": self._config.get("n_stacks"),
                "stack_types": self._config.get("stack_types"),
            },
        )

    def get_stack_decomposition(
        self, X: np.ndarray, sample_idx: int = 0
    ) -> dict[str, np.ndarray] | None:
        """
        Get stack-wise decomposition for interpretability.

        Returns the forecast contribution from each stack (trend, seasonality, etc.)

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            sample_idx: Index of sample to analyze

        Returns:
            Dictionary mapping stack names to their forecast contributions,
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
            _, stack_forecasts = self._model(X_tensor, return_components=True)

        stack_types = self._config.get("stack_types", ["generic", "trend", "seasonality"])
        n_stacks = self._config.get("n_stacks", 3)

        result = {}
        for i, forecast in enumerate(stack_forecasts):
            stack_name = stack_types[i] if i < len(stack_types) else f"stack_{i}"
            result[stack_name] = forecast[0].cpu().numpy()

        return result


__all__ = [
    "NBEATSModel",
    "NBEATSNetwork",
    "NBEATSStack",
    "NBEATSBlock",
    "GenericBasis",
    "TrendBasis",
    "SeasonalityBasis",
]
