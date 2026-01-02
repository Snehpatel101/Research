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
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseModel, PredictionOutput, TrainingMetrics
from ..common import map_classes_to_labels, map_labels_to_classes
from ..device import get_amp_dtype, get_best_gpu, get_mixed_precision_config

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
        self.rnn: nn.Module | None = None

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

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: nn.Module | None = None
        self._n_classes: int = 3
        self._n_features: int | None = None
        self._bidirectional_warning_logged: bool = False

        # Device setup with "auto" detection support
        device_config = self._config.get("device", "auto")
        if device_config == "auto":
            self._device = _get_device(use_cuda=True)
        elif device_config == "cuda":
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
            "batch_size": 256,
            "max_epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "early_stopping_patience": 15,
            "min_delta": 0.0001,
            "warmup_epochs": 5,
            "device": "auto",  # Auto-detect GPU/CPU
            "mixed_precision": True,  # Use GPU-appropriate precision
            "num_workers": 4,
            "pin_memory": True,
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

        # Log bidirectional warning if applicable (only logged once per model)
        self._log_bidirectional_warning()

        # Call training start hook for subclass-specific setup/logging
        extra_metadata = self._on_training_start(train_config, seq_len)

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

        # Training state
        early_stopping = EarlyStoppingState()
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        max_epochs = train_config.get("max_epochs", 100)
        patience = train_config.get("early_stopping_patience", 15)
        min_delta = train_config.get("min_delta", 0.0001)
        grad_clip = train_config.get("gradient_clip", 1.0)

        logger.info(
            f"Training {self._get_model_type().upper()}: "
            f"epochs={max_epochs}, batch_size={train_config.get('batch_size')}, "
            f"hidden={train_config.get('hidden_size')}, layers={train_config.get('num_layers')}, "
            f"mixed_precision={'on' if self._use_amp else 'off'}"
        )

        for epoch in range(max_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion, scheduler, scaler, amp_dtype, grad_clip
            )
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion, amp_dtype)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{max_epochs} - "
                    f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                    f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"
                )

            # Early stopping check
            if early_stopping.check(val_loss, epoch, self._model, patience, min_delta):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if early_stopping.best_state_dict is not None:
            self._model.load_state_dict(early_stopping.best_state_dict)

        training_time = time.time() - start_time
        epochs_trained = len(history["train_loss"])

        # Compute final metrics
        train_metrics = self._compute_final_metrics(train_loader, amp_dtype, y_train)
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

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions with class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        self._model.eval()
        amp_dtype = self._amp_dtype

        # Convert to tensor
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

        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)

        self._config = checkpoint["config"]
        self._n_features = checkpoint["n_features"]
        self._n_classes = checkpoint["n_classes"]

        # Recreate and load model
        self._model = self._create_network(self._n_features)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(self._device)
        self._model.eval()

        self._is_fitted = True
        logger.info(f"Loaded {self._get_model_type().upper()} model from {path}")

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None,
        config: dict[str, Any],
        shuffle: bool,
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(self._convert_labels_to_class(y), dtype=torch.long)

        if sample_weights is not None:
            weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)

        return DataLoader(
            dataset,
            batch_size=config.get("batch_size", 256),
            shuffle=shuffle,
            num_workers=config.get("num_workers", 4),
            pin_memory=config.get("pin_memory", True) and self._device.type == "cuda",
            drop_last=False,
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
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create cosine annealing scheduler with warmup."""
        max_epochs = config.get("max_epochs", 100)
        warmup_epochs = config.get("warmup_epochs", 5)

        total_steps = max_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler | None,
        amp_dtype: torch.dtype,
        grad_clip: float,
    ) -> tuple[float, float]:
        """Run one training epoch."""
        self._model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            if len(batch) == 3:
                X_batch, y_batch, weights = batch
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                weights = weights.to(self._device)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                weights = None

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
                logits = self._model(X_batch)
                if weights is not None:
                    # Use reduction='none' to get per-sample losses for weighting
                    criterion_unreduced = nn.CrossEntropyLoss(reduction="none")
                    per_sample_loss = criterion_unreduced(logits, y_batch)
                    loss = (per_sample_loss * weights).mean()
                else:
                    loss = criterion(logits, y_batch)

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()

            # Track metrics
            total_loss += loss.item() * len(y_batch)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)

        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total

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

        with torch.no_grad():
            for batch in loader:
                X_batch, y_batch = batch[0], batch[1]
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

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
        return map_labels_to_classes(labels)

    def _convert_labels_from_class(self, labels: np.ndarray) -> np.ndarray:
        return map_classes_to_labels(labels)


__all__ = [
    "BaseRNNModel",
    "RNNNetwork",
    "EarlyStoppingState",
]
