"""
Neural network model implementations.

Sequential and feedforward neural network models that require
feature scaling and may require sequential input format.

Models:
- LSTM: Long Short-Term Memory networks
- GRU: Gated Recurrent Unit networks
- TCN: Temporal Convolutional Networks
- MLP: Multi-Layer Perceptron (planned)

All models auto-register with ModelRegistry on import.

Example:
    # Models register automatically when imported
    from src.models.neural import LSTMModel, GRUModel, TCNModel

    # Or create via registry
    from src.models import ModelRegistry
    model = ModelRegistry.create("lstm", config={"hidden_size": 256})
    model = ModelRegistry.create("tcn", config={"num_channels": [64, 64, 64, 64]})
"""
from .base_rnn import BaseRNNModel, RNNNetwork
from .lstm_model import LSTMModel, LSTMNetwork
from .gru_model import GRUModel, GRUNetwork
from .tcn_model import TCNModel, TCNNetwork, TemporalBlock, CausalConv1d

__all__ = [
    # Base classes
    "BaseRNNModel",
    "RNNNetwork",
    # LSTM
    "LSTMModel",
    "LSTMNetwork",
    # GRU
    "GRUModel",
    "GRUNetwork",
    # TCN
    "TCNModel",
    "TCNNetwork",
    "TemporalBlock",
    "CausalConv1d",
]
