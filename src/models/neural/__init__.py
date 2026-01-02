"""
Neural network model implementations.

Sequential and feedforward neural network models that require
feature scaling and may require sequential input format.

Models:
- LSTM: Long Short-Term Memory networks
- GRU: Gated Recurrent Unit networks
- TCN: Temporal Convolutional Networks
- Transformer: Vanilla Transformer encoder with self-attention
- PatchTST: Patched Time Series Transformer (segment input into patches)
- iTransformer: Inverted Transformer (attention over features, not time)
- TFT: Temporal Fusion Transformer (interpretable multi-horizon forecasting)
- N-BEATS: Neural Basis Expansion Analysis for Time Series
- InceptionTime: Inception-based CNN for time series classification
- ResNet1D: 1D ResNet with residual blocks

All models auto-register with ModelRegistry on import.

Example:
    # Models register automatically when imported
    from src.models.neural import LSTMModel, GRUModel, TCNModel, TransformerModel
    from src.models.neural import NBEATSModel, InceptionTimeModel, ResNet1DModel
    from src.models.neural import PatchTSTModel, iTransformerModel, TFTModel

    # Or create via registry
    from src.models import ModelRegistry
    model = ModelRegistry.create("lstm", config={"hidden_size": 256})
    model = ModelRegistry.create("tcn", config={"num_channels": [64, 64, 64, 64]})
    model = ModelRegistry.create("transformer", config={"d_model": 256, "n_heads": 8})
    model = ModelRegistry.create("patchtst", config={"patch_len": 16, "stride": 8})
    model = ModelRegistry.create("itransformer", config={"d_model": 256})
    model = ModelRegistry.create("tft", config={"lstm_layers": 2, "attention_layers": 1})
    model = ModelRegistry.create("nbeats", config={"n_stacks": 3})
    model = ModelRegistry.create("inceptiontime", config={"n_blocks": 6})
    model = ModelRegistry.create("resnet1d", config={"channels": (64, 128, 256, 512)})
"""

from .base_rnn import BaseRNNModel, RNNNetwork
from .cnn import (
    InceptionBlock,
    InceptionModule,
    InceptionTimeModel,
    InceptionTimeNetwork,
    ResidualBlock1D,
    ResidualBlock1DBottleneck,
    ResNet1DModel,
    ResNet1DNetwork,
)
from .gru_model import GRUModel, GRUNetwork
from .lstm_model import LSTMModel, LSTMNetwork
from .nbeats import (
    GenericBasis,
    NBEATSBlock,
    NBEATSModel,
    NBEATSNetwork,
    NBEATSStack,
    SeasonalityBasis,
    TrendBasis,
)
from .tcn_model import CausalConv1d, TCNModel, TCNNetwork, TemporalBlock
from .transformer_model import PositionalEncoding, TransformerModel, TransformerNetwork
from .patchtst_model import (
    PatchTSTModel,
    PatchTSTNetwork,
    PatchEmbedding,
    LearnablePositionalEncoding,
)
from .itransformer_model import (
    iTransformerModel,
    iTransformerNetwork,
    TemporalEmbedding,
    FeaturePositionalEncoding,
)
from .tft_model import (
    TFTModel,
    TFTNetwork,
    GatedLinearUnit,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
)

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
    # Transformer
    "TransformerModel",
    "TransformerNetwork",
    "PositionalEncoding",
    # PatchTST
    "PatchTSTModel",
    "PatchTSTNetwork",
    "PatchEmbedding",
    "LearnablePositionalEncoding",
    # iTransformer
    "iTransformerModel",
    "iTransformerNetwork",
    "TemporalEmbedding",
    "FeaturePositionalEncoding",
    # TFT
    "TFTModel",
    "TFTNetwork",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "InterpretableMultiHeadAttention",
    # N-BEATS
    "NBEATSModel",
    "NBEATSNetwork",
    "NBEATSStack",
    "NBEATSBlock",
    "GenericBasis",
    "TrendBasis",
    "SeasonalityBasis",
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
