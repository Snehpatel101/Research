"""
CNN Models for Time Series - InceptionTime and ResNet1D.

This module provides backward compatibility re-exports.
The actual implementations are in:
- inceptiontime_model.py: InceptionTime components
- resnet1d_model.py: ResNet1D components

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

# Re-export from separate model files for backward compatibility
from .inceptiontime_model import (
    InceptionModule,
    InceptionBlock,
    InceptionTimeNetwork,
    InceptionTimeModel,
)
from .resnet1d_model import (
    ResidualBlock1D,
    ResidualBlock1DBottleneck,
    ResNet1DNetwork,
    ResNet1DModel,
)

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
