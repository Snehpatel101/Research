"""
Shared CNN utilities and base components.

Common imports, utilities, and base classes used by CNN models
(InceptionTime and ResNet1D).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import PredictionResult
from ..registry import register
from .base_rnn import BaseRNNModel

logger = logging.getLogger(__name__)

__all__ = [
    # Re-exports for convenience
    "logger",
    "np",
    "torch",
    "nn",
    "F",
    "Any",
    "PredictionResult",
    "register",
    "BaseRNNModel",
]
