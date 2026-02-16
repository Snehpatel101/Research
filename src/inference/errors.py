"""
Inference error types with shape-mismatch diagnostics.

Provides structured error messages for common inference failures,
especially adapter routing and tensor shape mismatches.
"""

from __future__ import annotations


class InferenceError(Exception):
    """Base class for inference errors."""


class ShapeMismatchError(InferenceError):
    """Raised when tensor shapes don't match model expectations.

    Provides diagnostic messages showing expected vs actual shapes
    and suggestions for resolution.
    """

    def __init__(
        self,
        expected_rank: int,
        actual_shape: tuple[int, ...],
        model_name: str,
        *,
        context: str = "",
    ) -> None:
        self.expected_rank = expected_rank
        self.actual_shape = actual_shape
        self.model_name = model_name
        self.context = context

        rank_labels = {
            2: "tabular (N, F)",
            3: "sequence (N, T, F)",
            4: "multi-stream (N, TF, T, F)",
        }
        expected_label = rank_labels.get(expected_rank, f"{expected_rank}D")

        msg = (
            f"Shape mismatch for model '{model_name}': "
            f"expected {expected_label}, got shape {actual_shape}."
        )
        if context:
            msg += f" {context}"

        super().__init__(msg)


class AdapterRoutingError(InferenceError):
    """Raised when adapter routing fails during inference.

    Indicates the model requires a data shape transformation that
    cannot be performed with the available bundle configuration.
    """

    def __init__(
        self,
        model_name: str,
        required_rank: int,
        *,
        reason: str = "",
    ) -> None:
        self.model_name = model_name
        self.required_rank = required_rank
        self.reason = reason

        msg = f"Cannot route data for model '{model_name}' " f"(requires {required_rank}D input)."
        if reason:
            msg += f" {reason}"

        super().__init__(msg)


class PreprocessingError(InferenceError):
    """Raised when preprocessing graph fails during inference."""


__all__ = [
    "InferenceError",
    "ShapeMismatchError",
    "AdapterRoutingError",
    "PreprocessingError",
]
