"""
Safe pickle loading utility.

Centralizes all pickle.load() calls behind a single function
with debug logging and optional type checking.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def safe_pickle_load(
    path: str | Path,
    allowed_types: tuple[type, ...] | None = None,
) -> Any:
    """
    Load a pickle file with logging and optional type validation.

    Args:
        path: Path to the pickle file.
        allowed_types: If provided, check that the loaded object is an
            instance of one of these types. Raises TypeError on mismatch.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If *path* does not exist.
        TypeError: If *allowed_types* is set and the loaded object
            does not match any of the given types.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    logger.debug("Loading pickle: %s", path)

    with open(path, "rb") as f:
        obj = pickle.load(f)  # noqa: S301

    if allowed_types is not None and not isinstance(obj, allowed_types):
        raise TypeError(
            f"Loaded object type {type(obj).__name__} not in " f"allowed types {allowed_types}"
        )

    return obj
