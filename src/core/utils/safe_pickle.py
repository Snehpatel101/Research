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


def safe_pickle_dump(obj: Any, path: str | Path) -> None:
    """
    Serialize *obj* to *path* in the format :func:`safe_pickle_load` reads.

    This is the single write-side counterpart for every artifact loaded via
    ``safe_pickle_load``. Writing with ``joblib.dump`` instead produces files
    that plain ``pickle.load`` cannot read whenever they contain numpy arrays.
    """
    path = Path(path)
    logger.debug("Saving pickle: %s", path)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
