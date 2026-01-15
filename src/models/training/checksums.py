"""
Model artifact integrity verification.

Provides SHA256 checksums for model artifacts to ensure integrity
during storage, transfer, and loading. Automatically registers
checksums when saving models and verifies on load.

Usage:
    >>> manager = ArtifactIntegrityManager(artifacts_dir)
    >>> checksum = manager.register_artifact(Path("model.pt"))
    >>> print(f"Registered: {checksum.sha256}")
    ...
    >>> is_valid = manager.verify_artifact(Path("model.pt"))
    >>> if not is_valid:
    ...     raise RuntimeError("Model file corrupted!")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArtifactChecksum:
    """Checksum record for a model artifact."""

    file_path: str
    sha256: str
    size_bytes: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactChecksum:
        """Create from dictionary."""
        return cls(**data)


class ArtifactIntegrityManager:
    """
    Manage checksums for model artifacts.

    Computes and stores SHA256 checksums for model files to verify
    integrity during loading. Useful for:
    - Detecting corrupted model files
    - Verifying model transfers (cloud storage, network)
    - Audit trail for model versioning

    Example:
        >>> manager = ArtifactIntegrityManager(Path("experiments/run_001"))
        >>> manager.register_artifact(Path("experiments/run_001/checkpoints/best_model/model.pt"))
        >>> manager.save_checksums()
        ...
        >>> # Later, verify integrity
        >>> results = manager.verify_all()
        >>> assert all(results.values()), "Some artifacts failed verification!"
    """

    # Files to automatically checksum
    DEFAULT_PATTERNS = [
        "**/*.pt",  # PyTorch models
        "**/*.pkl",  # Pickled files (calibrators, etc.)
        "**/*.joblib",  # Joblib serialized models
        "**/*.json",  # Config files
        "**/*.parquet",  # Data files
    ]

    def __init__(self, artifacts_dir: Path) -> None:
        """
        Initialize ArtifactIntegrityManager.

        Args:
            artifacts_dir: Directory containing model artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.checksums_file = self.artifacts_dir / "checksums.json"
        self._checksums: dict[str, ArtifactChecksum] = {}

        # Load existing checksums if available
        if self.checksums_file.exists():
            self._checksums = self.load_checksums()

    def compute_checksum(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Compute SHA256 checksum of file.

        Uses chunked reading for memory efficiency with large files.

        Args:
            file_path: Path to file
            chunk_size: Bytes to read per chunk

        Returns:
            Hex-encoded SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def register_artifact(self, file_path: Path) -> ArtifactChecksum:
        """
        Register artifact and compute checksum.

        Args:
            file_path: Path to artifact file

        Returns:
            ArtifactChecksum with computed hash

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {file_path}")

        # Compute checksum
        sha256 = self.compute_checksum(file_path)
        size_bytes = file_path.stat().st_size

        # Create checksum record
        # Store relative path for portability
        try:
            rel_path = str(file_path.relative_to(self.artifacts_dir))
        except ValueError:
            # File is outside artifacts_dir, use absolute path
            rel_path = str(file_path.absolute())

        checksum = ArtifactChecksum(
            file_path=rel_path,
            sha256=sha256,
            size_bytes=size_bytes,
            created_at=datetime.now().isoformat(),
        )

        # Store in index
        self._checksums[rel_path] = checksum

        logger.debug(f"Registered artifact: {rel_path} ({size_bytes} bytes, sha256={sha256[:16]}...)")

        return checksum

    def register_all(self, patterns: list[str] | None = None) -> list[ArtifactChecksum]:
        """
        Register all matching artifacts in directory.

        Args:
            patterns: Glob patterns to match (defaults to DEFAULT_PATTERNS)

        Returns:
            List of registered ArtifactChecksums
        """
        if patterns is None:
            patterns = self.DEFAULT_PATTERNS

        registered = []
        for pattern in patterns:
            for file_path in self.artifacts_dir.glob(pattern):
                if file_path.is_file() and file_path.name != "checksums.json":
                    checksum = self.register_artifact(file_path)
                    registered.append(checksum)

        logger.info(f"Registered {len(registered)} artifacts")
        return registered

    def verify_artifact(self, file_path: Path) -> bool:
        """
        Verify artifact integrity against stored checksum.

        Args:
            file_path: Path to artifact file

        Returns:
            True if checksum matches, False otherwise
        """
        file_path = Path(file_path)

        # Get relative path for lookup
        try:
            rel_path = str(file_path.relative_to(self.artifacts_dir))
        except ValueError:
            rel_path = str(file_path.absolute())

        if rel_path not in self._checksums:
            logger.warning(f"No checksum registered for: {rel_path}")
            return False

        if not file_path.exists():
            logger.error(f"Artifact file missing: {file_path}")
            return False

        stored = self._checksums[rel_path]
        current_sha256 = self.compute_checksum(file_path)

        if current_sha256 != stored.sha256:
            logger.error(
                f"Checksum mismatch for {rel_path}: "
                f"expected {stored.sha256[:16]}..., got {current_sha256[:16]}..."
            )
            return False

        # Also verify size
        current_size = file_path.stat().st_size
        if current_size != stored.size_bytes:
            logger.error(
                f"Size mismatch for {rel_path}: "
                f"expected {stored.size_bytes}, got {current_size}"
            )
            return False

        logger.debug(f"Verified: {rel_path}")
        return True

    def verify_all(self) -> dict[str, bool]:
        """
        Verify all registered artifacts.

        Returns:
            Dict mapping file paths to verification status
        """
        results = {}
        for rel_path in self._checksums:
            file_path = self.artifacts_dir / rel_path
            results[rel_path] = self.verify_artifact(file_path)

        n_passed = sum(results.values())
        n_total = len(results)

        if n_passed == n_total:
            logger.info(f"All {n_total} artifacts verified successfully")
        else:
            logger.error(f"Verification failed: {n_passed}/{n_total} artifacts passed")

        return results

    def save_checksums(self) -> Path:
        """
        Save checksums to JSON file.

        Returns:
            Path to saved checksums file
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "artifacts_dir": str(self.artifacts_dir),
            "checksums": {k: v.to_dict() for k, v in self._checksums.items()},
        }

        with open(self.checksums_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved checksums to {self.checksums_file}")
        return self.checksums_file

    def load_checksums(self) -> dict[str, ArtifactChecksum]:
        """
        Load checksums from JSON file.

        Returns:
            Dict mapping file paths to ArtifactChecksum objects
        """
        if not self.checksums_file.exists():
            return {}

        with open(self.checksums_file) as f:
            data = json.load(f)

        checksums = {}
        for path, checksum_data in data.get("checksums", {}).items():
            checksums[path] = ArtifactChecksum.from_dict(checksum_data)

        logger.debug(f"Loaded {len(checksums)} checksums from {self.checksums_file}")
        return checksums

    def get_artifact_info(self, file_path: Path) -> ArtifactChecksum | None:
        """
        Get checksum info for an artifact.

        Args:
            file_path: Path to artifact

        Returns:
            ArtifactChecksum if registered, None otherwise
        """
        try:
            rel_path = str(file_path.relative_to(self.artifacts_dir))
        except ValueError:
            rel_path = str(file_path.absolute())

        return self._checksums.get(rel_path)

    def list_artifacts(self) -> list[ArtifactChecksum]:
        """
        List all registered artifacts.

        Returns:
            List of ArtifactChecksum objects
        """
        return list(self._checksums.values())

    def clear(self) -> None:
        """Clear all registered checksums."""
        self._checksums.clear()
        if self.checksums_file.exists():
            self.checksums_file.unlink()
        logger.debug("Cleared all checksums")

    @staticmethod
    def quick_verify(file_path: Path, expected_sha256: str) -> bool:
        """
        Quick verification without manager instance.

        Args:
            file_path: Path to file
            expected_sha256: Expected SHA256 hash

        Returns:
            True if checksum matches
        """
        if not file_path.exists():
            return False

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest() == expected_sha256


def compute_file_checksum(file_path: Path) -> str:
    """
    Compute SHA256 checksum of a file.

    Convenience function for one-off checksum computation.

    Args:
        file_path: Path to file

    Returns:
        Hex-encoded SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


__all__ = [
    "ArtifactChecksum",
    "ArtifactIntegrityManager",
    "compute_file_checksum",
]
