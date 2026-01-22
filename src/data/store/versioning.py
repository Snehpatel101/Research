"""
Feature versioning with semantic versioning support.

This module provides version management for feature sets, using semantic versioning
(major.minor.patch) to track changes to feature definitions and computations.

Version meanings:
- Major: Breaking changes to feature schema (columns added/removed, types changed)
- Minor: Backward-compatible additions (new derived features, config changes)
- Patch: Bug fixes, performance improvements (same output)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import total_ordering
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator

logger = logging.getLogger(__name__)


@total_ordering
@dataclass(frozen=True)
class SemanticVersion:
    """
    Semantic version representation (major.minor.patch).

    Supports comparison, sorting, and parsing from strings.

    Examples
    --------
    >>> v1 = SemanticVersion(1, 0, 0)
    >>> v2 = SemanticVersion.parse("1.2.3")
    >>> v1 < v2
    True
    >>> str(v2)
    '1.2.3'
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        """Validate version components are non-negative."""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError(
                f"Version components must be non-negative, got "
                f"{self.major}.{self.minor}.{self.patch}"
            )

    @classmethod
    def parse(cls, version_str: str) -> SemanticVersion:
        """
        Parse a version string into a SemanticVersion.

        Parameters
        ----------
        version_str : str
            Version string in format "major.minor.patch"

        Returns
        -------
        SemanticVersion
            Parsed version object

        Raises
        ------
        ValueError
            If version string is invalid
        """
        pattern = r"^(\d+)\.(\d+)\.(\d+)$"
        match = re.match(pattern, version_str.strip())
        if not match:
            raise ValueError(
                f"Invalid version string: '{version_str}'. "
                f"Expected format: major.minor.patch (e.g., '1.2.3')"
            )
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
        )

    def __str__(self) -> str:
        """Return version string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"SemanticVersion({self.major}, {self.minor}, {self.patch})"

    def __eq__(self, other: object) -> bool:
        """Check version equality."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other: SemanticVersion) -> bool:
        """Check if this version is less than other."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __hash__(self) -> int:
        """Return hash for use in sets and dicts."""
        return hash((self.major, self.minor, self.patch))

    def bump_major(self) -> SemanticVersion:
        """Return new version with major incremented, minor/patch reset."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> SemanticVersion:
        """Return new version with minor incremented, patch reset."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> SemanticVersion:
        """Return new version with patch incremented."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)


@dataclass
class VersionInfo:
    """
    Complete version information for a feature set.

    Includes version number, creation timestamp, description, and optional metadata.
    """

    version: SemanticVersion
    created_at: datetime
    description: str = ""
    schema_hash: str = ""  # Hash of column names + dtypes
    config_hash: str = ""  # Hash of feature config
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "version": str(self.version),
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "schema_hash": self.schema_hash,
            "config_hash": self.config_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VersionInfo:
        """Create from dictionary."""
        return cls(
            version=SemanticVersion.parse(data["version"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description", ""),
            schema_hash=data.get("schema_hash", ""),
            config_hash=data.get("config_hash", ""),
            metadata=data.get("metadata", {}),
        )


class VersionManager:
    """
    Manages version history for a feature set.

    Tracks all versions, provides version comparison, and suggests
    appropriate version bumps based on schema changes.

    Parameters
    ----------
    feature_set : str
        Name of the feature set being versioned

    Examples
    --------
    >>> manager = VersionManager("core_features")
    >>> manager.register_version(
    ...     version=SemanticVersion(1, 0, 0),
    ...     description="Initial release",
    ...     schema_hash="abc123"
    ... )
    >>> manager.latest_version
    SemanticVersion(1, 0, 0)
    """

    def __init__(self, feature_set: str) -> None:
        """
        Initialize version manager.

        Parameters
        ----------
        feature_set : str
            Name of the feature set
        """
        if not feature_set or not feature_set.strip():
            raise ValueError("feature_set name cannot be empty")

        self.feature_set = feature_set
        self._versions: dict[SemanticVersion, VersionInfo] = {}
        self._ordered_versions: list[SemanticVersion] = []

    def register_version(
        self,
        version: SemanticVersion,
        description: str = "",
        schema_hash: str = "",
        config_hash: str = "",
        metadata: dict | None = None,
        created_at: datetime | None = None,
    ) -> VersionInfo:
        """
        Register a new version.

        Parameters
        ----------
        version : SemanticVersion
            Version to register
        description : str, optional
            Description of changes
        schema_hash : str, optional
            Hash of feature schema
        config_hash : str, optional
            Hash of feature config
        metadata : dict, optional
            Additional metadata
        created_at : datetime, optional
            Creation timestamp (defaults to now)

        Returns
        -------
        VersionInfo
            Registered version info

        Raises
        ------
        ValueError
            If version already exists
        """
        if version in self._versions:
            raise ValueError(
                f"Version {version} already exists for feature set '{self.feature_set}'"
            )

        info = VersionInfo(
            version=version,
            created_at=created_at or datetime.now(),
            description=description,
            schema_hash=schema_hash,
            config_hash=config_hash,
            metadata=metadata or {},
        )

        self._versions[version] = info
        self._ordered_versions.append(version)
        self._ordered_versions.sort()

        logger.info(
            f"Registered version {version} for feature set '{self.feature_set}': {description}"
        )

        return info

    def get_version(self, version: SemanticVersion | str) -> VersionInfo | None:
        """
        Get version info.

        Parameters
        ----------
        version : SemanticVersion or str
            Version to look up

        Returns
        -------
        VersionInfo or None
            Version info if found
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return self._versions.get(version)

    @property
    def latest_version(self) -> SemanticVersion | None:
        """Get the latest registered version."""
        if not self._ordered_versions:
            return None
        return self._ordered_versions[-1]

    @property
    def all_versions(self) -> list[SemanticVersion]:
        """Get all versions in sorted order."""
        return list(self._ordered_versions)

    def suggest_version(
        self,
        current_schema_hash: str,
        current_config_hash: str,
        base_version: SemanticVersion | None = None,
    ) -> SemanticVersion:
        """
        Suggest next version based on changes.

        Compares current hashes against the base version (or latest) to
        determine appropriate version bump:
        - Schema change -> major bump
        - Config change only -> minor bump
        - No changes -> patch bump

        Parameters
        ----------
        current_schema_hash : str
            Hash of current schema
        current_config_hash : str
            Hash of current config
        base_version : SemanticVersion, optional
            Version to compare against (defaults to latest)

        Returns
        -------
        SemanticVersion
            Suggested next version
        """
        # First version
        if not self._ordered_versions:
            return SemanticVersion(1, 0, 0)

        base = base_version or self.latest_version
        if base is None:
            return SemanticVersion(1, 0, 0)

        base_info = self._versions.get(base)
        if base_info is None:
            # No base info, suggest major bump
            return base.bump_major()

        # Compare hashes to determine bump type
        if base_info.schema_hash != current_schema_hash:
            # Schema changed - major bump
            logger.info(
                f"Schema hash changed ({base_info.schema_hash} -> {current_schema_hash}), "
                f"suggesting major version bump"
            )
            return base.bump_major()

        if base_info.config_hash != current_config_hash:
            # Config changed but schema same - minor bump
            logger.info(
                f"Config hash changed ({base_info.config_hash} -> {current_config_hash}), "
                f"suggesting minor version bump"
            )
            return base.bump_minor()

        # No changes - patch bump (for bug fixes, re-computation)
        logger.info("No schema or config changes, suggesting patch version bump")
        return base.bump_patch()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "feature_set": self.feature_set,
            "versions": {str(v): info.to_dict() for v, info in self._versions.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> VersionManager:
        """Create from dictionary."""
        manager = cls(data["feature_set"])
        for version_str, info_dict in data.get("versions", {}).items():
            info = VersionInfo.from_dict(info_dict)
            manager._versions[info.version] = info
            manager._ordered_versions.append(info.version)
        manager._ordered_versions.sort()
        return manager

    def __iter__(self) -> Iterator[VersionInfo]:
        """Iterate over version infos in order."""
        for version in self._ordered_versions:
            yield self._versions[version]

    def __len__(self) -> int:
        """Return number of versions."""
        return len(self._versions)

    def __contains__(self, version: SemanticVersion | str) -> bool:
        """Check if version exists."""
        if isinstance(version, str):
            try:
                version = SemanticVersion.parse(version)
            except ValueError:
                return False
        return version in self._versions


def compute_schema_hash(columns: list[str], dtypes: dict[str, str] | None = None) -> str:
    """
    Compute hash of feature schema (columns + dtypes).

    Parameters
    ----------
    columns : list[str]
        Sorted list of column names
    dtypes : dict[str, str], optional
        Mapping of column names to dtype strings

    Returns
    -------
    str
        SHA256 hash of schema
    """
    import hashlib
    import json

    schema_data = {
        "columns": sorted(columns),
        "dtypes": {k: str(v) for k, v in sorted((dtypes or {}).items())},
    }

    schema_str = json.dumps(schema_data, sort_keys=True)
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def compute_config_hash(config: dict) -> str:
    """
    Compute hash of feature configuration.

    Parameters
    ----------
    config : dict
        Feature configuration dictionary

    Returns
    -------
    str
        SHA256 hash of config
    """
    import hashlib
    import json

    # Sort and serialize config
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
