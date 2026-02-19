"""
Feature lineage tracking module.

Tracks the computation history of features from raw data through transformations
to final feature sets. Provides auditability and reproducibility guarantees.

Lineage captures:
- Raw data source (path, checksum, date range)
- Transformations applied (feature engineering config, version)
- Output features (columns, schema, version)
- Dependencies between feature sets
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# Type alias for config dictionaries
ConfigDict = dict[str, Any]

logger = logging.getLogger(__name__)


class TransformationType(StrEnum):
    """Types of transformations tracked in lineage."""

    INGESTION = "ingestion"  # Raw data loading
    CLEANING = "cleaning"  # Data cleaning, gap filling
    RESAMPLING = "resampling"  # Timeframe conversion
    FEATURE_ENGINEERING = "feature_engineering"  # Indicator computation
    MTF_AGGREGATION = "mtf_aggregation"  # Multi-timeframe features
    LABELING = "labeling"  # Target label generation
    SCALING = "scaling"  # Feature scaling/normalization
    SELECTION = "selection"  # Feature selection


@dataclass
class DataSource:
    """
    Represents a data source in the lineage graph.

    Tracks the origin of data including file path, checksum,
    and date range of the data.
    """

    path: str
    checksum: str  # SHA256 of file content
    symbol: str
    timeframe: str
    row_count: int
    date_start: datetime
    date_end: datetime
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "checksum": self.checksum,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "row_count": self.row_count,
            "date_start": self.date_start.isoformat(),
            "date_end": self.date_end.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataSource:
        """Create from dictionary."""
        return cls(
            path=data["path"],
            checksum=data["checksum"],
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            row_count=data["row_count"],
            date_start=datetime.fromisoformat(data["date_start"]),
            date_end=datetime.fromisoformat(data["date_end"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Transformation:
    """
    Represents a transformation step in the lineage.

    Captures what transformation was applied, its configuration,
    and input/output information.
    """

    type: TransformationType
    name: str  # Human-readable name (e.g., "add_rsi", "add_macd")
    config: dict[str, Any]  # Parameters used
    config_hash: str  # Hash of config for comparison
    input_columns: list[str]  # Columns consumed
    output_columns: list[str]  # Columns produced
    executed_at: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "name": self.name,
            "config": self.config,
            "config_hash": self.config_hash,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "executed_at": self.executed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transformation:
        """Create from dictionary."""
        return cls(
            type=TransformationType(data["type"]),
            name=data["name"],
            config=data["config"],
            config_hash=data["config_hash"],
            input_columns=data["input_columns"],
            output_columns=data["output_columns"],
            executed_at=datetime.fromisoformat(data["executed_at"]),
            duration_seconds=data.get("duration_seconds", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FeatureLineage:
    """
    Complete lineage record for a feature set.

    Tracks the full history from raw data through all transformations
    to the final feature output.

    Parameters
    ----------
    feature_set : str
        Name of the feature set
    version : str
        Version of the feature set
    symbol : str
        Symbol this lineage applies to
    """

    feature_set: str
    version: str
    symbol: str
    source: DataSource | None = None
    transformations: list[Transformation] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)
    output_checksum: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    parent_lineages: list[str] = field(default_factory=list)  # IDs of parent lineages
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def lineage_id(self) -> str:
        """Generate unique lineage ID from key attributes."""
        key = f"{self.feature_set}:{self.version}:{self.symbol}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def add_transformation(
        self,
        type: TransformationType,
        name: str,
        config: dict[str, Any],
        input_columns: list[str],
        output_columns: list[str],
        duration_seconds: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Transformation:
        """
        Add a transformation step to the lineage.

        Parameters
        ----------
        type : TransformationType
            Type of transformation
        name : str
            Name of the transformation
        config : dict
            Configuration parameters
        input_columns : list[str]
            Columns consumed by transformation
        output_columns : list[str]
            Columns produced by transformation
        duration_seconds : float, optional
            Execution duration
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        Transformation
            The added transformation
        """
        config_hash = _compute_hash(config)
        transformation = Transformation(
            type=type,
            name=name,
            config=config,
            config_hash=config_hash,
            input_columns=input_columns,
            output_columns=output_columns,
            duration_seconds=duration_seconds,
            metadata=metadata or {},
        )
        self.transformations.append(transformation)
        logger.debug(
            f"Added transformation '{name}' ({type.value}) to lineage "
            f"for {self.feature_set}:{self.version}"
        )
        return transformation

    def set_source(
        self,
        path: str,
        checksum: str,
        symbol: str,
        timeframe: str,
        row_count: int,
        date_start: datetime,
        date_end: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> DataSource:
        """
        Set the data source for this lineage.

        Parameters
        ----------
        path : str
            Path to source file
        checksum : str
            Checksum of source file
        symbol : str
            Symbol of source data
        timeframe : str
            Timeframe of source data
        row_count : int
            Number of rows in source
        date_start : datetime
            Start of data range
        date_end : datetime
            End of data range
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        DataSource
            The created data source
        """
        self.source = DataSource(
            path=path,
            checksum=checksum,
            symbol=symbol,
            timeframe=timeframe,
            row_count=row_count,
            date_start=date_start,
            date_end=date_end,
            metadata=metadata or {},
        )
        logger.debug(f"Set source for lineage {self.lineage_id}: {path}")
        return self.source

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lineage_id": self.lineage_id,
            "feature_set": self.feature_set,
            "version": self.version,
            "symbol": self.symbol,
            "source": self.source.to_dict() if self.source else None,
            "transformations": [t.to_dict() for t in self.transformations],
            "output_columns": self.output_columns,
            "output_checksum": self.output_checksum,
            "created_at": self.created_at.isoformat(),
            "parent_lineages": self.parent_lineages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureLineage:
        """Create from dictionary."""
        lineage = cls(
            feature_set=data["feature_set"],
            version=data["version"],
            symbol=data["symbol"],
            output_columns=data.get("output_columns", []),
            output_checksum=data.get("output_checksum", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            parent_lineages=data.get("parent_lineages", []),
            metadata=data.get("metadata", {}),
        )
        if data.get("source"):
            lineage.source = DataSource.from_dict(data["source"])
        for t_dict in data.get("transformations", []):
            lineage.transformations.append(Transformation.from_dict(t_dict))
        return lineage

    def get_transformation_chain(self) -> list[str]:
        """Get ordered list of transformation names."""
        return [t.name for t in self.transformations]

    def get_all_output_columns(self) -> set[str]:
        """Get all columns produced by all transformations."""
        columns = set()
        for t in self.transformations:
            columns.update(t.output_columns)
        return columns


class LineageTracker:
    """
    Tracks and manages feature lineages across feature sets.

    Provides storage, retrieval, and query capabilities for
    feature lineage information.

    Parameters
    ----------
    storage_path : Path, optional
        Path to store lineage data. If None, operates in memory only.

    Examples
    --------
    >>> tracker = LineageTracker(storage_path=Path("data/lineage"))
    >>> lineage = tracker.create_lineage(
    ...     feature_set="core_features",
    ...     version="1.0.0",
    ...     symbol="MES"
    ... )
    >>> lineage.set_source(path="data/raw/MES_1m.parquet", ...)
    >>> lineage.add_transformation(type=TransformationType.FEATURE_ENGINEERING, ...)
    >>> tracker.save_lineage(lineage)
    """

    def __init__(self, storage_path: Path | str | None = None) -> None:
        """
        Initialize lineage tracker.

        Parameters
        ----------
        storage_path : Path or str, optional
            Path to store lineage files
        """
        self._lineages: dict[str, FeatureLineage] = {}
        self._storage_path: Path | None = None

        if storage_path is not None:
            self._storage_path = Path(storage_path)
            self._storage_path.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    def _load_existing(self) -> None:
        """Load existing lineages from storage."""
        if self._storage_path is None:
            return

        for lineage_file in self._storage_path.glob("*.json"):
            try:
                with open(lineage_file) as f:
                    data = json.load(f)
                lineage = FeatureLineage.from_dict(data)
                self._lineages[lineage.lineage_id] = lineage
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load lineage from {lineage_file}: {e}")

        logger.info(f"Loaded {len(self._lineages)} existing lineages from {self._storage_path}")

    def create_lineage(
        self,
        feature_set: str,
        version: str,
        symbol: str,
        parent_lineages: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureLineage:
        """
        Create a new lineage record.

        Parameters
        ----------
        feature_set : str
            Name of the feature set
        version : str
            Version string
        symbol : str
            Symbol this lineage applies to
        parent_lineages : list[str], optional
            IDs of parent lineages (for derived features)
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        FeatureLineage
            New lineage record
        """
        lineage = FeatureLineage(
            feature_set=feature_set,
            version=version,
            symbol=symbol,
            parent_lineages=parent_lineages or [],
            metadata=metadata or {},
        )
        self._lineages[lineage.lineage_id] = lineage
        logger.info(f"Created lineage {lineage.lineage_id} for {feature_set}:{version}:{symbol}")
        return lineage

    def get_lineage(self, lineage_id: str) -> FeatureLineage | None:
        """
        Get lineage by ID.

        Parameters
        ----------
        lineage_id : str
            Lineage ID to retrieve

        Returns
        -------
        FeatureLineage or None
            Lineage if found
        """
        return self._lineages.get(lineage_id)

    def get_lineage_for_feature_set(
        self, feature_set: str, version: str, symbol: str
    ) -> FeatureLineage | None:
        """
        Get lineage for a specific feature set version and symbol.

        Parameters
        ----------
        feature_set : str
            Feature set name
        version : str
            Version string
        symbol : str
            Symbol name

        Returns
        -------
        FeatureLineage or None
            Matching lineage if found
        """
        key = f"{feature_set}:{version}:{symbol}"
        lineage_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._lineages.get(lineage_id)

    def save_lineage(self, lineage: FeatureLineage) -> Path | None:
        """
        Save lineage to storage.

        Parameters
        ----------
        lineage : FeatureLineage
            Lineage to save

        Returns
        -------
        Path or None
            Path to saved file, or None if no storage configured
        """
        self._lineages[lineage.lineage_id] = lineage

        if self._storage_path is None:
            return None

        file_path = self._storage_path / f"{lineage.lineage_id}.json"
        with open(file_path, "w") as f:
            json.dump(lineage.to_dict(), f, indent=2, default=str)

        logger.debug(f"Saved lineage {lineage.lineage_id} to {file_path}")
        return file_path

    def find_lineages(
        self,
        feature_set: str | None = None,
        symbol: str | None = None,
        version: str | None = None,
    ) -> list[FeatureLineage]:
        """
        Find lineages matching criteria.

        Parameters
        ----------
        feature_set : str, optional
            Filter by feature set name
        symbol : str, optional
            Filter by symbol
        version : str, optional
            Filter by version

        Returns
        -------
        list[FeatureLineage]
            Matching lineages
        """
        results = []
        for lineage in self._lineages.values():
            if feature_set and lineage.feature_set != feature_set:
                continue
            if symbol and lineage.symbol != symbol:
                continue
            if version and lineage.version != version:
                continue
            results.append(lineage)
        return results

    def get_downstream_lineages(self, lineage_id: str) -> list[FeatureLineage]:
        """
        Get all lineages that depend on the given lineage.

        Parameters
        ----------
        lineage_id : str
            Parent lineage ID

        Returns
        -------
        list[FeatureLineage]
            Lineages that have this as a parent
        """
        downstream = []
        for lineage in self._lineages.values():
            if lineage_id in lineage.parent_lineages:
                downstream.append(lineage)
        return downstream

    def get_upstream_lineages(self, lineage_id: str) -> list[FeatureLineage]:
        """
        Get all lineages that the given lineage depends on.

        Parameters
        ----------
        lineage_id : str
            Child lineage ID

        Returns
        -------
        list[FeatureLineage]
            Parent lineages
        """
        lineage = self._lineages.get(lineage_id)
        if lineage is None:
            return []

        upstream = []
        for parent_id in lineage.parent_lineages:
            parent = self._lineages.get(parent_id)
            if parent:
                upstream.append(parent)
        return upstream

    def __iter__(self) -> Iterator[FeatureLineage]:
        """Iterate over all lineages."""
        return iter(self._lineages.values())

    def __len__(self) -> int:
        """Return number of lineages tracked."""
        return len(self._lineages)


def _compute_hash(data: Any) -> str:
    """Compute SHA256 hash of data."""
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]


def compute_file_checksum(file_path: Path | str) -> str:
    """
    Compute SHA256 checksum of a file.

    Parameters
    ----------
    file_path : Path or str
        Path to file

    Returns
    -------
    str
        SHA256 hex digest

    Raises
    ------
    FileNotFoundError
        If file does not exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()
