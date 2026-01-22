"""
Query builder for trained model registry.

Provides a fluent interface for building complex model queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ModelQuery:
    """
    Builder for model registry queries.

    Provides a fluent interface for constructing queries to filter
    and sort trained models.

    Example:
        query = (
            ModelQuery()
            .model_name("xgboost")
            .horizon(20)
            .min_metric("val_f1", 0.5)
            .sort_by("val_f1", descending=True)
            .limit(10)
        )
        results = registry.query(query)
    """

    # Filter criteria
    model_names: list[str] = field(default_factory=list)
    model_families: list[str] = field(default_factory=list)
    horizons: list[int] = field(default_factory=list)
    feature_sets: list[str] = field(default_factory=list)
    min_metrics: dict[str, float] = field(default_factory=dict)
    max_metrics: dict[str, float] = field(default_factory=dict)
    date_after: datetime | None = None
    date_before: datetime | None = None
    tags: dict[str, str] = field(default_factory=dict)

    # Sort and limit
    sort_metric: str | None = None
    sort_descending: bool = True
    result_limit: int | None = None

    def model_name(self, *names: str) -> ModelQuery:
        """Filter by model name(s)."""
        self.model_names.extend(names)
        return self

    def model_family(self, *families: str) -> ModelQuery:
        """Filter by model family(s)."""
        self.model_families.extend(families)
        return self

    def horizon(self, *horizons: int) -> ModelQuery:
        """Filter by prediction horizon(s)."""
        self.horizons.extend(horizons)
        return self

    def feature_set(self, *sets: str) -> ModelQuery:
        """Filter by feature set(s)."""
        self.feature_sets.extend(sets)
        return self

    def min_metric(self, metric: str, value: float) -> ModelQuery:
        """Filter for minimum metric value."""
        self.min_metrics[metric] = value
        return self

    def max_metric(self, metric: str, value: float) -> ModelQuery:
        """Filter for maximum metric value."""
        self.max_metrics[metric] = value
        return self

    def after(self, date: datetime | str) -> ModelQuery:
        """Filter for models trained after date."""
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        self.date_after = date
        return self

    def before(self, date: datetime | str) -> ModelQuery:
        """Filter for models trained before date."""
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        self.date_before = date
        return self

    def with_tag(self, key: str, value: str) -> ModelQuery:
        """Filter by tag key-value pair."""
        self.tags[key] = value
        return self

    def sort_by(self, metric: str, descending: bool = True) -> ModelQuery:
        """Sort results by metric."""
        self.sort_metric = metric
        self.sort_descending = descending
        return self

    def limit(self, n: int) -> ModelQuery:
        """Limit number of results."""
        self.result_limit = n
        return self

    def matches(self, entry: dict[str, Any]) -> bool:
        """
        Check if an entry matches all filter criteria.

        Args:
            entry: Model entry dictionary

        Returns:
            True if entry matches all criteria
        """
        # Model name filter
        if self.model_names and entry.get("model_name") not in self.model_names:
            return False

        # Model family filter
        if self.model_families and entry.get("model_family") not in self.model_families:
            return False

        # Horizon filter
        if self.horizons and entry.get("horizon") not in self.horizons:
            return False

        # Feature set filter
        if self.feature_sets and entry.get("feature_set") not in self.feature_sets:
            return False

        # Metric filters
        metrics = entry.get("metrics", {})
        for metric, min_val in self.min_metrics.items():
            if metrics.get(metric, float("-inf")) < min_val:
                return False
        for metric, max_val in self.max_metrics.items():
            if metrics.get(metric, float("inf")) > max_val:
                return False

        # Date filters
        if self.date_after or self.date_before:
            train_date = entry.get("train_date")
            if train_date:
                if isinstance(train_date, str):
                    train_date = datetime.fromisoformat(train_date)
                if self.date_after and train_date < self.date_after:
                    return False
                if self.date_before and train_date > self.date_before:
                    return False

        # Tag filters
        entry_tags = entry.get("tags", {})
        for key, value in self.tags.items():
            if entry_tags.get(key) != value:
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert query to dictionary for serialization."""
        return {
            "model_names": self.model_names,
            "model_families": self.model_families,
            "horizons": self.horizons,
            "feature_sets": self.feature_sets,
            "min_metrics": self.min_metrics,
            "max_metrics": self.max_metrics,
            "date_after": self.date_after.isoformat() if self.date_after else None,
            "date_before": self.date_before.isoformat() if self.date_before else None,
            "tags": self.tags,
            "sort_metric": self.sort_metric,
            "sort_descending": self.sort_descending,
            "result_limit": self.result_limit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelQuery:
        """Create query from dictionary."""
        query = cls(
            model_names=data.get("model_names", []),
            model_families=data.get("model_families", []),
            horizons=data.get("horizons", []),
            feature_sets=data.get("feature_sets", []),
            min_metrics=data.get("min_metrics", {}),
            max_metrics=data.get("max_metrics", {}),
            tags=data.get("tags", {}),
            sort_metric=data.get("sort_metric"),
            sort_descending=data.get("sort_descending", True),
            result_limit=data.get("result_limit"),
        )
        if data.get("date_after"):
            query.date_after = datetime.fromisoformat(data["date_after"])
        if data.get("date_before"):
            query.date_before = datetime.fromisoformat(data["date_before"])
        return query


__all__ = ["ModelQuery"]
