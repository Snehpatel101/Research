from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    total_return: float | None = None


@dataclass
class DatasetMetrics:
    split: str
    n_samples: int
    n_features: int
    metrics: ModelMetrics


@dataclass
class TrainingInfo:
    start_time: str
    end_time: str
    duration_seconds: float
    epochs_trained: int
    early_stopped: bool
    best_epoch: int | None = None


@dataclass
class ModelConfig:
    model_name: str
    model_family: str
    horizon: int
    feature_set: str
    sequence_length: int | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineInfo:
    pipeline_run_id: str | None = None
    target_timeframe: str = "5min"
    data_version: str | None = None
    feature_count: int = 0
    data_checksums: dict[str, str] = field(default_factory=dict)
    lineage_validated: bool = False
    lineage_issues: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    report_version: str = "1.0"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str = ""
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig("", "", 0, "", 0))
    pipeline_info: PipelineInfo = field(default_factory=PipelineInfo)
    training_info: TrainingInfo = field(default_factory=lambda: TrainingInfo("", "", 0.0, 0, False))
    train_metrics: DatasetMetrics = field(
        default_factory=lambda: DatasetMetrics("train", 0, 0, ModelMetrics(0, 0, 0, 0))
    )
    val_metrics: DatasetMetrics = field(
        default_factory=lambda: DatasetMetrics("val", 0, 0, ModelMetrics(0, 0, 0, 0))
    )
    test_metrics: DatasetMetrics | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "generated_at": self.generated_at,
            "run_id": self.run_id,
            "model_config": {
                "model_name": self.model_config.model_name,
                "model_family": self.model_config.model_family,
                "horizon": self.model_config.horizon,
                "feature_set": self.model_config.feature_set,
                "sequence_length": self.model_config.sequence_length,
                "hyperparameters": self.model_config.hyperparameters,
            },
            "pipeline_info": {
                "pipeline_run_id": self.pipeline_info.pipeline_run_id,
                "target_timeframe": self.pipeline_info.target_timeframe,
                "data_version": self.pipeline_info.data_version,
                "feature_count": self.pipeline_info.feature_count,
                "data_checksums": self.pipeline_info.data_checksums,
                "lineage_validated": self.pipeline_info.lineage_validated,
                "lineage_issues": self.pipeline_info.lineage_issues,
            },
            "training_info": {
                "start_time": self.training_info.start_time,
                "end_time": self.training_info.end_time,
                "duration_seconds": self.training_info.duration_seconds,
                "epochs_trained": self.training_info.epochs_trained,
                "early_stopped": self.training_info.early_stopped,
                "best_epoch": self.training_info.best_epoch,
            },
            "train_metrics": {
                "split": self.train_metrics.split,
                "n_samples": self.train_metrics.n_samples,
                "n_features": self.train_metrics.n_features,
                "metrics": {
                    "accuracy": self.train_metrics.metrics.accuracy,
                    "precision": self.train_metrics.metrics.precision,
                    "recall": self.train_metrics.metrics.recall,
                    "f1_score": self.train_metrics.metrics.f1_score,
                    "sharpe_ratio": self.train_metrics.metrics.sharpe_ratio,
                    "max_drawdown": self.train_metrics.metrics.max_drawdown,
                    "win_rate": self.train_metrics.metrics.win_rate,
                    "total_return": self.train_metrics.metrics.total_return,
                },
            },
            "val_metrics": {
                "split": self.val_metrics.split,
                "n_samples": self.val_metrics.n_samples,
                "n_features": self.val_metrics.n_features,
                "metrics": {
                    "accuracy": self.val_metrics.metrics.accuracy,
                    "precision": self.val_metrics.metrics.precision,
                    "recall": self.val_metrics.metrics.recall,
                    "f1_score": self.val_metrics.metrics.f1_score,
                    "sharpe_ratio": self.val_metrics.metrics.sharpe_ratio,
                    "max_drawdown": self.val_metrics.metrics.max_drawdown,
                    "win_rate": self.val_metrics.metrics.win_rate,
                    "total_return": self.val_metrics.metrics.total_return,
                },
            },
            "test_metrics": (
                {
                    "split": self.test_metrics.split,
                    "n_samples": self.test_metrics.n_samples,
                    "n_features": self.test_metrics.n_features,
                    "metrics": {
                        "accuracy": self.test_metrics.metrics.accuracy,
                        "precision": self.test_metrics.metrics.precision,
                        "recall": self.test_metrics.metrics.recall,
                        "f1_score": self.test_metrics.metrics.f1_score,
                        "sharpe_ratio": self.test_metrics.metrics.sharpe_ratio,
                        "max_drawdown": self.test_metrics.metrics.max_drawdown,
                        "win_rate": self.test_metrics.metrics.win_rate,
                        "total_return": self.test_metrics.metrics.total_return,
                    },
                }
                if self.test_metrics
                else None
            ),
            "artifacts": self.artifacts,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationReport":
        model_config = ModelConfig(
            model_name=data["model_config"]["model_name"],
            model_family=data["model_config"]["model_family"],
            horizon=data["model_config"]["horizon"],
            feature_set=data["model_config"]["feature_set"],
            sequence_length=data["model_config"].get("sequence_length"),
            hyperparameters=data["model_config"].get("hyperparameters", {}),
        )

        pipeline_info = PipelineInfo(
            pipeline_run_id=data["pipeline_info"].get("pipeline_run_id"),
            target_timeframe=data["pipeline_info"].get("target_timeframe", "5min"),
            data_version=data["pipeline_info"].get("data_version"),
            feature_count=data["pipeline_info"].get("feature_count", 0),
            data_checksums=data["pipeline_info"].get("data_checksums", {}),
            lineage_validated=data["pipeline_info"].get("lineage_validated", False),
            lineage_issues=data["pipeline_info"].get("lineage_issues", []),
        )

        training_info = TrainingInfo(
            start_time=data["training_info"]["start_time"],
            end_time=data["training_info"]["end_time"],
            duration_seconds=data["training_info"]["duration_seconds"],
            epochs_trained=data["training_info"]["epochs_trained"],
            early_stopped=data["training_info"]["early_stopped"],
            best_epoch=data["training_info"].get("best_epoch"),
        )

        def _metrics_from_dict(m: dict[str, Any]) -> ModelMetrics:
            return ModelMetrics(
                accuracy=m["accuracy"],
                precision=m["precision"],
                recall=m["recall"],
                f1_score=m["f1_score"],
                sharpe_ratio=m.get("sharpe_ratio"),
                max_drawdown=m.get("max_drawdown"),
                win_rate=m.get("win_rate"),
                total_return=m.get("total_return"),
            )

        train_metrics = DatasetMetrics(
            split=data["train_metrics"]["split"],
            n_samples=data["train_metrics"]["n_samples"],
            n_features=data["train_metrics"]["n_features"],
            metrics=_metrics_from_dict(data["train_metrics"]["metrics"]),
        )

        val_metrics = DatasetMetrics(
            split=data["val_metrics"]["split"],
            n_samples=data["val_metrics"]["n_samples"],
            n_features=data["val_metrics"]["n_features"],
            metrics=_metrics_from_dict(data["val_metrics"]["metrics"]),
        )

        test_metrics = None
        if data.get("test_metrics"):
            test_metrics = DatasetMetrics(
                split=data["test_metrics"]["split"],
                n_samples=data["test_metrics"]["n_samples"],
                n_features=data["test_metrics"]["n_features"],
                metrics=_metrics_from_dict(data["test_metrics"]["metrics"]),
            )

        return cls(
            report_version=data.get("report_version", "1.0"),
            generated_at=data.get("generated_at", datetime.now().isoformat()),
            run_id=data.get("run_id", ""),
            model_config=model_config,
            pipeline_info=pipeline_info,
            training_info=training_info,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            artifacts=data.get("artifacts", {}),
            notes=data.get("notes", ""),
        )

    def save_json(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "EvaluationReport":
        import json

        with path.open("r") as f:
            data = json.load(f)
        return cls.from_dict(data)
