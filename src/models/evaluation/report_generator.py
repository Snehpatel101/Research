from pathlib import Path

from .report_schema import EvaluationReport, DatasetMetrics


def generate_markdown_report(report: EvaluationReport) -> str:
    lines = []

    lines.append(f"# Model Evaluation Report")
    lines.append("")
    lines.append(f"**Run ID:** `{report.run_id}`")
    lines.append(f"**Generated:** {report.generated_at}")
    lines.append("")

    lines.append("## Model Configuration")
    lines.append("")
    lines.append(f"- **Model:** {report.model_config.model_name}")
    lines.append(f"- **Family:** {report.model_config.model_family}")
    lines.append(f"- **Horizon:** {report.model_config.horizon}")
    lines.append(f"- **Feature Set:** {report.model_config.feature_set}")
    if report.model_config.sequence_length:
        lines.append(f"- **Sequence Length:** {report.model_config.sequence_length}")
    lines.append("")

    if report.model_config.hyperparameters:
        lines.append("### Hyperparameters")
        lines.append("")
        for key, value in sorted(report.model_config.hyperparameters.items()):
            lines.append(f"- `{key}`: {value}")
        lines.append("")

    lines.append("## Pipeline Information")
    lines.append("")
    if report.pipeline_info.pipeline_run_id:
        lines.append(f"- **Pipeline Run ID:** `{report.pipeline_info.pipeline_run_id}`")
    lines.append(f"- **Target Timeframe:** {report.pipeline_info.target_timeframe}")
    lines.append(f"- **Feature Count:** {report.pipeline_info.feature_count}")
    if report.pipeline_info.data_version:
        lines.append(f"- **Data Version:** {report.pipeline_info.data_version}")
    lines.append("")

    lines.append("## Training Information")
    lines.append("")
    lines.append(f"- **Start Time:** {report.training_info.start_time}")
    lines.append(f"- **End Time:** {report.training_info.end_time}")
    lines.append(f"- **Duration:** {report.training_info.duration_seconds:.2f}s")
    lines.append(f"- **Epochs Trained:** {report.training_info.epochs_trained}")
    lines.append(f"- **Early Stopped:** {'Yes' if report.training_info.early_stopped else 'No'}")
    if report.training_info.best_epoch is not None:
        lines.append(f"- **Best Epoch:** {report.training_info.best_epoch}")
    lines.append("")

    lines.append("## Performance Metrics")
    lines.append("")

    def _format_metrics_table(dataset: DatasetMetrics) -> list[str]:
        table = []
        table.append(f"### {dataset.split.upper()} Set")
        table.append("")
        table.append(
            f"**Dataset Size:** {dataset.n_samples} samples, {dataset.n_features} features"
        )
        table.append("")
        table.append("| Metric | Value |")
        table.append("|--------|-------|")
        table.append(f"| Accuracy | {dataset.metrics.accuracy:.4f} |")
        table.append(f"| Precision | {dataset.metrics.precision:.4f} |")
        table.append(f"| Recall | {dataset.metrics.recall:.4f} |")
        table.append(f"| F1 Score | {dataset.metrics.f1_score:.4f} |")

        if dataset.metrics.sharpe_ratio is not None:
            table.append(f"| Sharpe Ratio | {dataset.metrics.sharpe_ratio:.4f} |")
        if dataset.metrics.max_drawdown is not None:
            table.append(f"| Max Drawdown | {dataset.metrics.max_drawdown:.4f} |")
        if dataset.metrics.win_rate is not None:
            table.append(f"| Win Rate | {dataset.metrics.win_rate:.4f} |")
        if dataset.metrics.total_return is not None:
            table.append(f"| Total Return | {dataset.metrics.total_return:.4f} |")

        table.append("")
        return table

    lines.extend(_format_metrics_table(report.train_metrics))
    lines.extend(_format_metrics_table(report.val_metrics))

    if report.test_metrics:
        lines.extend(_format_metrics_table(report.test_metrics))

    if report.artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for name, path in sorted(report.artifacts.items()):
            lines.append(f"- **{name}:** `{path}`")
        lines.append("")

    if report.notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(report.notes)
        lines.append("")

    return "\n".join(lines)


def save_report(report: EvaluationReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_report.json"
    md_path = output_dir / "evaluation_report.md"

    report.save_json(json_path)

    markdown = generate_markdown_report(report)
    md_path.write_text(markdown)

    return json_path, md_path
