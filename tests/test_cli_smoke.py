"""CLI smoke tests for the unified Typer app.

Pure-parse tests: nothing here trains a model or loads real data (the end-to-end
runs live in tests/test_cli_e2e.py).

Covers:
- The typer app imports and exposes the expected command set (one pipeline:
  MLFactory behind every command; the old runner/train commands are gone).
- `--help` parses cleanly for every command.
- `ml run` error paths exit nonzero with a readable message (no raw traceback).
- `ml status` / `ml models` on cheap inputs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli.unified_cli import app

EXPECTED_COMMANDS = {
    "run",
    "data",
    "status",
    "models",
    "cv",
    "walk-forward",
    "cpcv-pbo",
    "version",
}

# Commands of the removed PipelineRunner / container-parquet flow
REMOVED_COMMANDS = {"resume", "train"}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# =============================================================================
# 1. App imports and exposes the expected command set
# =============================================================================


class TestCommandRegistration:
    def test_app_is_typer_app(self):
        import typer

        assert isinstance(app, typer.Typer)

    def test_exact_command_set(self):
        names = {cmd.name for cmd in app.registered_commands}
        assert names == EXPECTED_COMMANDS
        assert not names & REMOVED_COMMANDS
        assert not app.registered_groups

    def test_package_exports_app_and_main(self):
        from src.cli import app as pkg_app
        from src.cli import main as pkg_main

        assert pkg_app is app
        assert callable(pkg_main)

    def test_pipeline_cli_entrypoint_imports(self):
        from src import pipeline_cli

        assert callable(pipeline_cli.main)

    def test_runner_stack_is_gone(self):
        import importlib.util

        for module in (
            "src.data.pipeline.runner",
            "src.data.pipeline.data_config",
            "src.data.pipeline.stage_registry",
        ):
            assert importlib.util.find_spec(module) is None, module


# =============================================================================
# 2. --help parses for every command
# =============================================================================


class TestHelp:
    def test_main_help_lists_every_command(self, runner: CliRunner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        out = result.output.lower()
        assert "ml factory" in out
        for cmd in EXPECTED_COMMANDS:
            assert cmd in out, f"Command '{cmd}' not mentioned in main --help"

    @pytest.mark.parametrize("command", sorted(EXPECTED_COMMANDS))
    def test_command_help(self, runner: CliRunner, command: str):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output

    def test_run_help_documents_pipeline_options(self, runner: CliRunner):
        result = runner.invoke(app, ["run", "--help"])
        for option in ("--data-path", "--models", "--training-mode", "--resume", "--n-trials"):
            assert option in result.output


# =============================================================================
# 3. `ml run` error paths: nonzero exit, readable error, no raw traceback
# =============================================================================


class TestRunErrorHandling:
    def test_run_missing_data_path_fails_cleanly(self, runner: CliRunner, tmp_path: Path):
        """Nonexistent --data-path: clean nonzero exit with a readable error."""
        result = runner.invoke(
            app,
            [
                "run",
                "--data-path",
                str(tmp_path / "does_not_exist.parquet"),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit), (
            f"Expected clean SystemExit, got raw {type(result.exception).__name__}: "
            f"{result.exception}"
        )
        out = result.output.lower()
        assert "pipeline failed" in out
        assert "no such file" in out or "does_not_exist" in out

    def test_run_without_data_path_is_a_configuration_error(
        self, runner: CliRunner, tmp_path: Path
    ):
        result = runner.invoke(app, ["run", "--output-dir", str(tmp_path / "out")])
        assert result.exit_code == 1
        assert "--data-path is required" in result.output

    def test_run_missing_config_path_fails_cleanly(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(app, ["run", "--config", str(tmp_path / "does_not_exist.yaml")])
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "configuration error" in result.output.lower()

    def test_resume_without_saved_config_fails_cleanly(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(app, ["run", "--resume", str(tmp_path)])
        assert result.exit_code == 1
        assert "experiment_config.yaml" in result.output

    def test_resume_rejects_other_options(self, runner: CliRunner, tmp_path: Path):
        (tmp_path / "experiment_config.yaml").write_text("name: x\n")
        result = runner.invoke(
            app, ["run", "--resume", str(tmp_path), "-m", "lstm", "--horizons", "5", "-v"]
        )
        assert result.exit_code == 1
        assert "--models" in result.output and "--horizons" in result.output
        assert "--verbose" not in result.output

    def test_run_rejects_unknown_horizon_format(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "run",
                "-d",
                str(tmp_path / "x.parquet"),
                "--horizons",
                "five",
                "-o",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1


# =============================================================================
# 4. status / models
# =============================================================================


class TestStatusCommand:
    def test_status_unknown_run_dir_exits_nonzero(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(app, ["status", str(tmp_path / "no_such_run")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_status_run_without_checkpoints(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(app, ["status", str(tmp_path)])
        assert result.exit_code == 0
        assert "no checkpoints" in result.output.lower()
        assert not (tmp_path / "checkpoints").exists()  # status never writes

    def test_status_lists_checkpoint_stages(self, runner: CliRunner, tmp_path: Path):
        from src.core.checkpoint import CheckpointState

        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir()
        state = CheckpointState(
            stage_name="data_pipeline",
            stage_index=0,
            completed_at=datetime(2026, 1, 1, 12, 0, 0),
            artifacts={},
            config_hash="abc",
            metadata={"n_rows": 2800},
        )
        (checkpoints / "checkpoint_000_data_pipeline.json").write_text(json.dumps(state.to_dict()))
        (checkpoints / "latest.json").write_text(json.dumps(state.to_dict()))

        result = runner.invoke(app, ["status", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "data_pipeline" in result.output
        assert "n_rows=2800" in result.output
        assert "--resume" in result.output


class TestModelsCommand:
    def test_models_lists_registered_models(self, runner: CliRunner):
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0, result.output
        for model in ("xgboost", "lstm", "patchtst"):
            assert model in result.output

    def test_models_shows_one_model(self, runner: CliRunner):
        result = runner.invoke(app, ["models", "xgboost"])
        assert result.exit_code == 0, result.output
        assert "Default Configuration" in result.output

    def test_models_unknown_model_exits_1(self, runner: CliRunner):
        result = runner.invoke(app, ["models", "no_such_model"])
        assert result.exit_code == 1


# =============================================================================
# 5. evaluation commands: model validation, no leftovers on failure
# =============================================================================


class TestEvaluationCommandValidation:
    @pytest.mark.parametrize("command", ["cv", "walk-forward", "cpcv-pbo"])
    def test_sequence_model_rejected_before_any_work(
        self, runner: CliRunner, tmp_path: Path, command: str
    ):
        out = tmp_path / "out"
        result = runner.invoke(
            app, [command, "-d", str(tmp_path / "x.parquet"), "-m", "lstm", "-o", str(out)]
        )
        assert result.exit_code == 1
        assert "tabular" in result.output.lower()
        assert not out.exists()

    def test_all_means_tabular_base_models(self):
        from src.cli.utils import parse_tabular_model_list

        models = parse_tabular_model_list("all")
        assert {"xgboost", "lightgbm", "catboost", "logistic", "random_forest", "svm"} == set(
            models
        )

    @pytest.mark.parametrize("command", ["cv", "walk-forward", "cpcv-pbo", "data"])
    def test_failed_data_step_leaves_no_run_directory(
        self, runner: CliRunner, tmp_path: Path, command: str
    ):
        out = tmp_path / "out"
        args = [command, "-d", str(tmp_path / "missing.parquet"), "-o", str(out)]
        if command != "data":
            args += ["-m", "logistic"]
        result = runner.invoke(app, args)
        assert result.exit_code == 1
        assert not out.exists() or not any(out.iterdir())
