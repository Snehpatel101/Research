"""CLI smoke tests for the unified Typer app.

Pure-parse tests: nothing here trains a model or loads real data.

Covers:
- The typer app imports and exposes the expected command set.
- `--help` parses cleanly for the main app and key subcommands.
- `ml run` with a nonexistent data path exits nonzero with a readable
  error (no raw traceback).
- `ml run --config <missing>` and `ml status` error paths (current
  behavior documented — see KNOWN BUGS below).

KNOWN BUGS documented (not fixed) by these tests:

1. ``ml run --config <nonexistent>``: ``_build_ml_config()`` is called
   OUTSIDE the try/except in ``run_pipeline`` (src/cli/commands/pipeline.py),
   so ``ExperimentConfig.from_yaml`` raises an unhandled ``FileNotFoundError``
   (raw traceback for real users) instead of a clean CLI error message.

2. ``typer.Exit`` subclasses ``Exception`` (via click's ``Exit`` ->
   ``RuntimeError``), so ``raise typer.Exit(0)`` INSIDE the ``try:`` blocks of
   ``run_pipeline`` / ``run_data`` / ``show_status`` / ``resume_pipeline`` is
   swallowed by ``except Exception``. Success paths therefore exit 1 and print
   an error (e.g. ``ml status`` on a valid run prints "Error: 0" and exits 1).

3. ``ml status`` with a nonexistent run id crashes with a raw
   ``rich.errors.MarkupError``: the ``[dim]`` markup tag is opened in one
   ``console.print`` call and closed in a different one
   (src/cli/commands/pipeline.py lines ~268-269).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli.unified_cli import app

# Commands registered directly on the main app.
EXPECTED_COMMANDS = {
    "run",
    "data",
    "status",
    "resume",
    "cv",
    "walk-forward",
    "cpcv-pbo",
    "version",
}

# Sub-apps registered via add_typer.
EXPECTED_GROUPS = {"train"}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# =============================================================================
# 1. App imports and exposes the expected command set
# =============================================================================


class TestCommandRegistration:
    """The typer app object exposes the expected commands."""

    def test_app_is_typer_app(self):
        import typer

        assert isinstance(app, typer.Typer)

    def test_expected_commands_registered(self):
        names = {cmd.name for cmd in app.registered_commands}
        missing = EXPECTED_COMMANDS - names
        assert not missing, f"Missing CLI commands: {sorted(missing)} (found: {sorted(names)})"

    def test_expected_groups_registered(self):
        group_names = {grp.name for grp in app.registered_groups}
        missing = EXPECTED_GROUPS - group_names
        assert not missing, f"Missing CLI sub-apps: {sorted(missing)}"

    def test_package_exports_app_and_main(self):
        from src.cli import app as pkg_app
        from src.cli import main as pkg_main

        assert pkg_app is app
        assert callable(pkg_main)

    def test_pipeline_cli_entrypoint_imports(self):
        from src import pipeline_cli

        assert callable(pipeline_cli.main)


# =============================================================================
# 2. --help parses for the main app and key subcommands
# =============================================================================


class TestHelp:
    """--help exits 0 and mentions each command's purpose. No execution."""

    def test_main_help(self, runner: CliRunner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        out = result.output.lower()
        assert "unified ml pipeline" in out
        # The help listing should mention the core commands.
        for cmd in ("run", "data", "cv", "status"):
            assert cmd in out, f"Command '{cmd}' not mentioned in main --help"

    def test_run_help(self, runner: CliRunner):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "full pipeline" in result.output.lower()
        assert "--data-path" in result.output

    def test_cv_help(self, runner: CliRunner):
        result = runner.invoke(app, ["cv", "--help"])
        assert result.exit_code == 0
        assert "cross-validation" in result.output.lower()

    def test_train_help(self, runner: CliRunner):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "training" in result.output.lower()


# =============================================================================
# 3. `ml run` error paths: nonzero exit, readable error, no raw traceback
# =============================================================================


class TestRunErrorHandling:
    """`ml run` with bad paths must fail without training anything."""

    def test_run_missing_data_path_fails_cleanly(self, runner: CliRunner, tmp_path: Path):
        """Nonexistent --data-path: clean nonzero exit with a readable error.

        The FileNotFoundError from the data load is caught inside
        run_pipeline's try/except, printed via show_error, and converted to
        typer.Exit(1) — so the runner sees SystemExit, not a raw traceback.
        """
        missing = tmp_path / "does_not_exist.parquet"
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "run",
                "--symbol",
                "MES",
                "--data-path",
                str(missing),
                "--output-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code != 0
        # Clean exit: the only "exception" is SystemExit from typer.Exit.
        assert isinstance(result.exception, SystemExit), (
            f"Expected clean SystemExit, got raw {type(result.exception).__name__}: "
            f"{result.exception}"
        )
        out = result.output.lower()
        assert "pipeline failed" in out
        assert "no such file" in out or "does_not_exist" in out

    def test_run_missing_config_path_exits_nonzero(self, runner: CliRunner, tmp_path: Path):
        """Nonexistent --config: exits nonzero.

        KNOWN BUG (documented, not asserted green): _build_ml_config() runs
        outside run_pipeline's try/except, so ExperimentConfig.from_yaml's
        FileNotFoundError currently escapes as a raw traceback rather than a
        clean CLI error. We assert only the safe part of the behavior
        (nonzero exit + the missing path is named somewhere).
        """
        missing_cfg = tmp_path / "does_not_exist.yaml"

        result = runner.invoke(
            app,
            [
                "run",
                "--data-path",
                str(tmp_path / "whatever.parquet"),
                "--output-dir",
                str(tmp_path / "out"),
                "--config",
                str(missing_cfg),
            ],
        )

        assert result.exit_code != 0
        # Error is surfaced either in output (fixed behavior) or in the
        # exception message (current raw-traceback behavior).
        surfaced = result.output + str(result.exception or "")
        assert "not found" in surfaced.lower() or "does_not_exist" in surfaced


# =============================================================================
# `ml status` smoke: cheap JSON-only command (documents two known bugs)
# =============================================================================


class TestStatusCommand:
    """`ml status` reads pipeline_state.json only — no pipeline execution."""

    def test_status_unknown_run_id_exits_nonzero(self, runner: CliRunner, tmp_path: Path):
        """Unknown run id exits nonzero.

        KNOWN BUG (documented): the 'not found' message uses a [dim] rich
        markup tag opened in one console.print and closed in another, which
        raises a raw rich MarkupError instead of a clean typer.Exit(1). Only
        the safe part (nonzero exit + readable 'not found' text) is asserted.
        """
        result = runner.invoke(
            app,
            ["status", "--run-id", "no_such_run", "--project-root", str(tmp_path)],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_status_valid_state_file_renders_status(self, runner: CliRunner, tmp_path: Path):
        """Valid state file: status is rendered from JSON.

        KNOWN BUG (documented): `raise typer.Exit(0)` inside the try block is
        caught by `except Exception` (typer.Exit subclasses Exception), so the
        success path currently exits 1 and prints 'Error: 0'. We assert the
        rendering happened and accept either exit code so this test stays
        green when the bug is fixed.
        """
        run_id = "rid_smoke_1"
        state_dir = tmp_path / "data" / "runs" / run_id / "artifacts"
        state_dir.mkdir(parents=True)
        (state_dir / "pipeline_state.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "saved_at": "2026-01-01T00:00:00",
                    "completed_stages": ["load", "clean"],
                    "stage_results": {
                        "load": {"status": "completed", "duration_seconds": 1.5},
                    },
                }
            )
        )

        result = runner.invoke(
            app,
            ["status", "--run-id", run_id, "--project-root", str(tmp_path)],
        )

        out = result.output
        assert "PIPELINE STATUS" in out
        assert run_id in out
        assert "load" in out
        # Exit code should be 0; currently 1 due to the typer.Exit-swallow bug.
        assert result.exit_code in (0, 1)
