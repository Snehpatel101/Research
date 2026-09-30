"""CLI tests read Rich-rendered help; pin plain, wide output so CI terminals match local runs."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def plain_wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    # GitHub Actions (GITHUB_ACTIONS / FORCE_COLOR) makes Rich color and wrap option names
    for name in ("GITHUB_ACTIONS", "FORCE_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "200")
