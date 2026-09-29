"""Vulture whitelist: names that are used only dynamically (signal handlers,
context-manager protocol, registry decorators, CLI commands).

Run ``vulture`` (config in pyproject.toml). Add an entry here only when the
name is genuinely reached at runtime but vulture cannot see it; delete real
dead code instead of whitelisting it.
"""
