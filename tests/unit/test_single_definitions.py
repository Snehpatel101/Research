"""Structural invariant: each core concept is defined exactly once in src/.

Duplicate definitions cause import confusion and divergent behavior (the reason
ExperimentConfig, FeatureSelectionResult and the type enums were deduplicated).
"""

from __future__ import annotations

import ast
from collections import defaultdict

import pytest

from tests.helpers import REPO_ROOT

CANONICAL_CLASSES = [
    "ExperimentConfig",
    "FeatureSelectionResult",
    "DataRank",
    "ModelFamily",
    "PipelineConfig",
    "BacktestConfig",
]


@pytest.fixture(scope="module")
def definitions() -> dict[str, list[str]]:
    found: dict[str, list[str]] = defaultdict(list)
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in CANONICAL_CLASSES:
                found[node.name].append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return found


@pytest.mark.parametrize("name", CANONICAL_CLASSES)
def test_class_is_defined_exactly_once(name: str, definitions: dict[str, list[str]]) -> None:
    assert len(definitions[name]) == 1, f"{name} defined in {definitions[name]}"
