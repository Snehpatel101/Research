"""
Check that relative links in the repository's Markdown files resolve.

Scans every ``*.md`` file (outside build, VCS, virtualenv and experiment
output directories) for inline links ``[text](target)`` and reference
definitions ``[label]: target``, ignoring external URLs, pure ``#anchor``
links and anything inside code spans or fenced code blocks, and reports
targets that do not exist on disk.

Usage:
    python scripts/check_md_links.py            # exit 1 when a link is broken
    python scripts/check_md_links.py docs README.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "site", "experiments", "node_modules", "worktrees", "__pycache__"}
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)  # http:, mailto:, //host
INLINE = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*(<[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)")
FENCE = re.compile(r"^\s*(```|~~~)")
CODE_SPAN = re.compile(r"`+[^`]*`+")


def markdown_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        for md in sorted(path.rglob("*.md")):
            if not SKIP_DIRS.intersection(md.relative_to(REPO_ROOT).parts[:-1]):
                files.append(md)
    return files


def broken_links(md: Path) -> list[tuple[int, str]]:
    problems: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        text = CODE_SPAN.sub("", line)
        targets = [m.group(1) for m in INLINE.finditer(text)]
        ref = REFERENCE.match(text)
        if ref:
            targets.append(ref.group(1))
        for raw in targets:
            target = raw.strip("<>")
            if EXTERNAL.match(target) or target.startswith("#"):
                continue
            path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not path_part:
                continue
            resolved = (
                REPO_ROOT / path_part.lstrip("/")
                if path_part.startswith("/")
                else md.parent / path_part
            )
            if not resolved.exists():
                problems.append((lineno, target))
    return problems


def main() -> None:
    args = [Path(a).resolve() for a in sys.argv[1:]] or [REPO_ROOT]
    files = markdown_files(args)
    n_broken = 0
    for md in files:
        for lineno, target in broken_links(md):
            n_broken += 1
            print(f"{md.relative_to(REPO_ROOT)}:{lineno}: broken link -> {target}")
    print(f"Checked {len(files)} Markdown files: {n_broken} broken relative link(s)")
    sys.exit(1 if n_broken else 0)


if __name__ == "__main__":
    main()
