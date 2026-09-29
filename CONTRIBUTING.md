# Contributing

## Setup

```bash
make install-dev          # uv venv (Python 3.11, CPU torch) + dev tools + git hooks
```

## Before every commit

```bash
make check                # ruff, black --check, pyright (0 errors), vulture, fast tests
```

Slow end-to-end tests: `make test-slow`. The full mix-and-match verification:
`make matrix` (hours on CPU; `scripts/mix_match.py <kind> --resume` continues an
interrupted run).

## Rules of the house

- **Behaviour first:** every bug fix comes with a test that fails before the fix.
  Tests assert behaviour, never source text.
- **No leakage:** anything that fits on data (scalers, feature selection,
  calibration, tuning, early stopping) sees training rows only; CV splits purge
  on label spans and embargo after each test block.
- **Train/serve parity:** inference replays the recorded feature spec, scaler and
  model; `scripts/mix_match.py` checks it.
- **Delete, don't adapt:** remove dead code instead of keeping compatibility shims.
- **One definition per concept**, imported from its canonical location.

## Adding a model

1. Implement `BaseModel` (`fit`, `predict`, `save`, `load`) in `src/models/<family>/`.
2. Register it with `@register(name=..., family=...)` (`src/models/registry.py`).
3. Add its contract (input rank, sequence length, feature limits) in
   `src/core/contracts/model_contract.py`.
4. Run `python scripts/mix_match.py custom <your_model>,xgboost` — it must pass
   training, stacking, deploy and prediction-parity checks.

The full walkthrough is in [docs/mix-and-match.md](docs/mix-and-match.md#adding-a-model).

## Documentation

User docs live in `docs/` and build into a site with mkdocs-material
(`mkdocs.yml`); the API reference is generated from docstrings.

```bash
make docs-gen             # regenerate docs/configuration.md and docs/cli.md from the code
make docs                 # fail on stale generated pages, then mkdocs build --strict
make docs-serve           # live preview on http://127.0.0.1:8000
python scripts/check_md_links.py   # every relative link in every *.md resolves
```

Changing `ExperimentConfig` or a CLI option? Run `make docs-gen` and commit the
regenerated pages — CI fails when they are stale. Historical audits and phase
reports go to `docs/archive/` (excluded from the site), not the repository root.

## Project documents

`CLAUDE.md` (status and conventions) · `DIRECTION.md` (architecture) ·
`CLEANUP_PLAN.md` / `CLEANUP_TASKS.md` (phases) · `COMPLETION.md` (history) ·
`DECISIONS.md` (open decisions) · `CHANGELOG.md` (user-facing changes).
