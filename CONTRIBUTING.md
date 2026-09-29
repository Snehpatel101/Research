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

## Project documents

`CLAUDE.md` (status and conventions) · `DIRECTION.md` (architecture) ·
`CLEANUP_PLAN.md` / `CLEANUP_TASKS.md` (phases) · `COMPLETION.md` (history) ·
`DECISIONS.md` (open decisions) · `CHANGELOG.md` (user-facing changes).
