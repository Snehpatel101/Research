# Task Completion Checklist

## Before Marking Any Task Complete

### 1. Code Quality

- [ ] Code follows Black formatting (run `black src tests scripts`)
- [ ] Code passes Ruff linting (run `ruff check src tests scripts`)
- [ ] Type hints added for all public functions/methods
- [ ] Docstrings for classes and public methods (Google-style)
- [ ] No unused imports or dead code
- [ ] File size within limits (target 650, max 800 lines)

### 2. Testing

- [ ] Unit tests added for new functionality
- [ ] All existing tests pass: `pytest tests/ -v`
- [ ] Tests are deterministic and fast
- [ ] Integration tests if cross-module changes
- [ ] Coverage maintained or improved

### 3. Validation (ML-Specific)

- [ ] No lookahead bias in features/labels
- [ ] `shift(1)` applied to all computed features used at bar[i]
- [ ] Purge/embargo properly configured for CV
- [ ] Train-only transforms (scaling, feature selection)
- [ ] OOF predictions used for stacking (no leakage)

### 4. Error Handling

- [ ] Input validation at boundaries
- [ ] Error messages are actionable
- [ ] No exception swallowing
- [ ] Edge cases handled

### 5. Documentation

- [ ] Function docstrings complete
- [ ] CLAUDE.md updated if contracts changed
- [ ] No orphaned documentation files

## Commands to Run Before Completion

```bash
# Format
black src tests scripts

# Lint
ruff check src tests scripts

# Tests
pytest tests/ -v

# Type check (optional but recommended)
mypy src

# Lookahead validation (for ML code)
pytest tests/phase_1_tests/stages/test_lookahead_invariance.py -v
```

## What NOT to Do

- Do not commit files with secrets (.env, credentials)
- Do not leave TODO/FIXME without tracking in issues
- Do not add features beyond what was requested
- Do not create documentation unless explicitly asked
- Do not leave dead code "just in case" (git is the archive)
- Do not swallow exceptions with bare try/except

## Git Commit Guidelines

When asked to commit:

1. Stage only relevant changes (`git add -p` for interactive staging)
2. Write clear commit message:
   - Format: `type: description`
   - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
   - Focus on WHY, not WHAT
3. Do NOT force push unless explicitly requested
4. Include co-author line for AI assistance:
   ```
   Co-Authored-By: Claude <noreply@anthropic.com>
   ```

## Post-Completion Verification

After completing a task, use Serena's thinking tools:

```
think_about_whether_you_are_done
```

This ensures:
- All requirements addressed
- No regressions introduced
- Tests pass
- Documentation updated if needed
