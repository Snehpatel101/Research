## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Model implementation (new model added to registry)
- [ ] Feature engineering (new features or indicators)
- [ ] Configuration change

## Changes Made
- List key changes here
- One per line

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally: `pytest tests/`
- [ ] Manual testing completed

### Model-Specific Testing (if applicable)
- [ ] Model registered successfully in ModelRegistry
- [ ] Appears in `./train_model.py --list-models`
- [ ] Smoke test passes: `pytest tests/integration/models/test_{model}_integration.py`
- [ ] Training completes without errors
- [ ] Predictions have correct shape and format
- [ ] Save/load cycle works correctly

## Configuration
- [ ] Configuration file added/updated: `config/models/{model}.yaml`
- [ ] Configuration validates correctly
- [ ] Environment-specific configs tested (if applicable)

## Documentation
- [ ] Code comments added/updated
- [ ] Docstrings added/updated (Google style)
- [ ] Documentation files updated (if applicable)
- [ ] README updated (if applicable)

## File Size Compliance
- [ ] All modified/new files stay within limits (target: 650 lines, max: 800 lines)
- [ ] If any file exceeds 650 lines, justification provided below

### File Size Justification (if applicable)
[Explain why any file exceeds 650 lines and why it cannot be reasonably split]

## Validation Checklist
- [ ] No lookahead bias introduced
- [ ] Purge/embargo preserved in new code
- [ ] Input validation added at boundaries
- [ ] Fail-fast error handling implemented
- [ ] No exception swallowing (avoid bare try/except)

## Performance Impact
- [ ] No performance regression
- [ ] Benchmarked if performance-critical
- [ ] GPU memory usage documented (if applicable)

## Breaking Changes
- [ ] No breaking changes
- [ ] Breaking changes documented below

### Breaking Changes Description (if applicable)
[Describe any breaking changes and migration path]

## Related Issues
Closes #[issue_number]
Relates to #[issue_number]

## Checklist
- [ ] Code follows repository style guidelines (see CLAUDE.md)
- [ ] Self-review completed
- [ ] Code is modular and well-documented
- [ ] No legacy/dead code introduced
- [ ] Git history is clean (no merge commits, descriptive messages)
