# Developer Setup Guide

This guide covers the developer experience improvements for this project, including pre-commit hooks, linting, formatting, and useful commands.

## Quick Start

```bash
# 1. Install dev dependencies and pre-commit hooks
make install-dev

# 2. Verify setup
make pre-commit
```

## Pre-commit Hooks

Pre-commit hooks run automatically before each git commit to catch issues early.

### What Gets Checked

1. **Ruff Linting** - Auto-fixes common code issues
2. **Ruff Formatting** - Ensures consistent code style
3. **Mypy Type Checking** - Checks type hints in `src/` directory
4. **Quick Tests** - Runs essential tests to catch breaking changes

### Configuration

Pre-commit configuration is in `.pre-commit-config.yaml`:

- **Ruff**: Auto-fixes issues without blocking commits
- **Mypy**: Only checks `src/` directory, skips tests/scripts/examples
- **Pytest**: Runs quick smoke tests on staged files

### Manual Usage

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files src/my_file.py

# Run specific hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files

# Skip hooks for a commit (use sparingly!)
git commit --no-verify -m "message"

# Update hook versions
make pre-commit-update
```

## Code Quality Tools

### Ruff (Linting + Formatting)

Ruff is a fast Python linter and formatter that replaces multiple tools.

```bash
# Auto-fix linting issues
make lint

# Format code
make format

# Or run manually
ruff check --fix .
ruff format .
```

**Configuration**: `pyproject.toml` under `[tool.ruff]`
- Line length: 100
- Target version: Python 3.11
- Enabled rules: E, W, F, I, B, C4, UP, SIM

### Mypy (Type Checking)

Static type checker for Python.

```bash
# Run type checking
make type-check

# Or run manually
mypy src/ --ignore-missing-imports
```

**Configuration**: `pyproject.toml` under `[tool.mypy]`

### Pytest

```bash
# Run all tests with coverage
make test

# Run quick tests only
make test-quick

# Run specific test file
pytest tests/test_specific.py -v
```

## Makefile Commands

The project includes a Makefile with helpful shortcuts:

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install package in dev mode |
| `make install-dev` | Install with dev dependencies + pre-commit |
| `make test` | Run all tests with coverage |
| `make test-quick` | Run quick tests only |
| `make lint` | Run ruff linter with auto-fix |
| `make format` | Format code with ruff |
| `make type-check` | Run mypy type checking |
| `make pre-commit` | Run all pre-commit hooks |
| `make pre-commit-update` | Update hook versions |
| `make clean` | Remove build artifacts and cache |

## Git Workflow

### Recommended Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
# ... edit files ...

# 3. Run checks manually (optional, will run on commit anyway)
make lint
make format
make test-quick

# 4. Stage changes
git add .

# 5. Commit (pre-commit hooks run automatically)
git commit -m "feat: add new feature"

# 6. If hooks fail, fix issues and try again
# ... fix issues ...
git add .
git commit -m "feat: add new feature"
```

### Commit Message Conventions

Follow conventional commits format:

- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `docs:` - Documentation changes
- `test:` - Test changes
- `chore:` - Build/tooling changes

## Troubleshooting

### Pre-commit hooks failing

```bash
# Run hooks manually to see details
pre-commit run --all-files

# Fix auto-fixable issues
make lint
make format

# For type errors, may need to add type hints or ignore
```

### Hooks taking too long

Pre-commit only runs on staged files by default. If it's slow:

```bash
# Run on specific files only
pre-commit run --files src/changed_file.py

# Skip mypy if you know types are clean
SKIP=mypy git commit -m "message"
```

### Updating dependencies

```bash
# Update pre-commit hook versions
make pre-commit-update

# Update Python packages
pip install -e ".[dev]" --upgrade
```

## IDE Integration

### VSCode

Recommended settings for `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "none",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "python.analysis.typeCheckingMode": "basic",
  "mypy.enabled": true
}
```

Recommended extensions:
- `charliermarsh.ruff` - Ruff
- `ms-python.mypy-type-checker` - Mypy
- `ms-python.python` - Python

### PyCharm

1. Go to Settings → Tools → External Tools
2. Add Ruff as external tool
3. Enable "Run on save" for formatting

## Performance Tips

- Pre-commit caches environments, so it's fast after first run
- Use `make lint` and `make format` during development
- Run `make test-quick` before committing large changes
- Run `make clean` periodically to remove cache files

## Further Reading

- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Conventional Commits](https://www.conventionalcommits.org/)
