# Contributing to ML Model Factory

Thank you for contributing! This guide will help you get started.

## Code of Conduct
- Be respectful and professional
- Focus on constructive feedback
- Assume good intent

## Before You Start
1. Read [CLAUDE.md](../CLAUDE.md) - Repository constitution and engineering rules
2. Review [Project Charter](../docs/planning/PROJECT_CHARTER.md) - Current status and roadmap
3. Check existing issues to avoid duplicates

## Development Workflow

### 1. Setup Development Environment
```bash
# Clone repository
git clone <repo-url>
cd Research

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests to verify setup
pytest tests/ -v
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
# or
git checkout -b model/model-name
```

### 3. Make Changes
Follow engineering rules from CLAUDE.md:
- **File size limits:** Target 650 lines, max 800 lines
- **Fail fast:** Validate inputs at boundaries
- **No exception swallowing:** Let errors propagate naturally
- **Delete legacy code:** Remove unused code immediately
- **Modular design:** Small, composable modules with clear contracts

### 4. Test Your Changes
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/models/boosting/test_xgboost.py -v

# Run integration tests
pytest tests/integration/ -v
```

### 5. Commit Changes
```bash
# Add files
git add <files>

# Commit with descriptive message
git commit -m "feat: add InceptionTime model implementation

- Implement BaseModel interface
- Add multi-scale CNN architecture
- Add configuration file
- Add unit and integration tests
- Register in ModelRegistry"
```

**Commit Message Format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation update
- `test:` Test additions/updates
- `refactor:` Code refactoring
- `perf:` Performance improvement
- `chore:` Maintenance tasks

### 6. Push and Create Pull Request
```bash
# Push to remote
git push origin feature/your-feature-name

# Create PR on GitHub
# Use pull request template
```

## Adding a New Model

See [Model Integration Guide](../docs/guides/MODEL_INTEGRATION_GUIDE.md) for detailed instructions.

### Quick Checklist
1. **Create model class** implementing `BaseModel` interface
2. **Add registration** via `@register(name="...", family="...")`
3. **Create config file** at `config/models/{model_name}.yaml`
4. **Write tests:**
   - Unit tests: `tests/unit/models/{family}/test_{model_name}.py`
   - Integration tests: `tests/integration/models/test_{model_name}_integration.py`
5. **Update documentation:**
   - Add to `docs/implementation/PHASE_6_TRAINING.md`
   - Update model count in `docs/planning/PROJECT_CHARTER.md`
6. **Verify registration:**
   ```bash
   python scripts/train_model.py --list-models
   ```

## Adding New Features

See [Feature Engineering Guide](../docs/guides/FEATURE_ENGINEERING.md).

### Feature Engineering Workflow
1. **Add feature calculation** in `src/phase1/stages/features/`
2. **Update feature config** (if needed)
3. **Add feature tests** in `tests/unit/phase1/stages/features/`
4. **Validate no lookahead bias:**
   - Use only current + historical data
   - No future peeking
5. **Document feature:**
   - Add to `docs/guides/FEATURE_ENGINEERING.md`
   - Include formula and interpretation

## File Organization

### Root Directory
- `CLAUDE.md` - Repository constitution (do not modify without discussion)
- `CLAUDE2.md` - Extended instructions
- `README.md` - User-facing documentation
- `pyproject.toml` - Package configuration
- `requirements.txt` - Dependencies

### Source Code
```
src/
├── phase1/         → Data pipeline (14 stages)
├── models/         → Model implementations (13 models)
├── cross_validation/ → CV and OOF generation
└── utils/          → Shared utilities
```

### Configuration
```
config/
├── models/         → Model-specific configs
├── ga_results/     → Genetic algorithm optimization results
├── training.yaml   → Global training settings
└── cv.yaml         → Cross-validation settings
```

### Documentation
```
docs/
├── getting-started/ → Quickstart guides
├── guides/         → Implementation guides
├── phases/         → Phase-specific documentation
├── planning/       → Project planning documents
├── reference/      → Technical references
└── research/       → Research papers and analysis
```

## Testing Requirements

### Unit Tests
- Test individual functions/classes in isolation
- Use mocks for external dependencies
- Fast execution (< 1 second per test)
- Coverage target: > 80%

### Integration Tests
- Test complete workflows (e.g., model training)
- Use real data containers (synthetic OHLCV)
- Verify end-to-end functionality
- Execution time: < 5 minutes per test

### Regression Tests
- Lock down previously fixed bugs
- Prevent regressions
- Include reproduction steps in test docstring

## Code Style

### Python Style
- Follow PEP 8
- Use type hints
- Google-style docstrings
- Black formatter (line length: 100)

### Example
```python
from typing import Optional
import numpy as np

def calculate_rsi(
    close: np.ndarray,
    period: int = 14,
    fillna: Optional[float] = None
) -> np.ndarray:
    """Calculate Relative Strength Index (RSI).

    Args:
        close: Close prices (1D array).
        period: RSI period (default: 14).
        fillna: Value to fill NaN (default: None).

    Returns:
        RSI values (0-100 range).

    Raises:
        ValueError: If period < 1 or close is empty.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # Implementation...
```

## Documentation Style

### Markdown Guidelines
- Use headings hierarchically (H1 → H2 → H3)
- Include code blocks with language tags
- Add tables for structured data
- Link to related documents

### Code Comments
- Explain **why**, not **what**
- Document assumptions and invariants
- Include references to papers/formulas

## Review Process

### What Reviewers Look For
1. **Correctness:** Does it work? Are tests comprehensive?
2. **Code quality:** Is it modular? Does it follow style guidelines?
3. **File size:** Are files within limits (650 target, 800 max)?
4. **Documentation:** Is it well-documented?
5. **No regressions:** Does it break existing functionality?
6. **Validation:** Are inputs validated? Fail-fast implemented?

### How to Respond to Feedback
- Address all comments
- Ask questions if unclear
- Push updates to same branch (don't force-push)
- Mark conversations as resolved when addressed

## Common Pitfalls

### ❌ Don't Do This
- Don't swallow exceptions silently
- Don't create files > 800 lines without justification
- Don't introduce lookahead bias in features/labels
- Don't skip input validation
- Don't leave commented-out code
- Don't commit without running tests

### ✅ Do This
- Fail fast with clear error messages
- Keep files modular (< 650 lines)
- Validate all inputs at boundaries
- Delete unused code immediately
- Run full test suite before committing
- Write descriptive commit messages

## Questions?

- Open an issue with question label
- Check existing documentation first
- Be specific about your question

## Thank You!

Your contributions make this project better. We appreciate your time and effort!
