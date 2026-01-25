Run validation checklist for $ARGUMENTS:

```bash
ruff check src/
python -c "from src.core.types import DataRank; print('OK')"
python -c "from src.core.contracts import get_model_contract; print('OK')"
```

Use integration-checker subagent to verify imports resolve and no circular deps.

Return: ✅ PASS | ❌ FAIL with details and recommended next actions.
