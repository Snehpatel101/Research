Run validation checklist for $ARGUMENTS:
- `ruff check src/`
- Core imports work
- No circular deps

**Tiered alternatives:**
| Tier | Command | Use When |
|------|---------|----------|
| Light | `/check-standard(5a)` | Full validation suite |
| Medium | `/check-deep(5b)` | 4-way comprehensive |
| Heavy | `/check-behavior(5c)` | Execution tracing |
