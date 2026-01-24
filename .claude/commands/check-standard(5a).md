Run validation checklist for $ARGUMENTS:

Parallel task agents verify:
- `ruff check .` passes
- `pyright topstepx_backend/` passes (0 errors)
- Backend starts without errors (`python -m topstepx_backend`)
- Frontend builds (`cd topstepx_frontend && npm run build`)
- Frontend typecheck passes (`npm run typecheck`)

Report: Pass/Fail status, any warnings, recommended next actions.
