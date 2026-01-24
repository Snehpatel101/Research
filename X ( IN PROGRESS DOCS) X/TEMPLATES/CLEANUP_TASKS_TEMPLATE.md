# [System] Cleanup Tasks - Active Work

**Last Updated:** [DATE]
**Status:** Phases [X-Y] Complete | Phase [Z] [In Progress/Pending]
**Net Impact:** [+/- lines] | [Key metrics summary]

---

## Table of Contents
- [Post-Cleanup Verification](#post-cleanup-verification)
- [Phase N: Current Phase](#phase-n-theme)
- [Phase N Execution Priority](#phase-n-execution-priority)
- [Completed Phases](#completed-phases)

---

## Post-Cleanup Verification

**[DATE] Verification:**
- [N] parallel agents verified against [system/criteria]
- All checks passed / [X] issues identified
- Status: COMPLETE / IN PROGRESS

---

## Phase [N]: [Theme] ([DATE])

**Source:** [N-agent analysis / Manual review]
**Status:** Analysis Complete | Execution [In Progress/Pending/Complete]
**Target:** [Outcome metrics - e.g., "Remove ~500 lines of dead code"]

---

### [N]A: [Specific Task] - [STATUS]

**Priority:** [CRITICAL | HIGH | MEDIUM | LOW]
**Effort:** [LOW | MEDIUM | HIGH]
**Location:** `file.py:line-line`

**Problem:**
[Context about what's wrong and why it matters]

**Current Code:**
```python
# What exists now
existing_problematic_code()
```

**Fix:**
```python
# What it should be
corrected_code()
```

**Verification:**
- [ ] Code compiles
- [ ] Tests pass
- [ ] No regressions

**Commit:** [hash] (after completion)

---

### [N]B: [Specific Task] - [STATUS]

**Priority:** [LEVEL]
**Effort:** [LEVEL]
**Location:** `another_file.py:line-line`

| Subtask | Priority | Lines | Status |
|---------|----------|-------|--------|
| [Subtask 1 description] | HIGH | ~10 | Done |
| [Subtask 2 description] | MEDIUM | ~5 | Pending |
| [Subtask 3 description] | LOW | ~3 | Pending |

**Impact:** [Metric improvement after completion]

---

### [N]C: [Specific Task] - [STATUS]

**Priority:** [LEVEL]
**Effort:** [LEVEL]
**Blocked By:** [N]A, [N]B

[Description of task that depends on prior work]

---

## Phase [N] Execution Priority

| Sub-Phase | Priority | Tasks | Est. Effort | Dependencies | Status |
|-----------|----------|-------|-------------|--------------|--------|
| [N]A | CRITICAL | 3 | 2h | None | Pending |
| [N]B | HIGH | 2 | 1h | [N]A | Blocked |
| [N]C | MEDIUM | 4 | 3h | None | Pending |
| [N]D | LOW | 1 | 30m | [N]B, [N]C | Blocked |

**Total Estimated Effort:** [X]h
**Parallelizable Tasks:** [N]A, [N]C

---

## Completed Phases

### Phase [N-1]: [Theme] ([DATE]) - COMPLETE

**Summary:** [Brief description of what was accomplished]
**Lines Removed:** [XXX]
**Commit:** [hash]

| Task | Status |
|------|--------|
| [Task 1] | Done |
| [Task 2] | Done |
| [Task 3] | Deferred to Phase [M] |

---

### Phase [N-2]: [Theme] ([DATE]) - COMPLETE

[Similar format]

---

## False Positives / Verified OK

Items investigated but determined to NOT need fixing:

| Item | File | Reason OK |
|------|------|-----------|
| [Suspected issue] | `file.py` | [Why it's actually fine] |

---

## Deferred Items

Items explicitly postponed (not forgotten):

| Item | Original Phase | Reason Deferred | Target Phase |
|------|----------------|-----------------|--------------|
| [Task] | [N] | [Rationale] | [M] / Indefinite |

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| Done | Completed and verified |
| Pending | Ready to start |
| In Progress | Currently being worked on |
| Blocked | Waiting on dependency |
| Deferred | Explicitly postponed |
| Reserved | Claimed by specific agent/developer |

---

## Quick Reference

**File Locations:**
- Main config: `path/to/config.py`
- Core services: `path/to/services/`
- Entry point: `path/to/main.py`

**Commands:**
```bash
# Type check
mypy project/

# Lint
ruff check .

# Run
python -m project
```
