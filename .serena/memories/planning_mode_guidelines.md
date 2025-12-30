# Planning Mode Guidelines for Serena

## When to Use Planning Mode

Switch to planning mode (`--mode planning`) when:

1. **Designing new features** - Before implementing any significant new functionality
2. **Architectural decisions** - Choosing between patterns or technologies
3. **Refactoring** - Planning large-scale code changes
4. **Bug investigation** - Understanding root causes before fixing
5. **Phase transitions** - Moving from one phase to another

## Planning Mode Workflow

### Step 1: Gather Information
```
- Use find_symbol to locate relevant code
- Use get_symbols_overview to understand file structure
- Use find_referencing_symbols to trace dependencies
- Use search_for_pattern for text-based searches
```

### Step 2: Analyze & Document
```
- Use think_about_collected_information to verify completeness
- Create a plan document with clear steps
- Identify risks and dependencies
```

### Step 3: Validate Plan
```
- Check against project constraints (file size, leakage prevention)
- Verify compatibility with existing architecture
- Identify test requirements
```

### Step 4: Exit Planning
```
- Switch to editing mode for implementation
- Reference plan document during implementation
```

## Planning Templates

### Feature Planning Template
```markdown
## Feature: [Name]

### Goal
[What this feature accomplishes]

### Affected Files
- [file1.py] - [changes needed]
- [file2.py] - [changes needed]

### Implementation Steps
1. [Step 1]
2. [Step 2]
...

### Tests Required
- [ ] Unit test for [component]
- [ ] Integration test for [flow]

### Risks
- [Risk 1]: [Mitigation]
```

### Bug Fix Planning Template
```markdown
## Bug: [Description]

### Root Cause
[Analysis of why the bug occurs]

### Files Affected
- [file.py:line_number] - [what's wrong]

### Fix Approach
1. [Step 1]
2. [Step 2]

### Validation
- [ ] Existing tests still pass
- [ ] New test covers the bug scenario
```

## Key Questions for Planning

### For New Features
- Does this violate single-contract isolation?
- Could this introduce lookahead bias?
- Does this fit within the 800-line file limit?
- What tests are needed?

### For Bug Fixes
- Is this the root cause or a symptom?
- Are there other instances of this bug pattern?
- What's the minimal fix that doesn't introduce new issues?

### For Refactoring
- What's the motivation (file size, complexity, coupling)?
- What's the migration path?
- Can this be done incrementally?

## Planning for Critical Bug Fixes

### HMM Lookahead Fix
1. Review `hmm.py` lines 329-354
2. Decide: disable expanding OR fix incrementally
3. Plan test changes for lookahead_invariance
4. Document expected performance impact

### GA Optimization Fix
1. Review pipeline stage ordering
2. Plan split-before-optimize approach
3. Update stage dependencies
4. Add leakage validation test

### Transaction Costs Fix
1. Review `triple_barrier.py`
2. Calculate cost adjustments per symbol
3. Plan label recalculation
4. Document expected label distribution changes
