# Feature Selection Documentation Index

**Quick navigation guide** for all feature selection optimization documentation across the ML Factory.

---

## Primary Documentation

### 1. Comprehensive Feature Selection Optimization Guide
**File:** `/Users/sneh/research/docs/guides/FEATURE_SELECTION_OPTIMIZATION.md`

**Best for:** Complete reference with examples, best practices, and implementation details

**Contents:**
- Overview of feature selection pipeline (Stages 8-9)
- All 5 selection strategies explained with code examples
- Per-model feature selection strategies
- Running feature selection (CLI, Python API, unified pipeline)
- Interpreting results with visualizations
- Best practices and optimization tips
- Integration with 16-stage pipeline

**Use when:** You need detailed guidance on implementing or understanding feature selection

---

### 2. Feature Engineering Guide (Sections on Feature Selection)
**File:** `/Users/sneh/research/docs/guides/FEATURE_ENGINEERING.md`

**Relevant Sections:**
- Lines 718-868: Stage 8 - Feature Selection Optimization with Optuna
- Lines 873-1031: Stage 9 - Feature Pruning Optimization with Optuna
- Lines 1038-1639: Advanced Feature Selection with Optuna (all strategies)

**Best for:** Understanding how feature selection fits into feature engineering workflow

**Use when:** You want to see feature selection in the broader context of feature engineering

---

### 3. Hyperparameter Tuning Guide (Feature Optimization Sections)
**File:** `/Users/sneh/research/docs/guides/HYPERPARAMETER_TUNING.md`

**Relevant Sections:**
- Lines 352-521: Stage 8 - Feature Selection Optimization with Optuna
- Lines 523-719: Stage 9 - Feature Pruning Optimization with Optuna

**Best for:** Understanding feature selection as part of overall Optuna optimization workflow

**Use when:** You want to see how feature selection relates to label and hyperparameter optimization

---

## Configuration Files

### 4. Feature Selection Configuration
**File:** `/Users/sneh/research/config/optimization/feature_selection.yaml`

**Best for:** Complete YAML configuration with all options

**Key Sections:**
- Line 37-66: Selection strategies configuration
- Line 70-150: Feature groups definition
- Line 153-197: Importance-based selection parameters
- Line 200-237: RFE parameters
- Line 240-264: Correlation-based parameters

**Use when:** You need to configure feature selection for your experiments

---

### 5. Feature Pruning Configuration
**File:** `/Users/sneh/research/config/optimization/feature_pruning.yaml`

**Best for:** Complete YAML configuration for pruning stage

**Key Sections:**
- Line 97-156: Pruning search space
- Line 158-193: Advanced pruning strategies
- Line 195-241: Importance calculation settings
- Line 321-355: Constraints

**Use when:** You need to configure feature pruning parameters

---

### 6. Optimization README
**File:** `/Users/sneh/research/config/optimization/README.md`

**Best for:** High-level overview of all optimization stages

**Relevant Sections:**
- Line 146-291: Feature Selection Optimization
- Line 295-345: Feature Pruning Optimization

**Use when:** You want a quick overview of feature optimization in the optimization pipeline

---

## Architecture Documentation

### 7. Unified Pipeline Architecture
**File:** `/Users/sneh/research/docs/implementation/UNIFIED_PIPELINE_ARCHITECTURE.md`

**Relevant Sections:**
- Line 73-78: Feature selection in 16-stage pipeline flow
- Line 214-346: Stage 8-9 in Optuna optimization stages
- Line 253-297: Stage 8 detailed architecture
- Line 299-346: Stage 9 detailed architecture

**Best for:** Understanding how feature selection fits into the full ML pipeline

**Use when:** You want to understand the big picture and how stages connect

---

### 8. Per-Model Feature Selection Architecture
**File:** `/Users/sneh/research/.serena/knowledge/per_model_feature_selection.md`

**Best for:** Understanding why different models need different features

**Key Sections:**
- Line 54-108: Tabular models feature selection
- Line 111-156: Sequence models feature selection
- Line 159-197: Advanced transformers feature selection
- Line 239-275: Diversity mechanisms

**Use when:** You want to understand model-specific feature requirements

---

## Quick Reference Tables

### When to Use Which Document

| Your Question | Best Document |
|---------------|---------------|
| How do I run feature selection? | FEATURE_SELECTION_OPTIMIZATION.md |
| What are all the selection strategies? | FEATURE_SELECTION_OPTIMIZATION.md |
| How do I configure feature selection? | feature_selection.yaml + FEATURE_SELECTION_OPTIMIZATION.md |
| What features should XGBoost use? | per_model_feature_selection.md |
| How does feature selection fit into the pipeline? | UNIFIED_PIPELINE_ARCHITECTURE.md |
| What's the difference between Stage 8 and Stage 9? | FEATURE_SELECTION_OPTIMIZATION.md |
| How do I interpret feature selection results? | FEATURE_SELECTION_OPTIMIZATION.md |
| What are best practices? | FEATURE_SELECTION_OPTIMIZATION.md |
| How does Optuna optimize features? | HYPERPARAMETER_TUNING.md |

### Feature Selection Strategies Overview

| Strategy | Document | Lines | Best For |
|----------|----------|-------|----------|
| Binary Group Selection | FEATURE_SELECTION_OPTIMIZATION.md | 224-342 | Fast, coarse filtering |
| Binary Individual Selection | FEATURE_SELECTION_OPTIMIZATION.md | 344-401 | Fine-grained control |
| RFE (Recursive Elimination) | FEATURE_SELECTION_OPTIMIZATION.md | 403-482 | Model-aware selection |
| Importance-Based | FEATURE_SELECTION_OPTIMIZATION.md | 484-572 | Fast, interpretable |
| Correlation-Based | FEATURE_SELECTION_OPTIMIZATION.md | 574-641 | Redundancy removal |

### Configuration Quick Reference

| Configuration Aspect | File | Section |
|---------------------|------|---------|
| Enable/disable strategies | feature_selection.yaml | Line 37-66 |
| Feature groups | feature_selection.yaml | Line 76-150 |
| Importance methods | feature_selection.yaml | Line 155-197 |
| RFE parameters | feature_selection.yaml | Line 200-237 |
| Correlation parameters | feature_selection.yaml | Line 240-264 |
| Pruning thresholds | feature_pruning.yaml | Line 97-156 |
| Constraints | feature_selection.yaml & feature_pruning.yaml | Line 312-327 & 321-355 |

---

## Related Topics

### Complementary Documentation

1. **Feature Engineering** - How features are created (Stage 5)
   - File: `docs/guides/FEATURE_ENGINEERING.md`

2. **Label Optimization** - How labels are optimized (Stage 7)
   - File: `docs/guides/HYPERPARAMETER_TUNING.md` (lines 75-349)

3. **Hyperparameter Optimization** - Model hyperparameter tuning (Stage 13)
   - File: `docs/guides/HYPERPARAMETER_TUNING.md` (lines 722-1003)

4. **Cross-Validation** - How to validate feature selection
   - File: `docs/guides/CROSS_VALIDATION.md` (if exists)

---

## Quick Start

### Minimal Working Example

```python
# 1. Read comprehensive guide
open("docs/guides/FEATURE_SELECTION_OPTIMIZATION.md")

# 2. Configure your feature selection
open("config/optimization/feature_selection.yaml")

# 3. Run feature selection
# See FEATURE_SELECTION_OPTIMIZATION.md lines 690-730

# 4. Interpret results
# See FEATURE_SELECTION_OPTIMIZATION.md lines 732-815
```

### Recommended Reading Order

1. **First time:** FEATURE_SELECTION_OPTIMIZATION.md (full read)
2. **Quick reference:** INDEX_FEATURE_SELECTION.md (this file)
3. **Configuration:** feature_selection.yaml + feature_pruning.yaml
4. **Integration:** UNIFIED_PIPELINE_ARCHITECTURE.md (Stage 8-9 sections)
5. **Per-model tuning:** per_model_feature_selection.md

---

## Document Maintenance

**Last Updated:** 2026-01-18

**Primary Maintainer:** ML Factory Team

**Update Schedule:** When feature selection implementation changes

**Cross-References Verified:** 2026-01-18

---

## Feedback

If you find gaps in documentation or have suggestions:
1. Check if topic is covered in comprehensive guide
2. Consult configuration files for detailed parameters
3. Review architecture docs for pipeline integration
4. File issue or update documentation directly

---

*This index is automatically updated when feature selection documentation changes.*
