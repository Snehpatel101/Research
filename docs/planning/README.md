# Planning Documentation

Strategic planning documents for the ML model factory project.

## 📋 Planning Documents

### [Project Charter](PROJECT_CHARTER.md)
**Official project status and goals**

**Current Status:**
- ✅ **13 models implemented** across 4 families
- 🚧 **6 models planned** (advanced architectures)
- ✅ **Phase 1-5 complete** (data, training, CV, ensemble, walk-forward)

**Model Inventory:**
- Boosting: XGBoost, LightGBM, CatBoost (3)
- Neural: LSTM, GRU, TCN, Transformer (4)
- Classical: Random Forest, Logistic, SVM (3)
- Ensemble: Voting, Stacking, Blending (3)

**Planned Models:**
- InceptionTime, 1D ResNet, PatchTST, iTransformer, TFT, N-BEATS (6)

---

### [Alignment Plan](ALIGNMENT_PLAN.md)
**Repository alignment strategy**

Comprehensive analysis of:
- Documentation vs implementation alignment
- Current capabilities vs documented features
- Roadmap to close gaps
- Priority actions and sequencing

**Key Topics:**
- Model factory architecture
- Multi-timeframe (MTF) capabilities
- Configuration system design
- Testing and validation coverage
- Development workflow

---

### [Repository Organization Analysis](REPO_ORGANIZATION_ANALYSIS.md)
**Current vs target state analysis**

**Analysis Sections:**
1. Model inventory (13 actual vs 19 target)
2. MTF timeframe gaps (missing 20min, 25min)
3. Configuration system gaps
4. Documentation conflicts
5. Testing coverage assessment
6. Development workflow recommendations
7. Priority action items

**Goals Identified:**
- Expand to 19 models (6 new advanced models)
- Complete 9-timeframe MTF ladder
- Standardize configuration system
- Comprehensive testing coverage

---

## 🎯 Strategic Goals

### Short Term (Current Phase)
1. ✅ Complete Phase 1-5 pipeline
2. ✅ Document current implementation (13 models)
3. 🚧 Create implementation roadmaps for advanced models
4. 🚧 Organize repository structure

### Medium Term (Next 3-6 Months)
1. 🎯 Implement 6 advanced models (18 days effort)
2. 🎯 Complete 9-timeframe MTF ladder
3. 🎯 Comprehensive model comparison benchmarks
4. 🎯 Production deployment documentation

### Long Term (6-12 Months)
1. 🎯 Walk-forward validation on multiple contracts
2. 🎯 Real-time inference pipeline
3. 🎯 Model monitoring and drift detection
4. 🎯 Automated retraining workflows

---

## 📊 Implementation Timeline

**Advanced Models (18 days):**
- Week 1: InceptionTime (3d) + 1D ResNet (2d)
- Week 2: PatchTST (4d) + iTransformer (3d)
- Week 3: TFT (5d)
- Week 4: N-BEATS (1d) + integration testing (2d)

**MTF Expansion (Parallel):**
- Week 1-2: Add 20min/25min timeframes, validate alignment
- Week 3-4: Model-specific MTF strategies, testing

**Total:** ~1 month @ 2 engineers (overlapping work streams)

---

## 🔍 Decision Log

### Model Selection Rationale

**Why these 6 advanced models?**

1. **InceptionTime** - SOTA for time series classification, multi-scale learning
2. **1D ResNet** - Deep learning baseline, proven architecture
3. **PatchTST** - Current SOTA for long-term forecasting (2023)
4. **iTransformer** - Novel attention mechanism for multivariate TS (2024)
5. **TFT** - Interpretable quantile forecasting, variable selection
6. **N-BEATS** - Interpretable decomposition, no feature engineering needed

**Selection Criteria:**
- ✅ Published research with strong benchmarks
- ✅ Diverse architectural approaches
- ✅ Different strengths (classification vs forecasting vs decomposition)
- ✅ Manageable implementation complexity
- ✅ Active maintenance in research community

---

## 📖 Related Documentation

### Implementation Guides
- [Advanced Models Roadmap](../roadmaps/ADVANCED_MODELS_ROADMAP.md) - Detailed implementation plans
- [MTF Implementation Roadmap](../roadmaps/MTF_IMPLEMENTATION_ROADMAP.md) - Multi-timeframe expansion
- [Model Integration Guide](../guides/MODEL_INTEGRATION_GUIDE.md) - How to add models

### Reference
- [Architecture](../reference/ARCHITECTURE.md) - System design patterns
- [Pipeline Flow](../reference/PIPELINE_FLOW.md) - Phase 1-5 execution flow

### Phase Documentation
- [Phase 1](../phases/PHASE_1.md) - Data pipeline (complete)
- [Phase 2](../phases/PHASE_2.md) - Model training (complete)
- [Phase 3](../phases/PHASE_3.md) - Cross-validation (complete)
- [Phase 4](../phases/PHASE_4.md) - Ensembles (complete)
- [Phase 5](../phases/PHASE_5.md) - Walk-forward (complete)

---

*See [Documentation Index](../INDEX.md) for complete documentation overview*
