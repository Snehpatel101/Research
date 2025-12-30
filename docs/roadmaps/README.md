# Implementation Roadmaps

Detailed implementation plans for expanding the ML model factory.

## 📋 Available Roadmaps

### [Advanced Models Roadmap](ADVANCED_MODELS_ROADMAP.md)
**Effort:** 18 days total (3.5 weeks @ 1 engineer, 2 weeks @ 2 engineers)

Implementation guide for 6 advanced time-series models to expand from 13 to 19 models:

| Model | Family | Effort | Description |
|-------|--------|--------|-------------|
| **InceptionTime** | CNN | 3 days | Multi-scale temporal convolutions with kernel sizes 10, 20, 40 |
| **1D ResNet** | CNN | 2 days | Deep residual learning with skip connections |
| **PatchTST** | Transformer | 4 days | SOTA patch-based transformer for long-term forecasting |
| **iTransformer** | Transformer | 3 days | Inverted attention mechanism (features as tokens) |
| **TFT** | Hybrid | 5 days | Temporal fusion transformer with variable selection + quantiles |
| **N-BEATS** | Decomposition | 1 day | Interpretable trend/seasonal basis expansion |

Each model section includes:
- ✅ Complete architecture code snippets
- ✅ Day-by-day implementation steps
- ✅ Expected performance metrics
- ✅ Integration checklists
- ✅ Testing requirements

### [MTF Implementation Roadmap](MTF_IMPLEMENTATION_ROADMAP.md)
**Effort:** 6 phases

Multi-timeframe (MTF) feature expansion roadmap:

**Current State:**
- 7 timeframes implemented: 1m, 5m, 10m, 15m, 30m, 45m, 60m
- Missing from 9-TF ladder: 20min, 25min

**Planned Expansion:**
1. **Phase 1:** Add 20min/25min to MTF constants
2. **Phase 2:** Validate forward-fill alignment
3. **Phase 3:** MTF strategy configuration (`mtf_strategy` parameter)
4. **Phase 4:** Model-specific MTF features
5. **Phase 5:** Testing and validation
6. **Phase 6:** Documentation updates

**Three MTF Strategies:**
- **Single-TF:** Train on one timeframe (e.g., 5min only)
- **MTF Indicators:** Compute indicators at multiple timeframes
- **MTF Ingestion:** Ingest raw OHLCV at multiple timeframes

---

## 🎯 Roadmap Summary

**Total implementation effort:** ~22 days (4.5 weeks @ 1 engineer, 2.5 weeks @ 2 engineers)

**Deliverables:**
- 6 new advanced model implementations
- Complete 9-timeframe MTF ladder
- Model-specific MTF feature strategies
- Comprehensive testing coverage

---

## 📖 Related Documentation

- [Model Integration Guide](../guides/MODEL_INTEGRATION_GUIDE.md) - How to implement the BaseModel interface
- [Feature Engineering Guide](../guides/FEATURE_ENGINEERING_GUIDE.md) - Model-specific feature strategies
- [Hyperparameter Optimization Guide](../guides/HYPERPARAMETER_OPTIMIZATION_GUIDE.md) - Tuning strategies
- [Model Infrastructure Requirements](../guides/MODEL_INFRASTRUCTURE_REQUIREMENTS.md) - Hardware planning

---

*See [Documentation Index](../INDEX.md) for complete documentation overview*
