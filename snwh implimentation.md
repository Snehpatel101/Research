L001 | # Implementation Plan: Unified Multi‑Timeframe Model Factory
L002 | 
L003 | ## Document Purpose
L004 | This plan defines a **single unified pipeline** where **any model family** can be trained
L005 | on **any timeframe**, and **heterogeneous ensembles work by default**. It prescribes
L006 | contracts, adapter architecture, validation, and tests to guarantee compatibility.
L007 | 
L008 | ## Success Criteria (Binary Pass/Fail)
L009 | 1. **Per‑model timeframe config** works across all families (tabular, neural, transformer, ensemble, meta‑learner).
L010 | 2. **Adapters** convert canonical OHLCV → model‑specific input shapes (2D/3D/4D) with no manual reshaping.
L011 | 3. **Heterogeneous stacking** works with mixed families using OOF predictions and correct data routing.
L012 | 4. **Ensemble validation** fails fast with clear fixes (no runtime shape surprises).
L013 | 5. **OOF integrity** enforced: one prediction per sample, no leakage (purge/embargo).
L014 | 6. **Reproducibility**: every artifact includes config hash, data lineage, and input signature.
L015 | 7. **Single‑contract isolation** maintained end‑to‑end (no cross‑symbol leakage).
L016 | 
L017 | ---
L018 | 
L019 | ## Phase 0 — Canonical Contracts (Foundation)
L020 | ### 0.1 Canonical Data Contract (CDS)
L021 | **Goal:** one schema for all adapters.
L022 | **Fields:**
L023 | - `timestamp`, `open`, `high`, `low`, `close`, `volume`
L024 | - `label_h{h}`, `sample_weight_h{h}` for each horizon
L025 | - `symbol`, `timeframe`, `feature_version`, `label_version`
L026 | **Rules:**
L027 | - All timeframes derived from canonical 1‑min source
L028 | - All labels computed from the same primary series
L029 | - All splits aligned by timestamp and stored with schema hash
L030 | **Deliverable:** `DataContract` dataclass + schema hash stored alongside artifacts.
L031 | 
L032 | ### 0.2 Canonical Model Contract (MDS)
L033 | **Goal:** every model declares its input requirements in the registry.
L034 | **Fields (required):**
L035 | - `input_rank`: 2/3/4
L036 | - `feature_mode`: `engineered` | `raw` | `hybrid`
L037 | - `timeframe`: primary TF
L038 | - `mtf_mode`: `none` | `indicators` | `multi_stream`
L039 | - `sequence_length`, `patch_length` (if applicable)
L040 | **Deliverable:** `ModelDataRequirements` extended and validated at registration.
L041 | 
L042 | ### 0.3 Artifact Safety Contract
L043 | **Goal:** safe load/reproducibility.
L044 | **Rules:**
L045 | - Avoid unsafe deserialization by default (prefer native formats / joblib)
L046 | - Store `metadata.json` with: schema hash, config hash, code version, model signature
L047 | - Verify hashes + signature on load; fail hard on mismatch
L048 | **Deliverable:** `ArtifactManifest` validator invoked in training and inference.
L049 | 
L050 | ---
L051 | 
L052 | ## Phase 1 — Unified Configuration Layer
L053 | ### 1.1 Per‑Model Config in UnifiedConfig
L054 | **Goal:** allow fully heterogeneous configurations without manual glue.
L055 | **Add to ModelConfig:**
L056 | - `name`, `family`, `timeframe`, `feature_mode`, `mtf_mode`
L057 | - `adapter_id`, `input_rank`, `sequence_length`
L058 | - `feature_strategy`: `baseline` | `optimized` | `raw`
L059 | - `ooof_mode`: `purged_kfold` | `holdout`
L060 | **Deliverable:** config schema + validation function.
L061 | 
L062 | ### 1.2 Ensemble Config Resolution
L063 | **Goal:** expand, validate, and derive aggregate requirements.
L064 | **Steps:**
L065 | - Expand ensemble aliases into base models
L066 | - Validate compatibility (homogeneous vs heterogeneous)
L067 | - Resolve per‑model adapter + timeframe requirements
L068 | - Determine multi‑adapter loading plan
L069 | **Deliverable:** `EnsemblePlan` structure used by trainer.
L070 | 
L071 | ---
L072 | 
L073 | ## Phase 2 — Adapter Architecture (Core of “Just Works”)
L074 | ### 2.1 Adapter Types
L075 | **TabularAdapter (2D)**
L076 | - Input: engineered features
L077 | - Output: `(n_samples, n_features)`
L078 | 
L079 | **SequenceAdapter (3D)**
L080 | - Input: engineered features + windowing
L081 | - Output: `(n_samples, seq_len, n_features)`
L082 | - Ensures alignment of labels after window offset
L083 | 
L084 | **MultiStreamAdapter (4D)**
L085 | - Input: multiple TF streams
L086 | - Output: `(n_samples, n_streams, seq_len, n_features)`
L087 | - Aligns streams to anchor timeframe
L088 | 
L089 | ### 2.2 Adapter Registry
L090 | **Goal:** automatic routing by model requirements.
L091 | **Rules:**
L092 | - Map `ModelDataRequirements → adapter_id`
L093 | - Training uses adapter registry to produce correct shapes
L094 | - Inference uses same adapter + stored feature schema
L095 | 
L096 | ### 2.3 Adapter Validation
L097 | - Validate rank (2D/3D/4D)
L098 | - Validate feature ordering and dtype
L099 | - Validate sequence length and stream count
L100 | - Provide actionable error hints
L101 | 
L102 | ---
L103 | 
L104 | ## Phase 3 — Timeframe Coordination & Alignment
L105 | ### 3.1 TimeframeCoordinator
L106 | **Goal:** enforce alignment between model‑specific timeframes and ensemble requirements.
L107 | **Responsibilities:**
L108 | - Ensure base TF for each model exists
L109 | - Align timestamps for multi‑stream inputs
L110 | - Apply `shift(1)` on MTF indicators to prevent leakage
L111 | - Enforce anchor‑TF alignment for ensemble aggregation
L112 | 
L113 | ### 3.2 Temporal Rules
L114 | - Anchor TF = smallest timeframe among base models (default)
L115 | - Downsample or resample to anchor with deterministic policy
L116 | - Sequence window offset preserved across labels and weights
L117 | 
L118 | ---
L119 | 
L120 | ## Phase 4 — Ensemble Compatibility & OOF Integrity
L121 | ### 4.1 Compatibility Matrix
L122 | **Voting/Blending:** homogeneous only (all 2D or all 3D)
L123 | **Stacking:** heterogeneous allowed (2D + 3D + 4D) via adapters
L124 | **Meta‑learner:** always 2D OOF features
L125 | 
L126 | ### 4.2 OOF Validation Rules
L127 | - PurgedKFold + embargo for time series
L128 | - Coverage check: 1 OOF prediction per training sample
L129 | - Strict shape alignment across base models
L130 | - Probability/logit consistency check
L131 | 
L132 | ### 4.3 Ensemble Dataset Builder
L133 | - Accepts `EnsemblePlan` and adapter outputs
L134 | - Builds OOF features in a single canonical format
L135 | - Stores OOF metadata for reproducibility
L136 | 
L137 | ---
L138 | 
L139 | ## Phase 5 — Feature Strategy Unification
L140 | ### 5.1 Feature Strategy Router
L141 | - `baseline`: standard per‑model family set
L142 | - `optimized`: Optuna‑pruned from baseline
L143 | - `raw`: OHLCV only (transformer‑friendly)
L144 | 
L145 | ### 5.2 Mixed Family Resolution
L146 | - Heterogeneous stacking uses model‑specific feature sets
L147 | - Optional `ensemble_base` for meta‑learner stability
L148 | 
L149 | ---
L150 | 
L151 | ## Phase 6 — Validation & Diagnostics
L152 | ### 6.1 Config Validation
L153 | - Model exists in registry
L154 | - Adapter exists
L155 | - Timeframe data exists
L156 | - Feature strategy supported
L157 | - Ensemble compatibility checked before training
L158 | 
L159 | ### 6.2 Input Signature Checks
L160 | - Validate rank, dtype, feature count
L161 | - Validate sequence length and stream count
L162 | - Validate scaler alignment
L163 | 
L164 | ---
L165 | 
L166 | ## Phase 7 — Testing Strategy
L167 | ### 7.1 Unit Tests
L168 | - Adapter output shapes
L169 | - Compatibility validator
L170 | - Timeframe alignment rules
L171 | 
L172 | ### 7.2 Integration Tests
L173 | - 1 tabular + 1 sequence + 1 transformer in one run
L174 | - Heterogeneous stacking end‑to‑end
L175 | 
L176 | ### 7.3 Regression Tests
L177 | - OOF coverage and no‑leakage
L178 | - Sequence + tabular mixed stack validation
L179 | 
L180 | ---
L181 | 
L182 | ## Phase 8 — Documentation & UX
L183 | ### 8.1 Unified Config Examples
L184 | - Single‑family training
L185 | - Heterogeneous stacking
L186 | - Multi‑timeframe transformer + boosting
L187 | 
L188 | ### 8.2 Error Messages
L189 | - Explicit model list + expected input shape
L190 | - Fix guidance: choose stacking or provide seq data
L191 | 
L192 | ---
L193 | 
L194 | ## External Best‑Practice Alignment (2025)
L195 | - Strict input signature validation
L196 | - Purged OOF stacking for time series
L197 | - Artifact hashing + schema validation
L198 | - Reproducible data lineage
L199 | 
L200 | ---
L201 | 
L202 | ## Immediate Next Steps
L203 | 1. Implement adapter registry + signature enforcement
L204 | 2. Refactor training flow to load per‑model timeframes
L205 | 3. Add TimeframeCoordinator for multi‑stream alignment
L206 | 4. Build ensemble dataset builder with OOF checks
L207 | 5. Add hetero stacking integration tests
