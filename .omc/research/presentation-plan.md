# AICB Model Extensibility Presentation Plan

## Structure: 12 slides

### Slide 1: Title
AICB Workload Generator 模型可扩展性研究
副标题：功能设计说明书 & 竞争分析报告
日期：2026-06-15

### Slide 2: Research Questions (4 sub-questions)
(1) AICB 能否扩展到 Llama, GPT, Mistral, Qwen, Gemma, Falcon, DBRX?
(2) 2025-2026 是否有新模型引入新并行策略/collective?
(3) aicb 仓库近期更新 / 社区贡献?
(4) vs PARAM, Chakra, astra-sim 的模型覆盖差距?

### Slide 3: Finding #1 -- Most extensions are parametric
80%+ models are parametric configs, not code changes
Show: Decoder-Only Transformer layer structure diagram
Key: 4 mocked_model primitives (linear, attention, MLP, embeddings)

### Slide 4: Model Coverage Gap Summary
Side-by-side comparison table: AICB vs Chakra vs MLSynth vs Echo
Highlight: AICB covers 10 models, competitors cover more but with trade-offs
Parametric gaps: 6 models (config only, ~5 days)
Architectural gaps: 5 models (need code changes)

### Slide 5: Finding #2 -- 2025-2026 Convergence
Three independent model families with non-uniform per-layer comm:
- Jamba 2 (SSM/Transformer 1:7)
- Llama 4 Maverick (dense/MoE 1:1 alternating)
- DeepSeek V4 (hash-routed/learned MoE split)
Key insight: this is architectural convergence, not anomaly

### Slide 6: Finding #3 -- AICB's 3 Competitive Gaps
Gap 1: Format lock-in (SimAI proprietary vs Chakra ET industry standard)
Gap 2: Tunability (MLSynth paper: "AICB is not tunable")
Gap 3: Model coverage breadth (~10 vs Echo's any HF model)

### Slide 7: Competitive Landscape
17-dimension x 6-tool comparison matrix (simplified for presentation)
Highlight Chakra ecosystem convergence (40+ members)
Show MLSynth paper Table 1: AICB is accurate/reproducible but NOT tunable

### Slide 8: Design Solution -- 5 Functions
F001: Model config profiles (6 new families, 3 days)
F002: Chakra ET export (eliminates format lock-in, 5-8 days) ★ HIGHEST IMPACT
F003: Tunability wrapper (straggler + scaling + variability, 5-8 days)
F004: Per-layer communication profiles (Jamba/Llama4/DV4, 8-12 days)
F005: Falcon parallel sub-layer (3-5 days)

### Slide 9: Function F002 Deep Dive -- Chakra ET Export
Show: AICB ops -> Chakra ET DAG mapping
Show: Natural language sequence diagram
Impact: Opens entire Chakra ecosystem (ASTRA-sim, Multiverse, Keysight)

### Slide 10: Function F003 Deep Dive -- Tunability
Show: Decorator pattern (TunabilityWrapper)
Show: Straggler injection + workload scaling + variability modeling
Impact: Closes MLSynth's "not tunable" criticism

### Slide 11: Delivery Roadmap (4 Phases)
Phase 1: F001 (model configs) + F002 (Chakra export) = 8-11 days
Phase 2: F003 (tunability) = 5-8 days
Phase 3: F004 (per-layer profiles) + F005 (Falcon) = 11-17 days
Phase 4: SSM + Hash-Routed MoE (20+ days, deferred)

### Slide 12: Strategic Recommendation
EXTEND AICB; DO NOT SWITCH
Bridge strategy: Chakra ET export = preserves AICB strengths + eliminates biggest gap
Call to action: publish model config documentation, accept community PRs
