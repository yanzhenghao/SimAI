# SimAI Research Directory

## Reports

### AICB Model Extensibility -- 功能设计说明书

**File:** `aicb-model-extensibility.md` (1,193 lines, 28 sources)
**Type:** 功能设计说明书 + 竞争调研报告
**Date:** 2026-06-15

Comprehensive research and design specification for extending AICB (aliyun/aicb) workload generator to support additional LLM model architectures (LLaMA, GPT, Mistral/Mixtral, Qwen, Gemma, Falcon, DBRX, and others).

**Contents:**
- Section 1: Functional domain overview + AICB codebase structure
- Section 2: Design principles (5 rules), alternative evaluation (4 options), domain data model (UML), system element relationship diagram
- Section 3: Specification changes -- 14 model families analyzed, 2025-2026 new models with AICB implications, 5 competitive tool architectures (PARAM, Chakra, ASTRA-sim, MLSynth, Echo), 17-dimension x 6-tool comparison matrix, 13-row model coverage gap table, community contribution assessment
- Section 4: 5 implementation functions (F001-F005) with SR-to-AR allocation (19 SRs, 30 ARs), sequence diagrams, interface specifications, FMEA analysis (18 failure modes), 4-phase delivery roadmap

**Key findings:**
- Most decoder-only models are parametric variations -- 80%+ of new models can be added via config files only
- Chakra ET format lock-in is AICB's most severe competitive gap
- Three independent model families (Jamba 2, Llama 4, DeepSeek V4) confirm non-uniform per-layer communication as an architectural convergence trend
- Recommended strategy: extend AICB + add Chakra ET export (do not switch to MLSynth/Echo/Chakra)

### Presentation Files

- `presentation-plan.md` -- 12-slide presentation outline
- `presentation.mdx` -- Slidev-compatible MDX deck (14 slides, view with `npx slidev`)
