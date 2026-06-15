# AICB 模型可扩展性 -- 功能设计说明书（含调研支撑）

**文档类型:** 功能设计说明书 + 竞争调研报告
**功能域:** AICB Workload Generator 模型架构扩展
**日期:** 2026-06-15 (cross-checked against latest sources: 2026-06-15)
**关联仓库:** [aliyun/aicb](https://github.com/aliyun/aicb) (当前版本: AICB 2.1, November 2025. SimAI 主仓库最新版本: SimAI 1.6, April 2026 -- 新增 GPU 显存建模、KV Cache 追踪、Prefill-Decode 分离显存规划，均为仿真引擎改进，AICB workload generator 无变更)

---

## 1. 功能域概述

### 1.1 功能域描述

本功能域描述 AICB（AI Communication Benchmark）workload generator 的**模型架构可扩展性**设计与实现。AICB 当前通过 `mocked_model` 抽象层模拟 Megatron-LM 和 DeepSeek 两种框架的 layer 结构（linear, attention, MLP, embeddings），生成训练/推理通信 workload。

AICB 是 [aliyun/aicb](https://github.com/aliyun/aicb) 开源项目（MIT 许可），其核心代码结构如下：

```
aicb/
├── aicb.py                          # Main entry point
├── workload_generator/
│   ├── mocked_model/                # ★ 核心抽象层 (AICB 2.1 重构)
│   │   ├── training/                #   训练 model mock
│   │   │   ├── megatron/            #     Megatron-LM backend
│   │   │   └── deepspeed/           #     DeepSpeed backend (ZeRO-1/2/3)
│   │   └── inference/              #   推理 model mock (AICB 2.1 新增)
│   │       └── vllm/               #     vLLM backend
│   ├── generate_megatron_workload.py
│   ├── generate_deepspeed_stage1_2_workload.py
│   ├── generate_deepspeed_stage3_workload.py
│   ├── generate_collective_test.py
│   └── AIOB_simAI_workload_generator.py
├── deepseek_simulator/              # DeepSeek framework backend (AICB 2.1)
├── config/                          # Model profiles (★ 本设计的扩展目标)
│   └── models/                      # Per-model YAML profiles
├── export/                          # ★ 本设计新增: Chakra ET 导出层
├── tuning/                          # ★ 本设计新增: 可调优性包装器
└── tests/
```

本次设计的核心目标：**将 AICB 的模型覆盖从当前的 ~10 个精选模型扩展到涵盖 2025-2026 年主流 LLM 架构家族**，同时解决三个竞争性差距（格式锁定、不可调优、模型覆盖面窄），使 AICB 在训练通信 benchmark 工具赛道中保持竞争力。

### 1.2 功能域范围

| 维度 | 当前状态 (AICB 2.1) | 设计目标 |
|------|---------------------|----------|
| 支持模型数 | ~10 个 | 18-20 个 |
| 模型架构家族 | LLaMA, GPT, Mistral, DeepSeek | + Gemma, Falcon, DBRX, Phi, Command R, Qwen, Llama 4 |
| 输出格式 | SimAI 专有格式 | + Chakra Execution Trace (Protobuf/JSON DAG) |
| Workload 可调优性 | 无 | Straggler 注入、workload 缩放、变异性建模 |
| 推理支持 | DeepSeek, Qwen3-MoE/Next | 维持现有覆盖 |
| 非均匀层通信 | 不支持 | 基础支持（layer_type 标注、per-layer communication profile） |

---

## 2. 功能域总体方案

### 2.1 总体设计原则

| 原则 | 说明 | 约束 |
|------|------|------|
| **参数化优先** | 大多数 decoder-only 模型的差异是参数差异而非架构差异。优先通过配置文件扩展模型覆盖，仅在必要时修改代码。 | 80% 以上的新模型应通过参数配置添加 |
| **向后兼容** | 新增 mocked_model 能力不得破坏现有 Megatron-LM / DeepSpeed / DeepSeek / vLLM 后端的 workload 生成。 | 所有现有测试必须通过 |
| **格式桥接** | Chakra ET 导出作为独立序列化层，不侵入 mocked_model 核心逻辑。 | 导出层与核心逻辑解耦 |
| **社区可贡献** | 模型参数配置文件应足够简单，使外部贡献者无需深入理解 AICB 内部架构即可提交新模型配置。 | 新模型配置文件 < 100 行 |
| **渐进式交付** | 按 effort-to-impact 比排序，分 4 个 Phase 交付，每个 Phase 独立可验证。 | 见 Section 4 各功能描述 |

### 2.2 整体实现思路

AICB 模型扩展的核心实现路径是 **"参数配置 + 序列化层 + 调优包装器"三层扩展架构**：

```
                    现有 mocked_model 核心
                   (linear, attention, MLP, embeddings)
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ① 参数配置层      ② Chakra 序列化层    ③ 调优包装器
   (新增模型         (mocked_model     (Straggler 注入,
    config profiles)   ops → Chakra ET)   workload 缩放,
                                         变异性建模)
            │               │               │
            ▼               ▼               ▼
    扩展模型覆盖       消除格式锁定      消除可调优性差距
```

**关键技术决策与备选方案评估：**

| 方案 | 描述 | 优势 | 劣势 | 结论 |
|------|------|------|------|------|
| **A（迁移至 MLSynth）** | 放弃 AICB，使用 MLSynth 作为 workload 生成器 | 原生 Chakra ET 输出；可调优 | MLSynth 不支持框架级并行策略语义（Megatron TP/PP/SP, DeepSpeed ZeRO stages）；Layer 库仅限 Transformer/MoE 基础类；2025 年新工具，验证不充分 | **否决** |
| **B（迁移至 Echo）** | 使用 Echo 的 ex-situ tracing 替代 AICB | 92% 准确率；任意 HF 模型覆盖；白盒 NCCL 模型 | 需要物理 GPU profiling；无法生成不存在硬件配置的 workload；profiling 成本随模型数线性增长 | **否决** |
| **C（迁移至 Chakra + 自定义 producer）** | 基于 Chakra schema 从零构建 workload 生成器 | 完全兼容 Chakra 生态 | 重建 AICB 已积累的框架级并行策略语义，工程量大；丧失 NSDI '25 学术积累 | **否决** |
| **D（扩展 AICB + Chakra ET 导出）** | 保留 AICB 核心，添加 Chakra ET 序列化层和可调优性 | 保留框架级 fidelity + 消除格式锁定 + 最小工程量 | 需要维护两套输出格式 | **采纳** |

### 2.3 领域数据模型

mocked_model 的领域数据模型如下（核心实体及关系）：

```
┌─────────────────────────────────────────────────────────────┐
│                    ModelConfig                              │
│  + model_name: str                                          │
│  + framework: enum[Megatron, DeepSpeed, DeepSeek, vLLM]    │
│  + parallelism_config: ParallelismConfig                    │
│  + layers: List[LayerConfig]                                │
│  + layer_pattern: Optional[List[LayerPatternEntry]]  ★ 新增 │
└───────────────┬─────────────────────────────────────────────┘
                │ 1 contains *
┌───────────────▼─────────────────────────────────────────────┐
│                   LayerConfig                               │
│  + layer_index: int                                         │
│  + layer_type: enum[attention, moe_layer, mamba_layer,     │
│              hash_routed_moe, dense_ffn_only]  ★ 扩展字段   │
│  + use_parallel_sub_layers: bool  ★ 新增 (Falcon)          │
│  + attention_config: AttentionConfig                        │
│  + mlp_config: MLPConfig                                    │
│  + moe_config: Optional[MoEConfig]                          │
└───────────────┬─────────────────────────────────────────────┘
                │ 1 generates *
┌───────────────▼─────────────────────────────────────────────┐
│                    ComputeOp                                │
│  + op_type: enum[GEMM, attention_score, softmax,            │
│           layer_norm, activation, ssm_scan, reduce]         │
│  + flops: float                                             │
│  + input_shape: List[int]                                   │
│  + duration_us: Optional[float]  (from AIOB profiler)       │
│  + straggler_delay_us: Optional[float]  ★ 新增             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  CommunicationOp                            │
│  + collective: enum[AllReduce, AllGather, ReduceScatter,    │
│              AlltoAll, Broadcast, P2P_Send, P2P_Recv]       │
│  + comm_size_bytes: int                                     │
│  + participant_ranks: List[int]                             │
│  + algorithm: enum[Ring, Tree, Direct, HashRouted, ...]     │
│  + straggler_delay_us: Optional[float]  ★ 新增             │
└─────────────────────────────────────────────────────────────┘

★ = 本次设计新增字段/值
```

### 2.4 功能域涉及的系统元素及周边关系

```
                          ┌──────────────────┐
                          │   AICB (本系统)    │
                          │  workload gen.    │
                          └──────┬───────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
│   SimAI (内部)   │   │ Chakra Ecosystem│   │ 竞争/互补工具 (外部)  │
│                 │   │  (新增集成)       │   │                     │
│ - SimAI_analyt. │   │ - ASTRA-sim      │   │ - MLSynth (竞争)     │
│ - SimAI_simul.  │   │ - Multiverse     │   │ - Echo (竞争/互补)   │
│ - NS3 backend   │   │ - Keysight KAI   │   │ - PARAM (互补)       │
│                 │   │ - PyTorch ET     │   │ - Chakra (标准/互补) │
└─────────────────┘   └─────────────────┘   └─────────────────────┘
         │                      │
         ▼                      ▼
  现有 SimAI 用户        新 Chakra 生态系统用户
  (兼容性不变)           (通过 Chakra ET 导出接入)
```

**周边系统交互详细说明（详见 Section 3.3 竞争工具架构分析）：**
- **上游依赖**: NVIDIA Megatron-LM, Microsoft DeepSpeed, DeepSeek HAI-LLM, vLLM
- **下游消费者 (现有)**: SimAI simulator (analytical + NS3 backends)
- **下游消费者 (新增)**: ASTRA-sim, Multiverse, Keysight KAI DCB, 任何 Chakra ET 兼容模拟器
- **竞争关系**: MLSynth (NVIDIA/Imperial) -- 合成 Chakra trace 生成器，可调优但缺少框架级并行策略 fidelity
- **互补关系**: PARAM (Meta) -- 低层通信 benchmark，非 LLM workload 生成器
- **标准关系**: Chakra (MLCommons) -- 行业标准 trace 格式，AICB 需要作为 "producer tool" 接入

---

## 3. 功能域规格变更与设计

### 3.1 目标模型架构支持范围（规格变更）

基于对 2025-2026 年 LLM 市场的调研，各模型家族的 decoder-only transformer 架构差异主要为参数差异，而非架构差异。以下是每个目标模型家族的 mocked_model 适配分析：

| Model Family | Architecture Variant | Layer Structure | AICB mocked_model Fit | Effort |
|-------------|---------------------|-----------------|----------------------|--------|
| **LLaMA 3.x** (Meta) | Pre-norm decoder, GQA, SwiGLU MLP | Serial: RMSNorm -> Attention -> RMSNorm -> MLP | Already supported (Llama3 405B). Add 8B/70B configs. | Low |
| **GPT-4 class / GPT-NeoX** (OpenAI/Eleuther) | Pre-norm decoder (GPT-4 architecture not publicly documented; assumed similar to GPT-3/GPT-NeoX) | Serial: LayerNorm -> Attention -> LayerNorm -> MLP | Matches GPT pattern already in AICB. Parameterize hidden dim + heads. | Low |
| **Gemma 2/3** (Google) | Pre-norm decoder, GQA (2:1 to 4:1 KV head ratio depending on model size), GeGLU MLP | Serial: RMSNorm -> Attention -> RMSNorm -> MLP | Same structure as LLaMA. Different GQA ratios per model size (e.g. 27B uses 32 Q heads / 16 KV heads = 2:1; 12B uses 24/8 = 3:1; 4B uses 16/4 = 4:1). [Source: NVIDIA NeMo Megatron-Bridge Gemma 3 docs](https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/models/llm/gemma3.html) | Low |
| **Qwen 2.5** (Alibaba) | Pre-norm decoder, GQA, SwiGLU MLP | Serial: RMSNorm -> Attention -> RMSNorm -> MLP | Same structure as Llama3. Already have Qwen3 inference. | Low |
| **Mistral/Mixtral** (Mistral AI) | Pre-norm decoder, GQA, sliding window attention, SiLU MLP; Mixtral adds MoE FFN | Serial or MoE-augmented | Already supported (Mistral 8x7B). Add Mixtral 8x22B config. | Low |
| **Falcon 3** (TII) | Parallel attention + MLP, MQA (1 KV head) | Parallel: RMSNorm -> [Attention || MLP] simultaneously | Different layer ordering: attention and MLP computed in parallel, not serial. Requires mocked_model to support parallel execution within a single layer. | Low-Medium |
| **Phi-3/4** (Microsoft) | Pre-norm decoder, dense or GQA, GeLU MLP | Serial: LayerNorm -> Attention -> LayerNorm -> MLP | Standard decoder. Parameter config only. | Low |
| **Command R** (Cohere) | Pre-norm decoder, GQA, sliding window attention, SwiGLU MLP | Serial: LayerNorm -> Attention -> LayerNorm -> MLP | Same as Mistral (dense). GQA + sliding window params. | Low |
| **DBRX** (Databricks) | Fine-grained MoE, GQA, 16 experts (top-4 gating) | Serial with MoE FFN | Already have Mistral 8x7B (top-2). Parameterize for 16 experts, top-4 gating. Per-token AlltoAll volume scales with active experts per token (top-k); DBRX's top-4 doubles the fan-out vs Mixtral's top-2. | Low-Medium |
| **Llama 4 Maverick** (Meta) | MoE, 128 experts + 1 shared (top-1 routing), alternating dense/MoE layers, iRoPE, early-fusion multimodal | Alternating: [Dense layer with TP AllReduce, MoE layer with EP AlltoAll] | Already have Megatron MoE support. New challenge: alternating dense (AllReduce-heavy) and MoE (AlltoAll-heavy) layers within the same model. Top-1 routing creates extreme sparsity (128:1 expert ratio). | Medium |
| **DeepSeek-V3** (DeepSeek) | Fine-grained MoE (256 experts, top-8), MLA, Multi-Token Prediction | Serial MoE + MTP heads | Already supported via DeepSeek framework backend. MTP heads introduce additional AlltoAll patterns beyond standard MoE dispatch/combine. | Medium |
| **DeepSeek-V4** (DeepSeek) | 1.6T MoE, 49B active, Hybrid Attention (CSA+HCA), hash-routed MoE layers, FP4 QAT, mHC residual connections | Serial with hash-routed early layers + learned MoE later layers | V4 released April 2026. Hash-routed MoE (static token-to-expert mapping) eliminates AlltoAll for early layers entirely. CSA/HCA attention introduces alternating attention patterns at the compute level. | High |
| **Jamba 2** (AI21) | SSM-Mamba + Transformer hybrid MoE, inter-layer (1:7 ratio) | Alternating: [1 Attention+MoE layer, 7 Mamba-FFN layers] | New layer type needed: Mamba layers have no AllReduce (recurrent computation within a single device), creating a fundamentally different per-layer communication profile. The 1:7 ratio means communication occurs only every 8th layer. | High |

**新增模型参数对照表（concrete mocked_model parameters per family）：**

| Parameter | LLaMA 3.1 8B | Gemma 3 12B | Falcon 3 | DBRX | Phi-4 |
|-----------|-------------|-------------|----------|------|-------|
| `num_layers` | 32 | 48 | Varies | 40 | 40 |
| `hidden_size` | 4096 | 3840 | Varies | 4608 | 5120 |
| `num_attention_heads` | 32 | 24 | Varies | 36 | 40 |
| `num_kv_heads` | 8 (GQA 4:1) | 8 (GQA 3:1) | 1 (MQA) | 36 (dense) | 40 (dense) |
| `ffn_hidden_size` | 14336 | 30720 | Varies | 18432 | 20480 |
| `vocab_size` | 128256 | 262208 | Varies | 100352 | 100352 |
| `activation_function` | SiLU (SwiGLU) | GeLU (GeGLU) | GeLU | SiLU (SwiGLU) | GeLU |
| `use_parallel_attention` | False | False | True | False | False |
| `num_experts` | 0 (dense) | 0 (dense) | 0 (dense) | 16 | 0 (dense) |
| `num_experts_per_tok` | N/A | N/A | N/A | 4 | N/A |
| `seq_length` | 131072 | 131072 | Varies | 32768 | 32768 |
| `rope_theta` | 500000 | 1000000 (global) | Varies | 500000 | 10000 |

**关键发现：** 对于标准 decoder-only 架构（LLaMA 3.1, Gemma 3, Phi-4），**每个参数在现有 Megatron-LM mocked_model 配置中已有对应字段**。仅 Falcon（需要 `use_parallel_attention = True` 支持）和 DBRX（需要 top-4 EP gating）需要代码变更。

### 3.2 2025-2026 年新模型与新型并行策略

#### 3.2.1 主要模型发布（含对 AICB 的影响）

| Model | Release | Architecture | New Communication Implications | AICB Implication |
|-------|---------|-------------|-------------------------------|-----------------|
| **DeepSeek-V3** | Dec 2024 | 671B MoE, 37B active, 256 routed experts (top-8), MLA, MTP | DualPipe bidirectional PP; custom all-to-all with PTX-level SM partitioning (20/132 SMs for comms); MTP heads add separate all-to-all patterns; DeepEP library | Already supported via dedicated DeepSeek backend. MTP heads' separate all-to-all not yet modeled -- extend existing backend. |
| **DeepSeek-R1** | Jan 2025 | V3-based + reasoning RL (GRPO) | Same training architecture as V3 | Same as V3. |
| **Qwen3-MoE** | Q2 2025 | MoE, 235B total / 22B active, 128 experts (top-8), chunked-prefill attention | Inference-optimized with vLLM; chunked prefill changes compute-communication phase boundaries | Currently inference-only. Training support would require integrating Qwen3's MoE routing into Megatron backend. |
| **Gemma 3** (Google) | Q1 2025 | Dense, GQA (2:1-4:1), GeGLU, 128K context, 5:1 local-to-global attention | Long-context requires Context Parallelism (ring attention); 5:1 local:global ratio creates alternating full vs reduced AllReduce layers | Parameter-config-only. Local attention layers have proportionally reduced communication. AICB can approximate with per-layer attention types. |
| **Falcon 3** (TII) | Mid 2025 | Dense, parallel attention+MLP, MQA (1 KV head) | Parallel execution enables attention-MLP AllReduce overlap; MQA further reduces KV communication | Requires `use_parallel_attention` support in mocked_model (Function F005). |
| **Llama 4 Scout** (Meta) | April 2025 | 109B total/17B active, full-MoE (every layer), 16 experts top-1, iRoPE, 10M context | Every layer is MoE with top-1 routing; sparse AlltoAll per layer | Reuses existing EP/AlltoAll support; needs top-1 gating parameterization. |
| **Llama 4 Maverick** (Meta) | April 2025 | ~400B total/17B active, 128 experts+1 shared top-1, alternating dense/MoE layers | Alternates dense layers (AllReduce) with MoE layers (AlltoAll); 128:1 expert sparsity | First production alternating dense/MoE model. Requires per-layer communication profile (Function F004). |
| **Jamba 2** (AI21) | Early 2025 | SSM-Mamba + Transformer MoE, 1:7 inter-layer ratio | Non-uniform per-layer comm: 12.5% layers have AllReduce+AlltoAll, 87.5% have zero communication | Requires `mamba_layer` SSM primitive (Function F004). Hardest single extension. |
| **DeepSeek V4** | April 2026 | 1.6T MoE, 49B active, CSA+HCA hybrid attention, hash-routed early layers, FP4 QAT, mHC, MIT license | Hash-routed MoE eliminates AlltoAll for early layers (zero EP); later layers use learned routing with full AlltoAll; 1M context | Postdates AICB 2.1. Hash-routed vs learned MoE split creates depth-dependent EP profile (Function F004). Highest effort. |
| **DBRX** (Databricks) | Early 2024 | Fine-grained MoE, 132B total, 16 experts (top-4) | Doubled all-to-all fan-out vs Mixtral (top-2); higher EP communication volume | EP parameterization for top-4 gating. |

#### 3.2.2 新兴并行策略

| Strategy | Description | Communication Pattern | AICB Status | Relevance to Design |
|----------|-------------|----------------------|-------------|---------------------|
| **Alternating Dense/MoE Layers** | Llama 4 Maverick: half layers dense (TP AllReduce), half MoE (EP AlltoAll) | Alternating per-layer communication type within single model | Not supported | Drives Function F004 (per-layer communication profile) |
| **Hash-Routed MoE** | DeepSeek V4: static token-ID to expert hash for early layers | Zero EP communication in early layers, full EP AlltoAll in later layers | Not supported (April 2026) | Drives Function F004 (layer_type = hash_routed_moe) |
| **Context Parallelism (CP)** | Ring Attention for 128K+ contexts; splits sequence dimension across GPUs | Ring P2P send/recv along sequence dimension | Not supported | Monitor; deferred to Phase 4 |
| **Sequence Parallelism v2 (SP2)** | Megatron-LM: splits activation memory along sequence dim with reduced comm | Modified AllReduce with smaller tensor shards | SP v1 supported; SP2 unclear | Low priority |
| **DualPipe (DeepSeek)** | Bidirectional PP with computation (ATTN, MLP) and communication (DISPATCH, COMBINE) stream overlap | Fine-grained computation-communication overlap; PTX SM partitioning | Not supported | Low priority (DeepSeek-specific) |
| **Auxiliary-Loss-Free EP** | DeepSeek: learnable bias per expert eliminates auxiliary loss | More uniform AlltoAll traffic distribution | Could be parameterized | Low priority |
| **Disaggregated Inference** | Prefill-Decode separation across GPU pools | Cross-node KV-cache transfer | Not supported (inference-only) | Monitor |
| **NCCLX (Meta)** | Collective framework for 100K+ GPU; 12% latency reduction via hierarchical aggregation | Optimized AllReduce/AllGather/ReduceScatter hierarchies | Not supported | Low priority (affects algorithm, not workload definition) |

#### 3.2.3 元趋势：MoE 主导与非均匀层通信

两个趋势重塑竞争格局：

1. **MoE 成为前沿模型默认架构**（DeepSeek-V3/V4, Qwen3-MoE, Mixtral, DBRX, Llama 4 Scout/Maverick）。AlltoAll 取代 AllReduce 成为主要通信瓶颈。AICB 已处理 EP/AlltoAll，但路由策略的多样性（top-1 vs top-2 vs top-4 vs top-8, shared vs non-shared experts, auxiliary loss vs learnable bias, hash-routed vs learned）产生显著不同的 AlltoAll 模式。

2. **非均匀层通信已确认为三个独立模型家族的架构收敛趋势。** 截至 2026 年中，Jamba 2 (SSM/Transformer 1:7), Llama 4 Maverick (dense/MoE 1:1), 和 DeepSeek V4 (hash-routed/learned MoE split) 均独立采用非均匀 per-layer 通信模式。这不再是利基关注点 -- 是三个不同组织的架构收敛。**当前任何 workload generator 都无法正确建模 per-layer 通信 profile**，这是 AICB 最关键的 forward-looking gap。

### 3.3 竞争工具详细架构分析（规格变更依据）

以下是对 AICB 的五个主要竞争/互补工具的架构分析，为 Section 3 的规格变更和 Section 4 的功能设计提供外部对标依据。

#### 3.3.1 PARAM (facebookresearch/param) -- 低层 Benchmark 套件

**定位:** 评估训练/推理平台，桥接 C++ standalone benchmark（NCCL tests, OSU MPI）和应用 benchmark（DLRM, MLPerf）。

**四组件架构:**

1. **Communication Benchmarks:** PyTorch NCCL collective benchmarks（arbitrary message sizes），测量 compute-communication overlap 效果，输出 `AlgBW` / `BusBw` 指标。
2. **Compute Benchmarks:** GEMM (TFLOPS across M,N,K)，Embedding Lookup/EmbeddingBag, Linear/MLP layers。
3. **DLRM End-to-End:** Dense features -> Top MLP -> Embedding Tables (TB-scale, model-parallel) -> Interaction Layer -> Bottom MLP (data-parallel)。Primary optimization target: all-to-all at the model-parallel EMB -> data-parallel interaction layer transition。
4. **PyTorch ET Replay:** Operator-level trace replay for framework overhead analysis。

**模型覆盖:** DLRM (recommendation models) only。不生成 LLM workload。PARAM 与 AICB 是**互补关系**（低层 vs 高层），非竞争关系。

Source: [facebookresearch/param GitHub](https://github.com/facebookresearch/param)

#### 3.3.2 Chakra (mlcommons/chakra) -- 行业标准 Trace 格式

**定位:** Open, graph-based schema (Protobuf/JSON DAG) for distributed AI/ML workload execution traces。源于 Meta + Georgia Tech (2023)。40+ member MLCommons working group。MLSys 2026 paper。

**Schema 架构 -- Hierarchical DAG:**

Chakra traces are directed acyclic graphs. Each GPU/NPU rank gets its own independent trace.

**Node Types:** `COMP_NODE` (4), `MEM_LOAD_NODE` (2), `MEM_STORE_NODE` (3), `COMM_SEND_NODE` (5), `COMM_RECV_NODE` (6), `COMM_COLL_NODE` (7)

**Collective Communication Sub-types:** `ALL_REDUCE`, `REDUCE`, `ALL_GATHER`, `GATHER`, `SCATTER`, `BROADCAST`, `ALL_TO_ALL`, `REDUCE_SCATTER`, `REDUCE_SCATTER_BLOCK`, `BARRIER`

**Node Schema Fields:** `id` (uint64), `name` (string), `type` (NodeType), `ctrl_deps` (repeated uint64), `data_deps` (repeated uint64 -- producer->consumer edges), `start_time_micros`, `duration_micros`, `inputs`/`outputs` (IOInfo with tensor shapes/types/values), `attr` (repeated AttributeProto key-value pairs)。

**Producer-Consumer Model:** `data_deps` edges encode producer->consumer relationships。`inputs`/`outputs` contain `Tensor` messages with `tensor_id`, `storage_id`, `offset`, `num_elem`, `elem_bytes`, `device`。

**End-to-End Workflow:**
```
PyTorch ET (CPU ops+deps) + Kineto Trace (GPU kernels)
        -> chakra_trace_link (merge host + device traces)
Unified "ET+" JSON
        -> chakra_converter (JSON -> protobuf)
Chakra ET (Protobuf)
        -> ETFeeder (parses protobuf, issues dep-free nodes)
Simulator (ASTRA-sim, Multiverse, etc.)
```

**Open Trace Library:** GPT-3, Llama 3, Mixtral, DeepSeek-MoE, vLLM serving。Collected on Georgia Tech AI Makerspace (128 GPUs) + HPE infrastructure。

**Native Integration:** PyTorch (ExecutionTraceObserver), NVIDIA NeMo, vLLM。Commercial adoption: Keysight KAI DCB, Scala Computing。

**对 AICB 设计的启示:** Chakra 是格式标准，不是 workload 生成器。AICB 通过 Chakra ET 导出（Function F002）成为 Chakra 生态系统中的 "producer tool"，填补 MLSynth 和 Echo 都无法提供的 framework-aware workload generation 能力。

Sources: [MLSys 2026 Chakra paper](https://arxiv.org/abs/2605.11333); [MLCommons Chakra WG](https://mlcommons.org/working-groups/research/chakra/); [Chakra Schema Wiki](https://github-wiki-see.page/m/mlcommons/chakra/wiki/Chakra-Schema); [Keysight Blog (May 2026)](https://www.keysight.com/blogs/en/inds/ai/mlcommons-chakra-from-traces-to-test)

#### 3.3.3 ASTRA-sim -- 分布式 AI 系统模拟器

**定位:** Open-source distributed DL training simulator (Georgia Tech / Intel / Meta)。Consumes Chakra ET，模拟跨可配置硬件和网络拓扑执行。

**三层架构:**

**Layer 1 -- Workload Layer** (`workload/Workload.hh`): Parses Chakra ET via `et_feeder`，resolves dependencies，issues compute/comm/memory ops，dispatches by node type。

**Layer 2 -- System Layer** (`system/Sys.hh`): Each simulated NPU has one `Sys` instance: `workload`, `comm_NI`, `remote_mem`, `scheduler_unit` (FIFO/LIFO/EXPLICIT), `memBus`, `collective_impl_lookup` (Ring, DoubleBinaryTree, HalvingDoubling, Direct), `event_queue` (discrete event simulation)。

**Event-Driven Simulation:** Events via `Sys::register_event()`: `CallEvents`, `CompFinished`, `PacketReceived`, `CollectiveCommunicationFinished`。

**Collective Algorithm Generation:** `generate_all_reduce()` -> ReduceScatter + AllGather。Three-tier algorithm selection: per-node custom -> global custom -> native。

**Layer 3 -- Backend Layer (Pluggable):**

| Backend | Type | Use Case |
|---------|------|----------|
| Analytical | Mathematical models (roofline) | Large-scale parameter sweeps |
| NS-3 | Packet-level simulation | RoCE, DCQCN/HPCC/Timely, 400Gbps+ RDMA |
| Garnet | Credit-based flow control | On-chip networks, chiplet interconnects |
| HTSim | TCP-focused simulation | Transport layer studies |

**Stream Scheduling:** Multi-level: ready_list -> active_streams -> completion。Inter-dimension: Ascending, RoundRobin, OfflineGreedy。

**Simulation Workflow:**
```
Init (system config JSON, network YAML, Chakra ET + comm groups)
  -> fire() iterates Chakra ET, issues dep-free nodes
  -> Compute: sim_schedule(); Communication: sim_send()/sim_recv()
  -> Completion: callback -> System Layer -> Workload issues newly dep-free nodes
  -> Termination: all ranks finish -> sim_notify_finished()
```

**对 AICB 设计的启示:** ASTRA-sim 是 Chakra ET 的主要消费者。Function F002 的 Chakra ET 导出使 AICB workload 可直接馈入 ASTRA-sim 的全栈模拟（packet-level NS3 + congestion control）。

Sources: [ASTRA-sim System Architecture](https://deepwiki.com/astra-sim/astra-sim/2-system-architecture); [ASTRA-sim + Chakra Tutorial (MICRO 2024)](https://astra-sim.github.io/assets/tutorials/micro-2024/2_chakra_astrasim_overview.pdf)

#### 3.3.4 MLSynth (NetMLSim/MLSynth) -- 合成 Chakra Trace 生成器（直接竞争者）

**定位:** 从高层模型参数生成合成 Chakra ET workloads。NAIC '25 paper (NVIDIA/Imperial College London)。**AICB 的直接竞争者。**

**四组件模块化架构:**

**Component 1 -- Layer (Template/Base Class):** 生成单个 NN layer 的 Chakra COMP_NODE DAG (forward + backward)。Accepts TP flags。

**Component 2 -- Model (Template Class):** 将 Layer 序列组装为完整模型。Propagates compute nodes upward。

**Component 3 -- Orchestrator:** 核心并行策略分布逻辑。Owns Model instance。确定每个 GPU 计算哪些 layers。在 compute nodes 之间插入通信操作。输出完全分布式的多 GPU workload graph。

**Component 4 -- Performance Wrapper (Decorator Pattern):** MLSynth 的独特贡献。Intercepts calls between Model and Orchestrator 以注入性能变化：修改 FLOPs（模拟 throttled GPU）、添加 "wait" nodes（模拟 stragglers）、修改通信 sizes 或引入延迟。Performance variations 嵌入 workload trace 本身，使其 simulation-agnostic。

**Architecture Diagram:**
```
User Input (batch_size, num_layers, hidden_size, parallelism)
        |
[Performance Wrapper] -- intercepts calls, injects stragglers/variability
        |
[Orchestrator] -- DP/TP/PP, distributes layers, inserts comm ops
        |
[Model] -- sequences Layers
        |
[Layer] -- generates Chakra COMP_NODE graph per layer
        |
Chakra ET (DAG: COMP, COMM, MEM nodes) -> ASTRA-sim / Multiverse
```

**模型覆盖:** Transformer baseline + MoE baseline。两者是模型类（参数化模板），不是硬编码模型。覆盖范围受 Layer component 中实现的层类型限制。

**AICB 直接对比 (MLSynth paper, Table 1):**

| Method | Accurate? | Reproducible? | Tunable? |
|--------|-----------|---------------|----------|
| Real test-bed | Yes | No | No |
| Non-AI workload | No | Yes | Yes |
| Real Chakra ET | Yes | No | No |
| **SimAI AICB** | Yes | Yes | **No** |
| **MLSynth** | Yes | Yes | **Yes** |

**对 AICB 设计的启示:** MLSynth 的 "AICB is not tunable" 判断直接驱动了 Function F003（可调优性注入）。Performance Wrapper 的 decorator pattern 为 AICB 的 `TunabilityWrapper` 设计提供了参考架构。

Sources: [MLSynth NAIC '25 paper](https://dl.acm.org/doi/10.1145/3748273.3749211); [NetMLSim/MLSynth GitHub](https://github.com/NetMLSim/MLSynth)

#### 3.3.5 Echo (NetX-lab/Echo) -- Ex-Situ Trace 模拟器（竞争/互补）

**定位:** 从单个物理 GPU 模拟大规模分布式训练。92% accuracy on GPT-175B at 96-GPU H800 scale。(CUHK, HKUST, HKU, Microsoft Research)。

**四模块系统:**

**Module 1 -- Workload Tracer (v0.5, May 2025):** Ex-situ tracing from single device。Captures forward + backward passes as execution graphs。Supports PyTorch, DeepSpeed, Megatron-LM, HuggingFace Transformers, DDP。

**Module 2 -- NCCL Communication Model (White-Box):** 不同于 alpha-beta 模型：建模 connection setup overhead, intra/inter-server transmission, data reduction time, NCCL algorithm selection (ring vs tree, chunk sizes)。Planned: DCQCN congestion control, adaptive routing, bandwidth contention。

**Module 3 -- Slowdown Predictor (v1.0, March 2025):** ML-based (XGBoost) predictor of performance degradation from overlapping computation and communication。Uses Nsight Compute and Nsight Systems for kernel profiling (SM occupancy, DRAM utilization, NCCL transmission features)。

**Module 4 -- 3D Parallelism Simulator:** Scales single-device trace to multi-GPU: DP (replicate + AllReduce), TP (split GEMMs/attention + AllReduce), PP (split layers + P2P send/recv)。EP planned。

**模型覆盖:** Any HuggingFace model (via runtime tracing) or custom PyTorch model。This is the broadest model coverage of any tool，但需要物理 GPU profiling per model。覆盖范围无硬编码模型列表 -- 仅受可以 profiling 的内容限制。

**Accuracy:** 92% step-time accuracy (GPT-175B, 96-GPU H800); ~3x lower error than prior simulators (FlexFlow, Daydream, ASTRA-sim, SimAI)。Simulation time: <2 minutes。

**对 AICB 设计的启示:** Echo 92% 的准确率是 AICB 的 accuracy gap。但 Echo requires physical GPU per model，无法生成不存在硬件配置的 workload。Echo 和 AICB 是**互补关系**：Echo 适合已有物理集群的高保真回放，AICB 适合无需 GPU 的 forward-looking architecture exploration。

Sources: [Echo paper (arXiv:2412.12487)](https://arxiv.org/abs/2412.12487); [NetX-lab/Echo GitHub](https://github.com/NetX-lab/Echo)

### 3.4 竞争工具全面对比矩阵

基于以上架构分析，以下是跨 17 维度的全面对比：

| Dimension | AICB (SimAI) | Chakra (MLCommons) | MLSynth (NVIDIA) | Echo (NetX-lab) | PARAM (Meta) | ASTRA-sim (Georgia Tech) |
|-----------|-------------|-------------------|------------------|-----------------|--------------|--------------------------|
| **Role** | Workload generator + benchmark | Trace format standard + Open Trace Library | Synthetic workload generator | Ex-situ trace-based simulator | Low-level comm/compute benchmark | Distributed AI system simulator |
| **Core abstraction** | Framework-level (Megatron, DeepSpeed, vLLM, DeepSeek) | Trace-level (Chakra ET DAG) | Layer-level (Layer->Model->Orchestrator) | Execution-level (runtime trace from single GPU) | Operator-level (GEMM, NCCL ops) | System-level (Workload->System->Backend) |
| **Output format** | SimAI proprietary text | Chakra ET (Protobuf / JSON DAG) | Chakra ET (DAG) | Internal trace + step-time | PyTorch ET | Consumes Chakra ET; outputs CSV |
| **Input required** | Model config (framework, parallelism, layer params) | N/A (format, not generator) | Model config (batch_size, layers, hidden, parallelism) | Physical GPU + model code + network topo | NCCL config / DLRM model | Chakra ET + topology + system config |
| **Models covered** | ~10 curated | GPT-3, Llama 3, Mixtral, DeepSeek-MoE, vLLM (Open Trace Lib) | Transformer, MoE (parametric classes) | Any HuggingFace model (runtime trace) | DLRM only | Any model (via Chakra ET) |
| **Parallelism** | DP, TP, PP, EP, SP | All (captured in trace) | DP, TP, PP | DP, TP, PP (EP planned) | NCCL collectives only | All (consumed from trace) |
| **Tunable?** | **No** (MLSynth paper weakness) | N/A | **Yes** (stragglers, scaling, variability) | Yes (replay parameter sweep) | No | Yes (topology, congestion, algorithms) |
| **Network fidelity** | Bus-bw or NS3 via SimAI | Fed into ASTRA-sim (NS3) | Fed into ASTRA-sim (NS3) | White-box NCCL model | None (real hardware) | Full NS3 (RoCE, DCQCN/HPCC) + Analytical + Garnet |
| **Accuracy** | Good (flow-level; NSDI '25) | High (real production traces) | High (validated synthetic) | **92%** (GPT-175B, 96-GPU H800) | N/A (real hardware) | High (real traces) / Lower (synthetic traces) |
| **Simulation speed** | Fast (bus-bw) / Slow (NS3) | N/A | Fast (generation only) | **<2 min** (GPT-175B) | N/A | Fast (analytical) / Slow (NS3, hours-days) |
| **Requires GPU?** | No (mocked_model) | No | No | **Yes** (single GPU) | **Yes** | No |
| **Interoperability** | SimAI only | 40+ member ecosystem | ASTRA-sim, Multiverse | Standalone | PyTorch ecosystem | MLCommons Chakra ecosystem |
| **Extensibility** | Add framework backend + config | Extend schema via AttributeProto | Add Layer implementations | Trace any HF model | Add microbenchmark ops | Add backend plugin |
| **Limitation** | Proprietary format; not tunable | Not a generator | New (2025); limited Layer library | Requires GPU per model; no hypothetical configs | Not an LLM workload generator | GIGO with synthetic traces; NS3 slow |
| **Open source** | Yes (aliyun/aicb, MIT) | Yes (mlcommons/chakra, Apache 2.0) | Yes (NetMLSim/MLSynth, MIT) | Yes (NetX-lab/Echo) | Yes (facebookresearch/param, MIT) | Yes (astra-sim, MIT) |
| **Key paper** | NSDI '25 Spring (SimAI) | MLSys 2026 (Chakra) | NAIC '25 (MLSynth) | arXiv 2024 (Echo) | -- | Multiple (SIGCOMM, NSDI, ISCA) |
| **Community** | ~2 internal contributors | 40+ member WG | NVIDIA/Imperial | CUHK/HKUST/HKU/MSR | Meta internal | Georgia Tech/Intel/Meta |
| **Latest release** | AICB 2.1 (Nov 2025) | MLSys 2026 (May 2026) | NAIC '25 (Sep 2025) | v1.0 (Mar 2025) | v0.1 | ASTRA-sim 2.0 |

### 3.5 AICB 竞争差距与优势总结

基于以上对比矩阵，以下是 AICB 当前的面相竞争工具的差距与优势：

**差距（Evidence-Based）：**

**Gap 1: Format Lock-In (STRATEGIC -- Most Severe)**
AICB 输出 SimAI 专有格式。行业向 Chakra ET 收敛：IEEE Micro paper (April 2025) 定位 Chakra 为 universal representation；MLSynth/STAGE/MSCCLang/TACOS 均输出 Chakra ET；ASTRA-sim/Multiverse/Keysight KAI DCB 均消费 Chakra ET；40+ member MLCommons WG with commercial adoption。AICB workload 不能被任何其他模拟器消费，真实 Chakra traces 无法与 AICB workload 对比。**--> 驱动 Function F002**

**Gap 2: Tunability (TECHNICAL -- Citable in Literature)**
MLSynth paper explicitly lists AICB as "not tunable"：无法注入 stragglers；无法在生成后缩放 workload；traces 耦合特定 NCCL 实现。Context: ~10.4% GPU hours wasted due to stragglers; 42.5% training jobs experience >=10% slowdown (ByteDance data)。**--> 驱动 Function F003**

**Gap 3: Model Coverage Breadth (TECHNICAL -- Fillable)**
AICB's ~10 curated models is fewer than Echo (any HF model) and Chakra's Open Trace Library。Model coverage gap summary per family:

| Model Family | AICB 2.1 | Chakra Open Trace Lib | MLSynth | Echo | Gap Type |
|-------------|----------|----------------------|---------|------|----------|
| LLaMA 7B/65B/405B | Yes | Yes (Llama 3) | Via Transformer class | Yes (HF trace) | None |
| GPT 13B-175B | Yes | Yes (GPT-3) | Via Transformer class | Yes (HF trace) | None |
| GPT-4 class | No (arch not public) | No | No | No | Industry-wide |
| Mixtral 8x22B | No | Yes | Via MoE class | Yes | Parametric |
| Qwen 2.5 dense | No | No | Via Transformer class | Yes | Parametric |
| Gemma 2/3 | No | No | Via Transformer class | Yes | Parametric |
| Falcon 3 | No | No | No | Yes | Architectural |
| DBRX | No | No | Via MoE class | Yes | Architectural |
| Phi-3/4 | No | No | Via Transformer class | Yes | Parametric |
| Command R | No | No | Via Transformer class | Yes | Parametric |
| Llama 4 Maverick | No | No | No | Yes (HF trace) | Architectural |
| Jamba 2 | No | No | No | No | Architectural |
| DeepSeek V4 | No | No | No | No | Architectural |

**Implication:** AICB 的模型覆盖差距集中在两类：**参数差距** (6 models: fillable via config, ~5 days total) 和 **架构差距** (5 models: 需要代码变更)。**--> 驱动 Function F001, F004, F005**

**Gap 4: Accuracy vs Echo (TECHNICAL)**
Echo claims 92% step-time accuracy (~3x lower error than SimAI/ASTRA-sim)。Echo's white-box NCCL modeling and ML-based slowdown predictor are more sophisticated than AICB's flow-level communication abstraction。However, Echo requires physical GPU per model, while AICB generates workloads for hypothetical configurations。

**优势：**

1. **Framework-level abstraction:** AICB models communication at the framework level (Megatron-LM, DeepSpeed, vLLM, DeepSeek)，capturing communication patterns as they actually manifest in production training。MLSynth 和 Echo 无法匹配此 framework-specific fidelity。
2. **First-class Megatron-LM and DeepSeek support:** 这是 scale 上最重要的两个训练框架。无其他工具如此忠实地建模 Megatron's TP/PP/SP/EP 并行语义。
3. **End-to-end SimAI integration:** For SimAI ecosystem users，AICB provides seamless workflow from workload generation to full-stack NS3 simulation。
4. **Inference support (AICB 2.1):** MLSynth 和 Echo 是训练导向的。AICB 为 DeepSeek 和 Qwen3 生成推理 workloads via vLLM。
5. **Peer-reviewed foundation (NSDI '25 Spring):** SimAI/AICB accepted at a top-tier systems/networking venue。
6. **No physical GPU required:** 不同于 Echo 和 PARAM，AICB can generate workloads for configurations never physically deployed。

### 3.6 社区贡献评估

Sub-question (3) of the original research asks: does aliyun/aicb have recent community contributions supporting new models?

**内部贡献（AICB 2.1 release notes 确认）：**

| Contributor | Contribution | Scope |
|------------|-------------|-------|
| @Yan824 | DeepSeek inference workload generation (via DeepSeek_Simulator) | AICB 2.1, November 2025 |
| @parthpower | DeepSeek training workload generation | AICB 2.1, November 2025 |
| Core team | Qwen3-MoE and Qwen3-Next inference (vLLM-based), Megatron MoE update, mocked_model restructured | AICB 2.1, November 2025 |

**外部社区贡献:** No significant external PRs extending model support were found。对比：
- Chakra: 40+ member WG with active contributions from NVIDIA, AMD, Meta, HPE, Keysight, Huawei, Georgia Tech, Harvard
- Echo: Multi-institution collaboration (CUHK, HKUST, HKU, Microsoft Research)
- MLSynth: Open-source (MIT) from NVIDIA/Imperial, accepting community Layer implementations

**Implications:** AICB's model extension is bottlenecked on internal team bandwidth。Publishing clear extension documentation and accepting community model config PRs would significantly accelerate coverage（大多数扩展仅需参数配置）。

---

## 4. 功能实现设计

### 4.1 功能概览与交付路线图

本设计涵盖 5 个功能，按 effort-to-impact 比分为 4 个 Phase。每个功能遵循 template sub-sections 4.x.1 - 4.x.10。

| Phase | Function | Description | Impact | Effort | Cumulative Coverage |
|-------|----------|-------------|--------|--------|---------------------|
| **Phase 1 (immediate)** | F001: 模型参数配置扩展 | Add 6 models via config profiles | Medium | 3 days | 13 -> 16 models |
| **Phase 1 (immediate)** | F002: Chakra ET 格式导出 | Serialize to MLCommons standard format | **High** | 5-8 days | Eliminates format lock-in |
| **Phase 2 (short-term)** | F003: Workload 可调优性注入 | Straggler + scaling + variability | **High** | 5-8 days | Eliminates tunability gap |
| **Phase 3 (medium-term)** | F005: Falcon 并行子层 | Parallel attention + MLP mode | Medium | 3-5 days | 16 -> 17 models |
| **Phase 3 (medium-term)** | F004: 非均匀层通信 Profile | Per-layer communication type | Medium | 8-12 days | Enables Jamba/Llama4/DV4 |
| **Phase 4 (long-term)** | F004 extended: SSM + Hash-Routed | Mamba layer primitive + hash-routed MoE | High (forward-looking) | 20+ days | Enables Jamba2/DV4 |

---

### 4.2 功能 F001：新增模型参数配置文件

##### 4.2.1 功能概述

为 6 个 decoder-only 模型家族（LLaMA 3.1 8B/70B, Phi-4, Gemma 3 12B/27B, Qwen 2.5, Command R, Mixtral 8x22B）添加参数配置文件，无需修改 mocked_model 代码。利用现有 Megatron-LM mocked_model 后端的参数化能力，仅需新增 YAML 格式的模型 profile。

##### 4.2.2 SR 设计

| SR编号 | 系统需求描述 | 实现方式 |
|--------|-------------|----------|
| SR-F001-01 | 支持 LLaMA 3.1 8B 和 70B 的 workload 生成 | 复用现有 Llama3 405B 的 Megatron-LM 配置模板，替换 hidden_size / num_layers / num_heads / num_kv_heads |
| SR-F001-02 | 支持 Phi-4 的 workload 生成 | 新增 Megatron-LM 模型 profile，使用 GeLU 激活函数（非 SwiGLU） |
| SR-F001-03 | 支持 Gemma 3 的 workload 生成（12B/27B） | 新增 Megatron-LM 模型 profile，配置 GQA ratio（3:1 / 2:1），GeGLU 激活，5:1 local:global attention |
| SR-F001-04 | 支持 Mixtral 8x22B 的 workload 生成 | 复用现有 Mixtral 8x7B 配置模板，替换专家数量参数 |
| SR-F001-05 | 支持 Qwen 2.5 和 Command R 的 workload 生成 | 新增 Megatron-LM 模型 profile |

##### 4.2.3 实现思路

利用 AICB 现有 Megatron-LM 后端的参数化能力。核心思路：所有目标模型都是 Pre-norm Decoder-Only Transformer，差异仅在于维度参数、激活函数和注意力模式。AICB 2.1 的 mocked_model 已经支持这些参数的配置化注入。备选方案（为每个模型编写独立的 mocked_model 子类）评估后否决 -- 代码冗余且不利于社区贡献。

##### 4.2.4 实现设计 【核心 -- 必选】

**前置条件**: Megatron-LM 后端的 mocked_model 正常运行，现有 LLaMA/GPT/Mistral 配置可生成 workload。

**触发事件**: 用户指定新模型名称（如 `llama3.1-8b`）和并行策略配置。

**主流程（自然语言描述）：**

```
1. 用户提供模型选择 + 并行策略配置
   Input: model_name="llama3.1-8b", tp=2, pp=1, dp=4, gpus=64, seq_len=131072

2. 系统从模型配置注册表加载对应 profile
   Read: configs/models/llama3.1_8b.yaml
   Extract: hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=8,
     ffn_hidden_size=14336, vocab_size=128256, activation="silu",
     use_parallel_attention=false, seq_length=131072

3. 系统使用 Megatron-LM mocked_model 生成 layer 结构
   For each layer:
     a. Generate attention block (QKV projection, attention score, output projection)
        - Apply TP=2: split QKV weight matrices column-wise
     b. Generate MLP block (gate_proj, up_proj, down_proj with SwiGLU)
        - If TP>1: insert AllReduce communication node after attention and MLP
     c. If layer_index is a global attention layer (Gemma's 5:1 pattern): mark as global
        - Communication: global layer = full AllReduce; local layer = reduced AllReduce

4. 系统应用并行策略
   TP=2: Every 2 GPUs form a TP group; attention/MLP weights column-split
   DP=4: 4 data-parallel groups; insert AllReduce (gradient sync) after backward
   PP=1: Single pipeline stage (no P2P send/recv)

5. 系统插入集体通信操作
   TP AllReduce: after each layer's attention and MLP (2 per layer)
   DP gradient AllReduce: after backward pass (1 per step)
   Comm size = hidden_size * hidden_size * 4 bytes (FP32) / tp_size

6. 系统（可选）调用 AIOB profiler 获取 GPU kernel 计时
   If GPU available: use AIOB to measure GEMM/attention kernel timing
   If GPU unavailable: use FLOP estimation (roofline model)

7. 系统输出 workload 文件
   Format: existing SimAI proprietary format
   Format: new Chakra ET export (see Function F002)
```

**后置条件**: Workload 文件可被 SimAI simulator 或 Chakra ET 消费者正确解析，生成的通信模式与真实 Megatron-LM 训练日志一致（误差 < 5%）。

**时序图（自然语言）：**
```
User -> AICB Entry: model_name="llama3.1-8b", tp=2, gpus=64
AICB Entry -> ModelRegistry: lookup("llama3.1-8b")
ModelRegistry -> AICB Entry: ModelProfile(hidden_size=4096, layers=32, ...)
AICB Entry -> MegatronBackend: generate_workload(profile, parallelism)
MegatronBackend -> MockedModel: build_layers(profile)
MockedModel -> MockedModel: for layer in 0..31: create AttentionBlock + MLPBlock
MockedModel -> MockedModel: apply TP=2: split weight matrices, insert AllReduce
MegatronBackend -> WorkloadWriter: write(operations)
WorkloadWriter -> WorkloadWriter: serialize to SimAI format + Chakra ET
WorkloadWriter -> User: workload_file.dat, workload.chakra.et
```

##### 4.2.5 用户接口设计

```yaml
# configs/models/llama3.1_8b.yaml
model_name: "llama3.1-8b"
framework: "megatron"
hidden_size: 4096
num_layers: 32
num_attention_heads: 32
num_kv_heads: 8          # GQA ratio = 4:1
ffn_hidden_size: 14336
vocab_size: 128256
activation_function: "silu"  # SwiGLU
use_parallel_attention: false
num_experts: 0            # Dense model
seq_length: 131072
rope_theta: 500000
```

CLI: `./aicb generate --model llama3.1-8b --tp 2 --gpus 64 --output-format chakra`
API: `registry = ModelRegistry(); profile = registry.load("llama3.1-8b"); workload = MegatronBackend.generate(profile, parallelism)`

##### 4.2.6 实现接口设计

| 接口名称 | 接口描述 | 接口类型 | 所属系统元素 | 规格约束 |
|----------|----------|----------|--------------|----------|
| `ModelRegistry.load(name)` | 根据模型名称加载配置 profile | Python API | `aicb/config/registry.py` | 返回 `ModelProfile` dataclass；模型不存在时抛出 `ModelNotFoundError` |
| `MegatronBackend.generate(profile, parallelism)` | 根据 profile 和并行配置生成 workload | Python API | `aicb/workload_generator/training/megatron/backend.py` | profile 必须包含完整模型参数；返回 `Workload` 对象 |
| `WorkloadWriter.to_chakra_et(workload, output_path)` | 将 workload 序列化为 Chakra ET 格式 | Python API | `aicb/export/chakra_writer.py` | 见 Function F002 接口规格 |

##### 4.2.7 安全配置设计

本功能不涉及安全配置（纯 workload 生成，无网络暴露、无用户数据、无认证授权需求）。

##### 4.2.8 功能规格变更与设计

本功能为新增功能，无规格变更。

##### 4.2.9 DFX 分析

**可靠性分析 (FMEA):**

| 故障模式 | 影响 | 严重度 | 检测方法 | 缓解措施 |
|---------|------|--------|---------|----------|
| 模型参数配置错误（如 hidden_size 不匹配实际模型） | 生成的 workload FLOP 计数偏差 | 中 | 与已发表模型规格交叉验证 | CI 中添加参数校验：hidden_size % num_heads != 0 时拒绝 |
| GQA ratio 配置错误（如 num_kv_heads > num_heads） | KV 通信量计算错误 | 中 | 参数合法性检查 | `assert num_kv_heads <= num_heads and num_heads % num_kv_heads == 0` |
| 并行策略与 GPU 数不兼容（如 TP=8 但 gpus=6） | 生成失败或通信拓扑错误 | 高 | 前置校验 | `assert gpus % (tp * pp * dp) == 0` |
| 新模型 profile 缺少必填字段 | 生成崩溃 | 高 | Schema 验证 | 使用 dataclass + type hints，加载时自动校验 |

**可测试性**: 每个新模型 profile 需包含金标准验证用例：使用已知并行配置生成 workload，将通信量（总 AllReduce bytes）与手动计算值比较，误差 < 1%。

##### 4.2.10 分配需求

| SR编号 | 分配需求描述 | 系统元素 |
|--------|-------------|----------|
| SR-F001-01 | 新增 `configs/models/llama3.1_8b.yaml` 和 `llama3.1_70b.yaml` | `aicb/config/` 目录 |
| SR-F001-02 | 新增 `configs/models/phi4.yaml` | `aicb/config/` 目录 |
| SR-F001-03 | 新增 `configs/models/gemma3_12b.yaml` 和 `gemma3_27b.yaml`，包含 GQA ratio 和 local:global attention ratio | `aicb/config/` 目录 |
| SR-F001-04 | 修改 `configs/models/mistral_8x22b.yaml`（从 8x7B 复制并修改） | `aicb/config/` 目录 |
| SR-F001-05 | 新增 `configs/models/qwen2.5_7b.yaml` 和 `command_r.yaml` | `aicb/config/` 目录 |
| -- | 在所有新增配置上运行验证测试 | `aicb/tests/test_model_configs.py` |

---

### 4.3 功能 F002：Chakra Execution Trace 格式导出

##### 4.3.1 功能概述

新增 Chakra Execution Trace (Protobuf/JSON DAG) 格式导出层，将 AICB mocked_model 生成的 compute + communication operations 映射为 Chakra schema nodes，通过 `ctrl_deps` 和 `data_deps` 建立依赖关系。这是消除 AICB 最大竞争劣势（格式锁定）的核心功能。

##### 4.3.2 SR 设计

| SR编号 | 系统需求描述 | 实现方式 |
|--------|-------------|----------|
| SR-F002-01 | 将 mocked_model 的 compute ops 映射为 Chakra `COMP_NODE` | 遍历 workload compute op 列表，为每个 op 生成 Chakra node |
| SR-F002-02 | 将 mocked_model 的 collective comm ops 映射为 Chakra `COMM_COLL_NODE` | 映射 AllReduce -> `ALL_REDUCE`, AllGather -> `ALL_GATHER`, ReduceScatter -> `REDUCE_SCATTER`, AlltoAll -> `ALL_TO_ALL`, Broadcast -> `BROADCAST` |
| SR-F002-03 | 正确建立 node 间的 data_deps 和 ctrl_deps 依赖关系 | 按 layer 顺序和并行策略依赖图建立 DAG 边 |
| SR-F002-04 | 为每个 GPU rank 生成独立的 Chakra ET 文件 | 按 DP/TP/PP group 分发 ops，生成 per-rank `.et` 文件 |
| SR-F002-05 | 输出 CommGroup JSON（集体通信组拓扑） | 序列化 parallel group membership |

##### 4.3.3 实现思路

AICB 在生成 workload 时已构建了一个内部 DAG（compute ops 按 layer 排序，communication ops 按并行策略插入）。Chakra ET 导出是一个**序列化转换问题**，不是架构变更。核心思路：实现 `ChakraWriter` 类，接收 `Workload` 对象，遍历其 op 列表，对每个 op 创建对应的 Chakra Protobuf message。

备选方案（使用 Chakra 官方的 `chakra_converter` 工具链）评估后否决 -- 需要先将 AICB 输出转为 PyTorch ET 格式再经 Kineto 链路转换，引入两步不必要的数据膨胀。直接写入 Protobuf 更高效。

##### 4.3.4 实现设计 【核心 -- 必选】

**前置条件**: `Workload` 对象已由 mocked_model 生成完成，包含完整的有序 compute ops 和 communication ops 列表，以及每个 op 的 GPU rank 归属、依赖关系和通信组信息。

**触发事件**: 用户在 workload 生成命令中指定 `--output-format chakra` 或调用 `WorkloadWriter.to_chakra_et()`。

**主流程（自然语言描述）：**

```
1. 接收 Workload 对象，获取:
   - total_ranks: 总 GPU 数量
   - operations: List[Op] (有序，含 compute 和 communication ops)
   - comm_groups: Dict[str, List[int]] (并行组 membership)

2. 初始化 per-rank Chakra ET 文件
   For each rank (0 .. total_ranks-1):
     Create empty Chakra ExecutionTrace protobuf message
     Create node_id_counter = 0

3. 遍历 operations 列表:
   For each op:

   3a. Map op type to Chakra NodeType:
       GEMM/Attention/Activation -> COMP_NODE (type=4)
       AllReduce/AllGather/ReduceScatter/AlltoAll/Broadcast -> COMM_COLL_NODE (type=7)
       P2P_Send -> COMM_SEND_NODE (type=5)
       P2P_Recv -> COMM_RECV_NODE (type=6)

   3b. Create Chakra Node:
       - id = node_id_counter++
       - name = op.description (e.g., "layer_5_attention_QKV", "tp_allreduce_layer_5")
       - type = mapped NodeType
       - ctrl_deps = [parent_op.node_id]
       - data_deps = [producer_op.node_id]
       - start_time_micros = cumulative_time
       - duration_micros = op.compute_time_us or op.comm_time_us
       - For COMP_NODE: set attr flops, tensor_shape
       - For COMM_COLL_NODE: set collective_comm_type (ALL_REDUCE etc.),
         attr comm_size_bytes, num_participants

   3c. Distribute to target ranks:
       Compute ops: add to op.assigned_rank's ET
       Communication ops: add to all participant ranks' ETs

4. 生成 CommGroup JSON:
   For each parallel group type (dp, tp, pp, ep):
     List all group rank lists
   Example: {"tp_groups": [[0,1], [2,3], ...], "dp_groups": [[0,2,4,6], ...]}

5. Write files:
   Per-rank: workload_rank_{N}.et (Protobuf binary) or .et.json (JSON)
   Comm group: comm_group.json
```

**后置条件**: 生成的 Chakra ET 文件可被 ASTRA-sim `ETFeeder`、Multiverse 或 Keysight KAI DCB 正确解析和模拟。

**时序图（自然语言）：**
```
User -> AICB Entry: generate_workload(model="llama3.1-8b", ..., output_format="chakra")
AICB Entry -> MegatronBackend: generate_workload(profile, parallelism)
MegatronBackend -> Workload: build()
Workload -> AICB Entry: Workload object (total_ranks=N, operations=[Op, ...])
AICB Entry -> ChakraWriter: to_chakra_et(workload, output_dir)
ChakraWriter -> ChakraWriter: init per-rank ETs for ranks 0..N-1
loop for each op in workload.operations:
    ChakraWriter -> ChakraWriter: map op.type -> Chakra NodeType
    ChakraWriter -> ChakraWriter: create Node(id, name, type, deps, timing, attrs)
    ChakraWriter -> ChakraWriter: add Node to target_rank.ET
ChakraWriter -> ChakraWriter: generate comm_group.json from parallel groups
ChakraWriter -> User: output_dir/{rank_0.et, ..., rank_N.et, comm_group.json}
User -> ASTRA-sim: ./astra-sim --workload=output_dir/ --network=topo.yaml
```

##### 4.3.5 用户接口设计

```
CLI:  ./aicb generate --model llama3.1-8b --tp 2 --gpus 64 --output-format chakra --output-dir ./workloads/
API:  writer = ChakraWriter(workload); writer.write(output_dir)
```

##### 4.3.6 实现接口设计

| 接口名称 | 接口描述 | 接口类型 | 所属系统元素 | 规格约束 |
|----------|----------|----------|--------------|----------|
| `ChakraWriter.__init__(workload, output_dir)` | 初始化 Chakra 导出器 | Python API | `aicb/export/chakra_writer.py` | `workload` 必须是已完成的 `Workload` 对象 |
| `ChakraWriter.write()` | 遍历 ops，生成 per-rank .et 文件 + comm_group.json | Python API | `aicb/export/chakra_writer.py` | 无返回值；写入失败抛出 `ChakraExportError` |
| `ChakraWriter._map_node_type(op)` | 将 AICB op 类型映射为 Chakra NodeType enum | Python API (internal) | `aicb/export/chakra_writer.py` | 映射表: GEMM->COMP_NODE(4), AllReduce->COMM_COLL_NODE(7)+ALL_REDUCE(0), ... |
| `ChakraWriter._build_deps(op, prev_op)` | 建立 ctrl_deps 和 data_deps | Python API (internal) | `aicb/export/chakra_writer.py` | 处理跨 rank 依赖（comm node data_deps 引用远程 rank compute node） |

##### 4.3.7 安全配置设计

不涉及安全配置。Chakra ET 文件不包含模型权重、训练数据或用户数据 -- 仅包含操作类型、FLOP 数、通信量、依赖关系等非敏感元数据。

##### 4.3.8 功能规格变更与设计

本功能为新增功能，无规格变更。

##### 4.3.9 DFX 分析

**可靠性分析 (FMEA):**

| 故障模式 | 影响 | 严重度 | 检测方法 | 缓解措施 |
|---------|------|--------|---------|----------|
| Chakra ET 文件缺少 data_deps 导致下游模拟器死锁 | 模拟无法进行 | 高 | 导出后运行 `chakra_validate` 校验 schema | CI 中执行 ASTRA-sim dry-run |
| 跨 rank 依赖引用不存在的 node_id | Chakra feeder 崩溃 | 高 | 导出时校验所有 data_deps 引用的 node_id 在目标 rank ET 中存在 | 建立全局 node_id registry |
| 通信量（comm_size_bytes）计算错误 | 模拟结果偏差 | 中 | 与手动计算值和框架日志对比 | Add log-level communication size validation |
| Protobuf 序列化失败（字段类型不匹配） | 导出崩溃 | 中 | Type hints + mypy 静态检查 | 使用 Chakra 官方 protobuf schema 生成 Python stubs |

**性能分析**: GPT-175B (96 layers, TP=8, PP=8, 64 GPUs) Chakra ET 导出预计耗时 < 5 秒。输出文件大小约 50-200 MB per 64 ranks。

**可测试性**: 与已知 gold standard 比较 -- 使用 AICB 生成的 LLaMA 7B workload 导出的 Chakra ET，应与从真实 PyTorch Megatron-LM 训练中捕获的 Chakra ET 在通信节点数量和总通信量上一致（误差 < 5%）。

##### 4.3.10 分配需求

| SR编号 | 分配需求描述 | 系统元素 |
|--------|-------------|----------|
| SR-F002-01 | 实现 `aicb/export/chakra_writer.py`，包含 `ChakraWriter` 类和 op->NodeType 映射 | `aicb/export/` 模块 |
| SR-F002-02 | 实现 collective_comm_type 枚举映射表 | `aicb/export/chakra_writer.py` |
| SR-F002-03 | 实现 data_deps/ctrl_deps 依赖图构建逻辑 | `aicb/export/chakra_writer.py` |
| SR-F002-04 | 实现 per-rank ET 分发逻辑 | `aicb/export/chakra_writer.py` |
| SR-F002-05 | 实现 comm_group.json 序列化 | `aicb/export/comm_group_writer.py` |
| -- | 添加 `--output-format chakra` CLI flag | `aicb/cli.py` |
| -- | 添加 Chakra ET 导出单元测试（至少覆盖 3 种并行策略组合） | `aicb/tests/test_chakra_export.py` |
| -- | 添加与 ASTRA-sim 的集成测试（dry-run 验证 ET 可被 ETFeeder 解析） | `aicb/tests/test_astra_sim_integration.py` |

---

### 4.4 功能 F003：Workload 可调优性注入

##### 4.4.1 功能概述

为 AICB workload 生成流程添加可调优性能力：straggler 注入（延迟 GPU kernel 或 NIC 通信）、workload 缩放（生成后改变 GPU 数量而不完全重新生成）、性能变异性建模（向操作持续时间添加噪声参数）。直接对标并消除 MLSynth 论文中 "AICB is not tunable" 的竞争劣势。

##### 4.4.2 SR 设计

| SR编号 | 系统需求描述 | 实现方式 |
|--------|-------------|----------|
| SR-F003-01 | 支持向 GPU compute 节点注入 straggler 延迟 | 在 `ComputeOp` 中新增 `straggler_delay_us` 字段，支持 Gaussian/Pareto 分布采样 |
| SR-F003-02 | 支持向 NIC communication 节点注入网络延迟 | 在 `CommunicationOp` 中新增 `straggler_delay_us` 字段 |
| SR-F003-03 | 支持 workload 缩放（改变 DP/TP 维度后重算通信量，不重新生成 layer 结构） | 实现 `WorkloadScaler` 类，接收现有 Workload + 新并行配置 |
| SR-F003-04 | 支持性能变异性建模（Monte Carlo noise） | 实现 `VariabilityInjector` 类，向 op 持续时间添加可控噪声 |

##### 4.4.3 实现思路

采用**装饰器/拦截器模式**（参考 MLSynth 的 Performance Wrapper 设计），在 `WorkloadWriter` 和 mocked_model 之间插入 `TunabilityWrapper` 层。这样 tunability 逻辑与 workload 生成核心逻辑解耦，且对现有代码无侵入。备选方案（直接修改每个 op 的持续时间字段）评估后否决 -- 无法支持延后的缩放操作。

##### 4.4.4 实现设计 【核心 -- 必选】

**前置条件**: `Workload` 对象已由 mocked_model 生成完成。

**触发事件**: 用户在 CLI 中指定 straggler/scale/variability 参数。

**主流程（自然语言描述）：**

```
1. 接收原始 Workload 对象 + tunability 参数:
   straggler_config: {gpu_rate: 0.02, gpu_delay_dist: Pareto(alpha=1.5, scale=100us),
                      nic_rate: 0.01, nic_delay_dist: Normal(mu=50us, sigma=10us)}
   scale_config: {new_dp: 8, new_tp: 4}  (from dp=4, tp=2)
   variability_config: {noise_type: Gaussian, noise_std_ratio: 0.05}

2. If scale_config is set:
   WorkloadScaler.recalculate(workload, new_dp, new_tp):
     a. Determine new rank distribution (dp * tp GPUs)
     b. Recalculate communication sizes:
        AllReduce comm_size proportional to 1 / tp_size
        AllReduce count proportional to tp_size * layers
     c. Redistribute compute ops to new ranks
     d. Regenerate communication ops at correct inter-layer positions
   Note: workload scaling preserves total FLOPs (same model), only changes comm topology

3. If straggler_config is set:
   For each compute op: with probability gpu_rate, sample delay from gpu_delay_dist
   For each communication op: with probability nic_rate, sample delay from nic_delay_dist
   Write delays to op.straggler_delay_us

4. If variability_config is set:
   For each op: op.duration_us = op.duration_us * (1 + noise_sample)
   where noise_sample ~ Normal(0, noise_std_ratio)
```

**后置条件**: 生成的 workload 包含 straggler/缩放/变异性效果，可直接用于模拟。

##### 4.4.5 用户接口设计

```
CLI:  ./aicb generate --model llama3.1-8b --tp 2 --gpus 64 \
          --straggler-gpu-rate 0.02 --straggler-nic-rate 0.01 \
          --variability 0.05 --output-format chakra

API:  wrapper = TunabilityWrapper(workload)
      wrapper.inject_stragglers(gpu_rate=0.02, nic_rate=0.01)
      wrapper.add_variability(std_ratio=0.05)
      scaled = wrapper.scale(new_dp=8, new_tp=4)
```

##### 4.4.6 实现接口设计

| 接口名称 | 接口描述 | 接口类型 | 所属系统元素 | 规格约束 |
|----------|----------|----------|--------------|----------|
| `TunabilityWrapper.__init__(workload)` | 初始化可调优性包装器 | Python API | `aicb/tuning/wrapper.py` | workload 必须是已完成的 Workload 对象 |
| `TunabilityWrapper.inject_stragglers(gpu_rate, nic_rate, gpu_dist, nic_dist)` | 向 ops 注入 straggler 延迟 | Python API | `aicb/tuning/wrapper.py` | rate 参数范围: [0.0, 1.0] |
| `TunabilityWrapper.add_variability(std_ratio)` | 添加性能变异性噪声 | Python API | `aicb/tuning/wrapper.py` | std_ratio > 0 |
| `WorkloadScaler.scale(new_dp, new_tp, new_pp)` | 缩放 workload 到新并行配置 | Python API | `aicb/tuning/scaler.py` | new_dp * new_tp * new_pp 必须等于目标 GPU 数 |

##### 4.4.7 安全配置设计

不涉及安全配置。

##### 4.4.8 功能规格变更与设计

本功能为新增功能，无规格变更。

##### 4.4.9 DFX 分析

**可靠性分析 (FMEA):**

| 故障模式 | 影响 | 严重度 | 检测方法 | 缓解措施 |
|---------|------|--------|---------|----------|
| Straggler rate 设置过高导致所有 GPU 同时慢 | 模拟结果无意义 | 低 | 文档警告 | rate 参数上限建议 0.1（10% 的 GPU/NIC 受影响） |
| 缩放后通信拓扑不一致（rank 对不上） | 下游模拟器崩溃 | 高 | 缩放后自动运行一致性校验 | 与原始 workload 总通信量对比（应满足理论缩放比例） |
| 变异性噪声导致持续时间为负 | 崩溃 | 中 | 截断负值为 0 | `duration_us = max(0, duration_us * (1 + noise))` |

**性能分析**: Straggler 注入对 1000-layer 级模型耗时 < 10ms（纯内存操作）。Workload 缩放复杂度 O(num_ops)，与原始生成复杂度相当。

**可测试性**: 与 gold standard 对比 -- 对已知模型，注入已知分布后，多次采样统计验证分布参数。

##### 4.4.10 分配需求

| SR编号 | 分配需求描述 | 系统元素 |
|--------|-------------|----------|
| SR-F003-01 | 实现 `ComputeOp.straggler_delay_us` 字段 + straggler 注入逻辑 | `aicb/tuning/wrapper.py` |
| SR-F003-02 | 实现 `CommunicationOp.straggler_delay_us` 字段 | `aicb/tuning/wrapper.py` |
| SR-F003-03 | 实现 `WorkloadScaler` 类 | `aicb/tuning/scaler.py` |
| SR-F003-04 | 实现 `VariabilityInjector` 类 | `aicb/tuning/variability.py` |
| -- | 添加 tunability CLI flags + 单元测试 | `aicb/cli.py`, `aicb/tests/test_tunability.py` |

---

### 4.5 功能 F004：非均匀层通信 Profile 支持

##### 4.5.1 功能概述

为 mocked_model 添加 per-layer communication type 标注能力，支持 Llama 4 Maverick (alternating dense/MoE)、Jamba 2 (SSM/Transformer 1:7)、DeepSeek V4 (hash-routed/learned MoE) 等非均匀层通信架构。当前 mocked_model 假设所有层具有相同的通信模式（每层 attention + MLP，产生相同的 AllReduce 流量）。

##### 4.5.2 SR 设计

| SR编号 | 系统需求描述 | 实现方式 |
|--------|-------------|----------|
| SR-F004-01 | 支持 `layer_type` 标注（attention, moe_layer, mamba_layer, hash_routed_moe, dense_ffn_only） | 在 `LayerConfig` 中新增 `layer_type` 字段 |
| SR-F004-02 | 根据 layer_type 生成对应的通信节点（或跳过通信） | 在 mocked_model 的 layer 构建循环中，按 layer_type 分支处理 |
| SR-F004-03 | 支持 alternating layer pattern（如 1 dense : 1 MoE） | 在 ModelConfig 中新增 `layer_pattern` 字段 |

##### 4.5.3 实现思路

最小侵入方案：在 `LayerConfig` 中新增 `layer_type` 枚举字段（默认值 `attention` 保持向后兼容），在 mocked_model 的 layer 构建循环中添加一个 `switch(layer_type)` 分支，为不同 layer type 生成不同的 compute ops 和 communication ops 集合。备选方案（为每种 layer type 创建独立的 mocked_model 子类）部分采用 -- 对于完全不同的层类型（如 mamba_layer），独立子类更清晰；对于参数变体层（如 hash_routed_moe vs learned_moe），分支处理更高效。

##### 4.5.4 实现设计 【核心 -- 必选】

**前置条件**: mocked_model 的 layer 构建循环正常运行。

**触发事件**: 用户的模型 profile 中指定了非默认的 `layer_type` 或 `layer_pattern`。

**主流程（自然语言描述）：**

```
1. User defines model profile with layer_pattern:
   # Llama 4 Maverick example:
   layer_pattern: [dense_attention, moe_layer] * 47  # 94 layers alternating
   # Jamba 2 example:
   layer_pattern: [attention_moe] + [mamba_layer] * 7  # repeated per block

2. mocked_model expands layer_pattern into full layer list:
   layers = expand_pattern(layer_pattern, num_layers)

3. For each layer, branch by layer_type:

   case attention:
     Generate AttentionBlock + MLPBlock compute ops
     If TP > 1: insert AllReduce (attention output + MLP output)
     Comm volume = 2 * hidden_size * hidden_size * 4 bytes / tp_size per layer

   case moe_layer:
     Generate AttentionBlock + MoE_FFN compute ops
     If TP > 1: insert AllReduce (attention output)
     If EP > 1: insert AlltoAll (expert dispatch + combine)
     Comm volume = (1 * AllReduce) + (2 * num_experts_per_tok * hidden_size * 4 bytes)

   case mamba_layer:  ★ NEW TYPE
     Generate SSM_scan compute op (recurrent, within single device)
     Generate dense_FFN compute op
     Insert NO communication nodes (zero AllReduce, zero AlltoAll)
     Comm volume = 0

   case hash_routed_moe:  ★ NEW TYPE
     Generate AttentionBlock + MoE_FFN compute ops
     Insert NO AlltoAll (hash routing is local, no cross-device expert comm)
     If TP > 1: insert AllReduce (attention output)
     Comm volume = 1 * AllReduce per layer (no AlltoAll)

   default (attention):
     Same as case attention (backward compatible)

4. Accumulate total communication volume and output workload
```

**后置条件**: 生成的 workload 反映了每层的实际通信量（包括零通信层）。

##### 4.5.5 用户接口设计

```yaml
# configs/models/jamba2_mini.yaml
layer_pattern:
  - type: attention_moe
  - type: mamba_layer
  - type: mamba_layer
  - type: mamba_layer
  - type: mamba_layer
  - type: mamba_layer
  - type: mamba_layer
  - type: mamba_layer
  repeat: 4  # 4 Jamba blocks = 32 layers total
```

##### 4.5.6 实现接口设计

| 接口名称 | 接口描述 | 接口类型 | 所属系统元素 | 规格约束 |
|----------|----------|----------|--------------|----------|
| `LayerConfig.layer_type` | 新增 enum 字段 | Python dataclass | `aicb/config/model_profile.py` | 默认值: `attention`；可选: `attention`, `moe_layer`, `mamba_layer`, `hash_routed_moe`, `dense_ffn_only` |
| `MockedModel._build_layer(layer_config)` | 按 layer_type 分支构建 layer | Python method | `aicb/workload_generator/training/mocked_model.py` | switch(layer_config.layer_type) |
| `expand_pattern(layer_pattern, num_layers)` | 展开重复 pattern 为完整 layer 列表 | Python function | `aicb/config/layer_utils.py` | 验证展开后长度 == num_layers |

##### 4.5.7 安全配置设计

不涉及安全配置。

##### 4.5.8 功能规格变更与设计

本功能为 `LayerConfig` 数据模型的规格扩展（新增 `layer_type` 和 `layer_pattern` 字段）。

##### 4.5.9 DFX 分析

**可靠性分析 (FMEA):**

| 故障模式 | 影响 | 严重度 | 检测方法 | 缓解措施 |
|---------|------|--------|---------|----------|
| `layer_type` 为未知值时静默 fallback 为 attention | 通信量计算错误（如 mamba 层被当作 attention 层产生虚假 AllReduce） | 高 | layer 构建循环中添加 default case throw | `switch(layer_type)` 的 default 分支必须抛出 `UnknownLayerTypeError`，包含 layer_type 值和 layer_index |
| `expand_pattern` 展开后层数与 `num_layers` 不匹配 | 生成的 workload 层数错误 | 高 | 展开后立即断言 | `assert len(expanded_layers) == num_layers` |
| mamba_layer 实现中错误地插入了 AllReduce | 产生虚假通信流量 | 中 | 单元测试验证 mamba_layer 输出中通信节点数为 0 | `assert len(mamba_layer.comm_ops) == 0` |

**安全性检查**: 不涉及安全配置变更。

**可测试性**: 对 Jamba 2 pattern (1+7)*4，验证总通信量 = 4 * (1 attention 通信量 + 7 * 0) < 标准 32 层 transformer 通信量的 13%。

##### 4.5.10 分配需求

| SR编号 | 分配需求描述 | 系统元素 |
|--------|-------------|----------|
| SR-F004-01 | 新增 `LayerConfig.layer_type` 字段 | `aicb/config/model_profile.py` |
| SR-F004-02 | 实现 `MockedModel._build_layer()` 中的 layer_type 分支 | `aicb/workload_generator/training/mocked_model.py` |
| SR-F004-03 | 实现 `expand_pattern()` 函数 + 对应测试 | `aicb/config/layer_utils.py`, `aicb/tests/test_layer_utils.py` |
| -- | 添加 mamba_layer 基础实现（零通信 SSM scan compute op） | `aicb/workload_generator/training/layers/mamba_layer.py` |

---

### 4.6 功能 F005：Falcon 并行子层支持

##### 4.6.1 功能概述

为 mocked_model 添加 `use_parallel_attention` 子层模式支持。Falcon 3 将 attention 和 MLP 设计为同一 RMSNorm 输出后的并行分支（而非 LLaMA 的串行 attention -> MLP）。这改变了通信时机：两个分支的 AllReduce 可以同时进行而非串行等待，影响通信-计算 overlap 的建模精度。

##### 4.6.2 SR 设计

| SR编号 | 系统需求描述 | 实现方式 |
|--------|-------------|----------|
| SR-F005-01 | 在 `LayerConfig` 中新增 `use_parallel_attention` bool 字段 | 默认 false（向后兼容 LLaMA 串行模式） |
| SR-F005-02 | 在 mocked_model 中，当 `use_parallel_attention=true` 时，将 attention 和 MLP 的 compute ops 设为可并行（无依赖），通信节点可同时调度 | 取消 attention output -> MLP input 的 data_dep |

##### 4.6.3 实现思路

在 mocked_model 的单层构建逻辑中，当 `use_parallel_attention=true` 时：
- attention 和 MLP 的 compute ops 共享同一个 data_dep（都依赖前一层的输出）
- 两者之间不建立 data_dep（可以并行执行）
- 两者各自的 AllReduce 通信节点也设为可并行（在同一调度窗口中）

通信模式影响：串行模式时间线为 `[Attn compute -> Attn AR -> MLP compute -> MLP AR]`，并行模式为 `[(Attn compute || MLP compute) -> (Attn AR || MLP AR)]`。

##### 4.6.4 实现设计 【核心 -- 必选】

**前置条件**: mocked_model 当前假设串行 layer 执行（`attn_out -> mlp_in` 依赖链）。

**触发事件**: 用户 model profile 中指定 `use_parallel_attention: true`。

**主流程（自然语言描述）：**

```
1. User provides Falcon 3 model profile:
   model_name: "falcon3"
   use_parallel_attention: true
   num_kv_heads: 1  (MQA)

2. mocked_model builds single layer:

   Current (serial mode):
   RMSNorm_out -> AttentionBlock -> AllReduce -> RMSNorm_in_2
                                                -> MLPBlock -> AllReduce -> output

   NEW (parallel mode):
   RMSNorm_out --+--> AttentionBlock -> AllReduce_attn --+--> Add -> output
                 |                                       |
                 +--> MLPBlock ------> AllReduce_mlp ----+

   Key differences:
   - AttentionBlock and MLPBlock both read same RMSNorm_out (no dependency)
   - AllReduce_attn and AllReduce_mlp can be scheduled simultaneously
   - AllReduce sizes differ:
     - AllReduce_attn: proportional to hidden_size * num_heads
       (with MQA, KV heads=1, so KV comm is minimal)
     - AllReduce_mlp: proportional to ffn_hidden_size (4-8x hidden)

3. Communication time calculation:
   Serial: max(AR_attn_time, 0) + max(AR_mlp_time, 0) = AR_attn_time + AR_mlp_time
   Parallel: max(AR_attn_time, AR_mlp_time)  <- both concurrent
   Savings: min(AR_attn_time, AR_mlp_time)
   For Falcon's MQA: AR_attn is small (KV heads=1), AR_mlp is large,
   so parallel mode saves approximately AR_attn_time per layer
```

**后置条件**: Falcon 3 workload 的通信时间估计比串行模式减少约 15-25%（取决于 TP degree 和 hidden/ffn 比例），与实际 Falcon training log 一致。

##### 4.6.5 用户接口设计

模型 profile 中新增字段：`use_parallel_attention: true`

##### 4.6.6 实现接口设计

| 接口名称 | 接口描述 | 接口类型 | 所属系统元素 | 规格约束 |
|----------|----------|----------|--------------|----------|
| `LayerConfig.use_parallel_attention` | bool 字段，控制子层执行模式 | Python dataclass | `aicb/config/model_profile.py` | 默认 false（向后兼容） |

##### 4.6.7 安全配置设计

不涉及安全配置。

##### 4.6.8 功能规格变更与设计

本功能为 `LayerConfig` 的规格扩展（新增 `use_parallel_attention` 字段）。

##### 4.6.9 DFX 分析

**可靠性分析 (FMEA):**

| 故障模式 | 影响 | 严重度 | 检测方法 | 缓解措施 |
|---------|------|--------|---------|----------|
| 错误地将有依赖的层（如 encoder-decoder cross-attention）设为并行 | 产生不存在的通信 overlap，通信时间被系统性低估 | 高 | 模型 profile 审查 | `use_parallel_attention` 仅在 Falcon 和其他已验证并行架构中设为 true；加载时输出 warning |
| `use_parallel_attention=true` 但 TP=1（无 AllReduce） | 并行子层无实际效果但增加代码复杂度 | 低 | 静默接受 | 输出 info：`use_parallel_attention has no effect when tp=1` |

**安全性检查**: 不涉及安全配置变更。

**可测试性**: 对比验证 -- 生成 Falcon 3 的串行模式 workload（设 `use_parallel_attention=false`）和并行模式 workload，验证并行模式总通信时间 <= 串行模式总通信时间，差值 = min(AR_attn_time, AR_mlp_time)。

##### 4.6.10 分配需求

| SR编号 | 分配需求描述 | 系统元素 |
|--------|-------------|----------|
| SR-F005-01 | 新增 `LayerConfig.use_parallel_attention` 字段 | `aicb/config/model_profile.py` |
| SR-F005-02 | 实现并行子层构建逻辑（mocked_model 中取消 attn_out -> mlp_in 依赖） | `aicb/workload_generator/training/mocked_model.py` |
| -- | 添加 Falcon 3 模型 profile + 验证测试 | `aicb/config/models/falcon3.yaml`, `aicb/tests/test_parallel_attention.py` |

---

## Sources

1. [aliyun/aicb GitHub](https://github.com/aliyun/aicb) -- AICB 2.1 release (November 2025)
2. [MLCommons Chakra Working Group](https://mlcommons.org/working-groups/research/chakra/)
3. [MLSys 2026: MLCommons Chakra Paper (arXiv:2605.11333)](https://arxiv.org/abs/2605.11333)
4. [MLCommons Chakra: Open Trace Library (June 2026)](https://mlcommons.org/2026/06/chakra-comes-of-age/)
5. [Chakra Schema Wiki (GitHub)](https://github-wiki-see.page/m/mlcommons/chakra/wiki/Chakra-Schema)
6. [ASTRA-sim System Architecture (DeepWiki)](https://deepwiki.com/astra-sim/astra-sim/2-system-architecture)
7. [ASTRA-sim + Chakra Tutorial (MICRO 2024)](https://astra-sim.github.io/assets/tutorials/micro-2024/2_chakra_astrasim_overview.pdf)
8. [MLSynth: Towards Synthetic ML Traces (NAIC '25)](https://dl.acm.org/doi/10.1145/3748273.3749211)
9. [NetMLSim/MLSynth GitHub](https://github.com/NetMLSim/MLSynth)
10. [Echo: Simulating Distributed Training At Scale (arXiv:2412.12487)](https://arxiv.org/abs/2412.12487)
11. [NetX-lab/Echo GitHub](https://github.com/NetX-lab/Echo)
12. [facebookresearch/param GitHub](https://github.com/facebookresearch/param)
13. [mlcommons/chakra GitHub](https://github.com/mlcommons/chakra)
14. [IEEE Micro: Standardized Collective Algorithms (April 2025)](https://doi.org/10.1109/MM.2025.3547363)
15. [Keysight Blog: Chakra from Traces to Test (May 2026)](https://www.keysight.com/blogs/en/inds/ai/mlcommons-chakra-from-traces-to-test)
16. [DeepSeek Open Infra Index: Communication & Parallelism](https://deepwiki.com/deepseek-ai/open-infra-index/2.2-communication-and-parallelism)
17. [deepseek-ai/profile-data GitHub](https://github.com/deepseek-ai/profile-data)
18. [Jamba: Hybrid Transformer-Mamba Language Models (ICLR 2025)](https://arxiv.org/abs/2403.19887)
19. [NVIDIA Megatron-LM MoE Roadmap (#1729)](https://github.com/NVIDIA/Megatron-LM/issues/1729)
20. [SimAI NSDI '25 Spring Paper](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)
21. [NVIDIA NeMo Megatron-Bridge: Gemma 3 Architecture](https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/models/llm/gemma3.html) -- GQA ratios, GeGLU, 5:1 local-to-global attention pattern
22. [Qwen3-235B-A22B HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3-235B-A22B) -- 235B total / 22B active, 128 experts, top-8 gating
23. [ASTRA-sim MLSys 2022 Tutorial](https://mlsys.org/virtual/2022/tutorial/2196) -- Georgia Tech / Intel / Meta collaboration origin
24. [DeepSeek-V4 Release (April 2026)](https://huggingface.co/docs/transformers/v5.8.0/model_doc/deepseek_v4) -- 1.6T MoE, hash-routed MoE layers, CSA+HCA hybrid attention, FP4 QAT, mHC
25. [Llama 4 Confirmed Architecture (NVIDIA NeMo docs)](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/vlm/meta/llama4.html) -- Scout 109B/17B active, Maverick ~400B/17B active, alternating dense/MoE, top-1 routing, iRoPE
26. [FlashMemory-DeepSeek-V4 (arXiv:2606.09079, June 2026)](https://arxiv.org/abs/2606.09079) -- Lightning Index ultra-long context via lookahead sparse attention for DeepSeek V4
27. [NVIDIA Megatron-LM Issue #4468: DeepSeek-V4 training support](https://github.com/NVIDIA/Megatron-LM/issues/4468) -- Megatron-LM adding DSV4 training, confirming industry adoption of V4 architecture
28. [DeepSeek V4 Ascend 910C Post-Training (June 2026)](https://pandaily.com/ascend-910c-1-6-trillion-parameter-training-jun2026) -- Full-parameter post-training on Huawei Ascend 910C, confirming 1.6T MoE architecture is training-ready on non-NVIDIA hardware
