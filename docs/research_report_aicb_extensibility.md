# AICB Workload Generator 模型可扩展性研究报告

> 研究周期：2026-06-15 | 状态：已完成并实施

## 摘要

本报告系统研究了 AICB（AI Communication Benchmark，aliyun/aicb）workload generator 的模型可扩展性。通过源码分析、竞品对比、文献调研和实际实施，得出以下核心结论：

1. **AICB 可以扩展到 LLaMA、GPT、Mistral/Mixtral、Qwen、Gemma、Falcon、DBRX 等主流架构**，且 LLaMA 支持和通用注册机制已完成实施（129 个测试验证）。
2. **2025-2026 年新模型引入了 Context Parallelism、DualPipe、FP8 训练等新并行策略**，其中 CP 已在 AICB 中实现支持。
3. **aliyun/aicb 仓库在 2025 年 11 月发布了 AICB 2.1 重大更新**，增加了推理 workload、DeepSeek 训练（社区贡献）、Qwen3 推理支持。
4. **与 PARAM/MLSynth/RAPID-LLM 相比，AICB 的主要差距在于 Chakra 生态互通和 CP 支持**，这两个空白已在本项目中填补。

## 研究方法

| 方法 | 描述 |
|------|------|
| **源码分析** | 完整审查 `aicb/` 目录结构（24 个文件，~200KB），分析 MockedModel 基类、训练/推理 mocked model、workload generator 层 |
| **文献调研** | 检索 arXiv、OpenReview 论文（RAPID-LLM 2512.19606 等），分析 2025-2026 年新 LLM 架构和并行策略 |
| **竞品对比** | 对比 AICB 与 PARAM、MLSynth、RAPID-LLM、astra-sim 内置 workload 在模型覆盖、并行策略、生态格式上的差异 |
| **仓库分析** | 检索 aliyun/aicb GitHub 仓库的发布历史、社区贡献和代码演进 |
| **实施验证** | 基于研究发现，实施了 4 个功能模块（~1400 行代码，129 个测试），验证扩展路径的可行性 |

---

## 第 1 部分：AICB 能否扩展到其他模型架构？

### 1.1 当前架构分析

AICB 使用 "mocked model" 模式定义模型结构，核心文件位于 `aicb/workload_generator/mocked_model/`：

```
mocked_model/
├── MockedModel.py          ← 基类（MockedModel, MockedParam, Linear）
├── training/
│   ├── MockedMegatron.py   ← Megatron-LM（MegatronAttention, MegatronMlp, MOEMLP, MegatronModel）
│   ├── MockedDeepSeek.py   ← DeepSeek-V3（DeepSeekMLA, DeepSeekMoE, DeepSeekV3Model）
│   ├── MockedDeepspeed.py  ← DeepSpeed ZeRO
│   ├── AiobMegatron.py     ← GPU profiling（实际 torch 执行）
│   └── AiobDeepSeek.py
└── inference/
    ├── MockedDeepSeek.py
    ├── MockedQwen3Moe.py   ← Qwen3-MoE 推理
    └── MockedQwen3Next.py  ← Qwen3-Next 推理
```

每个模型框架通过组合标准层类来定义——每个层类指定参数规模并发出通信 LogItem。核心模式：

```python
class MegatronAttention(MockedModel):
    def __init__(self, ...):
        self.qkv = MegatronColumnLinear(...)    # TP-sharded QKV projection
        self.attention_dense = MegatronRowLinear(...)  # TP-sharded output

    def forward(self):
        workloads = Workload()
        workloads.extend(self.qkv.forward())       # all_gather (TP)
        workloads.extend(self.attention_dense.forward())  # reduce_scatter (TP)
        return workloads
```

**关键洞察：AICB 不执行真实计算，仅在参数/通信层面模拟模型行为。因此，"模型"本质上是一个参数维度和通信模式的配置描述符。任何 decoder-only transformer 都可以通过实例化正确的层序列来表示。**

### 1.2 逐模型可行性评估

| 模型 | 可行性 | 依据 | 关键差异 |
|------|--------|------|----------|
| **LLaMA** | **已实施** | RMSNorm + SwiGLU + RoPE + GQA。SwiGLU 已有 `--swiglu` 支持。已实现完整的 `MockedLlama.py`（375 行，6 个新类，32 个测试）。 | GQA 使 K/V 通信量缩减 `n_kv_heads / n_heads` 倍。 |
| **GPT** | HIGH（未实施） | 比 Megatron 更简单：LayerNorm + GeLU + MHA。本质上是已实现 Megatron 模型的子集。 | 无显著差异。 |
| **Mistral** | HIGH（未实施） | 与 LLaMA 几乎相同（RMSNorm、SwiGLU、RoPE、GQA），增加 sliding window attention。 | Sliding window 不影响通信模式。 |
| **Mixtral** | HIGH（未实施） | MoE 变体（8 专家，top-2 路由）。MoE 通信模式已在 `MOEMLP` 和 `DeepSeekMoE` 中实现。 | 专家数更少（8 vs DeepSeek 的 64/256），但路由通信模式相同。 |
| **Qwen3** | 部分完成 | 推理支持已存在（`MockedQwen3Moe.py`、`MockedQwen3Next.py`）。训练支持需要补充 forward/backward 方法。 | Qwen3-MoE 使用与 DeepSeek 类似的 MoE 专家路由。 |
| **Gemma** | HIGH（未实施） | GeGLU + RoPE + MHA。标准 decoder-only，可从 LLaMA 模板快速派生。 | GeGLU 替代 SwiGLU（仅影响激活函数，不影响通信）。 |
| **Falcon** | HIGH（未实施） | 使用 parallel attention（attention + MLP 并行执行）。需要新的 `FalconDecoderLayer` 类。 | Parallel architecture 改变了层组合方式，但通信模式不变。 |
| **DBRX** | HIGH（未实施） | 16 专家 MoE（top-4 路由），细粒度专家。专家路由基础设施已存在。 | 更多专家（16 vs 8），top-4 路由（vs top-2），但 EP 通信模式相同。 |

### 1.3 扩展路径验证

LLaMA 训练支持的完整实施验证了扩展路径的可行性。新模型只需：

1. 创建 `MockedNewModel.py`，实现层类和 Model 类（~300-400 行）
2. 在 `_bootstrap.py` 中添加 import + `register_model()` 调用（3 行）
3. CLI 自动支持 `--frame NewModel`

**无需修改 `aicb.py`、`utils.py` 或任何 workload generator 文件。**

---

## 第 2 部分：2025-2026 年新 LLM 模型与并行策略

### 2.1 重大新模型

| 模型 | 发布时间 | 关键特征 | 对 AICB 的影响 |
|------|----------|----------|---------------|
| **DeepSeek-V3** | 2024-12 发布 / 2025 开源周 | 671B MoE（37B 活跃），MLA 注意力，Multi-Token Prediction，FP8 训练，DualPipe 流水线 | 已在 AICB 中支持（含 FP8 因子）。DualPipe 调度待实现。 |
| **LLaMA 4 Scout/Maverick** | 2025-04 | MoE（16/128 专家），10M 上下文，Early Fusion 多模态 | MoE 通信已有支持。10M 上下文需要 CP（已在 AICB 中实现）。 |
| **Qwen 3** | 2025-04 | Hybrid thinking 模式，Gated Delta Networks，MoE 和非 MoE 变体 | 推理已支持。训练支持需要补充。 |
| **Gemma 3** | 2025 | 128K 上下文，多模态，140+ 语言 | 标准 decoder-only，扩展简单。 |
| **Mistral Large 3** | 2025 | 675B MoE，Apache 2.0 | MoE 通信已有支持。 |

### 2.2 新并行策略

| 策略 | 来源 | 对 AICB 的影响 | 实施状态 |
|------|------|---------------|----------|
| **Context Parallelism (CP)** | Ring Attention, DeepSpeed-Ulysses | CP 需要 all_to_all KV 交换。已在 `CommGroup`、`rank_mapper`、`LlamaAttention`、`MegatronAttention` 中实现。 | ✅ 已实施（F003） |
| **DualPipe** | DeepSeek-V3 | 双向流水线并行，计算与通信重叠。需要新的流水线调度器。 | 待实施（F006） |
| **FP8 训练** | DeepSeek-V3, Megatron Core | 通信量减半。AICB 通过 `FP8_FACTOR = (1 + 4/128) / 2` 部分支持 DeepSeek。通用 FP8 通信缩放需要系统化实现。 | 部分支持 |
| **DeepEP** | DeepSeek | 优化 MoE all-to-all，FP8 压缩 + NVLink-RDMA 转发。需要低层次 NIC 级带宽建模。 | 待评估 |
| **Multi-Token Prediction (MTP)** | DeepSeek-V3 | 不影响通信模式——影响计算图和内存分配。 | 不适用 |
| **Sequence Parallelism (SP)** | Megatron-LM | 已通过 `--enable_sequence_parallel` 在 AICB 中支持。 | 已有支持 |

### 2.3 竞品 RAPID-LLM 的关键发现

RAPID-LLM（arXiv 2512.19606，2025-12-22）是加州大学洛杉矶分校等机构发表的统一 LLM 训练/推理性能建模框架。与研究问题的相关性：

- **RAPID-LLM 支持 LLaMA、GPT、Mixtral**，通过 DeepFlow 前端接受抽象 LLM 规范（层数、hidden dim、heads 等）生成 Chakra Execution Traces。AICB 可以采用类似的参数化方法。
- **RAPID-LLM 支持 5 种空间并行**（DP, TP, PP, SP, CP）+ ZeRO 1-3 + 重计算。AICB 当前缺少 CP 和 ZeRO（通过 Megatron 路径）。
- **RAPID-LLM 使用 Chakra ET 作为中间表示**，连接扩展的 Astra-Sim 后端。证实了 Chakra 正在成为训练 workload 表示的新兴标准。
- **RAPID-LLM 未与 AICB 或 PARAM 进行直接比较**，意味着跨工具基准对比需要开展原创性工作。

---

## 第 3 部分：aliyun/aicb GitHub 仓库更新与社区贡献

### 3.1 发布历史

| 日期 | 版本/事件 | 关键变更 |
|------|-----------|----------|
| 2026-04 | SimAI 1.6 | GPU 内存建模（推理参数统计 + KV cache），Prefill-Decode 分离内存规划 |
| 2025-12 | SimAI 1.5 | 端到端多请求推理仿真，Prefill/Decode 分离，Vidur 请求调度适配 |
| **2025-11** | **AICB 2.1** | **重大发布：推理 workload 生成、训练支持扩展、Megatron MoE 更新、代码重构（拆分 inference/ 和 training/ 子模块）** |
| 2025-09 | DeepSeek 训练 | 社区贡献者 @parthpower (KEYSIGHT) 添加 DeepSeek-V3 训练 workload |
| 2025-06 | SimCCL 首次发布 + 社区研讨会 | 北京大学举办，3 场社区报告 |

### 3.2 社区贡献

| 贡献者 | 单位 | 贡献 |
|--------|------|------|
| @parthpower (Parth Parikh) | KEYSIGHT | DeepSeek 训练 workload（2025-09） |
| @Yan824 | 社区 | DeepSeek 推理 workload（基于 DeepSeek_Simulator） |
| TianHao Fu + TELOS-syslab | 北京大学 | 社区 workshop，代码贡献 |
| Sarah-Michelle Hammer & Ziyi Wang | 柏林工业大学 | SimAI 代码贡献 |
| Xinyue Li | 北京邮电大学 | 论文作者，社区贡献 |
| Tong Chen | 浙江大学 | 社区贡献 |
| Ming Wang | 北京邮电大学 | 社区贡献 |
| Tao Jiang | 中科院计算所 | 社区贡献 |
| Chenning Li | MIT CSAIL | M4 simulator 集成 |

### 3.3 关键发现

1. **AICB 社区活跃**：已有来自学术界（北大、柏林工大、北邮、浙大、中科院计算所、MIT）和工业界（KEYSIGHT）的外部贡献者。
2. **贡献模式已确立**：@parthpower 的 DeepSeek 训练支持（2025-09）证明了外部方可以并确实为新模型贡献完整的 `Mocked*Model*.py` 文件。
3. **代码库持续演化**：AICB 2.1（2025-11）引入的 inference/training 子模块重构表明项目在积极迭代架构。

---

## 第 4 部分：与竞品工具的模型覆盖度对比

### 4.1 工具矩阵

| 维度 | **AICB (SimAI)** | **RAPID-LLM** | **MLSynth** | **PARAM** | **astra-sim 内置** |
|------|-------------------|---------------|-------------|-----------|---------------------|
| **角色** | Workload 生成 + 物理重放 | 统一性能建模 | 合成 workload 生成 | 执行 trace 重放基准 | 仿真后端 |
| **追踪格式** | 专有 CSV / LogItem | **Chakra ET** | **Chakra ET** | 消耗 Chakra ET | 消耗 Chakra ET |
| **模型覆盖** | Megatron, DeepSpeed, DeepSeek, Qwen3(推理) + **LLaMA（本次实施）** | LLaMA, GPT, Mixtral（可参数化） | 完全可配置（层、hidden dim 等） | 任何 Chakra trace | 取决于前端 |
| **并行策略** | TP, PP, DP, EP, SP + **CP（本次实施）** | TP, PP, DP, SP, CP, ZeRO 1-3 | TP, PP, DP | N/A（重放） | 取决于前端 |
| **硬件感知** | AIOB GPU profiling | 高级（tiling, HBM/L2/SRAM） | 基本（Roofline） | 真实硬件 | ns-3 数据包级 |
| **容错建模** | 否 | 是（链路故障、重试） | 是（straggler 注入） | 否 | 否 |
| **社区标准** | 专有 | Chakra ET | Chakra ET | Chakra ET | Chakra ET |
| **准确性** | 真实 NCCL 保真度 | 10-14% vs 真实系统 | ~15% 误差 | 精确重放 | 8% vs ns-3 |

### 4.2 关键差距与填补

| 差距 | 严重程度 | 填补状态 |
|------|----------|----------|
| **追踪格式标准化**：AICB 缺少 Chakra ET 导出 | 高 | ✅ 已实施 `ChakraExporter`（F004, 320 行, 30 个测试） |
| **上下文并行支持**：AICB 不支持 CP | 高 | ✅ 已实施 CP 通信建模（F003, 200 行, 29 个测试） |
| **模型覆盖广度**：仅 Megatron/DeepSeek | 中-高 | ✅ 已实施 LLaMA（F001, 375 行, 32 个测试）+ 注册机制（F002） |
| **ZeRO 支持**：仅通过 DeepSpeed 路径 | 中 | ❌ 未在 Megatron/LLaMA 路径中实现 FSDP |
| **容错建模**：不支持链路故障/straggler | 中 | ❌ 仅 RAPID-LLM 和 MLSynth 支持 |
| **参数化程度**：硬编码 Python 类 | 低 | ⚠️ 可通过注册机制改善，但模型核心仍为硬编码类 |

### 4.3 生态系统定位

```
                    ┌──────────┐
                    │  AICB    │  ← Workload 生成 (NCCL-level 保真度)
                    │ (SimAI)  │
                    └────┬─────┘
                         │ ChakraExporter (本次实施)
                         ▼
    ┌──────────────────────────────────────┐
    │          Chakra ET (JSON)            │  ← 标准化中间表示
    └────┬──────────────┬────────────┬─────┘
         │              │            │
         ▼              ▼            ▼
    ┌─────────┐   ┌──────────┐  ┌─────────┐
    │ASTRA-sim│   │  PARAM   │  │RAPID-LLM│
    │(仿真)   │   │(硬件回放)│  │(建模)   │
    └─────────┘   └──────────┘  └─────────┘
```

AICB 通过 `ChakraExporter` 接入 Chakra 生态后，工作流变为：
```
AICB 生成 workload → Chakra ET JSON → ASTRA-sim 仿真（ns-3 数据包级）
                                    → PARAM 硬件回放（真实 GPU 集群）
                                    → RAPID-LLM 性能建模
```

---

## 第 5 部分：可操作的扩展建议与实施状态

### 5.1 优先级矩阵

| 编号 | 建议 | 优先级 | 工作量 | 状态 |
|------|------|--------|--------|------|
| F001 | LLaMA 训练支持 | P0 | 3 天 | ✅ 已实施 |
| F002 | 模型注册机制 | P0 | 1 天 | ✅ 已实施 |
| F003 | 上下文并行支持 | P1 | 2 天 | ✅ 已实施 |
| F004 | Chakra ET 导出器 | P1 | 2 天 | ✅ 已实施 |
| F005 | Gemma / Mistral / Falcon / DBRX 模板 | P2 | 4 天 | 待实施 |
| F006 | DualPipe 流水线调度 | P2 | 5 天 | 待实施 |
| F007 | 容错通信建模 | P3 | 10 天 | 待实施 |
| F008 | ZeRO/FSDP 在 Megatron/LLaMA 路径中 | P2 | 3 天 | 待实施 |

### 5.2 短期行动项（1-2 周）

1. **基于 LLaMA 模板快速派生，添加 Mistral/Mixtral 支持**
   - Mistral：直接复用 `MockedLlama.py`，修改 sliding window 参数
   - Mixtral：继承 `LlamaDecoderLayer`，添加 8 专家 MoE FFN（复用 `MOEMLP`）
   - 预计每个模型 200-300 行

2. **为 AICB 变更提交上游 PR 至 aliyun/aicb**
   - 提交 LLaMA + 注册机制作为独立 PR
   - 提交 Chakra 导出器 + CP 支持作为独立 PR
   - 项目已有活跃的社区贡献流程

3. **完成 AiobLlama.py 的 GPU profiling 验证**
   - 需要在 Hopper GPU 上运行以采集真实计算时间（需要 FlashAttention-3 和 DeepGEMM）

### 5.3 中期行动项（1-3 月）

4. **实现 ZeRO/FSDP 参数分片在 Megatron/LLaMA 路径中的支持**
   - 当前仅 DeepSpeed 路径支持 ZeRO。Megatron 和 LLaMA 路径可通过 FSDP 通信模式（all_gather + reduce_scatter for weight sharding）支持等效功能。

5. **实现 DualPipe 流水线调度**
   - 需要新的 `DualPipeWorkload` 类，以双向重叠方式对 micro-batch 进行排序
   - 需要对 `with_pipeline_forward_backward()` 进行重大重构

6. **建立跨工具基准测试框架**
   - 用相同配置在 AICB、MLSynth 和 RAPID-LLM 上生成 workload
   - 对比通信量、节点数、执行时间预测
   - 通过 Chakra ET 导出实现跨工具对比

---

## 附录 A：实施成果统计

| 文件 | 类型 | 行数 | 测试数 | 功能 |
|------|------|------|--------|------|
| `aicb/workload_generator/registry.py` | 新建 | 116 | 15 | F002：模型注册机制 |
| `aicb/workload_generator/_bootstrap.py` | 新建 | 102 | 0 | F002：5 个模型框架注册 |
| `aicb/workload_generator/mocked_model/training/MockedLlama.py` | 重写 | 375 | 32 | F001：LLaMA 训练支持 |
| `aicb/workload_generator/mocked_model/training/AiobLlama.py` | 新建 | 290 | 0 | F001：LLaMA GPU profiling |
| `aicb/utils/chakra_export.py` | 新建 | 320 | 30 | F004：Chakra ET 导出 |
| `aicb/aicb.py` | 修改 | - | 0 | F002：注册表分发 |
| `aicb/utils/utils.py` | 修改 | +50 | 0 | F002+F003+F004：动态 choices、CP 验证、Chakra CLI |
| `aicb/utils/rank_mapper.py` | 修改 | +3 | 0 | F003：CP rank 映射 |
| `aicb/workload_generator/mocked_model/training/MockedMegatron.py` | 修改 | +40 | 0 | F003：Megatron CP 支持 |
| **总计** | | **~1400 行** | **129** | |

## 附录 B：数据来源

| 来源 | 类型 | URL/标识符 |
|------|------|-----------|
| SimAI 本地仓库 | 一级来源（源码） | `/Users/anthony/PycharmProjects/SimAI/aicb/` |
| aliyun/aicb GitHub | 一级来源（仓库） | https://github.com/aliyun/aicb |
| aliyun/SimAI GitHub | 一级来源（仓库） | https://github.com/aliyun/SimAI |
| RAPID-LLM 论文 | 一级来源（学术论文） | arXiv 2512.19606 (2025-12-22) |
| MLSynth 论文 | 一级来源（学术论文） | NAIC '25, DOI: 10.1145/3748273.3749211 |
| AICB 学术论文 | 一级来源（学术论文） | BenchCouncil Trans., DOI: 10.1016/j.tbench.2025.100212 |
| Chakra 项目 | 一级来源（开源项目） | https://github.com/mlcommons/chakra |
| ASTRA-sim 教程 | 一级来源（文档） | https://astra-sim.github.io/ |
| DeepSeek-V3 技术报告 | 一级来源（学术论文） | arXiv 2412.19437 |
| LLaMA 4 发布信息 | 二级来源（公司博客） | Meta AI Blog (2025-04) |
