# SimAI 能力扩展：从双模型到全架构覆盖的 LLM 训练通信仿真

> 本文将介绍 SimAI/AICB 项目的最新扩展能力：如何通过插件化的模型注册机制、上下文并行通信建模、Chakra 生态互通三大核心能力，将训练通信 benchmark 的覆盖范围从 Megatron/DeepSeek 扩展到 LLaMA、GPT、Mistral 等主流模型架构，并提供端到端的自动化验证体系。

---

## 一、背景：LLM 训练通信仿真的意义与瓶颈

随着大语言模型规模的持续增长，分布式训练中的通信开销已成为制约训练效率的核心瓶颈。以 LLaMA 4 Maverick（400B 总参数，128 专家 MoE）和 DeepSeek-V3（671B 总参数，2048 卡 H800 集群）为代表的大规模训练任务，通信时间占比可达 30%-50%。

**SimAI**（已被 NSDI'25 Spring 接收）是阿里巴巴开源的高精度 LLM 训练模拟器，其核心组件 **AICB**（AI Communication Benchmark）通过 mocked model 模式模拟模型各层的通信行为，无需真实 GPU 即可生成高保真的通信 workload，用于评估网络拓扑、并行策略和通信算法的性能。

然而，在本次扩展之前，AICB 仅原生支持三种训练框架：

| 框架 | 模型架构 | 并行策略 |
|------|----------|----------|
| Megatron-LM | 标准 Transformer (GeLU) | TP, PP, DP, EP, SP |
| DeepSpeed | ZeRO Stage 1/2/3 | DP, PP |
| DeepSeek | DeepSeek-V3 (MLA + MoE) | TP, EP |

这意味着如果你想评估 LLaMA 架构在自定义网络拓扑上的训练性能，或者想对比 Mistral/Mixtral 与 DeepSeek 的通信模式差异，你需要自己读懂源码、编写 MockedModel、修改入口逻辑——门槛相当高。

**本次扩展的目标很明确：让 AICB 从"双模型专用工具"进化为"全架构可扩展平台"。**

---

## 二、四大扩展能力一览

我们为 AICB 实现了四个层级的能力扩展，从基础设施到应用层逐级构建：

```
┌─────────────────────────────────────────────────┐
│  F004: Chakra ET 导出器  ← 生态互通层           │
├─────────────────────────────────────────────────┤
│  F003: 上下文并行 (CP)  ← 并行策略补全层        │
├─────────────────────────────────────────────────┤
│  F001: LLaMA 训练支持   ← 模型覆盖扩展层        │
├─────────────────────────────────────────────────┤
│  F002: 模型注册机制     ← 基础设施层            │
└─────────────────────────────────────────────────┘
```

| 功能编号 | 能力 | 类型 | 代码量 | 测试数 |
|----------|------|------|--------|--------|
| F002 | 插件化模型注册机制 | 基础设施 | 120 行 | 15 |
| F001 | LLaMA 训练 workload 支持 | 模型扩展 | 375 行 | 32 |
| F003 | 上下文并行 (CP) 通信建模 | 策略扩展 | 200 行 | 29 |
| F004 | Chakra ET 导出器 | 生态互通 | 320 行 | 30 |
| E2E | 端到端集成测试 | 质量保障 | 400 行 | 23 |
| **总计** | | | **~1400 行** | **129** |

下面逐一展开每个能力的设计思路与关键实现。

---

## 三、能力一：插件化模型注册机制

### 问题

AICB 原有的模型分发逻辑是硬编码的 if/elif 链：

```python
# 旧代码 (aicb.py)
if args.frame == "Megatron":
    model = MegatronModel(args)
    workload_generator = MegatronWorkload(args, model)
elif args.frame == "DeepSpeed":
    model = DeepspeedForCausalLM(args)
    if args.stage == 1:
        workload_generator = DeepSpeedStage1(args, model)
    elif ...
elif args.frame == "DeepSeek":
    model = DeepSeekV3Model(args)
    workload_generator = MegatronWorkload(args, model)
```

每新增一个模型，需要修改 3 个文件（aicb.py、对应的 Mocked 文件、utils.py 的 choices 列表），且代码中混杂了框架无关的核心逻辑和框架相关的分发逻辑。

### 方案

引入 `registry.py` 模块，实现基于注册表的插件模式：

```python
# 新代码：一行注册
register_model("LLaMA", LlamaModel, MegatronWorkload,
               "LLaMA 2/3/4 training workload (GQA + SwiGLU + RMSNorm)")

# 入口自动分发
from workload_generator.registry import lookup
entry = lookup(args.frame)
model = entry.model_cls(args) if entry.model_cls else None
workload_generator = entry.wl_cls(args, model)
```

新增模型的步骤从"改 3 个文件，写 30 行"简化为"在 `_bootstrap.py` 中添加 3 行 import + register 调用"。

### 关键设计

- **避免循环导入**：MockedModel 文件与 WorkloadGenerator 文件存在双向依赖（MockedMegatron.py ⟷ generate_megatron_workload.py）。通过引入独立的 `_bootstrap.py` 模块（不被任何模块反向导入）来解决循环依赖。
- **DeepSpeed 工厂模式**：DeepSpeed 的 workload generator 取决于 `--stage` 参数，通过工厂函数 `_deepspeed_wl_factory(args, model)` 封装分发逻辑，保持注册表接口统一。
- **CLI 参数动态派生**：`--frame` 的 choices 列表从注册表自动生成，不再手动维护。

---

## 四、能力二：LLaMA 训练 workload 支持

### 架构差异建模

LLaMA 与 Megatron 的核心架构差异在 AICB 的 mocked model 层面表现为通信量差异：

| 组件 | Megatron | LLaMA | 对通信的影响 |
|------|----------|-------|-------------|
| 归一化 | LayerNorm (weight + bias) | RMSNorm (weight only) | 参数减半，无通信影响 |
| 激活函数 | GeLU / SwiGLU(可选) | SwiGLU | 无通信影响（已有 --swiglu 支持） |
| 位置编码 | Learned Position Embedding | RoPE (cos/sin 缓存) | 零可训练参数 |
| 注意力 | MHA (Q=K=V heads) | GQA (K/V heads < Q heads) | **K/V 投影通信量缩减 n_kv/n_heads 倍** |
| 前馈网络 | 2 投影 (w1, w2) | 3 投影 (gate, up, down) | gate+up 的 TP 通信可合并 |

### GQA 通信量缩放

GQA（Group Query Attention）是 LLaMA 相比 Megatron 在通信建模上最关键的区别。以 LLaMA-3-70B 为例：

```python
# LLaMA-3-70B: 64 Q heads, 8 KV heads
# K/V 投影通信量 = (8/64) * MHA 通信量 = 1/8
self.k_proj = MegatronColumnLinear(
    hidden_size,
    num_kv_heads * head_dim,  # 8 * 128 = 1024, not 64 * 128 = 8192
    kv_tp,                     # min(num_kv_heads, tp) = min(8, tp)
    ...
)
```

当 `num_kv_heads=8, tp=4` 时，`kv_tp = min(8, 4) = 4`，K/V 在每个 TP rank 上被分成 2 份。当 `num_kv_heads=2, tp=4` 时，`kv_tp` 被上限为 2，额外的 2 个 GPU 复制 K/V（无通信开销）。

### 新增类一览

| 类 | 职责 |
|----|------|
| `LlamaRMSNorm` | RMS 归一化，仅 weight 参数（无 bias），参数减半 |
| `LlamaRotaryEmbedding` | RoPE 位置编码，cos/sin 缓存，零可训练参数 |
| `LlamaMLP` | SwiGLU 三投影（gate/up/down），gate+up TP 通信可合并 |
| `LlamaAttention` | GQA 注意力，kv_tp 上限保护，TP 下 K/V 通信自动缩放 |
| `LlamaDecoderLayer` | Pre-norm 结构：RMSNorm -> Attention -> RMSNorm -> MLP |
| `LlamaModel` | 完整架构：Embedding -> RoPE -> [Layers × N] -> Final RMSNorm -> LM Head |

### 使用示例

```bash
# LLaMA-3-8B 配置 (GQA: 32 Q heads, 8 KV heads, SwiGLU)
python aicb/aicb.py --frame LLaMA \
  --hidden_size 4096 --ffn_hidden_size 14336 --num_layers 32 \
  --num_attention_heads 32 --num_kv_heads 8 \
  --seq_length 8192 --vocab_size 128256 --swiglu \
  --tensor_model_parallel_size 4 --global_batch 1024 --micro_batch 1
```

---

## 五、能力三：上下文并行通信建模

上下文并行（Context Parallelism, CP）是 2025-2026 年 LLM 训练中最重要但最容易被忽视的并行策略。随着 LLaMA 4 Scout 将上下文窗口扩展到 10M tokens，序列长度远远超出单 GPU 显存容量，CP 已成为必需。

### 填补的空白

在本次扩展之前，AICB 不支持上下文并行。`RankGenerator` 的默认排序 `tp-cp-ep-dp-pp` 暗示了 CP 的计划，但 `--context-parallel-size` 参数未被任何 workload generator 使用。RAPID-LLM（2025 年 12 月，arXiv）等竞品已经支持 CP，这是 AICB 最大的功能空白。

### 实现设计

CP 的通信发生在注意力层：每个 GPU 只持有 `seq_len / cp` 个 token 的 Q，但需要完整的 K 和 V 才能计算注意力。

```
Forward:  Q_proj -> K_proj -> V_proj -> [CP all_to_all K+V] -> O_proj
Backward: O_grad -> [CP all_to_all K/V_grad] -> V_grad -> K_grad -> Q_grad
```

**通信量公式**（BF16）：
```
cp_kv_size = 2 × num_kv_heads × head_dim × seq_len × batch_size (bytes)
```

对于 GQA 模型，`num_kv_heads < num_heads`，CP 通信量自动缩减。例如 LLaMA-3-8B 的 `num_kv_heads=8`，CP 通信量仅为同等规模 MHA 模型的 1/4。

### 基础设施变更

| 组件 | 变更 |
|------|------|
| `CommGroup` 枚举 | 新增 `cp_group`, `cp_dp_group`, `cp_tp_group` |
| `rank_mapper.py` | 新增 3 条 CP 组映射（cp, cp-dp, cp-tp） |
| `get_params()` | 新增 CP 验证：`seq_len % cp == 0`, `world_size % (tp*pp*cp) == 0`，dp_num 重计算 |
| `LlamaAttention` | 新增 `cp` 参数，forward/backward 各插入 1 个 all_to_all LogItem |
| `MegatronAttention` | 同上（覆盖 Megatron 路径的 CP 支持） |

### 使用示例

```bash
# LLaMA 长上下文训练: TP=4, CP=2 (seq_len=4096 -> 每个 CP rank 处理 2048 tokens)
python aicb/aicb.py --frame LLaMA --context-parallel-size 2 \
  --tensor_model_parallel_size 4 --seq_length 4096 ...
```

---

## 六、能力四：Chakra 生态互通

### 为什么要支持 Chakra？

Chakra Execution Trace（由 MLCommons 维护）正在成为 AI 训练 workload 的标准化表示格式。RAPID-LLM、MLSynth、PARAM 和 astra-sim 都使用 Chakra ET 作为输入/输出格式。AICB 原有的专有 CSV 格式无法与这些工具互操作。

### 实现

ChakraExporter 将 AICB 的 LogItem 列表映射为 Chakra ET JSON 格式：

| AICB CommType | Chakra NodeType | Chakra Collective |
|---------------|----------------|-------------------|
| `computation` | `4` (COMP_NODE) | N/A |
| `all_reduce` | `7` (COMM_COLL) | ALL_REDUCE |
| `all_gather` | `7` (COMM_COLL) | ALL_GATHER |
| `reduce_scatter` | `7` (COMM_COLL) | REDUCE_SCATTER |
| `all_to_all` | `7` (COMM_COLL) | ALL_TO_ALL |
| `isend/irecv` | `5/6` (SEND/RECV) | N/A (P2P) |

关键设计决策：

- **JSON 而非 Protobuf**：无外部依赖（不引入 protobuf 编译链），兼容 astra-sim 的 Chakra JSON loader
- **元数据首节点**：始终包含 world_size、tp、pp、dp、cp、num_layers、hidden_size 等全局配置
- **线性依赖链**：默认每个节点依赖前一个节点，保证 forward -> backward 的执行顺序

### 使用示例

```bash
python aicb/aicb.py --frame LLaMA --export-chakra llama_workload.et.json ...
```

输出的 JSON 文件可直接被 ASTRA-sim 消费：
```
simulator -> Chakra ET (JSON) -> ASTRA-sim -> ns-3 packet-level simulation
```

---

## 七、端到端验证体系

本次扩展交付了完整的测试金字塔：

```
           ┌─────────┐
           │ 23 E2E  │  ← 完整流水线: Config→Model→Workload→Chakra
           │  Tests  │
      ┌────┴─────────┴────┐
      │   30 + 29 + 32    │  ← 功能测试: Chakra导出 + CP通信 + LLaMA层
      │   Unit Tests      │
 ┌────┴───────────────────┴────┐
 │      15 Registry Tests      │  ← 基础设施测试
 └─────────────────────────────┘
        129 Tests Total
```

E2E 测试覆盖的关键场景：

- **多模型一致性**：相同 TP 配置下，Megatron 和 LLaMA 均产生 TP 通信
- **并行策略组合**：TP=4 + CP=2 的注意力层同时产生 TP 和 CP 通信
- **边界条件**：单层模型、100 层模型、128K 词表、空 workload 导出
- **Chakra 回环**：导出 JSON -> 重新解析 -> 验证结构完整性（强制字段、唯一 ID）

运行命令：
```bash
cd aicb && python3 -m pytest tests/ -v   # 129 tests in ~2s
```

---

## 八、如何贡献新模型：三步走

基于新的注册机制，为 AICB 添加一个新模型框架只需三步：

**Step 1**: 创建 `MockedNewModel.py`

```python
from workload_generator.mocked_model.MockedModel import MockedModel
from workload_generator.mocked_model.training.MockedMegatron import (
    MegatronColumnLinear, MegatronRowLinear, MegatronEmbedding
)

class NewAttention(MockedModel):
    def __init__(self, ...):
        # 定义 Q, K, V, O 投影的通信模式
        ...

class NewMLP(MockedModel):
    def __init__(self, ...):
        # 定义前馈网络的通信模式
        ...

class NewModel(MockedModel):
    def __init__(self, config):
        self.embedding = MegatronEmbedding(...)
        self.layers = [NewDecoderLayer(...) for _ in range(config.num_layers)]
        ...

    def forward(self):
        workloads = Workload()
        workloads.extend(self.embedding.forward())
        for layer in self.layers:
            workloads.extend(layer.forward())
        return workloads
```

**Step 2**: 在 `_bootstrap.py` 中注册

```python
from workload_generator.mocked_model.training.MockedNewModel import NewModel
register_model("NewFrame", NewModel, MegatronWorkload,
               "New model framework description")
```

**Step 3**: 运行测试

```bash
python3 -m pytest tests/ -v
```

无需修改 `aicb.py`、`utils.py` 或任何 workload generator 文件。CLI 自动支持 `--frame NewFrame`。

---

## 九、总结与展望

本次扩展将 AICB 从"双模型专用工具"进化为"全架构可扩展平台"，核心交付包括：

| 维度 | Before | After |
|------|--------|-------|
| 模型覆盖 | Megatron, DeepSeek | + LLaMA 2/3/4 (GQA), Qwen3 (已有推理), 可扩展到 Mistral/Gemma/Falcon |
| 并行策略 | TP, PP, DP, EP, SP | + 上下文并行 (CP), CP+TP 组合 |
| 生态互通 | 专有 CSV | + Chakra ET JSON (ASTRA-sim/PARAM/MLSynth 兼容) |
| 可扩展性 | 硬编码 if/elif 链 | 注册表插件模式 (3 行添加新模型) |
| 测试覆盖 | 2 个测试文件 | 129 个测试, 5 个测试文件, E2E 流水线验证 |

**后续规划**：

- **F005**: Gemma / Mistral / Falcon / DBRX 模型模板（基于 LLaMA 模板快速派生）
- **F006**: DualPipe 流水线调度建模（DeepSeek-V3 的双向流水线并行算法）
- **F007**: 容错通信建模（链路故障、straggler 注入）

我们欢迎社区贡献新模型架构。如果你有兴趣为 AICB 添加你正在使用的模型框架，请参考 `aicb/docs/feature_design_model_extensibility.md` 中的设计文档，或直接提交 PR 到 [aliyun/SimAI](https://github.com/aliyun/SimAI)。

---

*作者注：本文涉及的代码已经过完整的单元测试和端到端验证。关于 SimAI 项目的更多信息，请参考 [SimAI GitHub](https://github.com/aliyun/SimAI) 和 NSDI'25 Spring 论文。*
