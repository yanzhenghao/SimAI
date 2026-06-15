# AICB 模型框架可扩展性功能设计说明书

## 1、功能域概述

本功能域描述 AICB（AI Communication Benchmark）workload generator 的模型框架可扩展性设计。当前 AICB 通过 `args.frame` 参数支持 Megatron-LM、DeepSpeed（Stage 1/2/3）和 DeepSeek 三种训练框架，以及 DeepSeek、Qwen3-MoE、Qwen3-Next 三种推理框架。每个框架通过在 `workload_generator/mocked_model/` 目录下定义 MockedModel 子类来描述模型结构和通信模式。

本设计的核心目标是：建立一套标准化、低摩擦的机制，使开发者能够快速为新的模型架构（Llama、GPT、Mistral、Gemma、Falcon 等）添加 AICB 训练/推理 workload 生成支持，同时补齐当前缺失的并行策略（上下文并行）和生态系统互操作能力（Chakra ET 导出）。

功能域涉及以下子功能：

| 编号 | 功能名称 | 类型 | 优先级 |
|------|----------|------|--------|
| F001 | Llama 训练模型框架实现 | feature | P0 |
| F002 | 模型注册机制重构 | refactor | P0 |
| F003 | 上下文并行通信支持 | feature | P1 |
| F004 | Chakra ET 导出器 | feature | P1 |
| F005 | Gemma / Mistral / Falcon / DBRX 模型模板 | feature | P2 |

## 2、功能域总体方案

### 2.1 总体设计原则

1. **不可变性原则（CRITICAL）：** 所有模型配置对象和 Workload 对象均采用不可变模式，创建新对象而非修改已有对象。
2. **开闭原则：** 对模型扩展开放（通过注册机制添加新模型），对 workload generator 核心逻辑关闭（无需修改 `SimAI_training_workload_generator.py` 和 `workload_generator.py` 的核心遍历逻辑）。
3. **高内聚低耦合：** 每个模型框架的文件自包含（单文件 <= 800 行），仅依赖 `MockedModel` 基类和 `utils` 工具模块。
4. **向后兼容：** 现有 Megatron、DeepSpeed、DeepSeek 的命令行接口和输出格式保持不变。

### 2.2 整体实现思路

AICB 的 mocked model 本质上是对模型参数规模和通信模式的抽象描述，不执行真实计算。因此，添加新模型框架的核心工作是：

1. **定义模型结构：** 使用 MockedModel 层类描述 embedding、attention、MLP、MoE、norm 等组件的参数规模和通信行为。
2. **组装模型图：** 在主 Model 类中按架构顺序组合层（embedding -> [transformer layers] -> final projection）。
3. **注册到框架：** 通过模型注册表将新模型映射到对应的 Model 类和 WorkloadGenerator 类。
4. **添加 GPU 计算时间 profile：** 可选，通过 AIOB 机制采集真实 GPU 计算时间以校准 workload。

### 2.3 领域数据模型

```
MockedModel (抽象基类)
├── parameters() → List[MockedParam]     # 参数规模
├── child_modules() → List[MockedModel]  # 子模块遍历
├── forward() → Workload                 # 前向通信模式
└── backward() → Workload                # 反向通信模式

MockedParam
├── shape: Tuple          # 参数张量形状
├── elem_size: int        # 元素字节数 (2=BF16, 1=FP8)
├── numel() → int         # 元素总数
└── msg_size() → int      # 通信消息字节数

Workload / LogItem
├── comm_type: CommType   # 通信类型 (all_reduce, all_gather, ...)
├── comm_group: CommGroup # 通信组 (tp_group, dp_group, ...)
├── msg_size: int         # 通信量
└── stage: str            # 阶段标识
```

### 2.4 系统元素及周边关系

```
aicb.py (入口)
  └─► MODEL_REGISTRY (新增：模型注册表)
        ├── "Megatron"   → (MegatronModel, MegatronWorkload)
        ├── "DeepSeek"   → (DeepSeekV3Model, MegatronWorkload)
        ├── "Llama"      → (LlamaModel, MegatronWorkload)    [新增]
        └── ...
            │
            ▼
workload_generator/mocked_model/
  ├── MockedModel.py          ← 基类 (不变)
  ├── training/
  │   ├── MockedMegatron.py   ← Megatron 层定义 (不变)
  │   ├── MockedDeepSeek.py   ← DeepSeek 层定义 (不变)
  │   ├── MockedLlama.py      ← Llama 层定义 [新增]
  │   └── MockedGeneric.py    ← 通用层工厂 [新增]
  └── inference/
      ├── MockedDeepSeek.py
      ├── MockedQwen3Moe.py
      └── ...

utils/utils.py
  ├── CommType (enum)         ← 通信类型 [扩展: 添加上下文并行相关]
  ├── CommGroup (enum)        ← 通信组 [扩展: 添加 cp_group]
  └── RankGenerator           ← Rank 映射 [扩展: 支持 cp 维度]
```

## 3、功能域规格变更与设计

### 3.1 现有规格分析

当前 `aicb/aicb.py` 第 36-51 行的模型分发逻辑：

```python
if args.frame == "Megatron":
    model = MegatronModel(args)
    workload_generator = MegatronWorkload(args, model)
elif args.frame == "DeepSpeed":
    model = DeepspeedForCausalLM(args)
    ...
elif args.frame == "DeepSeek":
    model = DeepSeekV3Model(args)
    workload_generator = MegatronWorkload(args, model)
```

**问题：**
- 硬编码 `if/elif` 链，每新增一个模型需修改 `aicb.py`
- `MegatronWorkload` 和 `DeepSpeedStage*` 与模型类之间没有显式的契约/接口约束
- `utils.py` 第 603 行 `choices=["Megatron", "DeepSpeed", "collective_test", "DeepSeek"]` 硬编码了合法值

### 3.2 规格变更

**变更 S001：** `aicb.py` 改为从 `MODEL_REGISTRY` 注册表查找模型和 workload generator。

**变更 S002：** `utils.py` 中 `--frame` 参数的 `choices` 由注册表动态生成。

**变更 S003：** 新增 `CommGroup.cp_group = "cp_group"` 和 `CommGroup.cp_dp_group = "cp_dp_group"`。

**变更 S004：** `RankGenerator` 的默认排序 `tp-cp-ep-dp-pp` 中的 `cp` 维度正式启用。

**变更 S005：** 新增 Chakra ET 导出接口 `ChakraExporter`，不修改现有 `LogItem` / `Work_Item` 结构。

## 4、功能实现设计

---

### 4.1 F002：模型注册机制重构（前置依赖）

#### 4.1.1 功能概述

将 `aicb.py` 中硬编码的 `if/elif args.frame` 模型分发逻辑重构为基于注册表的插件模式，使新增模型框架无需修改入口文件。此功能是 F001（Llama 支持）和其他新模型的前置依赖。

#### 4.1.2 SR设计

| SR编号 | SR描述 |
|--------|--------|
| SR-REG-001 | `aicb.py` 通过 `MODEL_REGISTRY[args.frame]` 查找模型类和 workload generator 类，而非硬编码 `if/elif` 分支 |
| SR-REG-002 | 每个模型框架通过一个 `register()` 调用完成注册，提供：模型类、workload generator 类、框架描述 |
| SR-REG-003 | `--frame` 参数的 `choices` 从注册表动态派生，不再手动维护 |
| SR-REG-004 | 注册失败（未找到框架名）时给出包含所有已注册框架名的错误信息 |

#### 4.1.3 实现思路

采用 Python 字典作为注册表，每个模型框架在对应 `Mocked*Model*.py` 文件底部调用 `register()` 函数完成自注册。

**备选方案：**
- 方案 A（采用）：Python 字典 + 显式注册函数。简单、无外部依赖、符合现有代码风格。
- 方案 B：基于装饰器的自动注册（`@register_model("Llama")`）。更优雅但引入了隐式执行（import 时自动注册），与项目当前显式控制流风格不一致。
- 方案 C：基于 `importlib` 的插件发现（扫描目录自动加载）。过度设计，256 行代码 vs 50 行，且 import 副作用风险高。

选择方案 A：最小改动量，最大可控性。

#### 4.1.4 实现设计

**前置条件：** 无。

**触发事件：** AICB 启动时，`aicb.py` 导入各 MockedModel 模块，各模块在底部调用 `register()`。

**主流程：**

1. 在 `aicb/workload_generator/` 下新建 `registry.py`，定义全局 `MODEL_REGISTRY: Dict[str, ModelEntry]` 和 `register_model()` 函数。
2. 修改 `aicb/aicb.py`：删除硬编码的 `if/elif` 链，改为 `registry.lookup(args.frame)` 动态查找。
3. 修改 `aicb/utils/utils.py`：`get_params()` 中 `--frame` 的 `choices` 改为 `get_registered_frames()` 调用。
4. 修改各 MockedModel 文件底部：添加 `register_model("Megatron", MegatronModel, MegatronWorkload, ...)` 调用。
5. 保持旧代码路径兼容（先尝试注册表，回退到硬编码逻辑）。

**时序图：**

```
aicb.py (main)            registry.py          MockedLlama.py        utils.py (get_params)
     │                         │                      │                     │
     │  1. import MockedLlama  │                      │                     │
     │─────────────────────────│─────────────────────►│                     │
     │                         │  2. register_model(  │                     │
     │                         │     "Llama",         │                     │
     │                         │     LlamaModel,      │                     │
     │                         │     MegatronWorkload)│                     │
     │                         │◄─────────────────────│                     │
     │                         │                      │                     │
     │  3. get_args() ──────────────────────────────────────────────────────►│
     │                         │                      │                     │
     │  4. args (frame="Llama" in choices) ◄─────────────────────────────────│
     │◄──────────────────────────────────────────────────────────────────────│
     │                         │                      │                     │
     │  5. model_cls, wl_cls = │                      │                     │
     │     registry.lookup(    │                      │                     │
     │       args.frame)       │                      │                     │
     │────────────────────────►│                      │                     │
     │  6. (LlamaModel, MegatronWorkload)             │                     │
     │◄────────────────────────│                      │                     │
     │                         │                      │                     │
```

**后置条件：** `MODEL_REGISTRY` 包含所有已注册框架的条目，`aicb.py` 通过注册表分发。

#### 4.1.5 用户接口设计

无新增 OM 接口。现有 CLI 接口不变，`--frame Llama` 直接从注册表查找。

#### 4.1.6 实现接口设计

| 接口名称 | 接口描述 | 接口类型 | 所属系统元素 | 规格约束 |
|----------|----------|----------|--------------|----------|
| `register_model(name, model_cls, wl_cls, description)` | 注册一个模型框架 | Python 函数 | `registry.py` | name 唯一，重复注册抛出 ValueError |
| `lookup(name) → ModelEntry` | 按名称查找已注册模型 | Python 函数 | `registry.py` | name 不存在时抛出 KeyError 并列出所有已注册名称 |
| `get_registered_frames() → List[str]` | 获取所有已注册框架名 | Python 函数 | `registry.py` | 返回排序后的名称列表 |

#### 4.1.7 安全配置设计

无安全配置项。注册表为内存数据结构，无持久化或网络暴露。

#### 4.1.9 DFX分析

- **可靠性分析 (FMEA)：**
  - 风险：注册同名模型 → 影响：覆盖已有条目或崩溃 → 缓解：`register_model()` 检测重复并抛出 `ValueError`
  - 风险：模型类签名不匹配 → 影响：运行时 TypeError → 缓解：`ModelEntry` 为 dataclass，在注册时校验类型

- **安全检查：** 无敏感操作。

#### 4.1.10 分配需求

| 系统需求编号 | 分配需求描述 | 系统元素 |
|-------------|-------------|----------|
| SR-REG-001 | `register_model()` 函数和 `MODEL_REGISTRY` 字典 | `registry.py` |
| SR-REG-002 | 各 MockedModel 文件底部调用 `register_model()` | `MockedMegatron.py`、`MockedDeepSeek.py`、`MockedLlama.py` 等 |
| SR-REG-003 | `get_params()` 中动态生成 `choices` | `utils/utils.py` |
| SR-REG-004 | `aicb.py` 中 `lookup()` + `try/except KeyError` 逻辑 | `aicb/aicb.py` |

---

### 4.2 F001：Llama 训练模型框架实现

#### 4.2.1 功能概述

为 AICB 训练 workload generator 添加 Llama 模型架构支持。Llama 是最广泛使用的开源 LLM 架构之一（Llama 2/3/3.1/4），其架构特征为：RMSNorm、SwiGLU 激活、Rotary Position Embedding (RoPE)、Grouped Query Attention (GQA)。通过实现此功能，验证模型注册机制的可行性，并为后续 Gemma（使用 GeGLU）提供参考模板。

#### 4.2.2 SR设计

| SR编号 | SR描述 |
|--------|--------|
| SR-LLAMA-001 | `MockedLlama.py` 实现所有 Llama 特有层类：`LlamaRMSNorm`、`LlamaRotaryEmbedding`、`LlamaAttention`（含 GQA）、`LlamaMLP`（含 gate/up/down projection） |
| SR-LLAMA-002 | `LlamaModel` 组合层为：`Embedding -> [LlamaDecoderLayer * num_layers] -> RMSNorm -> LM Head` |
| SR-LLAMA-003 | Llama 的 forward/backward 通信模式与 Megatron 对齐（TP 下使用 `all_reduce` / `all_gather` / `reduce_scatter`），新增 GQA 参数 `num_kv_heads` 影响 TP 分片计算 |
| SR-LLAMA-004 | CLI 支持 `--frame Llama`，模型参数通过已有的 `--hidden_size`、`--num_layers`、`--num_attention_heads`、`--swiglu`、`--seq_length` 等参数配置，新增 `--num_kv_heads` 参数 |
| SR-LLAMA-005 | Llama 支持 `--moe_enable`，当启用时使用 MoE FFN 替代标准 FFN（为 Llama 4 MoE 做准备） |

#### 4.2.3 实现思路

**方案：** 基于 `MockedMegatron.py` 参考实现，创建 `MockedLlama.py`。

Llama 与 Megatron 的核心差异：

| 组件 | Megatron | Llama | 对 MockedModel 的影响 |
|------|----------|-------|----------------------|
| Normalization | LayerNorm | RMSNorm | 无通信影响（仅参数规模不同） |
| Activation | GeLU / SwiGLU(可选) | SwiGLU | 无影响（SwiGLU 已有 `--swiglu` 支持） |
| Position Embedding | Learned | RoPE | 需要 `cos_cached` / `sin_cached` 参数建模 |
| Attention | MHA | GQA | `num_kv_heads` 影响 TP 分片的 KV projection 规模 |
| FFN | 2-layer (w1, w2) | 3-layer (gate, up, down) | `LlamaMLP` 需三投影矩阵；gate 和 up 可合并为单一 TP 列分片 |
| Pre/Post Norm | Post-LN | Pre-Norm (RMSNorm before attention and MLP) | 层内 compute 顺序不同，但不影响总通信量 |

由于 AICB 只建模通信模式而非实际计算，Llama 和 Megatron 的通信拓扑基本相同（均使用 TP 列/行分片），主要差异在于参数规模和 GQA 带来的 KV 投影缩减。

**备选方案：**
- 方案 A（采用）：新建 `MockedLlama.py`，复用 Megatron 的 `ColumnLinear` 和 `RowLinear` 通信模式。代码量约 350 行。
- 方案 B：参数化 `MockedMegatron.py`，通过配置开关支持 Llama。更少的文件但增加单文件复杂度（Megatron 已 676 行，添加 Llama 将超过 800 行限制）。

选择方案 A：符合项目"多小文件"原则，每个模型的层定义自包含。

#### 4.2.4 实现设计

**前置条件：** F002（模型注册机制重构）已完成。

**触发事件：** 用户执行 `aicb.py --frame Llama --hidden_size 4096 --num_layers 32 --num_attention_heads 32 --num_kv_heads 8 --swiglu ...`

**主流程（自然语言）：**

1. `aicb.py` 解析 `--frame Llama`，从 `MODEL_REGISTRY` 查找 `LlamaModel` 和 `MegatronWorkload`。
2. 实例化 `LlamaModel(args)`：
   a. 创建 `LlamaEmbedding`（padded_vocab_size x hidden_size，含 TP 分片）
   b. 创建 `num_layers` 个 `LlamaDecoderLayer`，每层包含：
      - `LlamaRMSNorm(hidden_size)` -- 前 norm
      - `LlamaAttention(hidden_size, num_heads, num_kv_heads, tp, seq_len, batch)`
      - `LlamaRMSNorm(hidden_size)` -- 后 norm
      - `LlamaMLP(hidden_size, ffn_hidden_size, tp, seq_len, batch)` 或 `MoEMLP`（若启用）
   c. 创建 `LlamaRMSNorm(hidden_size)` 作为 final norm
3. `MegatronWorkload(args, model)` 遍历模型层树，收集每层的 `forward()` / `backward()` workload。
4. 后续流程与现有 Megatron 路径相同（AIOB 计算时间、CSV 输出、物理重放）。

**LlamaAttention 层通信设计（GQA 关键）：**

```
LlamaAttention.__init__():
  self.n_heads = num_attention_heads          # 32
  self.n_kv_heads = num_kv_heads              # 8 (GQA ratio = 4)
  self.head_dim = hidden_size // num_heads     # 128

  # Q projection: [hidden, n_heads * head_dim] = [4096, 4096]
  self.q_proj = MegatronColumnLinear(hidden_size, n_heads * head_dim, tp, ...)

  # K projection: [hidden, n_kv_heads * head_dim] = [4096, 1024]  ← 更小!
  self.k_proj = MegatronColumnLinear(hidden_size, n_kv_heads * head_dim, tp, ...)

  # V projection: [hidden, n_kv_heads * head_dim] = [4096, 1024]  ← 更小!
  self.v_proj = MegatronColumnLinear(hidden_size, n_kv_heads * head_dim, tp, ...)

  # O projection: [n_heads * head_dim, hidden] = [4096, 4096]
  self.o_proj = MegatronRowLinear(n_heads * head_dim, hidden_size, tp, ...)
```

GQA 使得 K 和 V 投影的 TP 通信量比标准 MHA 减少 `n_kv_heads / n_heads` 倍（本例为 1/4）。

**LlamaMLP 层通信设计（SwiGLU 三投影）：**

```
LlamaMLP.__init__():
  # gate_proj: [hidden, ffn_hidden]  → ColumnLinear (TP 分片)
  self.gate_proj = MegatronColumnLinear(hidden_size, ffn_hidden_size, tp, ...)
  # up_proj:   [hidden, ffn_hidden]  → ColumnLinear (TP 分片)
  self.up_proj   = MegatronColumnLinear(hidden_size, ffn_hidden_size, tp, ...)
  # down_proj:  [ffn_hidden, hidden] → RowLinear (TP 分片, reduce_scatter)
  self.down_proj = MegatronRowLinear(ffn_hidden_size, hidden_size, tp, ...)
```

gate 和 up 的 TP 分片通信（`all_gather`）可合并为一次（与 Megatron SwiGLU 的 `w1`/`w2` 相同）。

**文件结构：**

```
aicb/workload_generator/mocked_model/training/
  MockedLlama.py          ← 新建 (预计 ~350 行)
  AiobLlama.py            ← 新建 (预计 ~200 行, GPU profiling)
```

#### 4.2.5 用户接口设计

**CLI 接口：**
```bash
# Llama 3 8B 类似配置
python aicb/aicb.py --frame Llama \
  --hidden_size 4096 \
  --num_layers 32 \
  --num_attention_heads 32 \
  --num_kv_heads 8 \
  --ffn_hidden_size 14336 \
  --vocab_size 128256 \
  --seq_length 8192 \
  --swiglu \
  --tensor_model_parallel_size 4 \
  --pipeline_model_parallel 1 \
  --global_batch 1024 \
  --micro_batch 1

# Llama 4 Scout (MoE) 类似配置
python aicb/aicb.py --frame Llama \
  --hidden_size 5120 \
  --num_layers 48 \
  --num_attention_heads 40 \
  --num_kv_heads 8 \
  --moe_enable \
  --num_experts 16 \
  --moe_router_topk 2 \
  --expert_model_parallel_size 4 \
  ...
```

**新增参数：**
```
--num_kv_heads:  GQA 的 KV 头数（默认等于 num_attention_heads，退化为 MHA）
```

**约束和限制：**
- 要求 `num_attention_heads % num_kv_heads == 0`（GQA 分组约束）
- 要求 `num_attention_heads % tensor_model_parallel_size == 0`（TP 分片约束）
- 启用 MoE 时要求 `--swiglu`（Llama 4 的 MoE 层使用 SwiGLU 激活）
- Hopper 架构 GPU 需要用于 AIOB 计算时间 profiling（DeepGEMM 等专用库）

#### 4.2.6 实现接口设计

| 接口名称 | 接口描述 | 接口类型 | 所属系统元素 | 规格约束 |
|----------|----------|----------|--------------|----------|
| `LlamaRMSNorm(hidden_size, eps)` | RMS 归一化层 mock | Python 类 | `MockedLlama.py` | 无通信操作，仅参数化模型规模 |
| `LlamaRotaryEmbedding(dim, max_position_embeddings, base)` | RoPE 位置编码 mock | Python 类 | `MockedLlama.py` | 无通信操作，参数化为 `cos_cached` / `sin_cached` |
| `LlamaAttention(hidden_size, num_heads, num_kv_heads, tp, seq_len, batch_size, layer_id)` | GQA Attention 层 | Python 类 | `MockedLlama.py` | Q/K/V/O 投影；TP 通信量随 `num_kv_heads` 缩放 |
| `LlamaMLP(hidden_size, ffn_hidden_size, tp, seq_len, batch_size, layer_id)` | SwiGLU MLP 层 | Python 类 | `MockedLlama.py` | gate/up/down 三投影；gate+up 的 TP 通信可合并 |
| `LlamaDecoderLayer(hidden_size, ffn_hidden_size, tp, seq_len, batch_size, num_heads, num_kv_heads, layer_id)` | Decoder 层组合 | Python 类 | `MockedLlama.py` | Pre-Norm 结构：RMSNorm -> Attention -> RMSNorm -> MLP |
| `LlamaModel(config)` | 完整 Llama 模型 | Python 类 | `MockedLlama.py` | 组合 embedding + decoder layers + final norm |

#### 4.2.7 安全配置设计

无安全配置项。MockedModel 为纯参数化描述，不执行网络操作或访问文件系统。

#### 4.2.9 DFX分析

- **可靠性分析 (FMEA)：**
  - 风险：GQA 的 KV projection TP 通信量计算错误 → 影响：workload 通信量与实际不符 → 缓解：参考 Llama 官方实现的 weight shapes，编写单元测试验证参数规模
  - 风险：`num_kv_heads` 未设置时退化为 MHA → 影响：用户误用 → 缓解：默认值 `None` 时自动设为 `num_attention_heads`

- **安全检查：** 无敏感操作。MockedModel 为本地纯计算代码。

#### 4.2.10 分配需求

| 系统需求编号 | 分配需求描述 | 系统元素 |
|-------------|-------------|----------|
| SR-LLAMA-001 | 实现 `LlamaRMSNorm`、`LlamaRotaryEmbedding`、`LlamaAttention`、`LlamaMLP` 类 | `MockedLlama.py` |
| SR-LLAMA-002 | 实现 `LlamaModel` 和 `LlamaDecoderLayer` 组合 | `MockedLlama.py` |
| SR-LLAMA-003 | GQA TP 分片通信量实现 | `LlamaAttention` in `MockedLlama.py` |
| SR-LLAMA-004 | `--frame Llama --num_kv_heads` 参数支持 | `utils/utils.py`、`aicb/aicb.py` |
| SR-LLAMA-005 | MoE FFN 可选替代标准 FFN | `LlamaDecoderLayer` in `MockedLlama.py` |

---

### 4.3 F003：上下文并行通信支持

#### 4.3.1 功能概述

为 AICB 添加上下文并行（Context Parallelism, CP）的通信建模支持。CP 将长序列的 token 范围分配给不同 GPU 组，通过 all-to-all 或块交换传递 KV 张量。当前 AICB 的 `RankGenerator` 默认排序为 `tp-cp-ep-dp-pp`，暗示了 CP 的计划，但 `context_parallel_size` 参数未被 workload generator 使用。此功能填补与 RAPID-LLM 相比最大的并行策略空白。

#### 4.3.2 SR设计

| SR编号 | SR描述 |
|--------|--------|
| SR-CP-001 | `CommGroup` 新增 `cp_group`、`cp_dp_group`、`cp_tp_group` |
| SR-CP-002 | `RankGenerator` 的 rank 映射支持 cp 维度 |
| SR-CP-003 | Attention 层的 CP 通信模式：`all_to_all` 或 block-wise P2P exchange，通信量为 `num_kv_heads * head_dim * seq_len_per_cp * batch_size` |
| SR-CP-004 | CLI 新增 `--context-parallel-size`（已存在于 utils.py 第 613 行但未使用） |

#### 4.3.3 实现思路

CP 通信发生在 attention 层：每个 GPU 只持有部分序列的 Q，但需要完整的 K 和 V。因此：
- forward：发送本地 K/V 给其他 CP ranks，接收其他 ranks 的 K/V（all-to-all 或 ring P2P）
- backward：对称的反向通信

通信量公式（Ring Attention 风格）：
```
cp_comm_size = 2 * num_kv_heads * head_dim * seq_len * batch_size  # K + V
per_step_cp_comm_size = cp_comm_size / cp_size * 2  # send + recv (ring)
```

#### 4.3.4 实现设计

**时序图（CP Attention forward）：**

```
GPU0 (cp_rank=0)      GPU1 (cp_rank=1)      GPU2 (cp_rank=2)      GPU3 (cp_rank=3)
    │                      │                      │                      │
    │ 1. compute local Q,K,V (seq_len/cp tokens)                        │
    │                      │                      │                      │
    │ 2. send KV_chunk →  │ 3. recv ← send →    │ 4. recv ← send →    │
    │                      │                      │                      │
    │ 5. all-to-all (ring/fully_connected) exchange complete KV         │
    │                      │                      │                      │
    │ 6. compute attention with full KV (no further comm)               │
    │                      │                      │                      │
```

自然语言主流程：

1. `CommGroup` 枚举添加 `cp_group = "cp_group"` 和 `cp_dp_group = "cp_dp_group"`。
2. `RankGenerator.__init__()` 接受 `cp` 参数，生成 CP 组的 rank 列表。
3. 在 `LlamaAttention.forward()` 中：若 `cp > 1`，在 QKV 投影计算后插入 `all_to_all` 通信 LogItem。
4. 在 `LlamaAttention.backward()` 中：插入对应的反向 `all_to_all`。
5. `SIMAI_workload.workload_generate_aiob()` 中添加 "cp_kv_exchange" 名称匹配。

#### 4.3.6 实现接口设计

| 接口名称 | 接口描述 | 接口类型 | 所属系统元素 | 规格约束 |
|----------|----------|----------|--------------|----------|
| `CommGroup.cp_group` | 上下文并行通信组 | 枚举值 | `utils/utils.py` | 与 tp/dp/ep/pp 正交 |
| `RankGenerator(cp=1, ...)` | 支持 cp 维度的 rank 生成器 | Python 类构造函数 | `utils/utils.py` | cp 默认为 1（无 CP） |

#### 4.3.10 分配需求

| 系统需求编号 | 分配需求描述 | 系统元素 |
|-------------|-------------|----------|
| SR-CP-001 | 枚举值 `cp_group`、`cp_dp_group` | `utils/utils.py` |
| SR-CP-002 | RankGenerator cp 维度支持 | `utils/utils.py` |
| SR-CP-003 | Attention 层 CP 通信 LogItem 插入 | `MockedLlama.py`、`MockedMegatron.py` |
| SR-CP-004 | `--context-parallel-size` 参数启用 | `utils/utils.py` |

---

### 4.4 F004：Chakra ET 导出器

#### 4.4.1 功能概述

实现 Chakra Execution Trace 导出器，将 AICB 生成的 `LogItem` / `Work_Item` workload 转换为标准化的 Chakra ET 格式（基于 protocol buffer 的 DAG 表示），使 AICB 工作负载可被整个 Chakra 生态系统消费（astra-sim、PARAM、RAPID-LLM 后端、MLSynth Orchestrator）。

#### 4.4.2 SR设计

| SR编号 | SR描述 |
|--------|--------|
| SR-CK-001 | `ChakraExporter` 类将 `Workload` (LogItem 列表) 转换为 Chakra ET protobuf 文件 |
| SR-CK-002 | 通信节点映射：`CommType.all_reduce` -> Chakra `COMM_ALL_REDUCE`，`CommType.all_gather` -> `COMM_ALL_GATHER`，以此类推 |
| SR-CK-003 | 计算节点映射：`CommType.computation` -> Chakra `COMP_NODE`，含 FLOPs 和 memory 元数据 |
| SR-CK-004 | 依赖边：按时间顺序连接节点，forward 和 backward 间插入 barrier 依赖 |
| SR-CK-005 | CLI 新增 `--export-chakra <filepath>` 参数，导出 `.et` 文件 |

#### 4.4.3 实现思路

采用轻量级导出器，不引入 Chakra protobuf 的完整 Python 绑定（避免复杂编译依赖）。改用 JSON 格式的 Chakra ET 兼容输出（MLCommons 的 Chakra 工具链接受 JSON 输入）。

**备选方案：**
- 方案 A（采用）：JSON 格式 Chakra ET 导出。无需 protobuf 编译链，与 astra-sim 的 Chakra JSON loader 兼容。
- 方案 B：完整 protobuf 导出。需要安装 Chakra pip 包和 protobuf 编译器，增加部署复杂度。

选择方案 A：轻量、无外部依赖、与现有工具链兼容。

#### 4.4.4 实现设计

**主流程：**

1. `ChakraExporter.__init__(workload)` 接收 AICB workload。
2. `export(filepath)` 遍历 workload：
   a. 每个 `LogItem` 转换为一个 Chakra 节点（含 id、type、`attr` 字典）。
   b. 插入依赖边（默认线性顺序，支持通过 `dependency` 字段自定义）。
   c. 添加全局元数据节点（GPU 数量、TP/PP/DP 配置）。
3. 写入 JSON 文件。

**Chakra ET JSON 格式示例：**

```json
{
  "nodes": [
    {"id": 0, "type": "COMP", "name": "forward.attention.qkv",
     "attr": {"flops": 2.1e12, "memory": 2.5e9}, "inputs": {}, "outputs": {"0": [1, 2]}},
    {"id": 1, "type": "COMM_ALL_REDUCE", "name": "forward.tp_all_reduce",
     "attr": {"comm_size": 6.7e7, "comm_group": [0,1,2,3]}, "inputs": {"0": [0]}, "outputs": {}}
  ],
  "global_attr": {"world_size": 8, "tp": 4, "pp": 1, "dp": 2}
}
```

#### 4.4.6 实现接口设计

| 接口名称 | 接口描述 | 接口类型 | 所属系统元素 | 规格约束 |
|----------|----------|----------|--------------|----------|
| `ChakraExporter(workload)` | 构造函数，接收 AICB workload | Python 类 | `utils/chakra_export.py` | workload 必须已完成 `_fill_ranks()` |
| `export(filepath)` | 导出为 Chakra ET JSON 文件 | Python 方法 | `utils/chakra_export.py` | 文件路径可写 |
| `logitem_to_chakra_node(item) -> dict` | 映射单个 LogItem 到 Chakra 节点 | Python 函数 | `utils/chakra_export.py` | 内部使用 |

#### 4.4.10 分配需求

| 系统需求编号 | 分配需求描述 | 系统元素 |
|-------------|-------------|----------|
| SR-CK-001 | ChakraExporter 主类 | `utils/chakra_export.py` [新建] |
| SR-CK-002 | CommType -> Chakra COMM_* 映射表 | `utils/chakra_export.py` |
| SR-CK-003 | computation LogItem -> COMP_NODE 映射 | `utils/chakra_export.py` |
| SR-CK-004 | 依赖边生成逻辑 | `utils/chakra_export.py` |
| SR-CK-005 | `--export-chakra` CLI 参数 | `utils/utils.py`、`aicb/aicb.py` |

---

## 附录 A：完整实现文件清单

| 文件 | 类型 | 预计行数 | 所属功能 |
|------|------|----------|----------|
| `aicb/workload_generator/registry.py` | 新建 | ~50 | F002 |
| `aicb/workload_generator/mocked_model/training/MockedLlama.py` | 新建 | ~350 | F001 |
| `aicb/workload_generator/mocked_model/training/AiobLlama.py` | 新建 | ~200 | F001 |
| `aicb/utils/chakra_export.py` | 新建 | ~150 | F004 |
| `aicb/aicb.py` | 修改 | ~20 行变更 | F002 |
| `aicb/utils/utils.py` | 修改 | ~50 行变更 | F002, F003, F004, F001 (参数) |
| `aicb/workload_generator/mocked_model/training/MockedMegatron.py` | 修改 | ~5 行（底部注册） | F002 |
| `aicb/workload_generator/mocked_model/training/MockedDeepSeek.py` | 修改 | ~5 行（底部注册） | F002 |
| `aicb/workload_generator/mocked_model/training/MockedDeepspeed.py` | 修改 | ~5 行（底部注册） | F002 |

## 附录 B：扩展检查清单

为后续模型（Gemma、Mistral、Falcon、DBRX）添加支持时，按以下步骤操作：

1. [ ] 确定目标模型的层结构（参考 HuggingFace `modeling_*.py` 源码或官方论文）
2. [ ] 确定通信上与已有 Megatron/Llama 实现的差异
3. [ ] 创建 `Mocked<ModelName>.py`，继承 `MockedModel`
4. [ ] 实现特有层类（如 GeGLU、parallel attention、sliding window attention 等）
5. [ ] 实现主 Model 类的 `forward()` / `backward()`
6. [ ] 在文件底部调用 `register_model()`
7. [ ] 创建 `Aiob<ModelName>.py` 用于 GPU profiling（可选）
8. [ ] 添加模型特有 CLI 参数（如需要）
9. [ ] 编写单元测试验证参数规模和通信量
10. [ ] 更新 `--frame` choices（自动派生，无需手动）
