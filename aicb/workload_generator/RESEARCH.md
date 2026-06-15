# Qwen3 and Qwen3.5 AICB Workload Generator -- Complete Research Report

Verified 2025-06-15 by agent workflow-subagent. Primary sources: HF config.json
files (gitcode.com, hf-mirror.com), HuggingFace transformers source on GitHub,
AICB codebase analysis.

---

## 1. Architecture Comparison

| Feature | LLaMA 3 | Qwen3 | Qwen3.5 |
|---------|---------|-------|---------|
| Attention | GQA (derived head_dim) | GQA (explicit head_dim=128) | Hybrid: 75% DeltaNet + 25% GQA (head_dim=256) |
| QK-Norm | No | Yes (per-head RMSNorm before RoPE) | Full-attn: Yes. DeltaNet: RMSNormGated |
| RoPE theta | 500,000 | 1,000,000 | 10,000,000 |
| KV heads | varies (8-8) | fixed 8 (dense), 4 (MoE) | fixed 2-4 |
| Vocab | 128,000 | 151,936 | 248,320 |
| MoE shared expert | N/A | None | 1 per layer (sigmoid-gated) |
| MTP layers | No | No | 1 |
| Q_dim vs hidden | Always equal | 3 of 6 diverge | 3 of 5 diverge |

Critical finding: Qwen3/3.5 use explicit head_dim (128 or 256).
num_q * head_dim frequently differs from hidden_size. This breaks the LLaMA
assumption that QKV projection = 3 * hidden_size.

| Model | hidden | Q heads | Q dim | Mismatch |
|-------|--------|---------|-------|----------|
| Qwen3-0.6B | 1024 | 16 | 2048 | 2.0x |
| Qwen3-4B | 2560 | 32 | 4096 | 1.6x |
| Qwen3-32B | 5120 | 64 | 8192 | 1.6x |
| Qwen3-30B-A3B | 2048 | 32 | 4096 | 2.0x |
| Qwen3-235B-A22B | 4096 | 64 | 8192 | 2.0x |
| Qwen3.5-0.8B | 1024 | 8 | 2048 | 2.0x |
| Qwen3.5-27B | 5120 | 24 | 6144 | 1.2x |
| Qwen3.5-397B-A17B | 4096 | 32 | 8192 | 2.0x |

8 of 16 models diverge. All MoE models diverge by 2x.

---

## 2. Verified Model Configurations

All 16 configs verified from primary-source config.json files.

### 2.1 Qwen3 Dense (6 variants, MockedQwen3.py, moe_enable=False)

| Model | h | L | Q | KV | FFN | vocab | max_seq | tie |
|-------|---|---|----|-----|-----|-------|---------|-----|
| 0.6B | 1024 | 28 | 16 | 8 | 3072 | 151936 | 40960 | yes |
| 1.7B | 2048 | 28 | 16 | 8 | 6144 | 151936 | 40960 | yes |
| 4B | 2560 | 36 | 32 | 8 | 9728 | 151936 | 40960 | yes |
| 8B | 4096 | 36 | 32 | 8 | 12288 | 151936 | 40960 | no |
| 14B | 5120 | 40 | 40 | 8 | 17408 | 151936 | 40960 | no |
| 32B | 5120 | 64 | 64 | 8 | 25600 | 151936 | 40960 | no |

Common: head_dim=128, rope_theta=1M, rms_norm_eps=1e-6, qk_norm=yes,
use_sliding_window=false, attention_bias=false. Qwen3-14B has 40 layers
and Qwen3-32B has 64 heads (OLMo-core PR listing 48L, 40H was incorrect).

### 2.2 Qwen3 MoE (2 variants, MockedQwen3.py, moe_enable=True)

| Model | h | L | Q | KV | FFN | moe_ffn | exp | topk | shared | vocab |
|-------|---|---|----|-----|-----|---------|-----|------|--------|-------|
| 30B-A3B | 2048 | 48 | 32 | 4 | 6144 | 768 | 128 | 8 | none | 151936 |
| 235B-A22B | 4096 | 94 | 64 | 4 | 12288 | 1536 | 128 | 8 | none | 151936 |

Common: head_dim=128, rope_theta=1M, decoder_sparse_step=1.

Resolved: intermediate_size (6144/12288) is inherited but UNUSED. Verified from
HF source: Qwen3MoeDecoderLayer creates dense MLP then immediately replaces it
with SparseMoeBlock using moe_intermediate_size. Dense MLP is phantom-allocated
then garbage collected. Every layer is pure MoE: attention + sparse FFN only.

### 2.3 Qwen3.5 Dense (5 variants, MockedQwen3_5.py, moe_enable=False)

| Model | h | L | Q | KV | FFN | vocab | full:lin | tie |
|-------|---|---|----|-----|-----|-------|----------|-----|
| 0.8B | 1024 | 24 | 8 | 2 | 3584 | 248320 | 6:18 | yes |
| 2B | 2048 | 24 | 8 | 2 | 6144 | 248320 | 6:18 | yes |
| 4B | 2560 | 32 | 16 | 4 | 9216 | 248320 | 8:24 | yes |
| 9B | 4096 | 32 | 16 | 4 | 12288 | 248320 | 8:24 | no |
| 27B | 5120 | 64 | 24 | 4 | 17408 | 248320 | 16:48 | no |

Common: head_dim=256, rope_theta=10M, max_pos=262144, full_attn_interval=4,
linear_key_head_dim=128, linear_value_head_dim=128, linear_num_key_heads=16
(constant), linear_conv_kernel_dim=4, partial_rotary=0.25, attn_output_gate=true,
mrope_interleaved=true, mtp_num_hidden_layers=1. Linear V heads scale: 16 (0.8B,
2B), 32 (4B, 9B), 48 (27B).

### 2.4 Qwen3.5 MoE (3 variants, MockedQwen3_5.py, moe_enable=True)

| Model | h | L | Q | KV | hdim | exp | topk | moe_ffn | shared_ffn | lin_V | full:lin | vocab |
|-------|---|---|----|-----|------|-----|------|---------|------------|-------|----------|-------|
| 35B-A3B | 2048 | 40 | 16 | 2 | 256 | 256 | 8 | 512 | 512 | 32 | 10:30 | 248320 |
| 122B-A10B | 3072 | 48 | 32 | 2 | 256 | 256 | 8 | 1024 | 1024 | 64 | 12:36 | 248320 |
| 397B-A17B | 4096 | 60 | 32 | 2 | 256 | 512 | 10 | 1024 | 1024 | 64 | 15:45 | 248320 |

Common: rope_theta=10M, full_attn_interval=4, linear_num_key_heads=16,
linear_conv_kernel_dim=4, partial_rotary=0.25, attn_output_gate=true,
mrope_interleaved=true, mtp_num_hidden_layers=1, model_type=qwen3_5_moe.
Shared expert: 1 per layer, sigmoid-gated, operates on ALL tokens (not routed).
FP8 native (blog: ~50% activation memory reduction). 262K native context, 1M via YaRN.

Note: The A17B suffix in 397B-A17B refers to ~17B activated parameters per token
(10 routed experts + 1 shared expert, each with moe_ffn=1024), not 17 experts.
The Qwen3.5 blog incorrectly states "17 active" -- config.json confirms topk=10.

---

## 3. Primary Source Verification

| # | Model | Source |
|---|-------|--------|
| 1 | Qwen3-0.6B | gitcode.com config.json |
| 2 | Qwen3-1.7B | hf-mirror.com config.json (Base) |
| 3 | Qwen3-4B | hf-mirror.com config.json |
| 4 | Qwen3-8B | gitcode.com config.json |
| 5 | Qwen3-14B | hf-mirror.com config.json |
| 6 | Qwen3-32B | hf-mirror.com config.json |
| 7 | Qwen3-30B-A3B | hf-mirror.com config.json |
| 8 | Qwen3-235B-A22B | hf-mirror.com config.json |
| 9 | Qwen3.5-0.8B | gitcode.com config.json |
| 10 | Qwen3.5-2B | hf-mirror.com config.json |
| 11 | Qwen3.5-4B | gitcode.com config.json |
| 12 | Qwen3.5-9B | gitcode.com config.json |
| 13 | Qwen3.5-27B | gitcode.com config.json |
| 14 | Qwen3.5-35B-A3B | hf-mirror.com config.json |
| 15 | Qwen3.5-122B-A10B | hf-mirror.com config.json |
| 16 | Qwen3.5-397B-A17B | hf-mirror.com config.json |

Code sources: modeling_qwen3_moe.py v4.51.0, modeling_qwen3_5_moe.py main.

---

## 4. AICB Implementation

### 4.1 File Status

| File | Status | Variants |
|------|--------|----------|
| training/MockedQwen3.py | COMPLETE | Dense (6) + MoE (2) |
| training/MockedQwen3_5.py | COMPLETE | Dense (5) + MoE (3) |
| generate_megatron_workload.py | COMPLETE | --frame Qwen3 / Qwen3.5 |
| training/MockedMegatron.py | FIXED | MoE ep_size bug |
| training/MockedDeepSeek.py | FIXED | MoE ep_size bug |

### 4.2 Key Design Decisions

GatedDeltaNet: forward() returns Workload() (empty). All projections replicated
across TP. Parameters tracked via raw MockedParam for DP gradient sync.

Qwen3Attention: uses explicit head_dim and GQA-aware num_kv_heads. ColumnLinear/
RowLinear abstractions are dimension-agnostic, so TP communication sizes are
correct for all Q-dimension configurations.

MoE: Qwen3 uses MOEMLP (128 experts, top-8, no shared). Qwen3.5 uses
Qwen3_5MoEMLP (256-512 experts, top-8/10, 1 shared expert with learned gate).

---

## 5. Communication Formulas

### 5.1 Forward pass (TP=8, SP enabled)

```
Qwen3 dense:     fwd = 1 + L * 4 + 1      (emb + layers*4 + lm_head)
Qwen3 MoE:       fwd = 1 + L * 7 + 1      (emb + layers*7 + lm_head)
Qwen3.5 dense:   fwd = 1 + Lf*4 + Ld*2 + 1
Qwen3.5 MoE:     fwd = 1 + Lf*9 + Ld*7 + 1
```

Lf = L / 4, Ld = L - Lf.

### 5.2 Per-layer composition

| Layer type | Attn | MLP/MoE | Total | Components |
|---|---|---|---|---|
| Qwen3 standard | 2 | 2 | 4 | QKV(ag)+out(rs)+gate_up(ag)+down(rs) |
| Qwen3 MoE | 2 | 5 | 7 | +preprocess(ag)+dispatch(a2a)+perm(ag)+unperm(rs)+combine(a2a) |
| Qwen3.5 FullAttn | 2 | 2 or 2+5 | 4 or 9 | Same as Qwen3 |
| Qwen3.5 DeltaNet | 0 | 2 or 2+5 | 2 or 7 | MLP/MoE only |

ag=all_gather, rs=reduce_scatter, a2a=all_to_all

MoE msg_size formulas (divide by ep_size, fixed 2025-06-15):
```
dispatch:  seq_len * hidden * batch * topk / tp / ep * 2
permute:   2 * hidden * topk * batch * seq_len / ep
combine:   same as dispatch
unpermute: same as permute
```

### 5.3 Verified across all 16 variants

| Model | L | Type | Fwd |
|-------|---|------|-----|
| Qwen3-0.6B | 28 | Dense | 114 |
| Qwen3-1.7B | 28 | Dense | 114 |
| Qwen3-4B | 36 | Dense | 146 |
| Qwen3-8B | 36 | Dense | 146 |
| Qwen3-14B | 40 | Dense | 162 |
| Qwen3-32B | 64 | Dense | 258 |
| Qwen3-30B-A3B | 48 | MoE | 338 |
| Qwen3-235B-A22B | 94 | MoE | 660 |
| Qwen3.5-0.8B | 24 | Dense | 62 |
| Qwen3.5-2B | 24 | Dense | 62 |
| Qwen3.5-4B | 32 | Dense | 82 |
| Qwen3.5-9B | 32 | Dense | 82 |
| Qwen3.5-27B | 64 | Dense | 162 |
| Qwen3.5-35B-A3B | 40 | MoE | 302 |
| Qwen3.5-122B-A10B | 48 | MoE | 362 |
| Qwen3.5-397B-A17B | 60 | MoE | 452 |

DeepSeekV3 regression passed (538 fwd). Megatron/LLaMA 3.1 verified: 8B=129, 70B=321,
405B MoE=883.

Generated workload CSVs in `results/mocked_workload/` (18 files, 9 models):
- LLaMA: llama8b, llama_405b_moe
- Qwen3: qwen3_8b, qwen3_30b_a3b, qwen3_235b_a22b
- Qwen3.5: qwen35_9b, qwen35_35b_a3b, qwen35_122b_a10b, qwen35_397b_a17b

---

## 6. Training Parallelism Configurations

### 6.1 Verified (from official sources)

| Model | Source | TP | PP | EP | CP | Total GPUs |
|-------|--------|----|----|----|----|------------|
| Qwen3-235B-A22B (pretrain) | NVIDIA Megatron-Bridge recipe | **4** | **16** | **8** | **2** | 4x16x8 = 512 |
| Qwen3-235B-A22B (SFT) | NVIDIA Megatron-Bridge recipe | 4 | -- | 4 | -- | 64 (8 nodes) |

### 6.2 Estimated (from NVIDIA node counts, not verified)

| Model | Source | Nodes | GPUs | Likely TP/EP | Rationale |
|-------|--------|-------|------|-------------|-----------|
| Qwen3.5-35B-A3B | NVIDIA Megatron-Bridge | 2 | 16 | TP=2, EP=8 | 35B total, 3B active, 256 experts |
| Qwen3.5-122B-A10B | NVIDIA Megatron-Bridge | 4 | 32 | TP=4, EP=8 | 122B total, 10B active, 256 experts |
| Qwen3.5-397B-A17B | Not published | -- | -- | TP=4, EP=16? | 397B total, 17B active, 512 experts, topk=10 |

### 6.3 Data Gap

Qwen3.5 was released Feb-Mar 2026. As of 2025-06, no technical report or
pretraining paper exists for Qwen3.5. The NVIDIA Megatron-Bridge docs provide
inference node counts but not explicit TP/PP/EP training breakdowns. Estimates
above assume EP scaling with expert count (Qwen3: 128e/EP8; Qwen3.5: 256e/EP8
or 512e/EP16) and TP scaling with active parameters (~4-8 per active-param
tier). These should be treated as placeholders until official documentation
is published.

### 5.4 Head-to-head communication comparison

| Comparison | Model A | Items | Model B | Items | Reduction |
|---|---|---|---|---|---|
| Dense equiv | Qwen3-8B | 146 | Qwen3.5-9B | 82 | 44% |
| MoE equiv | Qwen3-30B | 338 | Qwen3.5-35B | 302 | 11% |
| Flagship | Qwen3-235B | 660 | Qwen3.5-397B | 452 | 32% |

The 44% dense reduction comes from 24 DeltaNet layers generating zero attention
communication. The smaller MoE reduction reflects MoE all-to-all dominating
per-layer communication, with Qwen3.5 adding shared expert TP overhead.

---

## 6. Bugs Found and Fixed

| # | Bug | Files | Impact | Fix |
|---|-----|-------|--------|-----|
| 1 | MoE msg_size overstated by ep_size (12 formulas) | MOEMLP, DeepSeekMoE, Qwen3_5MoEMLP | 64x overcount at EP=64; 50 GB phantom comm/iter at 30B | Added // ep_size |
| 2 | Qwen3 MoE not supported | MockedQwen3.py | 30B-A3B, 235B unusable | Added moe_enable + MOEMLP |
| 3 | DeltaNet overcounted TP comm (4/layer vs 0 correct) | MockedQwen3_5.py | 59% overestimate in 9B | Reverted to comm-accurate |
| 4 | OLMo-core PR wrong configs (14B: 48L, 32B: 40H) | Research | Wrong third-party specs | Primary config.json: 40L, 64H |
| 5 | CLAUDE.md described fixed bugs as current | CLAUDE.md | Stale docs | Updated all sections |

---

## 7. Known Limitations

1. MTP not modeled (~2% extra compute, negligible communication).
2. Inference mocks for Qwen3/Qwen3.5 are skeletal (no forward/backward).
3. AIOB training benchmarks for Qwen3/Qwen3.5 do not exist (inference AIOB only).
4. GatedDeltaNet parameter count approximate (linear weights only).
5. Qwen3Params training orchestration fields (pp_rank, use_distributed_optimizer)
   set by CLI get_params(), not by Params defaults.

---

## Qwen3.5 Official Blog (qwen-ai.com/qwen-3-5, March 2026)

The Qwen3.5 launch blog confirms several architectural findings and adds
performance context:

- **Hybrid ratio**: 75% GatedDeltaNet + 25% gated attention (3:1), matching
  config.json `full_attention_interval=4` and `layer_types` pattern.
- **Performance**: 8.6x faster decode at 32K tokens, 19x at 256K vs Qwen3-Max.
  Community: 35B-A3B hits 196 tok/s on RTX 4090.
- **Vocab**: 248,320 tokens (up from 152K in Qwen3), 201 languages.
- **FP8 native**: ~50% activation memory reduction.
- **All models multimodal**: Text, image, video via early fusion. No separate
  "VL" suffix -- every model handles all modalities.
- **Context**: 262K native, 1M via YaRN.
- **MoE 397B**: 512 total experts, 10 active + 1 shared (confirmed by config.json
  `num_experts_per_tok=10`). Blog says "17 active" -- this conflates the activated
  parameter count (~17B, hence the A17B suffix) with the number of routed experts.
- **Thinking control**: API parameter `enable_thinking` (not chat template
  toggles like Qwen3). On by default for 27B+, off for smaller.
- **Deployment caveats**: Ollama broken as of March 2026 (infinite CoT loops,
  broken tool calling, 5-7x slower than llama.cpp). QLoRA 4-bit incompatible
  with hybrid architecture (use bf16 LoRA instead).

Source: https://qwen-ai.com/qwen-3-5 (fetched 2025-06-15, 38KB).

---

## How to Use

### Quick smoke test (standalone Python, no GPU needed)

```bash
cd aicb
python3 workload_generator/mocked_model/training/MockedQwen3.py
python3 workload_generator/mocked_model/training/MockedQwen3_5.py
```

### Generate workload CSVs programmatically

```python
from workload_generator.mocked_model.training.MockedQwen3 import Qwen3Params, Qwen3Model

# Qwen3-8B dense
cfg = Qwen3Params()
cfg.hidden_size = 4096
cfg.intermediate_size = 12288
cfg.num_hidden_layers = 36
cfg.num_attention_heads = 32
cfg.num_key_value_heads = 8
cfg.head_dim = 128
cfg.vocab_size = 151936
cfg.tensor_model_parallel_size = 8
cfg.world_size = 8
cfg.seq_length = 4096
cfg.micro_batch = 2
cfg.enable_sequence_parallel = True

model = Qwen3Model(cfg)
fwd = model.forward()   # 146 ops
bwd = model.backward()  # 219 ops
fwd.dump("qwen3_8b_fwd")  # -> results/mocked_workload/qwen3_8b_fwd_workload.csv
```

### Qwen3.5-9B dense

```python
from workload_generator.mocked_model.training.MockedQwen3_5 import Qwen3_5Params, Qwen3_5Model

cfg = Qwen3_5Params()
cfg.hidden_size = 4096; cfg.intermediate_size = 12288
cfg.num_hidden_layers = 32; cfg.num_attention_heads = 16
cfg.num_key_value_heads = 4; cfg.head_dim = 256
cfg.vocab_size = 248320
cfg.full_attention_interval = 4
cfg.linear_key_head_dim = 128; cfg.linear_value_head_dim = 128
cfg.linear_num_key_heads = 16; cfg.linear_num_value_heads = 32
cfg.linear_conv_kernel_dim = 4
cfg.tensor_model_parallel_size = 8; cfg.world_size = 8
cfg.seq_length = 4096; cfg.micro_batch = 2
cfg.enable_sequence_parallel = True

model = Qwen3_5Model(cfg)
fwd = model.forward()   # 82 ops
bwd = model.backward()  # 123 ops
```

### Qwen3.5-397B-A17B MoE

```python
cfg = Qwen3_5Params()
cfg.hidden_size = 4096; cfg.intermediate_size = 12288
cfg.moe_intermediate_size = 1024; cfg.shared_expert_intermediate_size = 1024
cfg.num_hidden_layers = 60; cfg.num_attention_heads = 32
cfg.num_key_value_heads = 2; cfg.head_dim = 256
cfg.vocab_size = 248320
cfg.full_attention_interval = 4
cfg.linear_key_head_dim = 128; cfg.linear_value_head_dim = 128
cfg.linear_num_key_heads = 16; cfg.linear_num_value_heads = 64
cfg.linear_conv_kernel_dim = 4
cfg.tensor_model_parallel_size = 4; cfg.expert_model_parallel_size = 16
cfg.world_size = 128
cfg.seq_length = 4096; cfg.micro_batch = 2
cfg.enable_sequence_parallel = True
cfg.moe_enable = True; cfg.num_experts = 512; cfg.moe_router_topk = 10

model = Qwen3_5Model(cfg)
fwd = model.forward()   # 452 ops
bwd = model.backward()  # 468 ops
```

### CLI (requires --frame, --swiglu, --head_dim, --num_key_value_heads)

```bash
python -m workload_generator.generate_megatron_workload \
  --frame Qwen3 --model_name Qwen3-8B \
  --hidden_size 4096 --num_layers 36 \
  --num_attention_heads 32 --num_key_value_heads 8 --head_dim 128 \
  --intermediate_size 12288 --vocab_size 151936 \
  --world_size 8 --tensor_model_parallel_size 8 \
  --seq_length 4096 --micro_batch 2 \
  --enable_sequence_parallel --swiglu --workload_only
```
