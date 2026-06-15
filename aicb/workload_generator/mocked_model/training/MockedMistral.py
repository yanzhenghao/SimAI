"""
Mocked Mistral / Mixtral model for AICB workload generation.

Mistral-7B (dense): Same architecture as LLaMA (RMSNorm, SwiGLU, GQA, RoPE)
  with sliding window attention (window_size=4096 typically).
  Reuses LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaMLP.

Mixtral-8x7B (MoE): 8 experts, top-2 routing, shared attention across experts.
  Uses MOEMLP for expert FFN with EP support.
  Total 46.7B params, 12.9B active per token.

Supported configs:
  Mistral-7B:   hidden=4096,  intermediate=14336,  num_heads=32,  num_kv_heads=8,   layers=32
  Mixtral-8x7B: hidden=4096,  intermediate=14336,  num_heads=32,  num_kv_heads=8,   layers=32,
                num_experts=8, moe_router_topk=2, expert_model_parallel_size=1-8
  Mixtral-8x22B: hidden=6144, intermediate=16384, num_heads=56,  num_kv_heads=8,   layers=56,
                num_experts=8, moe_router_topk=2

Based on MockedLlama.py and MockedMegatron.py patterns.
File: MockedMistral.py
License: Apache 2.0
"""

from workload_generator.mocked_model.MockedModel import MockedModel
from workload_generator.mocked_model.training.MockedLlama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaAttention,
    LlamaMLP,
)
from workload_generator.mocked_model.training.MockedMegatron import (
    MegatronEmbedding,
    MegatronColumnLinear,
    MOEMLP,
)
from log_analyzer.log import Workload, LogItem


# ---------------------------------------------------------------------------
# MistralDecoderLayer -- standard dense decoder layer
# ---------------------------------------------------------------------------
class MistralDecoderLayer(MockedModel):
    """Mistral decoder layer: identical to LLaMA architecture.

    Structure (pre-norm):
      x -> RMSNorm -> GQA Attention (sliding window) -> residual(+x)
        -> RMSNorm -> SwiGLU MLP -> residual(+)

    The sliding window attention is a compute-only optimization (local
    attention) that does not affect communication patterns.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_kv_heads,
        tp,
        cp,
        seq_len,
        batch_size,
        layer_id,
        window_size=None,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        super().__init__()
        self.name = f"mistral_layer_{layer_id}"
        self.layer_id = layer_id
        self.window_size = window_size  # sliding window (no comm impact)

        self.input_layernorm = LlamaRMSNorm(
            hidden_size, name=f"input_norm_{layer_id}"
        )
        self.self_attn = LlamaAttention(
            num_attention_heads, num_kv_heads, hidden_size,
            tp, cp, seq_len, batch_size, layer_id,
            sequence_parallel_enabled, computation_enable, add_bias_linear,
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            hidden_size, name=f"post_attn_norm_{layer_id}"
        )
        self.mlp = LlamaMLP(
            hidden_size, intermediate_size, tp, seq_len, batch_size,
            layer_id, sequence_parallel_enabled, computation_enable, add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.self_attn.forward())
        workloads.extend(self.mlp.forward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.mlp.backward())
        workloads.extend(self.self_attn.backward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads


# ---------------------------------------------------------------------------
# MixtralDecoderLayer -- MoE decoder layer
# ---------------------------------------------------------------------------
class MixtralDecoderLayer(MockedModel):
    """Mixtral MoE decoder layer: shared attention, MoE FFN.

    Structure (pre-norm):
      x -> RMSNorm -> GQA Attention -> residual(+x)
        -> RMSNorm -> MoE FFN (8 experts, top-2 gating) -> residual(+)
                      + shared expert FFN (optional, not in original Mixtral)

    Communication:
      - Attention: same as LlamaAttention (TP all_gather + reduce_scatter)
      - MoE FFN: all_to_all for expert dispatch + combine (EP group)
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_kv_heads,
        tp,
        cp,
        seq_len,
        batch_size,
        layer_id,
        expert_model_parallel_size,
        num_experts,
        moe_router_topk,
        moe_grouped_gemm=True,
        num_shared_experts=0,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        super().__init__()
        self.name = f"mixtral_layer_{layer_id}"
        self.layer_id = layer_id

        self.input_layernorm = LlamaRMSNorm(
            hidden_size, name=f"input_norm_{layer_id}"
        )
        self.self_attn = LlamaAttention(
            num_attention_heads, num_kv_heads, hidden_size,
            tp, cp, seq_len, batch_size, layer_id,
            sequence_parallel_enabled, computation_enable, add_bias_linear,
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            hidden_size, name=f"post_attn_norm_{layer_id}"
        )

        # MoE FFN: reuses Megatron's MOEMLP with expert parallelism
        self.mlp = MOEMLP(
            batch_size, hidden_size, tp,
            expert_model_parallel_size, intermediate_size,
            seq_len, moe_router_topk, num_experts,
            layer_id,
            num_shared_experts=num_shared_experts,
        )

        # Router: the MoE router (top-k gating) is a compute-only operation
        # that produces routing indices. No communication impact.
        self.router_weight = None  # modeled as compute-only

    def forward(self):
        workloads = Workload()
        workloads.extend(self.self_attn.forward())
        workloads.extend(self.mlp.forward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.mlp.backward())
        workloads.extend(self.self_attn.backward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads


# ---------------------------------------------------------------------------
# MistralModel -- dense Mistral (7B, etc.)
# ---------------------------------------------------------------------------
class MistralModel(MockedModel):
    """Mistral dense model: Embedding -> N x MistralDecoderLayer -> FinalNorm -> LM Head.

    Identical architecture to LLaMA with sliding window attention.
    """

    def __init__(self, config):
        super().__init__()
        num_kv_heads = getattr(config, "num_kv_heads", None) or config.num_attention_heads
        cp = getattr(config, "context_parallel_size", 1) or 1
        window_size = getattr(config, "sliding_window", None)

        self.embedding = MegatronEmbedding(
            config.padded_vocab_size, config.hidden_size,
            config.tensor_model_parallel_size, config.seq_length, config.micro_batch,
        )

        self.rotary_emb = LlamaRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=getattr(config, "max_position_embeddings", config.seq_length),
            base=getattr(config, "rope_theta", 1000000.0),  # Mistral uses 1e6 base
        )

        self.layers = [
            MistralDecoderLayer(
                config.hidden_size, config.ffn_hidden_size,
                config.num_attention_heads, num_kv_heads,
                config.tensor_model_parallel_size, cp,
                config.seq_length, config.micro_batch,
                i, window_size=window_size,
                sequence_parallel_enabled=config.enable_sequence_parallel,
                computation_enable=config.computation_enable,
                add_bias_linear=config.add_bias_linear,
            )
            for i in range(config.num_layers)
        ]

        self.final_norm = LlamaRMSNorm(config.hidden_size, name="final_norm")
        self.lm_head = MegatronColumnLinear(
            config.hidden_size, config.padded_vocab_size,
            config.tensor_model_parallel_size, config.seq_length, config.micro_batch,
            config.num_layers + 1, "lm_head",
            sequence_parallel_enabled=config.enable_sequence_parallel,
            computation_enable=config.computation_enable,
            add_bias_linear=False,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.embedding.forward())
        for layer in self.layers:
            workloads.extend(layer.forward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()
        for layer in self.layers[::-1]:
            workloads.extend(layer.backward())
        workloads.extend(self.embedding.backward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads


# ---------------------------------------------------------------------------
# MixtralModel -- MoE Mixtral (8x7B, 8x22B)
# ---------------------------------------------------------------------------
class MixtralModel(MockedModel):
    """Mixtral MoE model: Embedding -> N x MixtralDecoderLayer -> FinalNorm -> LM Head.

    Mixtral-8x7B: 32 layers, 8 experts, top-2 routing, 46.7B total / 12.9B active.
    Mixtral-8x22B: 56 layers, 8 experts, top-2 routing, 141B total / 39B active.
    """

    def __init__(self, config):
        super().__init__()
        num_kv_heads = getattr(config, "num_kv_heads", None) or config.num_attention_heads
        cp = getattr(config, "context_parallel_size", 1) or 1
        num_experts = getattr(config, "num_experts", 8)
        moe_router_topk = getattr(config, "moe_router_topk", 2)
        expert_ep = getattr(config, "expert_model_parallel_size", 1)

        self.embedding = MegatronEmbedding(
            config.padded_vocab_size, config.hidden_size,
            config.tensor_model_parallel_size, config.seq_length, config.micro_batch,
        )

        self.rotary_emb = LlamaRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=getattr(config, "max_position_embeddings", config.seq_length),
            base=getattr(config, "rope_theta", 1000000.0),
        )

        self.layers = [
            MixtralDecoderLayer(
                config.hidden_size, config.ffn_hidden_size,
                config.num_attention_heads, num_kv_heads,
                config.tensor_model_parallel_size, cp,
                config.seq_length, config.micro_batch,
                i,
                expert_model_parallel_size=expert_ep,
                num_experts=num_experts,
                moe_router_topk=moe_router_topk,
                moe_grouped_gemm=getattr(config, "moe_grouped_gemm", True),
                sequence_parallel_enabled=config.enable_sequence_parallel,
                computation_enable=config.computation_enable,
                add_bias_linear=config.add_bias_linear,
            )
            for i in range(config.num_layers)
        ]

        self.final_norm = LlamaRMSNorm(config.hidden_size, name="final_norm")
        self.lm_head = MegatronColumnLinear(
            config.hidden_size, config.padded_vocab_size,
            config.tensor_model_parallel_size, config.seq_length, config.micro_batch,
            config.num_layers + 1, "lm_head",
            sequence_parallel_enabled=config.enable_sequence_parallel,
            computation_enable=config.computation_enable,
            add_bias_linear=False,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.embedding.forward())
        for layer in self.layers:
            workloads.extend(layer.forward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()
        for layer in self.layers[::-1]:
            workloads.extend(layer.backward())
        workloads.extend(self.embedding.backward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads
