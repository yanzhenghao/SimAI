"""
Mocked Gemma model for AICB workload generation.

Gemma (Google, 2024-2025): Decoder-only architecture with GeGLU activation.

Key differences from LLaMA:
  - GeGLU activation (GELU on gate_proj) instead of SwiGLU (SiLU on gate_proj)
  - Standard MHA (Gemma 2B/7B) or GQA (Gemma 3)
  - RMSNorm pre-normalization
  - RoPE position embeddings
  - 256K vocab size

Communication-wise, GeGLU is identical to SwiGLU: 3 projections (gate, up, down)
with the same TP sharding pattern. The activation function difference is
compute-only and has no impact on communication modeling.

Reuses LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaMLP from
MockedLlama.py, and MegatronEmbedding, MegatronColumnLinear from MockedMegatron.py.

Supported configs:
  Gemma-2B:   hidden=2048, intermediate=16384, num_heads=8,   num_kv_heads=8,  layers=18
  Gemma-7B:   hidden=3072, intermediate=24576, num_heads=16,  num_kv_heads=16, layers=28
  Gemma-3-12B: hidden=3840, intermediate=15360, num_heads=16, num_kv_heads=8,  layers=48

File: MockedGemma.py
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
)
from log_analyzer.log import Workload, LogItem


class GemmaDecoderLayer(MockedModel):
    """Gemma decoder layer: identical to LLaMA in communication structure.

    GeGLU (GELU on gate_proj) is compute-only. Communication patterns
    are modeled identically via LlamaMLP (SwiGLU) since TP sharding
    of gate/up/down projections is the same.
    """

    def __init__(self, hidden_size, intermediate_size,
                 num_attention_heads, num_kv_heads, tp, cp,
                 seq_len, batch_size, layer_id,
                 sequence_parallel_enabled=True, computation_enable=False,
                 add_bias_linear=False):
        super().__init__()
        self.name = f"gemma_layer_{layer_id}"
        self.layer_id = layer_id

        self.input_layernorm = LlamaRMSNorm(hidden_size,
                                            name=f"input_norm_{layer_id}")
        self.self_attn = LlamaAttention(
            num_attention_heads, num_kv_heads, hidden_size,
            tp, cp, seq_len, batch_size, layer_id,
            sequence_parallel_enabled, computation_enable, add_bias_linear,
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            hidden_size, name=f"post_attn_norm_{layer_id}")
        # GeGLU uses same gate/up/down pattern as SwiGLU
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


class GemmaModel(MockedModel):
    """Gemma model: Embedding -> N x GemmaDecoderLayer -> FinalNorm -> LM Head."""

    def __init__(self, config):
        super().__init__()
        num_kv_heads = getattr(config, "num_kv_heads", None) or config.num_attention_heads
        cp = getattr(config, "context_parallel_size", 1) or 1

        self.embedding = MegatronEmbedding(
            config.padded_vocab_size, config.hidden_size,
            config.tensor_model_parallel_size, config.seq_length, config.micro_batch,
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=getattr(config, "max_position_embeddings", config.seq_length),
            base=getattr(config, "rope_theta", 10000.0),
        )
        self.layers = [
            GemmaDecoderLayer(
                config.hidden_size, config.ffn_hidden_size,
                config.num_attention_heads, num_kv_heads,
                config.tensor_model_parallel_size, cp,
                config.seq_length, config.micro_batch, i,
                config.enable_sequence_parallel,
                config.computation_enable, config.add_bias_linear,
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
        return workloads

    def backward(self):
        workloads = Workload()
        for layer in self.layers[::-1]:
            workloads.extend(layer.backward())
        workloads.extend(self.embedding.backward())
        return workloads
